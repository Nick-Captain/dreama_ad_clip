"""
中间帧图层数据模型

「样式参数」列存储的 JSON 文档结构：
{
  "version": 1,
  "layers": [ {<layer>}, ... ]   # z 序 = 列表顺序，靠后的在上层
}

图层类型与坐标语义（均为相对画布的百分比，适配任意分辨率）：
- search_box      搜索框贴图。x=水平中心(0.5=居中)，y=顶边位置，scale=宽度占比
- guide_subtitle  引导语字幕。x=水平中心，y=垂直中心；文案来自记录「引导语」，
                  跟随 TTS 节奏分段显示（该行为由管线控制，图层只管样式）
- text            普通文字（角色名/自定义）。x/y=文字中心；text_source 指定取值来源
- image           图片贴图。x/y=图片中心，scale=宽度占比；可带 animation

文字样式：color(#RRGGBB)、stroke{color,width}、shadow{color,opacity,blur}
不透明度 opacity：0~1，作用于图片与文字整体（默认 1=不透明）；可重命名 name 字段
显示时机 timing："full"=全程显示，{"segment": N}=跟随引导语第 N 段（从 0 计）
动画 animation：{"type": "shake"|"float"|"blink", "amplitude":..., "speed":...}
"""

import copy
import json
import logging
import re

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# 内置默认图层：与图层化改造前的固定样式完全一致（保证存量行为无回归），
# 外加设计定稿新增的「角色名」默认图层（角色名列为空时自动隐藏）
BUILTIN_DEFAULT_LAYERS = {
    "version": SCHEMA_VERSION,
    "layers": [
        {
            "id": "search_box",
            "type": "search_box",
            "visible": True,
            "x": 0.5,
            "y": 0.15,
            "scale": 0.70,
        },
        {
            "id": "role_name",
            "type": "text",
            "visible": True,
            "text_source": "角色名",
            "text": "",
            "x": 0.76,
            "y": 0.175,
            "size": 0.035,
            "color": "#242424",
            "stroke": None,
            "shadow": None,
            "timing": "full",
        },
        {
            "id": "guide",
            "type": "guide_subtitle",
            "visible": True,
            "x": 0.5,
            "y": 0.70,
            "size": 0.06,
            "color": "#FFFFFF",
            "stroke": {"color": "#000000", "width": 3},
            "shadow": None,
        },
    ],
}


def default_layer_doc() -> dict:
    return copy.deepcopy(BUILTIN_DEFAULT_LAYERS)


def parse_layer_doc(raw: str) -> dict | None:
    """解析「样式参数」列的 JSON 文本，非法/为空返回 None。"""
    if not raw or not raw.strip():
        return None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[图层] 样式参数 JSON 解析失败: {raw[:120]}")
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("layers"), list) or not doc["layers"]:
        return None
    return doc


def resolve_layer_doc(record_raw: str = "", global_doc: dict | None = None) -> dict:
    """
    样式文档取用优先级：记录级参数 > 全局默认 > 内置默认。
    文档是整体替换关系（H5 保存的是完整图层列表），不做字段级合并。
    """
    doc = parse_layer_doc(record_raw)
    if doc is not None:
        return doc
    if isinstance(global_doc, dict) and isinstance(global_doc.get("layers"), list) and global_doc["layers"]:
        return copy.deepcopy(global_doc)
    return default_layer_doc()


def parse_hex_color(value: str, fallback: tuple = (255, 255, 255)) -> tuple:
    """#RGB / #RRGGBB → (r, g, b)，非法值回退 fallback。"""
    if not isinstance(value, str):
        return fallback
    v = value.strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{3}", v):
        v = "".join(ch * 2 for ch in v)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", v):
        return fallback
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def resolve_layer_text(layer: dict, context: dict | None) -> str:
    """按 text_source 解析文字图层的实际文案；返回空串表示该图层应跳过。"""
    source = layer.get("text_source", "")
    if source and source != "自定义":
        value = (context or {}).get(source, "")
        return str(value).strip()
    return str(layer.get("text", "")).strip()
