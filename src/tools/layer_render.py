"""
中间帧图层渲染器

把图层文档渲染为「拍平底图 + ffmpeg overlay 规格列表」：
- 静态且全程显示的图层（搜索框/文字/图片）直接拍平进底图，零额外开销
- 带动画或带显示时机的图层渲染为独立透明 PNG，由 ffmpeg overlay 按
  时间表达式合成（抖动/浮动=正弦位移，闪烁/分段显示=enable 表达式）
- 引导语字幕逐段渲染为全画布透明 PNG，按 TTS 节奏 enable 显示

所有文字（含描边、高斯模糊阴影）统一由 PIL 渲染——预览接口与成片
共用本模块，保证所见即所得。
"""

import logging
import math
import os
import requests
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tools.layer_model import parse_hex_color, resolve_layer_text

logger = logging.getLogger(__name__)


@dataclass
class OverlaySpec:
    """一条 ffmpeg overlay 输入的描述。x/y 为像素表达式（可含 t）。"""
    path: str
    x_expr: str
    y_expr: str
    enable_expr: Optional[str] = None


@dataclass
class FreezeRenderPlan:
    base_path: str
    overlays: list = field(default_factory=list)


def _load_font(font_path: str, size: int):
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")


def render_text_canvas(
    canvas_w: int,
    canvas_h: int,
    text: str,
    layer: dict,
    font_path: str,
) -> Image.Image:
    """把一个文字图层渲染为全画布透明 RGBA（含描边与高斯模糊阴影）。"""
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if not text:
        return img

    size_ratio = float(layer.get("size", 0.06) or 0.06)
    font_size = max(24, int(canvas_w * size_ratio))
    font = _load_font(font_path, font_size)

    stroke = layer.get("stroke") or {}
    stroke_width = int(stroke.get("width", 0) or 0)
    stroke_color = parse_hex_color(stroke.get("color", "#000000"), (0, 0, 0)) + (255,)
    fill_color = parse_hex_color(layer.get("color", "#FFFFFF")) + (255,)

    measurer = ImageDraw.Draw(img)
    bbox = measurer.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x_pct = float(layer.get("x", 0.5) or 0.5)
    y_pct = float(layer.get("y", 0.7) or 0.7)
    tx = int(canvas_w * x_pct - text_w / 2) - bbox[0]
    ty = int(canvas_h * y_pct - text_h / 2) - bbox[1]

    shadow = layer.get("shadow") or None
    if shadow:
        sh_color = parse_hex_color(shadow.get("color", "#000000"), (0, 0, 0))
        sh_alpha = max(0, min(255, int(255 * float(shadow.get("opacity", 0.6) or 0.6))))
        sh_blur = max(0.0, float(shadow.get("blur", 4) or 0))
        offset = max(2, font_size // 16)
        shadow_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_img)
        sd.text(
            (tx + offset, ty + offset), text, font=font,
            fill=sh_color + (sh_alpha,),
            stroke_width=stroke_width, stroke_fill=sh_color + (sh_alpha,),
        )
        if sh_blur > 0:
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(sh_blur))
        img = Image.alpha_composite(img, shadow_img)

    draw = ImageDraw.Draw(img)
    if stroke_width > 0:
        draw.text((tx, ty), text, font=font, fill=fill_color,
                  stroke_width=stroke_width, stroke_fill=stroke_color)
    else:
        draw.text((tx, ty), text, font=font, fill=fill_color)
    return img


def _segment_enable_expr(layer: dict, segment_bounds: list) -> Optional[str]:
    """显示时机 → enable 表达式；全程显示返回 None。"""
    timing = layer.get("timing", "full")
    if isinstance(timing, dict) and "segment" in timing and segment_bounds:
        idx = max(0, min(int(timing["segment"]), len(segment_bounds) - 1))
        start, end = segment_bounds[idx]
        return f"between(t,{start:.3f},{end:.3f})"
    return None


def _animation_exprs(layer: dict, base_x: int, base_y: int, canvas_w: int, canvas_h: int):
    """动画 → (x表达式, y表达式, 附加enable表达式)。"""
    anim = layer.get("animation") or {}
    kind = anim.get("type", "")
    speed = float(anim.get("speed", 0) or 0)
    if kind == "shake":
        amp = int(canvas_w * float(anim.get("amplitude", 0.012) or 0.012))
        freq = speed or 8.0
        return f"{base_x}+{amp}*sin(2*PI*{freq}*t)", str(base_y), None
    if kind == "float":
        amp = int(canvas_h * float(anim.get("amplitude", 0.018) or 0.018))
        freq = speed or 0.8
        return str(base_x), f"{base_y}+{amp}*sin(2*PI*{freq}*t)", None
    if kind == "blink":
        period = 1.0 / speed if speed else 0.9
        on_time = period * 0.6
        return str(base_x), str(base_y), f"lt(mod(t,{period:.3f}),{on_time:.3f})"
    return str(base_x), str(base_y), None


def _combine_enable(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return f"({a})*({b})"
    return a or b


def build_freeze_render_plan(
    frame_path: str,
    canvas_w: int,
    canvas_h: int,
    layer_doc: dict,
    layer_context: dict | None,
    search_box_image_url: str,
    subtitle_segments: list,
    segment_durations: list,
    font_path: str,
    tmp_dir: str,
    uid: str,
) -> FreezeRenderPlan:
    """
    输入图层文档与素材，输出：
    - base_path: 拍平了所有静态全程图层的底图 PNG
    - overlays:  动画/分段显示图层的 OverlaySpec 列表（z 序同图层顺序）
    """
    base = Image.open(frame_path).convert("RGBA")
    overlays: list = []
    overlay_idx = 0

    bounds = []
    cursor = 0.0
    for dur in segment_durations:
        bounds.append((cursor, cursor + dur))
        cursor += dur

    def _save_overlay(img: Image.Image, tag: str) -> str:
        nonlocal overlay_idx
        path = os.path.join(tmp_dir, f"ovl_{uid}_{overlay_idx}_{tag}.png")
        overlay_idx += 1
        img.save(path, "PNG")
        return path

    for layer in layer_doc.get("layers", []):
        if not layer.get("visible", True):
            continue
        ltype = layer.get("type", "")

        if ltype == "search_box":
            if not search_box_image_url:
                continue
            try:
                sb = _download_image(search_box_image_url)
            except Exception as e:
                logger.warning(f"[{uid}] 搜索框图片下载失败，跳过该图层: {e}")
                continue
            target_w = max(1, int(canvas_w * float(layer.get("scale", 0.7) or 0.7)))
            target_h = max(1, int(sb.height * target_w / sb.width))
            sb = sb.resize((target_w, target_h), Image.Resampling.LANCZOS)
            pos_x = int(canvas_w * float(layer.get("x", 0.5) or 0.5) - target_w / 2)
            pos_y = int(canvas_h * float(layer.get("y", 0.15) or 0.15))
            base.paste(sb, (pos_x, pos_y), sb)
            continue

        if ltype == "guide_subtitle":
            for i, seg_text in enumerate(subtitle_segments):
                seg_img = render_text_canvas(canvas_w, canvas_h, seg_text, layer, font_path)
                path = _save_overlay(seg_img, f"guide{i}")
                start, end = bounds[i] if i < len(bounds) else (0.0, cursor)
                overlays.append(OverlaySpec(
                    path=path, x_expr="0", y_expr="0",
                    enable_expr=f"between(t,{start:.3f},{end:.3f})",
                ))
            continue

        if ltype == "text":
            text = resolve_layer_text(layer, layer_context)
            if not text:
                continue
            txt_img = render_text_canvas(canvas_w, canvas_h, text, layer, font_path)
            seg_enable = _segment_enable_expr(layer, bounds)
            if seg_enable is None:
                base = Image.alpha_composite(base, txt_img)
            else:
                path = _save_overlay(txt_img, "text")
                overlays.append(OverlaySpec(path=path, x_expr="0", y_expr="0", enable_expr=seg_enable))
            continue

        if ltype == "image":
            url = (layer.get("url") or "").strip()
            if not url:
                continue
            try:
                im = _download_image(url)
            except Exception as e:
                logger.warning(f"[{uid}] 图片图层下载失败，跳过: {e}")
                continue
            target_w = max(1, int(canvas_w * float(layer.get("scale", 0.2) or 0.2)))
            target_h = max(1, int(im.height * target_w / im.width))
            im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
            pos_x = int(canvas_w * float(layer.get("x", 0.5) or 0.5) - target_w / 2)
            pos_y = int(canvas_h * float(layer.get("y", 0.5) or 0.5) - target_h / 2)
            seg_enable = _segment_enable_expr(layer, bounds)
            has_anim = bool((layer.get("animation") or {}).get("type"))
            if not has_anim and seg_enable is None:
                base.paste(im, (pos_x, pos_y), im)
                continue
            path = _save_overlay(im, "img")
            x_expr, y_expr, anim_enable = _animation_exprs(layer, pos_x, pos_y, canvas_w, canvas_h)
            overlays.append(OverlaySpec(
                path=path, x_expr=x_expr, y_expr=y_expr,
                enable_expr=_combine_enable(seg_enable, anim_enable),
            ))
            continue

        logger.warning(f"[{uid}] 未知图层类型已忽略: {ltype}")

    base_path = os.path.join(tmp_dir, f"freeze_base_{uid}.png")
    base.convert("RGB").save(base_path, "PNG")
    return FreezeRenderPlan(base_path=base_path, overlays=overlays)
