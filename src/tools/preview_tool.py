"""
预览图生成工具

功能：
1. 支持传入视频URL，自动提取最后一帧
2. 支持传入帧图片URL，直接合成
3. 在定格帧上叠加字幕和搜索框，生成预览图
4. 所有参数可调：字幕位置/颜色/字号/描边，搜索框位置/大小
"""

import os
import json
import logging
import tempfile
import uuid
import requests
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from langchain.tools import tool
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.runtime_ctx.context import new_context
from coze_coding_utils.log.write_log import request_context

# 复用 video_pipeline 中的函数
from tools.video_pipeline import (
    _download_file,
    _extract_clean_last_frame,
    _get_video_resolution,
    _find_chinese_font,
    _get_ctx,
    _get_storage,
    _upload_image_to_s3,
    _is_black_frame,
    _frame_has_subtitle,
    _remove_subtitle_with_seedream,
)

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_SUBTITLE_Y_PCT = 0.70       # 字幕垂直位置（画面70%处）
DEFAULT_SUBTITLE_FONT_SIZE_RATIO = 0.06  # 字号 = 视频宽度 × 6%
DEFAULT_SUBTITLE_FONT_COLOR = "white"
DEFAULT_SUBTITLE_BORDER_COLOR = "black"
DEFAULT_SUBTITLE_BORDER_WIDTH = 3
DEFAULT_SEARCH_BOX_Y_PCT = 0.15     # 搜索框垂直位置（画面15%处）
DEFAULT_SEARCH_BOX_SCALE = 0.70     # 搜索框宽度占画面70%


def _composite_preview_image(
    frame_image_url: str,
    subtitle_text: str,
    search_box_image_url: Optional[str],
    output_path: str,
    video_w: int,
    video_h: int,
    subtitle_y_pct: float = DEFAULT_SUBTITLE_Y_PCT,
    subtitle_font_size_ratio: float = DEFAULT_SUBTITLE_FONT_SIZE_RATIO,
    subtitle_font_color: str = DEFAULT_SUBTITLE_FONT_COLOR,
    subtitle_border_color: str = DEFAULT_SUBTITLE_BORDER_COLOR,
    subtitle_border_width: int = DEFAULT_SUBTITLE_BORDER_WIDTH,
    search_box_y_pct: float = DEFAULT_SEARCH_BOX_Y_PCT,
    search_box_scale: float = DEFAULT_SEARCH_BOX_SCALE,
) -> str:
    """
    合成预览图：定格帧 + 搜索框 + 字幕
    """
    # 下载帧图片
    resp = requests.get(frame_image_url, timeout=60)
    resp.raise_for_status()
    frame_img = Image.open(BytesIO(resp.content)).convert("RGBA")

    # 合成搜索框
    if search_box_image_url:
        resp2 = requests.get(search_box_image_url, timeout=60)
        resp2.raise_for_status()
        search_box = Image.open(BytesIO(resp2.content)).convert("RGBA")

        target_w = int(video_w * search_box_scale)
        ratio = target_w / search_box.width
        target_h = int(search_box.height * ratio)
        search_box = search_box.resize((target_w, target_h), Image.Resampling.LANCZOS)

        pos_x = (video_w - target_w) // 2
        pos_y = int(video_h * search_box_y_pct)
        frame_img.paste(search_box, (pos_x, pos_y), search_box)

    # 绘制字幕
    draw = ImageDraw.Draw(frame_img)
    font_path = _find_chinese_font()
    font_size = max(24, int(video_w * subtitle_font_size_ratio))

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # 计算字幕位置
    bbox = draw.textbbox((0, 0), subtitle_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (video_w - text_w) // 2
    text_y = int(video_h * subtitle_y_pct) - text_h // 2

    # 描边效果：在四个方向偏移绘制
    if subtitle_border_width > 0:
        for dx in range(-subtitle_border_width, subtitle_border_width + 1):
            for dy in range(-subtitle_border_width, subtitle_border_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), subtitle_text, font=font, fill=subtitle_border_color)

    # 绘制主文字
    draw.text((text_x, text_y), subtitle_text, font=font, fill=subtitle_font_color)

    # 保存
    frame_img = frame_img.convert("RGB")
    frame_img.save(output_path, "PNG")
    return output_path


@tool
def preview_frame(
    video_url: str = "",
    frame_image_url: str = "",
    subtitle_text: str = "后续剧情该如何选择？快来左下角造梦次元",
    search_box_image_url: str = "",
    subtitle_y_pct: float = DEFAULT_SUBTITLE_Y_PCT,
    subtitle_font_size_ratio: float = DEFAULT_SUBTITLE_FONT_SIZE_RATIO,
    subtitle_font_color: str = DEFAULT_SUBTITLE_FONT_COLOR,
    subtitle_border_color: str = DEFAULT_SUBTITLE_BORDER_COLOR,
    subtitle_border_width: int = DEFAULT_SUBTITLE_BORDER_WIDTH,
    search_box_y_pct: float = DEFAULT_SEARCH_BOX_Y_PCT,
    search_box_scale: float = DEFAULT_SEARCH_BOX_SCALE,
) -> str:
    """
    生成广告尾帧的静态预览图。支持传入视频URL（自动提取最后一帧）或帧图片URL。

    参数说明：
    - video_url: 用户上传的视频URL（与 frame_image_url 二选一，优先 video_url）
    - frame_image_url: 已有的帧图片URL（video_url 为空时使用）
    - subtitle_text: 字幕文字内容
    - search_box_image_url: 搜索框透明背景图片URL（可选）
    - subtitle_y_pct: 字幕垂直位置，0.0=顶部 1.0=底部，默认0.70
    - subtitle_font_size_ratio: 字号比例（相对视频宽度），默认0.06
    - subtitle_font_color: 字体颜色，默认"white"
    - subtitle_border_color: 描边颜色，默认"black"
    - subtitle_border_width: 描边粗细，默认3
    - search_box_y_pct: 搜索框垂直位置，默认0.15
    - search_box_scale: 搜索框宽度占比，默认0.70

    返回：包含预览图URL和当前参数的JSON字符串
    """
    ctx = _get_ctx()
    tmp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:12]

    try:
        # Step 1: 获取帧图片
        if video_url:
            logger.info(f"[{uid}] 从视频提取最后一帧: {video_url}")
            video_w, video_h = _get_video_resolution(video_url)
            frame_local = os.path.join(tmp_dir, f"preview_frame_{uid}.png")
            _extract_clean_last_frame(video_url, frame_local, video_w, video_h, uid)
            frame_image_url = _upload_image_to_s3(frame_local, f"temp/preview_frame_{uid}.png")
            logger.info(f"[{uid}] 帧图片URL: {frame_image_url}")
        elif frame_image_url:
            # 需要获取分辨率，从图片本身读取
            resp = requests.get(frame_image_url, timeout=60)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            video_w, video_h = img.size
            logger.info(f"[{uid}] 使用已有帧图片, 分辨率: {video_w}x{video_h}")
        else:
            return json.dumps({
                "success": False,
                "error": "请提供 video_url 或 frame_image_url"
            }, ensure_ascii=False)

        # Step 2: 合成预览图
        output_local = os.path.join(tmp_dir, f"preview_{uid}.png")
        _composite_preview_image(
            frame_image_url=frame_image_url,
            subtitle_text=subtitle_text,
            search_box_image_url=search_box_image_url if search_box_image_url else None,
            output_path=output_local,
            video_w=video_w,
            video_h=video_h,
            subtitle_y_pct=subtitle_y_pct,
            subtitle_font_size_ratio=subtitle_font_size_ratio,
            subtitle_font_color=subtitle_font_color,
            subtitle_border_color=subtitle_border_color,
            subtitle_border_width=subtitle_border_width,
            search_box_y_pct=search_box_y_pct,
            search_box_scale=search_box_scale,
        )

        # Step 3: 上传预览图到S3
        preview_url = _upload_image_to_s3(output_local, f"temp/preview_{uid}.png")

        # 清理临时文件
        try:
            os.remove(output_local)
        except Exception:
            pass

        return json.dumps({
            "success": True,
            "preview_url": preview_url,
            "current_params": {
                "subtitle_y_pct": subtitle_y_pct,
                "subtitle_font_size_ratio": subtitle_font_size_ratio,
                "subtitle_font_color": subtitle_font_color,
                "subtitle_border_color": subtitle_border_color,
                "subtitle_border_width": subtitle_border_width,
                "search_box_y_pct": search_box_y_pct,
                "search_box_scale": search_box_scale,
            },
            "video_resolution": f"{video_w}x{video_h}",
            "subtitle_text": subtitle_text,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[{uid}] 预览图生成失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"预览图生成失败: {str(e)}",
        }, ensure_ascii=False)
