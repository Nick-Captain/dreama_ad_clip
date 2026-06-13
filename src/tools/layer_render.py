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
    """一条 ffmpeg overlay 输入的描述。x/y 为像素表达式（可含 t、overlay 宽高变量 w/h）。"""
    path: str
    x_expr: str
    y_expr: str
    enable_expr: Optional[str] = None
    prefilter: Optional[str] = None  # 叠加前对该输入的滤镜链（fade/scale 等）
    cx: Optional[float] = None  # 静态预览用：叠加图中心像素（None=全画布叠加）
    cy: Optional[float] = None


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


def _apply_opacity(img: Image.Image, layer: dict) -> Image.Image:
    op = float(layer.get("opacity", 1) or 1)
    if op < 1:
        img.putalpha(img.getchannel("A").point(lambda a: int(a * op)))
    return img


def _rotate_expand(img: Image.Image, layer: dict) -> Image.Image:
    """绕图片中心旋转并扩展画布（Konva 正角=顺时针 → PIL 取负）。"""
    rot = float(layer.get("rotation", 0) or 0)
    if rot:
        img = img.rotate(-rot, expand=True, resample=Image.BICUBIC)
    return img


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

    rotation = float(layer.get("rotation", 0) or 0)
    if rotation:
        # 绕文字中心旋转（Konva 正角=顺时针，PIL 正角=逆时针，取负对齐）
        img = img.rotate(-rotation, center=(canvas_w * x_pct, canvas_h * y_pct), resample=Image.BICUBIC)

    opacity = float(layer.get("opacity", 1) or 1)
    if opacity < 1:
        img.putalpha(img.getchannel("A").point(lambda a: int(a * opacity)))
    return img


def render_text_tight(canvas_w: int, canvas_h: int, text: str, layer: dict, font_path: str):
    """把文字渲染为紧致 RGBA（含描边/高斯阴影/旋转/不透明度），返回 (img, center_x_px, center_y_px)。
    供带动画的文字图层做居中叠加（fade/slide/zoom 需要紧致图，而非全画布）。"""
    size_ratio = float(layer.get("size", 0.06) or 0.06)
    font_size = max(24, int(canvas_w * size_ratio))
    font = _load_font(font_path, font_size)
    stroke = layer.get("stroke") or {}
    stroke_width = int(stroke.get("width", 0) or 0)
    stroke_color = parse_hex_color(stroke.get("color", "#000000"), (0, 0, 0)) + (255,)
    fill_color = parse_hex_color(layer.get("color", "#FFFFFF")) + (255,)
    shadow = layer.get("shadow") or None

    measurer = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measurer.textbbox((0, 0), text or " ", font=font, stroke_width=stroke_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    sh_off = sh_blur = 0
    if shadow:
        sh_blur = max(0.0, float(shadow.get("blur", 4) or 0))
        sh_off = max(2, font_size // 16)
    pad = int(stroke_width + sh_off + 3 * sh_blur + 6)
    W, H = max(1, tw + 2 * pad), max(1, th + 2 * pad)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tx, ty = pad - bbox[0], pad - bbox[1]

    if shadow:
        sh_color = parse_hex_color(shadow.get("color", "#000000"), (0, 0, 0))
        sh_alpha = max(0, min(255, int(255 * float(shadow.get("opacity", 0.6) or 0.6))))
        sd_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sd_img).text(
            (tx + sh_off, ty + sh_off), text, font=font, fill=sh_color + (sh_alpha,),
            stroke_width=stroke_width, stroke_fill=sh_color + (sh_alpha,))
        if sh_blur > 0:
            sd_img = sd_img.filter(ImageFilter.GaussianBlur(sh_blur))
        img = Image.alpha_composite(img, sd_img)

    draw = ImageDraw.Draw(img)
    if stroke_width > 0:
        draw.text((tx, ty), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
    else:
        draw.text((tx, ty), text, font=font, fill=fill_color)

    rotation = float(layer.get("rotation", 0) or 0)
    if rotation:
        img = img.rotate(-rotation, expand=True, resample=Image.BICUBIC)
    img = _apply_opacity(img, layer)
    cx = canvas_w * float(layer.get("x", 0.5) or 0.5)
    cy = canvas_h * float(layer.get("y", 0.7) or 0.7)
    return img, cx, cy


def _segment_enable_expr(layer: dict, segment_bounds: list) -> Optional[str]:
    """显示时机 → enable 表达式；全程显示返回 None。"""
    timing = layer.get("timing", "full")
    if isinstance(timing, dict) and "segment" in timing and segment_bounds:
        idx = max(0, min(int(timing["segment"]), len(segment_bounds) - 1))
        start, end = segment_bounds[idx]
        return f"between(t,{start:.3f},{end:.3f})"
    return None


def _anim_dur(spec: dict, default: float = 0.5) -> float:
    try:
        return max(0.05, float(spec.get("duration", default) or default))
    except (TypeError, ValueError):
        return default


def _has_motion(layer: dict) -> bool:
    """图层是否带任何动画（入场/出场/循环），是则需做成 overlay 而非拍平。"""
    return bool((layer.get("animation") or {}).get("type")
                or (layer.get("entrance") or {}).get("type")
                or (layer.get("exit") or {}).get("type"))


def _build_overlay_motion(layer: dict, cx: float, cy: float, seg_dur: float, canvas_w: int, canvas_h: int, t0: float = 0.0):
    """入场/出场/循环 → 居中叠加的 (x_expr, y_expr, enable_expr, prefilter)。
    以 (cx,cy) 为叠加图中心，借 overlay 的 w/h 变量自动居中（兼容缩放）。
    t0：该图层所在片段的起始时刻（引导语分段时入场/出场相对段起算）。"""
    ent = layer.get("entrance") or {}
    ext = layer.get("exit") or {}
    loop = layer.get("animation") or {}
    D = max(0.1, float(seg_dur or 0.1))
    t0 = max(0.0, float(t0 or 0.0))
    tv = "t" if t0 <= 0 else f"(t-{t0:.3f})"  # 相对段起的时间
    x_terms = [f"({cx:.1f}-w/2)"]
    y_terms = [f"({cy:.1f}-h/2)"]
    prefilters = ["format=rgba"]
    enables = []

    # 入场/出场 淡入淡出（alpha 滤镜，st 为绝对时间）
    if ent.get("type") == "fade":
        prefilters.append(f"fade=t=in:st={t0:.3f}:d={_anim_dur(ent):.3f}:alpha=1")
    if ext.get("type") == "fade":
        dx = _anim_dur(ext)
        prefilters.append(f"fade=t=out:st={t0 + max(0, D - dx):.3f}:d={dx:.3f}:alpha=1")

    # 入场/出场 滑动（位移）
    offH, offV = int(canvas_w * 0.28), int(canvas_h * 0.28)

    def _slide(spec, entering):
        d = spec.get("dir", "down" if entering else "up")
        dur = _anim_dur(spec)
        coef = (offH if d in ("left", "right") else offV) * (-1 if d in ("left", "up") else 1)
        k = f"(1-min(1,{tv}/{dur:.3f}))" if entering else f"max(0,({tv}-{max(0, D - dur):.3f})/{dur:.3f})"
        return ("x" if d in ("left", "right") else "y"), f"{coef}*{k}"

    if ent.get("type") == "slide":
        ax, term = _slide(ent, True); (x_terms if ax == "x" else y_terms).append(term)
    if ext.get("type") == "slide":
        ax, term = _slide(ext, False); (x_terms if ax == "x" else y_terms).append(term)

    # 入场/出场 缩放（scale eval=frame；overlay 用 w/h 自动重新居中）
    zoom = []
    if ent.get("type") == "zoom":
        zoom.append(f"(0.3+0.7*min(1,{tv}/{_anim_dur(ent):.3f}))")
    if ext.get("type") == "zoom":
        dx = _anim_dur(ext)
        zoom.append(f"(1-0.7*max(0,({tv}-{max(0, D - dx):.3f})/{dx:.3f}))")
    if zoom:
        fexpr = "*".join(zoom)
        prefilters.append(f"scale=w='iw*{fexpr}':h='ih*{fexpr}':eval=frame")

    # 循环 shake/float/blink
    lt = loop.get("type", "")
    sp = float(loop.get("speed", 0) or 0)
    if lt == "shake":
        amp = int(canvas_w * float(loop.get("amplitude", 0.012) or 0.012)); fq = sp or 1.0
        x_terms.append(f"{amp}*sin(2*PI*{fq}*t)")
    elif lt == "float":
        amp = int(canvas_h * float(loop.get("amplitude", 0.018) or 0.018)); fq = sp or 0.8
        y_terms.append(f"{amp}*sin(2*PI*{fq}*t)")
    elif lt == "blink":
        period = 1.0 / sp if sp else 0.9
        enables.append(f"lt(mod(t,{period:.3f}),{period * 0.6:.3f})")

    x_expr = "+".join(x_terms).replace("+-", "-")
    y_expr = "+".join(y_terms).replace("+-", "-")
    enable_expr = "*".join(f"({e})" for e in enables) if enables else None
    return x_expr, y_expr, enable_expr, ",".join(prefilters)


def _combine_enable(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return f"({a})*({b})"
    return a or b


def render_static_preview(
    frame_path: str,
    canvas_w: int,
    canvas_h: int,
    layer_doc: dict,
    layer_context: dict | None,
    search_box_image_url: str,
    guide_text: str,
    font_path: str,
    output_path: str,
) -> str:
    """
    H5「精确预览」：与成片同一渲染器输出静态合成图。
    引导语整句显示（不分段）；动画/分段图层按静止状态叠加。
    """
    plan = build_freeze_render_plan(
        frame_path=frame_path,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        layer_doc=layer_doc,
        layer_context=layer_context,
        search_box_image_url=search_box_image_url,
        subtitle_segments=[guide_text] if guide_text else [],
        segment_durations=[1.0] if guide_text else [],
        font_path=font_path,
        tmp_dir=os.path.dirname(output_path) or ".",
        uid=f"preview_{os.getpid()}_{abs(hash(output_path)) % 100000}",
    )
    base = Image.open(plan.base_path).convert("RGBA")
    for spec in plan.overlays:
        try:
            ovl = Image.open(spec.path).convert("RGBA")
            # 预览取动画的静止状态：有中心坐标的按中心贴，否则全画布叠加（引导语等）
            if spec.cx is not None:
                base.paste(ovl, (int(spec.cx - ovl.width / 2), int(spec.cy - ovl.height / 2)), ovl)
            elif ovl.size == base.size:
                base = Image.alpha_composite(base, ovl)
            else:
                base.paste(ovl, (0, 0), ovl)
        finally:
            try:
                os.remove(spec.path)
            except Exception:
                pass
    try:
        os.remove(plan.base_path)
    except Exception:
        pass
    base.convert("RGB").save(output_path, "PNG")
    return output_path


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
    transparent_base: bool = False,
    base_t0: float = 0.0,
) -> FreezeRenderPlan:
    """
    输入图层文档与素材，输出：
    - base_path: 拍平了所有静态全程图层的底图 PNG（透明模式=只含静态层的透明 PNG）
    - overlays:  动画/分段显示图层的 OverlaySpec 列表（z 序同图层顺序）

    transparent_base=True：底图为透明（叠加到视频上而非静态帧，用于「黑屏渐显」）。
    base_t0：所有时间（分段窗口、入场/出场起点）整体偏移这么多秒（黑屏渐显叠加在视频末段）。
    """
    if transparent_base:
        base = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    else:
        base = Image.open(frame_path).convert("RGBA")
        # 抽出的帧若带旋转元数据（手机竖拍常见），ffmpeg 自动旋转后实际像素尺寸会与编码尺寸
        # 宽高互换。一律以底图实际尺寸为准，否则全画布文字图层与 base 尺寸不一致会报错。
        canvas_w, canvas_h = base.size
    overlays: list = []
    overlay_idx = 0

    bounds = []
    cursor = float(base_t0 or 0.0)
    for dur in segment_durations:
        bounds.append((cursor, cursor + dur))
        cursor += dur

    total_dur = cursor if cursor > base_t0 else (base_t0 + 1.0)

    def _save_overlay(img: Image.Image, tag: str) -> str:
        nonlocal overlay_idx
        path = os.path.join(tmp_dir, f"ovl_{uid}_{overlay_idx}_{tag}.png")
        overlay_idx += 1
        img.save(path, "PNG")
        return path

    def _motion_overlay(img: Image.Image, cx: float, cy: float, tag: str, seg_enable):
        path = _save_overlay(img, tag)
        win = max(0.1, total_dur - base_t0)  # 该图层动画窗口长度
        x_expr, y_expr, anim_en, prefilter = _build_overlay_motion(layer, cx, cy, win, canvas_w, canvas_h, t0=base_t0)
        enable = _combine_enable(seg_enable, anim_en)
        if transparent_base and base_t0 > 0:  # 黑屏渐显：图层仅在末段窗口显示
            enable = _combine_enable(f"gte(t,{base_t0:.3f})", enable)
        overlays.append(OverlaySpec(
            path=path, x_expr=x_expr, y_expr=y_expr,
            enable_expr=enable, prefilter=prefilter, cx=cx, cy=cy,
        ))

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
            sb = _apply_opacity(sb, layer)
            sb = _rotate_expand(sb, layer)
            cx = canvas_w * float(layer.get("x", 0.5) or 0.5)
            cy = canvas_h * float(layer.get("y", 0.18) or 0.18)  # 中心定位（与图片一致）
            if _has_motion(layer):
                _motion_overlay(sb, cx, cy, "sb", None)
            else:
                base.paste(sb, (int(cx - sb.width / 2), int(cy - sb.height / 2)), sb)
            continue

        if ltype == "guide_subtitle":
            for i, seg_text in enumerate(subtitle_segments):
                start, end = bounds[i] if i < len(bounds) else (0.0, cursor)
                img, cx, cy = render_text_tight(canvas_w, canvas_h, seg_text, layer, font_path)
                path = _save_overlay(img, f"guide{i}")
                x_expr, y_expr, anim_en, prefilter = _build_overlay_motion(
                    layer, cx, cy, end - start, canvas_w, canvas_h, t0=start)
                seg_en = f"between(t,{start:.3f},{end:.3f})"
                overlays.append(OverlaySpec(
                    path=path, x_expr=x_expr, y_expr=y_expr,
                    enable_expr=_combine_enable(seg_en, anim_en), prefilter=prefilter, cx=cx, cy=cy,
                ))
            continue

        if ltype == "text":
            text = resolve_layer_text(layer, layer_context)
            if not text:
                continue
            seg_enable = _segment_enable_expr(layer, bounds)
            if not _has_motion(layer) and seg_enable is None:
                base = Image.alpha_composite(base, render_text_canvas(canvas_w, canvas_h, text, layer, font_path))
            else:
                img, cx, cy = render_text_tight(canvas_w, canvas_h, text, layer, font_path)
                _motion_overlay(img, cx, cy, "text", seg_enable)
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
            im = _apply_opacity(im, layer)
            im = _rotate_expand(im, layer)  # 旋转后画布扩展，按中心重新定位
            cx = canvas_w * float(layer.get("x", 0.5) or 0.5)
            cy = canvas_h * float(layer.get("y", 0.5) or 0.5)
            seg_enable = _segment_enable_expr(layer, bounds)
            if not _has_motion(layer) and seg_enable is None:
                base.paste(im, (int(cx - im.width / 2), int(cy - im.height / 2)), im)
            else:
                _motion_overlay(im, cx, cy, "img", seg_enable)
            continue

        logger.warning(f"[{uid}] 未知图层类型已忽略: {ltype}")

    base_path = os.path.join(tmp_dir, f"freeze_base_{uid}.png")
    if transparent_base:
        base.save(base_path, "PNG")  # 保留 alpha，叠加到视频上
    else:
        base.convert("RGB").save(base_path, "PNG")
    return FreezeRenderPlan(base_path=base_path, overlays=overlays)
