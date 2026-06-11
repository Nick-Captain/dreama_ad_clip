"""
广告尾帧视频处理管线工具

功能：
1. 用 ffmpeg 提取用户视频最后一帧 + 检测原视频分辨率
2. 合成搜索框图片到定格帧
3. 字幕分段 → 每段分别 TTS 配音 → 拼接音频（按实际朗读节奏）
4. 用 ffmpeg 本地生成静止定格视频 + 字幕烧录 + 音频合成
5. 拼接：用户视频 + 定格视频 + 广告尾帧
6. 可选 BGM：裁剪、调音量、混入最终视频
"""

import os
import json
import logging
import math
import tempfile
import uuid
import re
import subprocess
import requests
from typing import Optional, List, Tuple

# Monkey-patch: TOS 预签名 URL 不支持 HEAD 请求（返回 403），
# 但 GET 下载正常。跳过 SDK 的 HEAD 预检以避免误判。
from coze_coding_dev_sdk.core import url_utils as _sdk_url_utils
_orig_validate_url = _sdk_url_utils.validate_url


def _patched_validate_url(url: str, **kwargs):
    kwargs.setdefault("allow_head_check", False)
    return _orig_validate_url(url, **kwargs)


_sdk_url_utils.validate_url = _patched_validate_url
from PIL import Image
from io import BytesIO

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from coze_coding_dev_sdk.video_edit import (
    VideoEditClient,
    OutputSync,
)
from coze_coding_dev_sdk import TTSClient, ImageGenerationClient
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.runtime_ctx.context import new_context, Context, default_headers
from coze_coding_utils.log.write_log import request_context

logger = logging.getLogger(__name__)

# ============================================================
# 内置广告尾帧库（已上传到对象存储的预签名URL）
# ============================================================
BUILTIN_TAILS = {
    "派对接引尾帧": "https://coze-coding-project.tos.coze.site/coze_storage_7649297641398009898/ad_tails/tail_paidui_f03fa26a.mp4?sign=1783590022-2718e160ef-0-28db0da062289806519930bc5ca01e69656c5f0ed039315998d00a8231e15b85",
    "短剧推广尾帧": "https://coze-coding-project.tos.coze.site/coze_storage_7649297641398009898/ad_tails/tail_duanju_e6f2d8d8.mp4?sign=1783590022-58e9d687cc-0-47ad19bd216e0355bdf67258f080ba8ea6adf59ebd58e2bca80ed3415dfbb725",
}

# ============================================================
# TTS 音色选项
# ============================================================
VOICE_OPTIONS = {
    "小荷（通用女声）": "zh_female_xiaohe_uranus_bigtts",
    "米仔（视频配音女声）": "zh_female_mizai_saturn_bigtts",
    "大奕（视频配音男声）": "zh_male_dayi_saturn_bigtts",
    "可爱女生": "saturn_zh_female_keainvsheng_tob",
}

# ============================================================
# 转场效果选项
# ============================================================
TRANSITION_OPTIONS = {
    "硬切（无转场）": None,
    "叶片翻转": "1182355",
    "百叶窗": "1182356",
    "风吹": "1182357",
    "交替出场": "1182359",
    "旋转放大": "1182360",
    "泛开": "1182358",
    "风车": "1182362",
    "多色混合": "1182363",
    "遮罩转场": "1182364",
    "六角形": "1182365",
    "心型打开": "1182366",
    "故障转换": "1182367",
    "飞眼": "1182368",
    "梦幻放大": "1182369",
    "开门展现": "1182370",
    "对角擦除": "1182371",
    "立方转换": "1182373",
    "透镜变换": "1182374",
    "晚霞转场": "1182375",
    "圆形打开": "1182376",
    "圆形擦开": "1182377",
    "圆形交替": "1182378",
    "时钟扫开": "1182379",
}

# ============================================================
# 默认文案
# ============================================================
DEFAULT_GUIDE_TEXT = "后续剧情该如何选择？快来左下角造梦次元"
DEFAULT_SUBTITLE_TEXT = "后续剧情该如何选择？快来左下角造梦次元"


def _get_ctx():
    """获取请求上下文"""
    ctx = request_context.get()
    if ctx is None:
        ctx = new_context(method="video_pipeline")
    return ctx


def _get_storage():
    """获取对象存储客户端"""
    return S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )


def _download_file(url: str, local_path: str) -> str:
    """下载文件到本地"""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return local_path


def _upload_image_to_s3(local_path: str, remote_name: str) -> str:
    """上传图片到对象存储并返回预签名URL"""
    storage = _get_storage()
    with open(local_path, "rb") as f:
        key = storage.stream_upload_file(
            fileobj=f,
            file_name=remote_name,
            content_type="image/png",
        )
    return storage.generate_presigned_url(key=key, expire_time=2592000)


def _upload_video_to_s3(local_path: str, remote_name: str) -> str:
    """上传视频到对象存储并返回预签名URL"""
    storage = _get_storage()
    with open(local_path, "rb") as f:
        key = storage.stream_upload_file(
            fileobj=f,
            file_name=remote_name,
            content_type="video/mp4",
        )
    return storage.generate_presigned_url(key=key, expire_time=2592000)


def _get_video_resolution(video_url: str) -> Tuple[int, int]:
    """获取视频分辨率 (width, height)，先下载再用 ffprobe"""
    tmp_v = os.path.join(tempfile.gettempdir(), f"vres_{uuid.uuid4().hex[:8]}.mp4")
    _download_file(video_url, tmp_v)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        tmp_v,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    w, h = result.stdout.strip().split(",")
    os.remove(tmp_v)
    return int(w), int(h)


def _get_video_duration(local_path: str) -> float:
    """获取本地视频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        local_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def _get_llm():
    """获取多模态 LLM 实例，用于字幕检测"""
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
    return ChatOpenAI(
        model="doubao-seed-2-0-lite-260215",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=60,
        default_headers=default_headers(_get_ctx()),
    )


def _is_black_frame(frame_image_path: str, threshold: float = 10.0) -> bool:
    """
    检测帧画面是否为黑屏（用户使用了渐隐出场特效）。
    计算画面平均亮度，低于阈值则判定为黑屏。
    threshold: 平均像素亮度阈值，默认 10（0-255 范围）
    """
    img = Image.open(frame_image_path).convert("L")  # 灰度
    pixels = list(img.getdata())
    avg_brightness = sum(pixels) / len(pixels)
    is_black = avg_brightness < threshold
    logger.info(f"黑屏检测: 平均亮度={avg_brightness:.1f}, 阈值={threshold}, is_black={is_black}")
    return is_black


def _frame_has_subtitle(frame_image_path: str) -> bool:
    """
    用多模态 LLM 检测帧画面中是否包含字幕/文字。
    返回 True 表示有字幕，False 表示无字幕。
    """
    frame_url = _upload_image_to_s3(frame_image_path, f"temp/frame_check_{uuid.uuid4().hex[:8]}.png")

    llm = _get_llm()
    msg = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": frame_url},
        },
        {
            "type": "text",
            "text": "请仔细观察这张视频截图，画面底部或中部是否有字幕文字（包括中文、英文、数字等任何文字叠加）？请只回答「有」或「无」，不要解释。",
        },
    ])
    resp = llm.invoke([msg])
    content = resp.content
    if isinstance(content, list):
        answer = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                answer = str(item.get("text", ""))
                break
        if not answer:
            answer = str(content[0]) if content else ""
    else:
        answer = str(content).strip()
    has_sub = "有" in answer and "无" not in answer
    logger.info(f"字幕检测结果: '{answer}' → has_subtitle={has_sub}")
    return has_sub


def _remove_subtitle_with_seedream(frame_image_path: str, output_path: str, video_w: int, video_h: int, uid: str) -> str:
    """
    使用 Seedream 4.0 的 Image-to-Image 能力去除帧画面中的字幕。
    传入有字幕的帧作为参考图，prompt 指示去除文字，输出干净帧。
    """
    logger.info(f"[{uid}] 使用 Seedream 4.0 去除字幕...")

    # 上传有字幕的帧到 S3
    frame_url = _upload_image_to_s3(frame_image_path, f"temp/frame_with_sub_{uid}.png")

    ctx = _get_ctx()
    client = ImageGenerationClient(ctx=ctx)

    size_str = f"{video_w}x{video_h}"

    response = client.generate(
        prompt="去除画面中所有文字和字幕，保持原始画面内容、色彩、构图完全不变，只移除文字叠加层",
        image=frame_url,
        size=size_str,
        model="doubao-seedream-5-0-260128",
    )

    if not response.success:
        error_msgs = response.error_messages if hasattr(response, 'error_messages') else "未知错误"
        logger.error(f"[{uid}] Seedream 4.0 去字幕失败: {error_msgs}")
        raise RuntimeError(f"Seedream 4.0 去字幕失败: {error_msgs}")

    clean_url = response.image_urls[0]
    logger.info(f"[{uid}] Seedream 4.0 去字幕成功: {clean_url}")

    # 下载去字幕后的帧
    _download_file(clean_url, output_path)
    return output_path


def _extract_frame_at_time(video_path: str, seek_time: float, output_path: str) -> str:
    """在指定时间点提取一帧"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"提取帧失败 (seek={seek_time}s): {result.stderr[-300:]}")
    return output_path


def _extract_clean_last_frame(video_url: str, output_path: str, video_w: int, video_h: int, uid: str) -> str:
    """
    提取视频干净的最后帧（无黑屏、无字幕）：

    流程：
    1. 提取最后一帧
    2. 检测是否黑屏 → 是：向前回退 0.5s（最多 5s）找非黑帧
    3. 检测是否有字幕 → 是：用 Seedream 4.0 去字幕
    4. 返回干净帧
    """
    tmp_v = os.path.join(tempfile.gettempdir(), f"lastframe_src_{uid}.mp4")
    _download_file(video_url, tmp_v)
    duration = _get_video_duration(tmp_v)

    # Step 1: 提取最后一帧
    seek_time = max(0, duration - 0.1)
    _extract_frame_at_time(tmp_v, seek_time, output_path)
    logger.info(f"[{uid}] 提取最后一帧 (t={seek_time:.2f}s)")

    # Step 2: 黑屏检测 → 回退找非黑帧
    if _is_black_frame(output_path):
        logger.info(f"[{uid}] 最后一帧为黑屏，向前回退寻找非黑帧")
        found_non_black = False

        for step in range(1, 11):  # 最多回退 5 秒 (10 * 0.5s)
            back_time = max(0, duration - 0.1 - step * 0.5)
            if back_time <= 0:
                break
            _extract_frame_at_time(tmp_v, back_time, output_path)
            if not _is_black_frame(output_path):
                logger.info(f"[{uid}] 在回退 {step * 0.5:.1f}s 处找到非黑帧 (t={back_time:.2f}s)")
                found_non_black = True
                break

        if not found_non_black:
            logger.warning(f"[{uid}] 回退范围内均为黑屏，使用回退最远的帧")

    # Step 3: 字幕检测 → Seedream 4.0 去字幕
    if _frame_has_subtitle(output_path):
        logger.info(f"[{uid}] 帧画面有字幕，使用 Seedream 4.0 去除")
        _remove_subtitle_with_seedream(output_path, output_path, video_w, video_h, uid)
    else:
        logger.info(f"[{uid}] 帧画面无字幕，直接使用")

    os.remove(tmp_v)
    return output_path


def _split_subtitle(text: str) -> List[str]:
    """
    将字幕文本按规则拆分：
    1. 优先按原标点位置自然分段
    2. 每段不超过12个字符
    3. 如果某段仍超12字，再均匀拆分
    """
    max_len = 12

    punct_pattern = re.compile(r'[^\u4e00-\u9fff\w\s]')
    punct_positions = [m.start() for m in punct_pattern.finditer(text)]

    cleaned = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
    cleaned = re.sub(r'\s+', '', cleaned)

    if not cleaned:
        return [text]

    if len(cleaned) <= max_len:
        return [cleaned]

    char_map = []
    cleaned_idx = 0
    for ch in text:
        if re.match(r'[^\u4e00-\u9fff\w\s]', ch) or ch.isspace():
            char_map.append(-1)
        else:
            char_map.append(cleaned_idx)
            cleaned_idx += 1

    split_positions = []
    for pos in punct_positions:
        if pos > 0 and char_map[pos - 1] >= 0:
            split_positions.append(char_map[pos - 1] + 1)

    split_positions = sorted(set(split_positions))

    segments = []
    start = 0
    for sp in split_positions:
        if sp <= start:
            continue
        if sp > len(cleaned):
            break
        seg = cleaned[start:sp]
        if len(seg) <= max_len:
            segments.append(seg)
            start = sp

    if start < len(cleaned):
        remaining = cleaned[start:]
        while remaining:
            if len(remaining) <= max_len:
                segments.append(remaining)
                break
            segments.append(remaining[:max_len])
            remaining = remaining[max_len:]

    return segments if segments else [cleaned]


def _composite_search_box(
    frame_image_path: str,
    search_box_image_url: Optional[str],
    output_path: str,
) -> str:
    """
    将搜索框图片合成到定格帧上
    - 搜索框居中偏上，占画面宽度70%
    """
    frame_img = Image.open(frame_image_path).convert("RGBA")

    if search_box_image_url:
        resp = requests.get(search_box_image_url, timeout=60)
        resp.raise_for_status()
        search_box = Image.open(BytesIO(resp.content)).convert("RGBA")

        frame_w, frame_h = frame_img.size
        target_w = int(frame_w * 0.7)
        ratio = target_w / search_box.width
        target_h = int(search_box.height * ratio)
        search_box = search_box.resize((target_w, target_h), Image.Resampling.LANCZOS)

        pos_x = (frame_w - target_w) // 2
        pos_y = int(frame_h * 0.15)

        frame_img.paste(search_box, (pos_x, pos_y), search_box)

    frame_img = frame_img.convert("RGB")
    frame_img.save(output_path, "PNG")
    return output_path


def _get_audio_duration(audio_url: str) -> float:
    """获取音频时长（秒），通过下载后用 ffprobe 获取"""
    tmp_audio = os.path.join(tempfile.gettempdir(), f"tmp_audio_{uuid.uuid4().hex[:8]}.mp3")
    _download_file(audio_url, tmp_audio)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        tmp_audio,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    duration = float(result.stdout.strip())
    os.remove(tmp_audio)
    return duration


def _find_chinese_font() -> str:
    """查找可用的中文字体"""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "file"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            return lines[0].split(":")[0].strip()
    except Exception:
        pass
    return ""


def _generate_still_video_with_subtitles_and_audio(
    composite_image_path: str,
    subtitle_segments: List[str],
    segment_durations: List[float],
    merged_audio_path: str,
    video_width: int,
    video_height: int,
    output_path: str,
    uid: str,
) -> str:
    """
    用 ffmpeg 本地完成：
    1. 从合成图生成静止视频
    2. 按实际 TTS 朗读节奏烧录分段字幕
    3. 合成配音

    segment_durations: 每段字幕的实际 TTS 朗读时长（秒）
    merged_audio_path: 已拼接好的完整配音文件路径

    返回输出视频路径
    """
    font_path = _find_chinese_font()
    if not font_path:
        logger.warning(f"[{uid}] 未找到中文字体，字幕可能无法正确显示")
        font_path = ""

    # 计算每段字幕的起止时间（累计时长）
    drawtext_filters = []
    cumulative = 0.0
    for i, (seg, dur) in enumerate(zip(subtitle_segments, segment_durations)):
        start_t = cumulative
        end_t = cumulative + dur
        cumulative = end_t

        if font_path:
            font_param = f"fontfile='{font_path}':"
        else:
            font_param = ""

        font_size = max(24, int(video_width * 0.06))

        # 字幕在画面中间偏下（y=70%位置）
        filter_str = (
            f"drawtext={font_param}"
            f"text='{seg}':"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"bordercolor=black:"
            f"borderw=3:"
            f"x=(w-text_w)/2:"
            f"y=h*0.70-text_h/2:"
            f"enable='between(t,{start_t},{end_t})'"
        )
        drawtext_filters.append(filter_str)

    vf_chain = ",".join(drawtext_filters)
    total_duration = cumulative

    logger.info(f"[{uid}] ffmpeg 字幕: {len(subtitle_segments)} 段, 总时长 {total_duration:.2f}s")

    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", composite_image_path,
        "-i", merged_audio_path,
        "-vf", vf_chain,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(total_duration),
        "-shortest",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]

    logger.info(f"[{uid}] 执行 ffmpeg 命令...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        logger.error(f"[{uid}] ffmpeg 失败: {stderr_tail}")
        raise RuntimeError(f"ffmpeg 生成定格视频失败: {stderr_tail}")

    logger.info(f"[{uid}] ffmpeg 定格视频生成成功: {output_path}")
    return output_path


def _concat_audio_files(audio_paths: List[str], output_path: str, uid: str) -> str:
    """
    用 ffmpeg concat 拼接多个音频文件。
    """
    # 创建 concat 文件列表
    concat_list_path = os.path.join(tempfile.gettempdir(), f"concat_{uid}.txt")
    with open(concat_list_path, "w") as f:
        for ap in audio_paths:
            f.write(f"file '{ap}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    os.remove(concat_list_path)

    if result.returncode != 0:
        raise RuntimeError(f"音频拼接失败: {result.stderr[-300:]}")

    return output_path


def _mix_bgm(
    video_url: str,
    bgm_url: str,
    bgm_volume: float,
    output_path: str,
    uid: str,
) -> str:
    """
    下载最终视频和 BGM，将 BGM 裁剪到视频时长、调整音量后混入视频。
    bgm_volume: 0.0 ~ 1.0，默认 0.6
    """
    tmp_dir = tempfile.gettempdir()

    # 下载最终视频
    tmp_video = os.path.join(tmp_dir, f"final_before_bgm_{uid}.mp4")
    _download_file(video_url, tmp_video)

    # 获取视频时长
    video_duration = _get_video_duration(tmp_video)

    # 下载 BGM
    tmp_bgm = os.path.join(tmp_dir, f"bgm_{uid}.mp3")
    _download_file(bgm_url, tmp_bgm)

    logger.info(f"[{uid}] BGM 混音: 视频时长={video_duration:.2f}s, BGM音量={bgm_volume:.0%}")

    # ffmpeg: 将 BGM 裁剪到视频时长 + 调整音量 + 混入视频
    # 使用 amovie 滤镜读取 BGM，atrim 裁剪，volume 调音量，amix 混合
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-stream_loop", "-1",
        "-i", tmp_bgm,
        "-filter_complex",
        (
            f"[1:a]atrim=0:{video_duration},volume={bgm_volume}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[outa]"
        ),
        "-map", "0:v:0",
        "-map", "[outa]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(video_duration),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # 清理
    try:
        os.remove(tmp_video)
        os.remove(tmp_bgm)
    except Exception:
        pass

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        logger.error(f"[{uid}] BGM 混音失败: {stderr_tail}")
        raise RuntimeError(f"BGM 混音失败: {stderr_tail}")

    logger.info(f"[{uid}] BGM 混音完成: {output_path}")
    return output_path


# ============================================================
# 核心管线函数（可被 batch_tool 等外部模块复用）
# ============================================================

def process_video_pipeline(
    video_url: str,
    guide_text: str = DEFAULT_GUIDE_TEXT,
    subtitle_text: str = DEFAULT_SUBTITLE_TEXT,
    voice_name: str = "米仔（视频配音女声）",
    tail_name: str = "短剧推广尾帧",
    tail_custom_url: str = "",
    transition1_name: str = "硬切（无转场）",
    transition2_name: str = "硬切（无转场）",
    search_box_image_url: str = "",
    bgm_url: str = "",
    bgm_volume: float = 0.6,
) -> dict:
    """
    核心视频处理管线，返回 dict 而非 JSON 字符串，便于程序化调用。
    """
    ctx = _get_ctx()
    tmp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:12]

    logger.info(f"[{uid}] 开始处理视频: {video_url}")

    # Step 0: 下载用户视频，检测分辨率，重新上传到自有S3
    logger.info(f"[{uid}] Step 0: 下载用户视频并检测分辨率")
    tmp_video = os.path.join(tmp_dir, f"input_{uid}.mp4")
    _download_file(video_url, tmp_video)
    video_w, video_h = _get_video_resolution(video_url)
    logger.info(f"[{uid}] 原视频分辨率: {video_w}x{video_h}")

    video_url = _upload_video_to_s3(tmp_video, f"temp/input_{uid}.mp4")
    os.remove(tmp_video)
    logger.info(f"[{uid}] 用户视频已重新上传到S3")

    # Step 1: 提取干净最后一帧（黑屏检测+字幕检测+去字幕）
    logger.info(f"[{uid}] Step 1: 提取干净最后一帧")
    last_frame_path = os.path.join(tmp_dir, f"lastframe_{uid}.png")
    _extract_clean_last_frame(video_url, last_frame_path, video_w, video_h, uid)
    logger.info(f"[{uid}] 干净最后一帧已提取: {last_frame_path}")

    # Step 2: 字幕分段
    logger.info(f"[{uid}] Step 2: 字幕分段")
    subtitle_segments = _split_subtitle(subtitle_text)
    logger.info(f"[{uid}] 字幕分为 {len(subtitle_segments)} 段: {subtitle_segments}")

    # Step 3: 逐段 TTS 配音
    logger.info(f"[{uid}] Step 3: 逐段 TTS 配音")
    speaker = VOICE_OPTIONS.get(voice_name, "zh_female_mizai_saturn_bigtts")
    tts_client = TTSClient(ctx=ctx)

    segment_durations = []
    segment_audio_paths = []
    segment_audio_urls = []

    for i, seg in enumerate(subtitle_segments):
        seg_uid = f"ad_tail_{uid}_seg{i}"
        audio_url, audio_size = tts_client.synthesize(
            uid=seg_uid,
            text=seg,
            speaker=speaker,
            audio_format="mp3",
        )
        dur = _get_audio_duration(audio_url)
        segment_durations.append(dur)
        segment_audio_urls.append(audio_url)

        local_seg = os.path.join(tmp_dir, f"tts_{uid}_seg{i}.mp3")
        _download_file(audio_url, local_seg)
        segment_audio_paths.append(local_seg)

        logger.info(f"[{uid}] 段{i}「{seg}」TTS时长: {dur:.2f}s")

    total_audio_duration = sum(segment_durations)
    logger.info(f"[{uid}] TTS 总时长: {total_audio_duration:.2f}s")

    merged_audio_path = os.path.join(tmp_dir, f"merged_audio_{uid}.mp3")
    _concat_audio_files(segment_audio_paths, merged_audio_path, uid)

    merged_audio_url = _get_storage().generate_presigned_url(
        key=_get_storage().stream_upload_file(
            fileobj=open(merged_audio_path, "rb"),
            file_name=f"temp/merged_audio_{uid}.mp3",
            content_type="audio/mpeg",
        ),
        expire_time=2592000,
    )

    # Step 4: 合成搜索框
    logger.info(f"[{uid}] Step 4: 合成搜索框")
    composite_path = os.path.join(tmp_dir, f"composite_{uid}.png")
    _composite_search_box(
        frame_image_path=last_frame_path,
        search_box_image_url=search_box_image_url if search_box_image_url else None,
        output_path=composite_path,
    )

    # Step 5: ffmpeg 生成定格视频
    logger.info(f"[{uid}] Step 5: ffmpeg 生成定格视频")
    freeze_video_local = os.path.join(tmp_dir, f"freeze_{uid}.mp4")
    _generate_still_video_with_subtitles_and_audio(
        composite_image_path=composite_path,
        subtitle_segments=subtitle_segments,
        segment_durations=segment_durations,
        merged_audio_path=merged_audio_path,
        video_width=video_w,
        video_height=video_h,
        output_path=freeze_video_local,
        uid=uid,
    )

    freeze_video_url = _upload_video_to_s3(freeze_video_local, f"temp/freeze_{uid}.mp4")
    logger.info(f"[{uid}] 定格视频URL: {freeze_video_url}")

    # Step 6: 拼接三段视频
    logger.info(f"[{uid}] Step 6: 拼接视频")

    if tail_custom_url and tail_custom_url.strip():
        tail_url = tail_custom_url.strip()
    else:
        tail_url = BUILTIN_TAILS.get(tail_name)
        if not tail_url:
            raise ValueError(f"未找到内置尾帧「{tail_name}」，可选：{list(BUILTIN_TAILS.keys())}")

    transitions = []
    t1_id = TRANSITION_OPTIONS.get(transition1_name)
    t2_id = TRANSITION_OPTIONS.get(transition2_name)
    if t1_id:
        transitions.append(t1_id)
    if t2_id:
        transitions.append(t2_id)

    video_edit_client = VideoEditClient(ctx=ctx)
    concat_resp = video_edit_client.concat_videos(
        videos=[video_url, freeze_video_url, tail_url],
        transitions=transitions if transitions else None,
    )
    final_video_url = concat_resp.url
    logger.info(f"[{uid}] 拼接后视频URL: {final_video_url}")

    # Step 7: 可选 BGM 混音
    if bgm_url and bgm_url.strip():
        logger.info(f"[{uid}] Step 7: BGM 混音 (volume={bgm_volume})")
        bgm_output = os.path.join(tmp_dir, f"final_with_bgm_{uid}.mp4")
        _mix_bgm(
            video_url=final_video_url,
            bgm_url=bgm_url.strip(),
            bgm_volume=bgm_volume,
            output_path=bgm_output,
            uid=uid,
        )
        final_video_url = _upload_video_to_s3(bgm_output, f"temp/final_with_bgm_{uid}.mp4")
        logger.info(f"[{uid}] BGM混音后视频URL: {final_video_url}")

    # 清理临时文件
    for tmp_file in [last_frame_path, composite_path, freeze_video_local, merged_audio_path] + segment_audio_paths:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass

    return {
        "success": True,
        "final_video_url": final_video_url,
        "details": {
            "audio_url": merged_audio_url,
            "audio_duration_sec": round(total_audio_duration, 2),
            "freeze_video_url": freeze_video_url,
            "tail_used": tail_name if not tail_custom_url else "自定义尾帧",
            "voice_used": voice_name,
            "transition1": transition1_name,
            "transition2": transition2_name,
            "subtitle_segments": subtitle_segments,
            "segment_durations": [round(d, 2) for d in segment_durations],
            "video_resolution": f"{video_w}x{video_h}",
            "bgm_applied": bool(bgm_url and bgm_url.strip()),
            "bgm_volume": bgm_volume if bgm_url and bgm_url.strip() else None,
        }
    }


# ============================================================
# 主工具：广告尾帧视频处理（Agent 工具包装）
# ============================================================

@tool
def process_ad_tail_video(
    video_url: str,
    guide_text: str = DEFAULT_GUIDE_TEXT,
    subtitle_text: str = DEFAULT_SUBTITLE_TEXT,
    voice_name: str = "米仔（视频配音女声）",
    tail_name: str = "短剧推广尾帧",
    tail_custom_url: str = "",
    transition1_name: str = "硬切（无转场）",
    transition2_name: str = "硬切（无转场）",
    search_box_image_url: str = "",
    bgm_url: str = "",
    bgm_volume: float = 0.6,
) -> str:
    """
    处理单个视频：提取最后一帧 → 合成搜索框 → 字幕分段TTS配音 → ffmpeg生成静止定格视频（含字幕+配音） → 拼接 → 可选BGM混音。

    参数说明：
    - video_url: 用户上传的视频URL（必填）
    - guide_text: 引导语文字内容，默认"后续剧情该如何选择？快来左下角造梦次元"
    - subtitle_text: 固定字幕文字内容，默认同上
    - voice_name: TTS配音音色，可选：小荷（通用女声）/ 米仔（视频配音女声）/ 大奕（视频配音男声）/ 可爱女生
    - tail_name: 内置广告尾帧名称，可选：派对接引尾帧 / 短剧推广尾帧
    - tail_custom_url: 用户自定义广告尾帧URL（如果提供则优先使用）
    - transition1_name: 用户视频→定格帧的转场效果
    - transition2_name: 定格帧→广告尾帧的转场效果
    - search_box_image_url: 搜索框透明背景图片URL（可选）
    - bgm_url: 用户上传的BGM音频URL（可选）
    - bgm_volume: BGM音量，0.0~1.0，默认0.6（60%）

    返回：包含最终视频URL和处理详情的JSON字符串
    """
    ctx = _get_ctx()
    tmp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:12]

    logger.info(f"[{uid}] 开始处理视频: {video_url}")

    try:
        # ============================================================
        # Step 0: 下载用户视频，检测分辨率，重新上传到自有S3
        # ============================================================
        logger.info(f"[{uid}] Step 0: 下载用户视频并检测分辨率")
        tmp_video = os.path.join(tmp_dir, f"input_{uid}.mp4")
        _download_file(video_url, tmp_video)
        video_w, video_h = _get_video_resolution(video_url)
        logger.info(f"[{uid}] 原视频分辨率: {video_w}x{video_h}")

        video_url = _upload_video_to_s3(tmp_video, f"temp/input_{uid}.mp4")
        os.remove(tmp_video)
        logger.info(f"[{uid}] 用户视频已重新上传到S3")

        # ============================================================
        # Step 1: 用 ffmpeg 提取最后一帧（黑屏检测 + 字幕检测 + Seedream 4.0 去字幕）
        # ============================================================
        logger.info(f"[{uid}] Step 1: 提取干净最后一帧（黑屏检测+字幕检测+去字幕）")
        last_frame_path = os.path.join(tmp_dir, f"lastframe_{uid}.png")
        _extract_clean_last_frame(video_url, last_frame_path, video_w, video_h, uid)
        logger.info(f"[{uid}] 干净最后一帧已提取: {last_frame_path}")

        # ============================================================
        # Step 2: 字幕分段
        # ============================================================
        logger.info(f"[{uid}] Step 2: 字幕分段")
        subtitle_segments = _split_subtitle(subtitle_text)
        logger.info(f"[{uid}] 字幕分为 {len(subtitle_segments)} 段: {subtitle_segments}")

        # ============================================================
        # Step 3: 每段字幕分别 TTS → 获取实际时长 → 拼接音频
        # ============================================================
        logger.info(f"[{uid}] Step 3: 逐段 TTS 配音")
        speaker = VOICE_OPTIONS.get(voice_name, "zh_female_mizai_saturn_bigtts")
        tts_client = TTSClient(ctx=ctx)

        segment_durations = []
        segment_audio_paths = []
        segment_audio_urls = []

        for i, seg in enumerate(subtitle_segments):
            seg_uid = f"ad_tail_{uid}_seg{i}"
            audio_url, audio_size = tts_client.synthesize(
                uid=seg_uid,
                text=seg,
                speaker=speaker,
                audio_format="mp3",
            )
            dur = _get_audio_duration(audio_url)
            segment_durations.append(dur)
            segment_audio_urls.append(audio_url)

            # 下载音频到本地用于后续拼接
            local_seg = os.path.join(tmp_dir, f"tts_{uid}_seg{i}.mp3")
            _download_file(audio_url, local_seg)
            segment_audio_paths.append(local_seg)

            logger.info(f"[{uid}] 段{i}「{seg}」TTS时长: {dur:.2f}s")

        total_audio_duration = sum(segment_durations)
        logger.info(f"[{uid}] TTS 总时长: {total_audio_duration:.2f}s")

        # 拼接所有音频片段
        merged_audio_path = os.path.join(tmp_dir, f"merged_audio_{uid}.mp3")
        _concat_audio_files(segment_audio_paths, merged_audio_path, uid)

        # 上传拼接后的音频到 S3（用于返回给用户）
        merged_audio_url = _get_storage().generate_presigned_url(
            key=_get_storage().stream_upload_file(
                fileobj=open(merged_audio_path, "rb"),
                file_name=f"temp/merged_audio_{uid}.mp3",
                content_type="audio/mpeg",
            ),
            expire_time=2592000,
        )

        # ============================================================
        # Step 4: 合成搜索框到定格帧
        # ============================================================
        logger.info(f"[{uid}] Step 4: 合成搜索框")
        composite_path = os.path.join(tmp_dir, f"composite_{uid}.png")
        _composite_search_box(
            frame_image_path=last_frame_path,
            search_box_image_url=search_box_image_url if search_box_image_url else None,
            output_path=composite_path,
        )

        # ============================================================
        # Step 5: ffmpeg 生成定格视频（含字幕 + 配音）
        # ============================================================
        logger.info(f"[{uid}] Step 5: ffmpeg 生成定格视频（含字幕+配音）")
        freeze_video_local = os.path.join(tmp_dir, f"freeze_{uid}.mp4")
        _generate_still_video_with_subtitles_and_audio(
            composite_image_path=composite_path,
            subtitle_segments=subtitle_segments,
            segment_durations=segment_durations,
            merged_audio_path=merged_audio_path,
            video_width=video_w,
            video_height=video_h,
            output_path=freeze_video_local,
            uid=uid,
        )

        freeze_video_url = _upload_video_to_s3(freeze_video_local, f"temp/freeze_{uid}.mp4")
        logger.info(f"[{uid}] 定格视频URL: {freeze_video_url}")

        # ============================================================
        # Step 6: 拼接三段视频
        # ============================================================
        logger.info(f"[{uid}] Step 6: 拼接视频")

        if tail_custom_url and tail_custom_url.strip():
            tail_url = tail_custom_url.strip()
        else:
            tail_url = BUILTIN_TAILS.get(tail_name)
            if not tail_url:
                return json.dumps({
                    "success": False,
                    "error": f"未找到内置尾帧「{tail_name}」，可选：{list(BUILTIN_TAILS.keys())}"
                }, ensure_ascii=False)

        transitions = []
        t1_id = TRANSITION_OPTIONS.get(transition1_name)
        t2_id = TRANSITION_OPTIONS.get(transition2_name)
        if t1_id:
            transitions.append(t1_id)
        if t2_id:
            transitions.append(t2_id)

        video_edit_client = VideoEditClient(ctx=ctx)
        concat_resp = video_edit_client.concat_videos(
            videos=[video_url, freeze_video_url, tail_url],
            transitions=transitions if transitions else None,
        )
        final_video_url = concat_resp.url
        logger.info(f"[{uid}] 拼接后视频URL: {final_video_url}")

        # ============================================================
        # Step 7: 可选 BGM 混音
        # ============================================================
        if bgm_url and bgm_url.strip():
            logger.info(f"[{uid}] Step 7: BGM 混音 (volume={bgm_volume})")
            bgm_output = os.path.join(tmp_dir, f"final_with_bgm_{uid}.mp4")
            _mix_bgm(
                video_url=final_video_url,
                bgm_url=bgm_url.strip(),
                bgm_volume=bgm_volume,
                output_path=bgm_output,
                uid=uid,
            )
            final_video_url = _upload_video_to_s3(bgm_output, f"temp/final_with_bgm_{uid}.mp4")
            logger.info(f"[{uid}] BGM混音后视频URL: {final_video_url}")

        # ============================================================
        # 清理临时文件
        # ============================================================
        for tmp_file in [last_frame_path, composite_path, freeze_video_local, merged_audio_path] + segment_audio_paths:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass

        return json.dumps({
            "success": True,
            "final_video_url": final_video_url,
            "details": {
                "audio_url": merged_audio_url,
                "audio_duration_sec": round(total_audio_duration, 2),
                "freeze_video_url": freeze_video_url,
                "tail_used": tail_name if not tail_custom_url else "自定义尾帧",
                "voice_used": voice_name,
                "transition1": transition1_name,
                "transition2": transition2_name,
                "subtitle_segments": subtitle_segments,
                "segment_durations": [round(d, 2) for d in segment_durations],
                "video_resolution": f"{video_w}x{video_h}",
                "bgm_applied": bool(bgm_url and bgm_url.strip()),
                "bgm_volume": bgm_volume if bgm_url and bgm_url.strip() else None,
            }
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[{uid}] 处理失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"视频处理失败: {str(e)}",
        }, ensure_ascii=False)


@tool
def list_available_options() -> str:
    """
    列出所有可用的配置选项：内置尾帧、TTS音色、转场效果。

    返回：JSON格式的选项列表
    """
    return json.dumps({
        "builtin_tails": list(BUILTIN_TAILS.keys()),
        "voice_options": list(VOICE_OPTIONS.keys()),
        "transition_options": list(TRANSITION_OPTIONS.keys()),
    }, ensure_ascii=False)
