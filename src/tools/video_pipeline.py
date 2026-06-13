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
import threading
import uuid
import re
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 本地重型转码的并发闸门：沙箱 CPU 有限，批量并发时多个 libx264 编码
# 互相拖慢会直接撞超时（实测两条并发即触发 120s 超时被杀）。
# 只限制本地编码，云端调用（TTS/拼接/生图）不占名额。
_HEAVY_FFMPEG_SEMAPHORE = threading.Semaphore(2)

# 尾帧规格化结果缓存：同一尾帧+同一目标尺寸在一批里只重编码一次。
# 键 (tail_url, w, h) → 规格化后的成片 URL（我们自有 S3，30 天预签名）。
# 内置尾帧 URL 稳定故命中率高；自定义临时 URL 不同键不命中，至多退化为现状（无害）。
_TAIL_NORM_CACHE: dict = {}
_TAIL_NORM_LOCK = threading.Lock()

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


def _video_resolution_of_file(local_path: str) -> Tuple[int, int]:
    """对已存在的本地视频文件用 ffprobe 读分辨率 (width, height)。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        local_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def _get_video_resolution(video_url: str) -> Tuple[int, int]:
    """获取视频分辨率 (width, height)，先下载再用 ffprobe（仅供只有 URL 的调用方）。"""
    tmp_v = os.path.join(tempfile.gettempdir(), f"vres_{uuid.uuid4().hex[:8]}.mp4")
    _download_file(video_url, tmp_v)
    try:
        return _video_resolution_of_file(tmp_v)
    finally:
        try:
            os.remove(tmp_v)
        except Exception:
            pass


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


def _vision_detect_subtitle(image_url: str) -> dict:
    """
    调用多模态网关检测图片是否含字幕，返回完整诊断信息。

    直接走 HTTP 而非 langchain：网关返回非 JSON 响应时，langchain 只会抛
    "'str' object has no attribute 'model_dump'" 并吞掉网关的真实错误。
    """
    diag = {
        "ok": False, "has_subtitle": False, "answer": "",
        "status": None, "content_type": "", "body_head": "", "error": "",
    }
    try:
        api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY", "")
        base_url = (os.getenv("COZE_INTEGRATION_MODEL_BASE_URL") or "").rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(default_headers(_get_ctx()))
        payload = {
            "model": "doubao-seed-2-0-lite-260215",
            "temperature": 0,
            "stream": False,
            "thinking": {"type": "disabled"},  # 检测只需答有/无，关掉深度思考提速
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请仔细观察这张视频截图，画面底部或中部是否有字幕文字（包括中文、英文、数字等任何文字叠加）？请只回答「有」或「无」，不要解释。"},
                ],
            }],
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.encoding = "utf-8"
        diag["status"] = resp.status_code
        diag["content_type"] = resp.headers.get("Content-Type", "")
        diag["body_head"] = resp.text[:300]
        if resp.status_code != 200:
            diag["error"] = f"网关返回 HTTP {resp.status_code}"
            return diag

        content_type = diag["content_type"].lower()
        if "event-stream" in content_type:
            # 扣子模型网关即使收到非流式请求也固定返回 SSE 流，逐行拼接增量内容
            answer_parts = []
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # 末尾的 usage 统计分片 choices 为空数组，需跳过
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                answer_parts.append(delta.get("content") or "")
            answer = "".join(answer_parts).strip()
        elif "json" in content_type:
            data = resp.json()
            choices = data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            answer = str(message.get("content", "")).strip()
        else:
            diag["error"] = f"网关返回无法识别的 Content-Type: {diag['content_type']}"
            return diag

        diag["answer"] = answer
        diag["has_subtitle"] = "有" in answer and "无" not in answer
        diag["ok"] = bool(answer)
        if not answer:
            diag["error"] = "网关响应中未解析到回答内容"
        return diag
    except Exception as e:
        diag["error"] = str(e)
        return diag


def _frame_has_subtitle(frame_image_path: str) -> bool:
    """
    用多模态 LLM 检测帧画面中是否包含字幕/文字。
    返回 True 表示有字幕，False 表示无字幕。
    检测失败不阻断管线（按无字幕处理继续出片），但完整记录网关响应。
    """
    try:
        frame_url = _upload_image_to_s3(frame_image_path, f"temp/frame_check_{uuid.uuid4().hex[:8]}.png")
        diag = _vision_detect_subtitle(frame_url)
        if not diag["ok"]:
            logger.warning(f"[字幕检测] 检测失败，按无字幕继续: {json.dumps(diag, ensure_ascii=False)[:600]}")
            return False
        logger.info(f"字幕检测结果: '{diag['answer']}' → has_subtitle={diag['has_subtitle']}")
        return diag["has_subtitle"]
    except Exception as e:
        logger.warning(f"[字幕检测] 调用失败，按无字幕继续: {e}", exc_info=True)
        return False


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


def _extract_clean_last_frame(video_url: str, output_path: str, video_w: int, video_h: int, uid: str,
                              skip_subtitle_removal: bool = False) -> str:
    """
    提取视频干净的最后帧（无黑屏、无字幕）：

    流程：
    1. 提取最后一帧
    2. 检测是否黑屏 → 是：向前回退 0.5s（最多 5s）找非黑帧
    3. 检测是否有字幕 → 是：用 Seedream 4.0 去字幕
    4. 返回干净帧

    skip_subtitle_removal=True 时跳过去字幕（H5 快速预览、「标准」模式用）。
    注：旧的黑屏自动回退方案已废弃——是否去字幕/是否走黑屏渐显由「末帧模式」显式决定。
    """
    tmp_v = os.path.join(tempfile.gettempdir(), f"lastframe_src_{uid}.mp4")
    _download_file(video_url, tmp_v)
    duration = _get_video_duration(tmp_v)

    # 提取最后一帧（不再做黑屏回退）
    seek_time = max(0, duration - 0.1)
    _extract_frame_at_time(tmp_v, seek_time, output_path)
    logger.info(f"[{uid}] 提取最后一帧 (t={seek_time:.2f}s)")

    # 去字幕（仅「去字幕」模式）：Seedream 4.0
    if skip_subtitle_removal:
        os.remove(tmp_v)
        return output_path
    logger.info(f"[{uid}] 末帧模式=去字幕，使用 Seedream 4.0 去除字幕")
    _remove_subtitle_with_seedream(output_path, output_path, video_w, video_h, uid)
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


def _generate_freeze_video_from_plan(
    plan,
    merged_audio_path: str,
    total_duration: float,
    output_path: str,
    uid: str,
) -> str:
    """
    按图层渲染计划生成定格视频：
    - 输入0 = 拍平底图（静态全程图层已包含其中）
    - 输入1 = 完整配音
    - 输入2.. = 动态 overlay（分段字幕 enable / 动画 x,y 时间表达式）

    plan: tools.layer_render.FreezeRenderPlan
    返回输出视频路径
    """
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", plan.base_path,
        "-i", merged_audio_path,
    ]
    for spec in plan.overlays:
        cmd += ["-loop", "1", "-i", spec.path]

    filter_parts = ["[0:v]fps=30[base0]"]
    last_label = "[base0]"
    for idx, spec in enumerate(plan.overlays):
        ov_in = f"[{idx + 2}:v]"
        prefilter = getattr(spec, "prefilter", None)
        if prefilter:
            pf_label = f"[pf{idx}]"
            filter_parts.append(f"{ov_in}{prefilter}{pf_label}")
            ov_in = pf_label
        # x/y 表达式可能含逗号(min/max)，必须单引号包裹，否则被当成滤镜分隔符
        opts = f"x='{spec.x_expr}':y='{spec.y_expr}'"
        if spec.enable_expr:
            opts += f":enable='{spec.enable_expr}'"
        out_label = f"[v{idx}]"
        filter_parts.append(f"{last_label}{ov_in}overlay={opts}{out_label}")
        last_label = out_label

    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", last_label,
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "veryfast",  # 静帧+贴图内容，对画质影响可忽略，显著降低CPU耗时
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(total_duration),
        "-shortest",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]

    logger.info(f"[{uid}] 执行 ffmpeg 定格视频合成: overlay {len(plan.overlays)} 个, 时长 {total_duration:.2f}s")
    try:
        with _HEAVY_FFMPEG_SEMAPHORE:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 生成定格视频超时（600秒），可能是并发过高或沙箱资源不足，请稍后重试")

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        logger.error(f"[{uid}] ffmpeg 失败: {stderr_tail}")
        raise RuntimeError(f"ffmpeg 生成定格视频失败: {stderr_tail}")

    logger.info(f"[{uid}] ffmpeg 定格视频生成成功: {output_path}")
    return output_path


def _normalize_tail(tail_url: str, target_w: int, target_h: int, output_path: str, uid: str) -> str:
    """把广告尾帧重编码为 H.264 / 30fps / 目标尺寸（缩放保比+黑边填充），
    使云端转场不因编解码(HEVC)、帧率(60)、尺寸不匹配而退化为硬切。"""
    tmp_tail = os.path.join(tempfile.gettempdir(), f"tail_src_{uid}.mp4")
    _download_file(tail_url, tmp_tail)
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,fps=30,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-i", tmp_tail,
        "-vf", vf, "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    try:
        with _HEAVY_FFMPEG_SEMAPHORE:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError("尾帧规格化超时")
    finally:
        try:
            os.remove(tmp_tail)
        except Exception:
            pass
    if result.returncode != 0:
        raise RuntimeError(f"尾帧规格化失败: {result.stderr[-300:]}")
    return output_path


def _prepare_tail_url(tail_url: str, target_w: int, target_h: int, tmp_dir: str, uid: str) -> str:
    """规格化尾帧并上传，返回新URL；失败则返回原URL（至少不阻断出片）。
    按 (tail_url, w, h) 进程级缓存：批内同尾帧只重编码一次。"""
    cache_key = (tail_url, target_w, target_h)
    with _TAIL_NORM_LOCK:
        cached = _TAIL_NORM_CACHE.get(cache_key)
    if cached:
        logger.info(f"[{uid}] 尾帧规格化命中缓存 {target_w}x{target_h}，跳过重编码")
        return cached
    try:
        norm = os.path.join(tmp_dir, f"tailn_{uid}.mp4")
        # 计算不持锁：避免阻塞其它记录的缓存读取；最坏情况批首几条重复编码一次，可接受
        _normalize_tail(tail_url, target_w, target_h, norm, uid)
        url = _upload_video_to_s3(norm, f"temp/tailn_{uid}.mp4")
        try:
            os.remove(norm)
        except Exception:
            pass
        with _TAIL_NORM_LOCK:
            _TAIL_NORM_CACHE[cache_key] = url
        logger.info(f"[{uid}] 尾帧已规格化为 {target_w}x{target_h} H.264/30fps（已缓存）")
        return url
    except Exception as e:
        logger.warning(f"[{uid}] 尾帧规格化失败，用原尾帧（转场可能不生效）: {e}")
        return tail_url


def _generate_overlays_over_video(video_local: str, plan, base_t0: float, output_path: str, uid: str) -> str:
    """黑屏渐显：把图层（静态底图 + 各自动画 overlay）叠加到用户视频末段（无配音、无中间帧）。
    plan: 由 build_freeze_render_plan(transparent_base=True, base_t0=...) 生成。"""
    dur = _get_video_duration(video_local)
    cmd = ["ffmpeg", "-y", "-i", video_local, "-loop", "1", "-i", plan.base_path]
    for spec in plan.overlays:
        cmd += ["-loop", "1", "-i", spec.path]
    en0 = f":enable='gte(t,{base_t0:.3f})'" if base_t0 > 0 else ""
    filter_parts = ["[0:v]fps=30[bv]", f"[bv][1:v]overlay=0:0{en0}[s0]"]
    last = "[s0]"
    for idx, spec in enumerate(plan.overlays):
        ov_in = f"[{idx + 2}:v]"
        if getattr(spec, "prefilter", None):
            filter_parts.append(f"{ov_in}{spec.prefilter}[pf{idx}]"); ov_in = f"[pf{idx}]"
        opts = f"x='{spec.x_expr}':y='{spec.y_expr}'"
        if spec.enable_expr:
            opts += f":enable='{spec.enable_expr}'"
        filter_parts.append(f"{last}{ov_in}overlay={opts}[o{idx}]"); last = f"[o{idx}]"
    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", last, "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(dur), "-shortest",
        output_path,
    ]
    logger.info(f"[{uid}] 黑屏渐显合成: 视频时长={dur:.2f}s, 末段起点={base_t0:.2f}s, overlay {len(plan.overlays)} 个")
    try:
        with _HEAVY_FFMPEG_SEMAPHORE:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 黑屏渐显合成超时（600秒），请稍后重试")
    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        logger.error(f"[{uid}] 黑屏渐显 ffmpeg 失败: {stderr_tail}")
        raise RuntimeError(f"黑屏渐显合成失败: {stderr_tail}")
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
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> str:
    """
    下载最终视频和 BGM，将 BGM 裁剪到视频时长、调整音量后混入视频。
    bgm_volume: 0.0 ~ 1.0，默认 0.6
    fade_in/fade_out: 淡入/淡出时长（秒），0=不启用
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

    # 渐入/渐出：afade 滤镜，淡出从 (时长-淡出时长) 开始
    afade = ""
    if fade_in and float(fade_in) > 0:
        afade += f",afade=t=in:st=0:d={float(fade_in):.2f}"
    if fade_out and float(fade_out) > 0:
        st = max(0.0, video_duration - float(fade_out))
        afade += f",afade=t=out:st={st:.2f}:d={float(fade_out):.2f}"

    logger.info(f"[{uid}] BGM 混音: 视频时长={video_duration:.2f}s, BGM音量={bgm_volume:.0%}, "
                f"渐入={fade_in}s, 渐出={fade_out}s")

    # ffmpeg: 将 BGM 裁剪到视频时长 + 调整音量 + 可选渐入渐出 + 混入视频
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-stream_loop", "-1",
        "-i", tmp_bgm,
        "-filter_complex",
        (
            f"[1:a]atrim=0:{video_duration},volume={bgm_volume}{afade}[bgm];"
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError("BGM 混音超时（300秒），请稍后重试")

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
    bgm_fade_in: float = 0.0,
    bgm_fade_out: float = 0.0,
    frame_mode: str = "标准",
    fade_seconds: float = 4.0,
    style_layers: Optional[dict] = None,
    layer_context: Optional[dict] = None,
) -> dict:
    """
    核心视频处理管线，返回 dict 而非 JSON 字符串，便于程序化调用。

    style_layers: 图层文档（见 tools.layer_model），None 时用内置默认（与旧版固定样式一致）
    layer_context: 文字图层 text_source 的取值上下文，如 {"角色名": "梦宝"}
    """
    ctx = _get_ctx()
    tmp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:12]

    logger.info(f"[{uid}] 开始处理视频: {video_url}")

    # Step 0: 下载用户视频，检测分辨率，重新上传到自有S3
    logger.info(f"[{uid}] Step 0: 下载用户视频并检测分辨率")
    tmp_video = os.path.join(tmp_dir, f"input_{uid}.mp4")
    _download_file(video_url, tmp_video)
    video_w, video_h = _video_resolution_of_file(tmp_video)  # 复用已下载文件，避免整片二次下载
    logger.info(f"[{uid}] 原视频分辨率: {video_w}x{video_h}")

    # ===== 黑屏渐显模式：无中间帧、无配音；图层逐个叠加到用户视频末段、各自动画生效 =====
    if frame_mode == "黑屏渐显":
        from tools.layer_render import build_freeze_render_plan
        from tools.layer_model import resolve_layer_doc as _resolve_doc
        # 取显示尺寸（带旋转视频的实际像素），抽一帧量尺寸
        probe_frame = os.path.join(tmp_dir, f"dim_{uid}.png")
        try:
            _extract_frame_at_time(tmp_video, max(0.0, _get_video_duration(tmp_video) * 0.5), probe_frame)
            with Image.open(probe_frame) as _im:
                cw, ch = _im.size
        finally:
            try:
                os.remove(probe_frame)
            except Exception:
                pass
        _doc = style_layers if (isinstance(style_layers, dict) and style_layers.get("layers")) else _resolve_doc()
        v_dur = _get_video_duration(tmp_video)
        win = float(fade_seconds or 4.0)
        start = max(0.0, v_dur - win)
        # 字幕分段（去标点、≤12字，与其它模式一致），无配音故时长平均分配
        segs = _split_subtitle(subtitle_text or guide_text or DEFAULT_GUIDE_TEXT)
        k = max(1, len(segs))
        seg_durs = [(v_dur - start) / k] * k
        plan = build_freeze_render_plan(
            frame_path="", canvas_w=cw, canvas_h=ch, layer_doc=_doc,
            layer_context=layer_context or {}, search_box_image_url=(search_box_image_url or "").strip(),
            subtitle_segments=segs, segment_durations=seg_durs, font_path=_find_chinese_font(),
            tmp_dir=tmp_dir, uid=uid, transparent_base=True, base_t0=start,
        )
        fadein_local = os.path.join(tmp_dir, f"fadein_{uid}.mp4")
        _generate_overlays_over_video(tmp_video, plan, start, fadein_local, uid)
        main_url = _upload_video_to_s3(fadein_local, f"temp/fadein_{uid}.mp4")
        if tail_custom_url and tail_custom_url.strip():
            tail_url = tail_custom_url.strip()
        else:
            tail_url = BUILTIN_TAILS.get(tail_name)
            if not tail_url:
                raise ValueError(f"未找到内置尾帧「{tail_name}」，可选：{list(BUILTIN_TAILS.keys())}")
        tail_url = _prepare_tail_url(tail_url, cw, ch, tmp_dir, uid)  # 规格化，避免转场退化硬切
        t2_id = TRANSITION_OPTIONS.get(transition2_name)
        logger.info(f"[{uid}] 黑屏渐显拼接: [末段视频, 尾帧] 转场2={transition2_name}({t2_id})")
        concat_resp = VideoEditClient(ctx=ctx).concat_videos(
            videos=[main_url, tail_url], transitions=[t2_id] if t2_id else None)
        final_video_url = concat_resp.url
        logger.info(f"[{uid}] 黑屏渐显拼接结果: {final_video_url}")
        if bgm_url and bgm_url.strip():
            bgm_out = os.path.join(tmp_dir, f"final_bgm_{uid}.mp4")
            _mix_bgm(final_video_url, bgm_url.strip(), bgm_volume, bgm_out, uid,
                     fade_in=bgm_fade_in, fade_out=bgm_fade_out)
            final_video_url = _upload_video_to_s3(bgm_out, f"temp/final_bgm_{uid}.mp4")
            try:
                os.remove(bgm_out)
            except Exception:
                pass
        for p in [tmp_video, fadein_local, plan.base_path] + [s.path for s in plan.overlays]:
            try:
                os.remove(p)
            except Exception:
                pass
        logger.info(f"[{uid}] 黑屏渐显模式完成: {final_video_url}")
        return {
            "success": True,
            "final_video_url": final_video_url,
            "freeze_video_url": main_url,
            "frame_mode": frame_mode,
            "transition2": transition2_name,
        }

    video_url = _upload_video_to_s3(tmp_video, f"temp/input_{uid}.mp4")
    os.remove(tmp_video)
    logger.info(f"[{uid}] 用户视频已重新上传到S3")

    # Step 1: 提取干净最后一帧（黑屏检测+字幕检测+去字幕）
    logger.info(f"[{uid}] Step 1: 提取干净最后一帧")
    last_frame_path = os.path.join(tmp_dir, f"lastframe_{uid}.png")
    _extract_clean_last_frame(video_url, last_frame_path, video_w, video_h, uid,
                              skip_subtitle_removal=(frame_mode != "去字幕"))
    logger.info(f"[{uid}] 干净最后一帧已提取 (模式={frame_mode}): {last_frame_path}")

    # Step 2: 字幕分段
    logger.info(f"[{uid}] Step 2: 字幕分段")
    subtitle_segments = _split_subtitle(subtitle_text)
    logger.info(f"[{uid}] 字幕分为 {len(subtitle_segments)} 段: {subtitle_segments}")

    # Step 3: 逐段 TTS 配音（并发：云端调用不占 ffmpeg 名额；每段只下载一次，probe 本地取时长）
    logger.info(f"[{uid}] Step 3: 逐段 TTS 配音（并发）")
    speaker = VOICE_OPTIONS.get(voice_name, "zh_female_mizai_saturn_bigtts")

    def _tts_one(i: int, seg: str):
        # 每任务新建 TTSClient，避免并发共享同一实例的线程安全问题
        seg_uid = f"ad_tail_{uid}_seg{i}"
        audio_url, _ = TTSClient(ctx=ctx).synthesize(
            uid=seg_uid,
            text=seg,
            speaker=speaker,
            audio_format="mp3",
        )
        local_seg = os.path.join(tmp_dir, f"tts_{uid}_seg{i}.mp3")
        _download_file(audio_url, local_seg)            # 只下载一次
        dur = _get_video_duration(local_seg)            # probe 本地文件，避免二次下载
        logger.info(f"[{uid}] 段{i}「{seg}」TTS时长: {dur:.2f}s")
        return i, audio_url, local_seg, dur

    _tts_slots = [None] * len(subtitle_segments)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(subtitle_segments)))) as _tp:
        for _fut in as_completed([_tp.submit(_tts_one, i, s) for i, s in enumerate(subtitle_segments)]):
            i, audio_url, local_seg, dur = _fut.result()
            _tts_slots[i] = (audio_url, local_seg, dur)

    segment_audio_urls = [s[0] for s in _tts_slots]
    segment_audio_paths = [s[1] for s in _tts_slots]
    segment_durations = [s[2] for s in _tts_slots]

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

    # Step 4: 图层合成（静态全程图层拍平进底图；动画/分段显示图层生成 overlay 规格）
    logger.info(f"[{uid}] Step 4: 图层合成")
    from tools.layer_model import resolve_layer_doc
    from tools.layer_render import build_freeze_render_plan
    if isinstance(style_layers, dict) and style_layers.get("layers"):
        layer_doc = style_layers
    else:
        layer_doc = resolve_layer_doc()
    plan = build_freeze_render_plan(
        frame_path=last_frame_path,
        canvas_w=video_w,
        canvas_h=video_h,
        layer_doc=layer_doc,
        layer_context=layer_context or {},
        search_box_image_url=search_box_image_url.strip() if search_box_image_url else "",
        subtitle_segments=subtitle_segments,
        segment_durations=segment_durations,
        font_path=_find_chinese_font(),
        tmp_dir=tmp_dir,
        uid=uid,
    )
    logger.info(f"[{uid}] 图层合成完成: 动态 overlay {len(plan.overlays)} 个")

    # Step 5: ffmpeg 生成定格视频
    logger.info(f"[{uid}] Step 5: ffmpeg 生成定格视频")
    freeze_video_local = os.path.join(tmp_dir, f"freeze_{uid}.mp4")
    _generate_freeze_video_from_plan(
        plan=plan,
        merged_audio_path=merged_audio_path,
        total_duration=total_audio_duration,
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

    # 规格化尾帧到内容显示尺寸/H.264/30fps，避免转场→尾帧因格式不匹配退化硬切
    try:
        with Image.open(last_frame_path) as _fim:
            _dw, _dh = _fim.size
    except Exception:
        _dw, _dh = video_w, video_h
    tail_url = _prepare_tail_url(tail_url, _dw, _dh, tmp_dir, uid)

    t1_id = TRANSITION_OPTIONS.get(transition1_name)
    t2_id = TRANSITION_OPTIONS.get(transition2_name)

    video_edit_client = VideoEditClient(ctx=ctx)
    logger.info(f"[{uid}] 转场: 转场1={transition1_name}({t1_id}) 转场2={transition2_name}({t2_id})")
    # 始终两两拼接：云端对"3 段一次拼 + 转场数组"不生效（退化硬切），
    # 而"2 段拼 + 单个转场"已验证可用。逐段拼接让两个转场都落位。
    logger.info(f"[{uid}] 拼接① [用户视频→定格帧] transitions={[t1_id] if t1_id else None}")
    first_resp = video_edit_client.concat_videos(
        videos=[video_url, freeze_video_url],
        transitions=[t1_id] if t1_id else None,
    )
    logger.info(f"[{uid}] 拼接①响应: {first_resp!r}")
    logger.info(f"[{uid}] 拼接② [→广告尾帧] transitions={[t2_id] if t2_id else None}")
    concat_resp = video_edit_client.concat_videos(
        videos=[first_resp.url, tail_url],
        transitions=[t2_id] if t2_id else None,
    )
    logger.info(f"[{uid}] 拼接②响应: {concat_resp!r}")
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
            fade_in=bgm_fade_in,
            fade_out=bgm_fade_out,
        )
        final_video_url = _upload_video_to_s3(bgm_output, f"temp/final_with_bgm_{uid}.mp4")
        logger.info(f"[{uid}] BGM混音后视频URL: {final_video_url}")

    # 清理临时文件
    overlay_paths = [spec.path for spec in plan.overlays]
    for tmp_file in [last_frame_path, plan.base_path, freeze_video_local, merged_audio_path] + segment_audio_paths + overlay_paths:
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
    bgm_fade_in: float = 0.0,
    bgm_fade_out: float = 0.0,
    frame_mode: str = "标准",
    fade_seconds: float = 4.0,
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
    # 委托核心管线执行（此前这里是一份 230 行的重复实现，曾导致两处逻辑漂移）
    try:
        result = process_video_pipeline(
            video_url=video_url,
            guide_text=guide_text,
            subtitle_text=subtitle_text,
            voice_name=voice_name,
            tail_name=tail_name,
            tail_custom_url=tail_custom_url,
            transition1_name=transition1_name,
            transition2_name=transition2_name,
            search_box_image_url=search_box_image_url,
            bgm_url=bgm_url,
            bgm_volume=bgm_volume,
            bgm_fade_in=bgm_fade_in,
            bgm_fade_out=bgm_fade_out,
            frame_mode=frame_mode,
            fade_seconds=fade_seconds,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"视频处理失败: {str(e)}", exc_info=True)
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
