"""
H5 中间帧编辑器后端接口

设计要点：
- 快速抽帧：跳过 AI 去字幕（秒级返回），按视频指纹缓存（h5_kv）
- 精确预览：与成片同一渲染器（tools.layer_render）静态出图
- 上传走原始字节流（octet-stream）而非 multipart，规避 python-multipart 依赖
- 所有阻塞操作经 asyncio.to_thread，不卡事件循环
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/h5")

DEFAULT_APP_TOKEN = os.getenv("DEFAULT_BITABLE_APP_TOKEN", "JOMibWw3wa6TzYsaHSIcAG27n2f")
DEFAULT_TABLE_ID = os.getenv("DEFAULT_BITABLE_TABLE_ID", "tblWNUywhvrkJ54u")
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://a0357594-98ce-46e4-a7fe-ac797d969b21.dev.coze.site",
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

# v2：帧尺寸改用抽出 PNG 的实际像素（修复旋转视频编码/显示尺寸不一致），旧缓存作废
FRAME_CACHE_PREFIX = "frame:v2:"


def _table_of(payload: dict) -> tuple:
    return (
        payload.get("app_token") or DEFAULT_APP_TOKEN,
        payload.get("table_id") or DEFAULT_TABLE_ID,
    )


async def _json_body(request: Request) -> dict:
    try:
        return await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")


# ------------------------------------------------------------
# 记录与视频源解析
# ------------------------------------------------------------

def _fetch_record_fields(app_token: str, table_id: str, record_id: str) -> dict:
    from tools.bitable_tool import BitableClient
    client = BitableClient()
    resp = client.get_record(app_token, table_id, record_id)
    return resp.get("data", {}).get("record", {}).get("fields", {})


def _resolve_video_source(fields: dict) -> tuple:
    """返回 (video_url, cache_key)。优先视频URL列，其次附件列。"""
    from tools.bitable_tool import BitableClient, field_to_text, attachment_to_download_url
    url = field_to_text(fields.get("视频URL")).strip()
    if url:
        parsed = urlparse(url)
        return url, f"{parsed.netloc}{parsed.path}"
    for att_name in ("视频附件", "附件"):
        value = fields.get(att_name)
        if not value:
            continue
        file_token = value[0].get("file_token", "") if isinstance(value, list) and isinstance(value[0], dict) else ""
        download_url = attachment_to_download_url(BitableClient(), value)
        if download_url:
            return download_url, file_token or download_url
    return "", ""


def _record_summary(item: dict) -> dict:
    from tools.bitable_tool import field_to_text
    fields = item.get("fields", {})
    name = field_to_text(fields.get("视频名")).strip()
    if not name:
        for att_name in ("视频附件", "附件"):
            value = fields.get(att_name)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                name = value[0].get("name", "")
                break
    if not name:
        raw_url = field_to_text(fields.get("视频URL")).strip()
        if raw_url:
            name = os.path.basename(urlparse(raw_url).path) or raw_url[:40]
    output = field_to_text(fields.get("输出视频URL")).strip()
    created = fields.get("创作日期")
    return {
        "record_id": item.get("record_id"),
        "name": name or "(未命名素材)",
        "role_name": field_to_text(fields.get("角色名")).strip(),
        "guide_text": field_to_text(fields.get("引导语")).strip(),
        "status": field_to_text(fields.get("处理状态")).strip(),
        "output_video_url": output,
        "preview_url": field_to_text(fields.get("预览图URL")).strip(),
        "thumbnail_url": field_to_text(fields.get("缩略图URL")).strip(),
        "created_at": created if isinstance(created, (int, float)) else None,
        "error": field_to_text(fields.get("错误信息")).strip(),
        "has_style": bool(field_to_text(fields.get("样式参数")).strip()),
    }


# ------------------------------------------------------------
# 写回辅助：优先填第一个空白行 + 自动创作日期
# ------------------------------------------------------------

def _today_ms() -> int:
    from datetime import datetime
    return int(datetime.now().timestamp() * 1000)


def _is_blank_row(fields: dict) -> bool:
    """空白行判定：无视频源、无名字、无状态、无产出，视为可复用空行。"""
    from tools.bitable_tool import field_to_text
    if field_to_text(fields.get("视频URL")).strip():
        return False
    for att in ("视频附件", "附件", "搜索框图片", "BGM"):
        if fields.get(att):
            return False
    for col in ("视频名", "处理状态", "输出视频URL", "样式参数", "角色名", "素材URL", "自定义尾帧URL"):
        if field_to_text(fields.get(col)).strip():
            return False
    return True


def _find_first_blank_record(client, app_token: str, table_id: str) -> str:
    page_token = None
    while True:
        resp = client.search_records(app_token=app_token, table_id=table_id, page_token=page_token)
        for item in resp.get("data", {}).get("items", []):
            if _is_blank_row(item.get("fields", {})):
                return item.get("record_id", "")
        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp.get("data", {}).get("page_token")
    return ""


def _write_video_record(client, app_token: str, table_id: str, fields: dict) -> str:
    """优先写入第一个空白行，无空白行才新建；自动补「创作日期」。"""
    fields = {**fields, "创作日期": _today_ms()}
    blank_id = _find_first_blank_record(client, app_token, table_id)
    if blank_id:
        client.update_records(app_token, table_id, [{"record_id": blank_id, "fields": fields}])
        return blank_id
    resp = client.create_record(app_token, table_id, fields)
    return resp.get("data", {}).get("record", {}).get("record_id", "")


# ------------------------------------------------------------
# 快速抽帧（带缓存）
# ------------------------------------------------------------

def _probe_local(path: str) -> tuple:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(out.stdout)
    stream = data.get("streams", [{}])[0]
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    return int(stream.get("width", 0)), int(stream.get("height", 0)), duration


def _quick_frame(video_url: str, cache_key: str) -> dict:
    from storage.database.db import get_session
    from storage.database.shared.model import H5KeyValue
    from tools.video_pipeline import (
        _download_file, _extract_frame_at_time, _is_black_frame, _upload_image_to_s3,
    )

    digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:20]
    kv_key = FRAME_CACHE_PREFIX + digest

    try:
        session = get_session()
        try:
            row = session.get(H5KeyValue, kv_key)
            if row and isinstance(row.value, dict) and row.value.get("frame_url"):
                return {**row.value, "cached": True}
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"[h5/frame] 缓存读取失败: {e}")

    uid = uuid.uuid4().hex[:10]
    tmp_dir = tempfile.gettempdir()
    tmp_v = os.path.join(tmp_dir, f"h5src_{uid}.mp4")
    frame_path = os.path.join(tmp_dir, f"h5frame_{uid}.png")
    try:
        _download_file(video_url, tmp_v)
        width, height, duration = _probe_local(tmp_v)
        if not width or not height:
            raise RuntimeError("无法读取视频分辨率")
        seek = max(0, duration - 0.1)
        _extract_frame_at_time(tmp_v, seek, frame_path)
        if _is_black_frame(frame_path):
            for step in range(1, 11):
                back = max(0, duration - 0.1 - step * 0.5)
                if back <= 0:
                    break
                _extract_frame_at_time(tmp_v, back, frame_path)
                if not _is_black_frame(frame_path):
                    break
        # 以抽出帧的实际像素尺寸为准（带旋转元数据的视频，编码尺寸与显示尺寸宽高互换），
        # 否则编辑器画布按编码尺寸绘制会把竖拍视频拉伸变形，且与成片渲染尺寸不一致。
        from PIL import Image
        with Image.open(frame_path) as _im:
            width, height = _im.size
        frame_url = _upload_image_to_s3(frame_path, f"h5/frames/{digest}.png")
    finally:
        for p in (tmp_v, frame_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    result = {"frame_url": frame_url, "width": width, "height": height}
    try:
        session = get_session()
        try:
            row = session.get(H5KeyValue, kv_key)
            if row is None:
                session.add(H5KeyValue(key=kv_key, value=result))
            else:
                row.value = result
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"[h5/frame] 缓存写入失败: {e}")
    return {**result, "cached": False}


def _kickoff_thumbnail(app_token: str, table_id: str, record_id: str, video_url: str) -> None:
    """后台抽取视频原始最后一帧，写入「缩略图URL」（与编辑器 /frame 共用帧缓存）。"""
    def _run():
        try:
            cache_key = urlparse(video_url).path or video_url
            res = _quick_frame(video_url, cache_key)
            thumb = res.get("frame_url", "")
            if thumb:
                from tools.bitable_tool import BitableClient
                BitableClient().update_records(
                    app_token, table_id,
                    [{"record_id": record_id, "fields": {"缩略图URL": thumb}}],
                )
        except Exception as e:
            logger.warning(f"[h5/thumbnail] 后台生成缩略图失败 record_id={record_id}: {e}")

    threading.Thread(target=_run, daemon=True).start()


@router.post("/frame")
async def api_frame(request: Request):
    """快速抽取记录视频的最后一帧（跳过AI去字幕，结果按视频指纹缓存）"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    video_url = payload.get("video_url", "")

    def _run():
        url, cache_key = video_url, video_url and urlparse(video_url).path or ""
        if record_id:
            fields = _fetch_record_fields(app_token, table_id, record_id)
            url, cache_key = _resolve_video_source(fields)
        if not url:
            raise HTTPException(status_code=400, detail="记录没有可用的视频来源（视频URL/视频附件均为空）")
        return _quick_frame(url, cache_key or url)

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 样式参数读写
# ------------------------------------------------------------

@router.post("/params/get")
async def api_params_get(request: Request):
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")

    def _run():
        from tools.bitable_tool import field_to_text, GUIDE_TEXT_OPTIONS, attachment_to_download_url, BitableClient
        from tools.layer_model import parse_layer_doc, resolve_layer_doc
        from tools.h5_store import get_global_layer_doc
        fields = _fetch_record_fields(app_token, table_id, record_id)
        record_raw = field_to_text(fields.get("样式参数"))
        global_doc = get_global_layer_doc()
        doc = resolve_layer_doc(record_raw, global_doc)
        if parse_layer_doc(record_raw):
            source = "record"
        elif global_doc:
            source = "global"
        else:
            source = "builtin"
        search_box_url = ""
        if fields.get("搜索框图片"):
            try:
                search_box_url = attachment_to_download_url(BitableClient(), fields.get("搜索框图片"))
            except Exception as e:
                logger.warning(f"[h5/params] 搜索框附件读取失败: {e}")
        if not search_box_url:
            search_box_url = field_to_text(fields.get("搜索框图片URL")).strip()

        # BGM 直链：附件优先，其次「BGM URL」列
        bgm_url = ""
        if fields.get("BGM"):
            try:
                bgm_url = attachment_to_download_url(BitableClient(), fields.get("BGM"))
            except Exception as e:
                logger.warning(f"[h5/params] BGM 附件读取失败: {e}")
        if not bgm_url:
            bgm_url = field_to_text(fields.get("BGM URL")).strip()

        from tools.video_pipeline import BUILTIN_TAILS, VOICE_OPTIONS, TRANSITION_OPTIONS

        def _num(name):
            v = fields.get(name)
            return v if isinstance(v, (int, float)) else None

        return {
            "layer_doc": doc,
            "source": source,
            "context": {"角色名": field_to_text(fields.get("角色名")).strip()},
            "guide_text": field_to_text(fields.get("引导语")).strip(),
            "guide_options": GUIDE_TEXT_OPTIONS,
            "search_box_url": search_box_url,
            "record": _record_summary({"record_id": record_id, "fields": fields}),
            "settings": {
                "tail_name": field_to_text(fields.get("广告尾帧")).strip(),
                "tail_custom_url": field_to_text(fields.get("自定义尾帧URL")).strip(),
                "voice_name": field_to_text(fields.get("配音音色")).strip(),
                "transition1": field_to_text(fields.get("转场1")).strip(),
                "transition2": field_to_text(fields.get("转场2")).strip(),
                "bgm_url": bgm_url,
                "bgm_volume": _num("BGM音量"),
                "bgm_fade_in": _num("BGM渐入"),
                "bgm_fade_out": _num("BGM渐出"),
                "frame_mode": field_to_text(fields.get("末帧模式")).strip(),
                "fade_seconds": _num("渐显时长"),
            },
            "options": {
                "tails": list(BUILTIN_TAILS.keys()),
                "voices": list(VOICE_OPTIONS.keys()),
                "transitions": list(TRANSITION_OPTIONS.keys()),
                "frame_modes": ["标准", "去字幕", "黑屏渐显"],
            },
        }

    return await asyncio.to_thread(_run)


@router.post("/params/save")
async def api_params_save(request: Request):
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    layer_doc = payload.get("layer_doc")
    if not record_id or not isinstance(layer_doc, dict):
        raise HTTPException(status_code=400, detail="record_id 和 layer_doc 必填")

    def _run():
        from tools.bitable_tool import BitableClient
        from tools.layer_model import parse_layer_doc
        from tools.h5_store import set_global_layer_doc
        raw = json.dumps(layer_doc, ensure_ascii=False)
        if parse_layer_doc(raw) is None:
            raise HTTPException(status_code=400, detail="layer_doc 不是合法的图层文档")
        client = BitableClient()
        update_fields = {"样式参数": raw}
        guide_text = payload.get("guide_text")
        if isinstance(guide_text, str):
            update_fields["引导语"] = guide_text
        client.update_records(app_token, table_id, [{"record_id": record_id, "fields": update_fields}])
        saved_default = False
        if payload.get("set_as_default"):
            set_global_layer_doc(layer_doc)
            saved_default = True
        return {"success": True, "set_as_default": saved_default}

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 精确预览
# ------------------------------------------------------------

@router.post("/settings/save")
async def api_settings_save(request: Request):
    """保存视频级设置（广告尾帧/配音/转场/BGM音量与渐变）到记录"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")
    s = payload.get("settings") or {}

    def _run():
        from tools.bitable_tool import BitableClient
        fields = {}
        text_map = {
            "tail_name": "广告尾帧",
            "tail_custom_url": "自定义尾帧URL",
            "voice_name": "配音音色",
            "transition1": "转场1",
            "transition2": "转场2",
            "frame_mode": "末帧模式",
        }
        for k, col in text_map.items():
            if k in s and s[k] is not None:
                fields[col] = s[k]
        num_map = {"bgm_volume": "BGM音量", "bgm_fade_in": "BGM渐入", "bgm_fade_out": "BGM渐出", "fade_seconds": "渐显时长"}
        for k, col in num_map.items():
            if k in s and s[k] is not None and s[k] != "":
                try:
                    fields[col] = float(s[k])
                except (TypeError, ValueError):
                    pass
        if fields:
            BitableClient().update_records(app_token, table_id, [{"record_id": record_id, "fields": fields}])
        return {"success": True, "updated": list(fields.keys())}

    return await asyncio.to_thread(_run)


@router.post("/render")
async def api_render(request: Request):
    """服务端精确预览：与成片同一渲染器输出静态合成图"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    frame_url = payload.get("frame_url", "")
    layer_doc = payload.get("layer_doc")
    if not frame_url or not isinstance(layer_doc, dict):
        raise HTTPException(status_code=400, detail="frame_url 和 layer_doc 必填")

    def _run():
        from tools.layer_render import render_static_preview
        from tools.video_pipeline import _download_file, _upload_image_to_s3, _find_chinese_font
        from PIL import Image
        uid = uuid.uuid4().hex[:10]
        tmp_dir = tempfile.gettempdir()
        frame_path = os.path.join(tmp_dir, f"h5pf_{uid}.png")
        out_path = os.path.join(tmp_dir, f"h5pv_{uid}.png")
        try:
            _download_file(frame_url, frame_path)
            with Image.open(frame_path) as im:
                width, height = im.size
            render_static_preview(
                frame_path=frame_path,
                canvas_w=width,
                canvas_h=height,
                layer_doc=layer_doc,
                layer_context=payload.get("context") or {},
                search_box_image_url=payload.get("search_box_url", ""),
                guide_text=payload.get("guide_text", ""),
                font_path=_find_chinese_font(),
                output_path=out_path,
            )
            preview_url = _upload_image_to_s3(out_path, f"h5/previews/pv_{uid}.png")
        finally:
            for p in (frame_path, out_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        record_id = payload.get("record_id", "")
        if record_id:
            try:
                from tools.bitable_tool import BitableClient
                BitableClient().update_records(
                    app_token, table_id,
                    [{"record_id": record_id, "fields": {"预览图URL": preview_url}}],
                )
            except Exception as e:
                logger.warning(f"[h5/render] 预览图URL写回失败: {e}")
        return {"preview_url": preview_url}

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 素材库与上传
# ------------------------------------------------------------

@router.post("/assets/upload")
async def api_asset_upload(request: Request, filename: str = "asset.png"):
    """上传图片素材（原始字节流），入共享素材库"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持图片格式: {IMAGE_EXTS}")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空文件")
    if len(body) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片不能超过20MB")

    def _run():
        from tools.video_pipeline import _get_storage
        storage = _get_storage()
        from io import BytesIO
        key = storage.stream_upload_file(
            fileobj=BytesIO(body),
            file_name=f"h5/assets/{uuid.uuid4().hex[:10]}{ext}",
            content_type=f"image/{ext.lstrip('.').replace('jpg', 'jpeg')}",
        )
        url = storage.generate_presigned_url(key=key, expire_time=2592000)
        from tools.h5_store import add_asset
        return add_asset(name=filename, url=url, content_type=f"image/{ext.lstrip('.')}")

    return await asyncio.to_thread(_run)


@router.post("/search-box/upload")
async def api_search_box_upload(request: Request, record_id: str = "", filename: str = "search_box.png"):
    """上传搜索框图片（原始字节流）→ 存对象存储 → 写入记录「搜索框图片URL」列（预览与成片都读它）"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持图片格式: {IMAGE_EXTS}")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空文件")
    if len(body) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片不能超过20MB")

    def _run():
        from tools.video_pipeline import _get_storage
        from tools.bitable_tool import BitableClient
        from io import BytesIO
        storage = _get_storage()
        key = storage.stream_upload_file(
            fileobj=BytesIO(body),
            file_name=f"h5/search_box/{uuid.uuid4().hex[:10]}{ext}",
            content_type=f"image/{ext.lstrip('.').replace('jpg', 'jpeg')}",
        )
        url = storage.generate_presigned_url(key=key, expire_time=2592000)
        BitableClient().update_records(
            DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID,
            [{"record_id": record_id, "fields": {"搜索框图片URL": url}}],
        )
        return {"url": url}

    return await asyncio.to_thread(_run)


@router.post("/bgm/upload")
async def api_bgm_upload(request: Request, record_id: str = "", filename: str = "bgm.mp3"):
    """上传 BGM 音频（原始字节流）→ 存对象存储 → 写入记录「BGM URL」列"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持音频格式: {AUDIO_EXTS}")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空文件")
    if len(body) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="音频不能超过30MB")

    def _run():
        from tools.video_pipeline import _get_storage
        from tools.bitable_tool import BitableClient
        from io import BytesIO
        storage = _get_storage()
        key = storage.stream_upload_file(
            fileobj=BytesIO(body),
            file_name=f"h5/bgm/{uuid.uuid4().hex[:10]}{ext}",
            content_type="audio/mpeg",
        )
        url = storage.generate_presigned_url(key=key, expire_time=2592000)
        BitableClient().update_records(
            DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID,
            [{"record_id": record_id, "fields": {"BGM URL": url}}],
        )
        return {"url": url, "name": filename}

    return await asyncio.to_thread(_run)


@router.get("/assets")
async def api_assets():
    def _run():
        from tools.h5_store import list_assets
        return {"assets": list_assets()}
    return await asyncio.to_thread(_run)


@router.post("/upload-video")
async def api_upload_video(request: Request, filename: str = "video.mp4"):
    """上传视频（原始字节流）→ 存对象存储 → 自动创建表格记录"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持视频格式: {VIDEO_EXTS}")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空文件")
    if len(body) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="视频不能超过200MB")

    def _run():
        from tools.video_pipeline import _get_storage
        from tools.bitable_tool import BitableClient
        from io import BytesIO
        storage = _get_storage()
        key = storage.stream_upload_file(
            fileobj=BytesIO(body),
            file_name=f"h5/videos/{uuid.uuid4().hex[:10]}{ext}",
            content_type="video/mp4",
        )
        url = storage.generate_presigned_url(key=key, expire_time=2592000)
        client = BitableClient()
        record_id = _write_video_record(
            client, DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID,
            {"视频URL": url, "视频名": filename},
        )
        if record_id:
            _kickoff_thumbnail(DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID, record_id, url)
        return {"success": True, "record_id": record_id, "video_url": url, "name": filename}

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 大文件分片上传（扣子网关对请求体限长：9MB可过、46MB被拒，
# 浏览器按 6MB 分片依次上传，后端顺序拼装后转存对象存储）
# ------------------------------------------------------------

MAX_UPLOAD_TOTAL = 500 * 1024 * 1024


def _part_path(upload_id: str) -> str:
    safe = "".join(c for c in upload_id if c.isalnum())[:32]
    if not safe:
        raise HTTPException(status_code=400, detail="非法 upload_id")
    return os.path.join(tempfile.gettempdir(), f"h5up_{safe}.part")


@router.post("/upload-video/init")
async def api_upload_init():
    upload_id = uuid.uuid4().hex[:16]
    open(_part_path(upload_id), "wb").close()
    return {"upload_id": upload_id, "chunk_size": 6 * 1024 * 1024}


@router.post("/upload-video/chunk")
async def api_upload_chunk(request: Request, upload_id: str, index: int = 0):
    path = _part_path(upload_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="upload_id 不存在或已过期，请重新 init")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空分片")
    if os.path.getsize(path) + len(body) > MAX_UPLOAD_TOTAL:
        try:
            os.remove(path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="文件超过500MB上限")

    def _append():
        with open(path, "ab") as f:
            f.write(body)
        return os.path.getsize(path)

    size = await asyncio.to_thread(_append)
    return {"received": len(body), "total": size, "index": index}


@router.post("/upload-video/finish")
async def api_upload_finish(request: Request):
    payload = await _json_body(request)
    upload_id = payload.get("upload_id", "")
    filename = payload.get("filename", "video.mp4")
    ext = os.path.splitext(filename)[1].lower() or ".mp4"
    if ext not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"仅支持视频格式: {VIDEO_EXTS}")
    path = _part_path(upload_id)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise HTTPException(status_code=400, detail="分片数据不存在或为空")

    def _run():
        from tools.video_pipeline import _get_storage
        from tools.bitable_tool import BitableClient
        storage = _get_storage()
        try:
            with open(path, "rb") as f:
                key = storage.stream_upload_file(
                    fileobj=f,
                    file_name=f"h5/videos/{uuid.uuid4().hex[:10]}{ext}",
                    content_type="video/mp4",
                )
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        url = storage.generate_presigned_url(key=key, expire_time=2592000)
        client = BitableClient()
        record_id = _write_video_record(
            client, DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID,
            {"视频URL": url, "视频名": filename},
        )
        if record_id:
            _kickoff_thumbnail(DEFAULT_APP_TOKEN, DEFAULT_TABLE_ID, record_id, url)
        return {"success": True, "record_id": record_id, "video_url": url, "name": filename}

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 记录列表与单条处理
# ------------------------------------------------------------

@router.post("/records")
async def api_records(request: Request):
    """记录列表（H5 首页），顺带回填缺失的「调整链接」"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)

    def _run():
        from tools.bitable_tool import BitableClient
        client = BitableClient()
        items, page_token = [], None
        while True:
            resp = client.search_records(app_token=app_token, table_id=table_id, page_token=page_token)
            items.extend(resp.get("data", {}).get("items", []))
            if not resp.get("data", {}).get("has_more"):
                break
            page_token = resp.get("data", {}).get("page_token")

        backfill = []
        for item in items:
            if not item.get("fields", {}).get("调整链接"):
                link = f"{PUBLIC_BASE_URL}/h5/?record_id={item.get('record_id')}"
                backfill.append({
                    "record_id": item.get("record_id"),
                    "fields": {"调整链接": {"link": link, "text": "调整样式"}},
                })
        if backfill:
            try:
                client.update_records(app_token, table_id, backfill)
            except Exception as e:
                logger.warning(f"[h5/records] 调整链接回填失败: {e}")

        records = [_record_summary(it) for it in items]
        records.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return {"records": records}

    return await asyncio.to_thread(_run)


@router.post("/process")
async def api_process(request: Request):
    """单条记录触发处理：置为待处理后后台执行，H5 轮询状态"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")

    def _kickoff():
        from tools.bitable_tool import BitableClient, field_to_text, attachment_to_download_url
        client = BitableClient()
        fields = _fetch_record_fields(app_token, table_id, record_id)
        update = {"处理状态": "待处理"}
        # 无视频URL时先从附件补URL；都为空才提示用户先上传，而非交给管线深层报错
        if not field_to_text(fields.get("视频URL")).strip():
            resolved = ""
            for att in ("视频附件", "附件"):
                if fields.get(att):
                    try:
                        resolved = attachment_to_download_url(client, fields.get(att))
                    except Exception as e:
                        logger.warning(f"[h5/process] 附件解析失败 record_id={record_id}: {e}")
                    if resolved:
                        break
            if resolved:
                update["视频URL"] = resolved
            else:
                raise HTTPException(status_code=400, detail="该记录没有视频源，请先上传视频后再提交处理")
        client.update_records(
            app_token, table_id,
            [{"record_id": record_id, "fields": update}],
        )

    await asyncio.to_thread(_kickoff)

    def _background():
        try:
            from tools.batch_tool import batch_process_from_bitable
            batch_process_from_bitable.invoke({
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
                "max_concurrency": 1,
                "send_notification": False,
            })
        except Exception as e:
            logger.error(f"[h5/process] 单条处理失败 record_id={record_id}: {e}", exc_info=True)

    threading.Thread(target=_background, daemon=True).start()
    return {"started": True, "record_id": record_id}


@router.post("/record-status")
async def api_record_status(request: Request):
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")

    def _run():
        fields = _fetch_record_fields(app_token, table_id, record_id)
        return _record_summary({"record_id": record_id, "fields": fields})

    return await asyncio.to_thread(_run)


@router.post("/record-delete")
async def api_record_delete(request: Request):
    """删除单条记录"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")

    def _run():
        from tools.bitable_tool import BitableClient
        BitableClient().delete_record(app_token, table_id, record_id)
        return {"success": True, "record_id": record_id}

    return await asyncio.to_thread(_run)


@router.post("/thumbnail")
async def api_thumbnail(request: Request):
    """懒加载缩略图：已有「缩略图URL」直接返回，否则抽原始最后一帧并回写。"""
    payload = await _json_body(request)
    app_token, table_id = _table_of(payload)
    record_id = payload.get("record_id", "")
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id 必填")

    def _run():
        from tools.bitable_tool import field_to_text, BitableClient
        fields = _fetch_record_fields(app_token, table_id, record_id)
        existing = field_to_text(fields.get("缩略图URL")).strip()
        if existing:
            return {"thumbnail_url": existing, "cached": True}
        url, cache_key = _resolve_video_source(fields)
        if not url:
            return {"thumbnail_url": "", "cached": False}
        res = _quick_frame(url, cache_key or url)
        thumb = res.get("frame_url", "")
        if thumb:
            try:
                BitableClient().update_records(
                    app_token, table_id,
                    [{"record_id": record_id, "fields": {"缩略图URL": thumb}}],
                )
            except Exception as e:
                logger.warning(f"[h5/thumbnail] 回写失败 record_id={record_id}: {e}")
        return {"thumbnail_url": thumb, "cached": False}

    return await asyncio.to_thread(_run)


# ------------------------------------------------------------
# 命名样式库（可复用样式）
# ------------------------------------------------------------

@router.post("/styles/list")
async def api_styles_list():
    def _run():
        from tools.h5_store import list_named_styles
        return {"styles": list_named_styles()}
    return await asyncio.to_thread(_run)


@router.post("/styles/save")
async def api_styles_save(request: Request):
    payload = await _json_body(request)
    name = payload.get("name", "")
    layer_doc = payload.get("layer_doc")
    if not isinstance(layer_doc, dict):
        raise HTTPException(status_code=400, detail="layer_doc 必填")

    def _run():
        from tools.layer_model import parse_layer_doc
        from tools.h5_store import save_named_style
        if parse_layer_doc(json.dumps(layer_doc, ensure_ascii=False)) is None:
            raise HTTPException(status_code=400, detail="layer_doc 不是合法的图层文档")
        return save_named_style(name, layer_doc, payload.get("guide_text", ""))

    return await asyncio.to_thread(_run)


@router.post("/styles/delete")
async def api_styles_delete(request: Request):
    payload = await _json_body(request)
    style_id = payload.get("style_id", "")
    if not style_id:
        raise HTTPException(status_code=400, detail="style_id 必填")

    def _run():
        from tools.h5_store import delete_named_style
        return {"success": delete_named_style(style_id)}

    return await asyncio.to_thread(_run)
