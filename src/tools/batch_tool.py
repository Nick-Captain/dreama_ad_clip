"""
批量处理编排工具

功能：
1. 从飞书多维表格读取待处理记录
2. 并发调用视频处理管线
3. 逐条写回结果到表格
4. 处理完成后飞书推送通知
"""

import json
import logging
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlparse

import requests
from langchain.tools import tool
from coze_workload_identity import Client

logger = logging.getLogger(__name__)

# ============================================================
# 飞书消息推送
# ============================================================

_client = Client()


def _get_webhook_url() -> str:
    cred = _client.get_integration_credential("integration-feishu-message")
    return json.loads(cred)["webhook_url"]


def _send_feishu_text(text: str) -> dict:
    """发送飞书文本消息"""
    payload = {"msg_type": "text", "content": {"text": text}}
    resp = requests.post(_get_webhook_url(), json=payload, timeout=10)
    return resp.json()


def _send_feishu_card(title: str, content: str, actions: Optional[list] = None) -> dict:
    """发送飞书卡片消息"""
    elements = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": content},
        }
    ]
    if actions:
        elements.append({"tag": "action", "actions": actions})

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": elements,
        },
    }
    resp = requests.post(_get_webhook_url(), json=payload, timeout=10)
    return resp.json()


# ============================================================
# 素材URL分类：备用列里图片归搜索框、音频归BGM
# ============================================================

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma")


def _classify_material_url(url: str) -> str:
    """按扩展名归类素材URL（image/audio/unknown），无扩展名时读响应头判断。

    用流式 GET 而非 HEAD：TOS 预签名链接不支持 HEAD（返回403）。
    """
    path = urlparse(url).path.lower()
    if path.endswith(IMAGE_EXTS):
        return "image"
    if path.endswith(AUDIO_EXTS):
        return "audio"
    try:
        resp = requests.get(url, stream=True, timeout=15)
        content_type = resp.headers.get("Content-Type", "").lower()
        resp.close()
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("audio/"):
            return "audio"
    except Exception as e:
        logger.warning(f"[素材URL] 类型探测失败: {url[:100]} - {e}")
    return "unknown"


# 汉字→拼音首字母（基于 GBK 编码区间，覆盖常用简体字，零依赖）
_PINYIN_RANGES = [
    (0xB0A1, 0xB0C4, "a"), (0xB0C5, 0xB2C0, "b"), (0xB2C1, 0xB4ED, "c"), (0xB4EE, 0xB6E9, "d"),
    (0xB6EA, 0xB7A1, "e"), (0xB7A2, 0xB8C0, "f"), (0xB8C1, 0xB9FD, "g"), (0xB9FE, 0xBBF6, "h"),
    (0xBBF7, 0xBFA5, "j"), (0xBFA6, 0xC0AB, "k"), (0xC0AC, 0xC2E7, "l"), (0xC2E8, 0xC4C2, "m"),
    (0xC4C3, 0xC5B5, "n"), (0xC5B6, 0xC5BD, "o"), (0xC5BE, 0xC6D9, "p"), (0xC6DA, 0xC8BA, "q"),
    (0xC8BB, 0xC8F5, "r"), (0xC8F6, 0xCBF9, "s"), (0xCBFA, 0xCDD9, "t"), (0xCDDA, 0xCEF3, "w"),
    (0xCEF4, 0xD1B8, "x"), (0xD1B9, 0xD4D0, "y"), (0xD4D1, 0xD7F9, "z"),
]


def _pinyin_initial(ch: str) -> str:
    try:
        gbk = ch.encode("gbk")
    except Exception:
        return ""
    if len(gbk) != 2:
        return ""
    v = gbk[0] * 256 + gbk[1]
    for lo, hi, letter in _PINYIN_RANGES:
        if lo <= v <= hi:
            return letter
    return ""


def _ascii_name(s: str) -> str:
    """中文→拼音首字母、英文数字保留(小写)、其余丢弃。结果仅含 a-z0-9。"""
    out = []
    for ch in (s or ""):
        if ch.isascii():
            if ch.isalnum():
                out.append(ch.lower())
        else:
            out.append(_pinyin_initial(ch))
    return "".join(out)


def _short_tail_name(tail_name: str, tail_custom_url: str) -> str:
    """尾帧简称：派对接引尾帧→派对尾帧、短剧推广尾帧→短剧尾帧、自定义→自定义尾帧。"""
    if tail_custom_url and tail_custom_url.strip():
        return "自定义尾帧"
    t = (tail_name or "").strip()
    if t.endswith("尾帧") and len(t) > 2:
        return t[:2] + "尾帧"
    return t or "尾帧"


def _rename_output(final_url: str, video_name: str, created_ms, tail_name: str, tail_custom_url: str, uid: str):
    """把成片转存到我们的存储，命名为「日期-原文件名-尾帧简称.mp4」。
    返回 (新URL, 状态备注)；失败则返回 (原URL, 失败备注)。"""
    import os
    import tempfile
    import datetime
    from tools.video_pipeline import _download_file, _get_storage
    try:
        if isinstance(created_ms, (int, float)) and created_ms:
            date_s = datetime.datetime.fromtimestamp(created_ms / 1000).strftime("%Y%m%d")
        else:
            date_s = datetime.datetime.now().strftime("%Y%m%d")
        # 文件名仅含英文字母/数字/横杠：中文转拼音首字母
        base = _ascii_name(os.path.splitext((video_name or "").strip())[0]) or "video"
        tail_part = _ascii_name(_short_tail_name(tail_name, tail_custom_url).replace("尾帧", "")) or "tail"
        fname = f"{date_s}-{base}-{tail_part}.mp4"
        # 注：扣子 SDK 的 S3SyncStorage.stream_upload_file 不支持 content_disposition，
        # 故下载名只能是对象 key 的 basename = {fname}_{8位hash}.mp4（哈希去不掉）。
        # 要彻底纯净名需走 H5 后端下载代理（另行实现）。
        tmp = os.path.join(tempfile.gettempdir(), f"named_out_{uid}.mp4")
        _download_file(final_url, tmp)
        storage = _get_storage()
        obj_key = f"ad_tail_output/{fname}"
        with open(tmp, "rb") as f:
            key = storage.stream_upload_file(
                fileobj=f,
                file_name=obj_key,
                content_type="video/mp4",
            )
        note = f"改名成功({fname})"
        logger.info(f"[批量处理] 成片改名成功: {fname}")
        try:
            os.remove(tmp)
        except Exception:
            pass
        return storage.generate_presigned_url(key=key, expire_time=2592000), note
    except Exception as e:
        logger.warning(f"[批量处理] 成片规范命名失败，用原URL: {e}")
        return final_url, f"改名:失败用原URL({str(e)[:120]})"


# ============================================================
# 批量处理工具
# ============================================================

@tool
def batch_process_from_bitable(
    app_token: str,
    table_id: str,
    max_concurrency: int = 3,
    send_notification: bool = True,
    record_id: str = "",
) -> str:
    """
    从飞书多维表格批量处理广告尾帧视频。

    流程：
    1. 读取表格中「待处理」状态的记录
    2. 逐条调用视频处理管线
    3. 将结果写回表格（成功→输出URL，失败→错误信息）
    4. 全部完成后推送飞书通知

    参数说明：
    - app_token: 多维表格 Base 的 app_token（必填）
    - table_id: 数据表的 table_id（必填）
    - max_concurrency: 最大并发数，默认3
    - send_notification: 是否发送飞书通知，默认True
    - record_id: 只处理指定记录（可选，H5 单条处理用）

    返回：处理结果摘要的 JSON 字符串
    """
    from tools.bitable_tool import BitableClient
    from tools.video_pipeline import process_video_pipeline

    client = BitableClient()
    summary = {"total": 0, "success": 0, "failed": 0, "details": []}

    try:
        # Step 1: 读取待处理记录
        logger.info(f"[批量处理] 读取表格记录: app_token={app_token}, table_id={table_id}")
        filter_dict = {
            "conjunction": "and",
            "conditions": [
                {"field_name": "处理状态", "operator": "is", "value": ["待处理"]},
            ],
        }

        all_items = []
        page_token = None
        while True:
            resp = client.search_records(
                app_token=app_token,
                table_id=table_id,
                filter_dict=filter_dict,
                page_token=page_token,
            )
            items = resp.get("data", {}).get("items", [])
            all_items.extend(items)
            if not resp.get("data", {}).get("has_more"):
                break
            page_token = resp.get("data", {}).get("page_token")

        if record_id:
            all_items = [it for it in all_items if it.get("record_id") == record_id]

        summary["total"] = len(all_items)
        logger.info(f"[批量处理] 共 {len(all_items)} 条待处理记录")

        if not all_items:
            return json.dumps({
                "success": True,
                "message": "没有待处理的记录",
                "summary": summary,
            }, ensure_ascii=False)

        # Step 2: 并发处理。
        # 处理管线是阻塞同步调用，必须用线程池才能真正并发；
        # 之前的 asyncio.run 方案在 FastAPI/飞书回调（已有事件循环）中会直接抛 RuntimeError。
        from tools.bitable_tool import field_to_text, attachment_to_download_url, GUIDE_TEXT_OPTIONS
        from tools.layer_model import resolve_layer_doc
        from tools.h5_store import get_global_layer_doc

        # 全局默认样式整批只读一次（DB 故障时降级为内置默认）
        global_layer_doc = get_global_layer_doc()

        def process_one(item: dict) -> dict:
                record_id = item.get("record_id")
                fields = item.get("fields", {})

                # search 接口的文本字段返回富文本片段数组，统一转成字符串再用
                video_url = field_to_text(fields.get("视频URL")).strip()

                # URL 为空时回退到附件列：用户可直接把视频文件传进表格
                attachment_error = ""
                if not video_url:
                    for att_field_name in ("视频附件", "附件"):
                        att_value = fields.get(att_field_name)
                        if not att_value:
                            continue
                        try:
                            video_url = attachment_to_download_url(client, att_value)
                        except Exception as att_err:
                            attachment_error = f"读取附件「{att_field_name}」失败: {att_err}"
                            logger.warning(f"[批量处理] record_id={record_id} {attachment_error}")
                        if video_url:
                            break
                tail_name = field_to_text(fields.get("广告尾帧"))
                voice_name = field_to_text(fields.get("配音音色"))
                guide_text = field_to_text(fields.get("引导语"))
                # 记录里已有旧的输出链接时（重跑场景），覆盖前备份到错误信息列
                old_output_url = field_to_text(fields.get("输出视频URL")).strip()
                bgm_volume = fields.get("BGM音量", None)
                transition1 = field_to_text(fields.get("转场1"))
                transition2 = field_to_text(fields.get("转场2"))

                # 搜索框/BGM：直接读 URL 列（附件列与「素材URL」备用列已废弃删除）
                search_box_url = field_to_text(fields.get("搜索框图片URL")).strip()
                bgm_url = field_to_text(fields.get("BGM URL")).strip()

                logger.info(f"[批量处理] 开始处理: record_id={record_id}")

                if not video_url:
                    error_msg = attachment_error or "「视频URL」和「视频附件/附件」均为空，已跳过"
                    try:
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {"处理状态": "失败", "错误信息": error_msg},
                            }],
                        )
                    except Exception as update_err:
                        logger.error(f"[批量处理] 更新失败状态出错: {update_err}")
                    return {"record_id": record_id, "status": "failed", "error": error_msg}

                # 更新状态为「处理中」
                try:
                    client.update_records(
                        app_token=app_token,
                        table_id=table_id,
                        records=[{"record_id": record_id, "fields": {"处理状态": "处理中"}}],
                    )
                except Exception as e:
                    logger.warning(f"[批量处理] 更新状态失败: {e}")

                # 调用视频处理管线
                try:
                    # 默认值处理：空字段使用内置默认值
                    # 引导语（=中间帧字幕，同一内容）：未选择时从文案池随机选取
                    _guide_text = guide_text.strip() if guide_text and guide_text.strip() else random.choice(GUIDE_TEXT_OPTIONS)
                    _subtitle_text = _guide_text
                    _voice_name = voice_name.strip() if voice_name and voice_name.strip() else "米仔（视频配音女声）"
                    _tail_name = tail_name.strip() if tail_name and tail_name.strip() else "短剧推广尾帧"
                    _tail_custom_url = ""
                    if _tail_name == "自定义":
                        _tail_custom_url = field_to_text(fields.get("自定义尾帧URL")).strip()
                    _transition1 = transition1.strip() if transition1 and transition1.strip() else "硬切（无转场）"
                    _transition2 = transition2.strip() if transition2 and transition2.strip() else "硬切（无转场）"
                    _search_box_url = search_box_url.strip() if search_box_url and search_box_url.strip() else ""
                    _bgm_url = bgm_url.strip() if bgm_url and bgm_url.strip() else ""
                    try:
                        _bgm_volume = float(bgm_volume) if bgm_volume else 0.6
                    except (TypeError, ValueError):
                        _bgm_volume = 0.6

                    def _num(name):
                        try:
                            return float(fields.get(name) or 0)
                        except (TypeError, ValueError):
                            return 0.0
                    _bgm_fade_in = _num("BGM渐入")
                    _bgm_fade_out = _num("BGM渐出")
                    _frame_mode = field_to_text(fields.get("末帧模式")).strip() or "标准"
                    _fade_seconds = _num("渐显时长") or 4.0

                    # 图层样式：记录级「样式参数」> 全局默认 > 内置默认
                    _layer_doc = resolve_layer_doc(
                        field_to_text(fields.get("样式参数")),
                        global_layer_doc,
                    )
                    _layer_ctx = {"角色名": field_to_text(fields.get("角色名")).strip()}

                    result = process_video_pipeline(
                        video_url=video_url,
                        guide_text=_guide_text,
                        subtitle_text=_subtitle_text,
                        voice_name=_voice_name,
                        tail_name=_tail_name,
                        tail_custom_url=_tail_custom_url,
                        transition1_name=_transition1,
                        transition2_name=_transition2,
                        search_box_image_url=_search_box_url,
                        bgm_url=_bgm_url,
                        bgm_volume=_bgm_volume,
                        bgm_fade_in=_bgm_fade_in,
                        bgm_fade_out=_bgm_fade_out,
                        frame_mode=_frame_mode,
                        fade_seconds=_fade_seconds,
                        style_layers=_layer_doc,
                        layer_context=_layer_ctx,
                    )
                    # process_video_pipeline 返回 dict，直接使用
                    result_data = result

                    if result_data.get("success"):
                        # 成功：成片规范命名（日期-原名-尾帧）后写回输出URL
                        final_url = result_data.get("final_video_url", "")
                        rename_note = ""
                        if final_url:
                            final_url, rename_note = _rename_output(
                                final_url, field_to_text(fields.get("视频名")),
                                fields.get("创作日期"), _tail_name, _tail_custom_url, record_id,
                            )
                        if old_output_url:
                            note = f"提示：本条为重新处理，旧输出视频已被覆盖。旧链接备份：{old_output_url}"
                        else:
                            note = ""
                        # 关键结果先写（这几列必定存在），保证出片成功一定能写回 URL/状态
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {
                                    "处理状态": "成功",
                                    "输出视频URL": final_url,
                                    "错误信息": note,
                                },
                            }],
                        )
                        # 诊断信息（替代扣子运行时拿不到的日志）：每步耗时 + 改名/CD 结果。
                        # 单独写、容错：若「调试信息」列未 migrate 出来，写失败也不影响主结果。
                        debug_text = result_data.get("debug_info", "")
                        if rename_note:
                            debug_text = (debug_text + "\n" + rename_note) if debug_text else rename_note
                        if debug_text:
                            try:
                                client.update_records(
                                    app_token=app_token,
                                    table_id=table_id,
                                    records=[{
                                        "record_id": record_id,
                                        "fields": {"调试信息": debug_text[:2000]},
                                    }],
                                )
                            except Exception as dbg_err:
                                logger.warning(f"[批量处理] 写「调试信息」列失败（可能未 migrate），忽略: {dbg_err}")
                        logger.info(f"[批量处理] 成功: record_id={record_id}")
                        return {"record_id": record_id, "status": "success", "url": final_url}
                    else:
                        raise Exception(result_data.get("error", "未知错误"))

                except Exception as e:
                    error_msg = str(e)
                    tb = traceback.format_exc()
                    logger.error(f"[批量处理] 失败: record_id={record_id}, error={error_msg}\n{tb}")
                    # 把出错位置（最后两层调用帧）一并写回表格，便于不翻服务端日志就能定位
                    frame_lines = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
                    location = " ← ".join(frame_lines[-2:]) if frame_lines else ""
                    detail = f"{error_msg}\n位置: {location}" if location else error_msg
                    # 失败：写回错误信息
                    try:
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {
                                    "处理状态": "失败",
                                    "错误信息": detail[:500],
                                },
                            }],
                        )
                    except Exception as update_err:
                        logger.error(f"[批量处理] 更新失败状态出错: {update_err}")
                    return {"record_id": record_id, "status": "failed", "error": error_msg}

        # 线程池并发执行
        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
            futures = [pool.submit(process_one, item) for item in all_items]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)

        # 统计结果
        for r in results:
            if isinstance(r, Exception):
                summary["failed"] += 1
                summary["details"].append({"status": "failed", "error": str(r)})
            elif isinstance(r, dict) and r.get("status") == "success":
                summary["success"] += 1
                summary["details"].append(r)
            else:
                summary["failed"] += 1
                summary["details"].append(r)

        # Step 3: 发送飞书通知
        if send_notification:
            card_content = (
                f"**批量处理完成**\n\n"
                f"📊 总计：{summary['total']} 条\n"
                f"✅ 成功：{summary['success']} 条\n"
                f"❌ 失败：{summary['failed']} 条\n\n"
                f"请查看多维表格获取详细结果。"
            )
            try:
                _send_feishu_card("广告尾帧批量处理", card_content)
            except Exception as e:
                logger.warning(f"[批量处理] 发送飞书通知失败: {e}")

        return json.dumps({
            "success": True,
            "message": f"批量处理完成：总计 {summary['total']}，成功 {summary['success']}，失败 {summary['failed']}",
            "summary": summary,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[批量处理] 整体失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"批量处理失败: {str(e)}",
        }, ensure_ascii=False)


@tool
def send_feishu_notification(message: str, title: str = "通知") -> str:
    """
    发送飞书消息通知。

    参数说明：
    - message: 消息内容（支持 Markdown 格式）
    - title: 通知标题，默认「通知」

    返回：发送结果的 JSON 字符串
    """
    try:
        if len(message) > 500:
            # 长消息用卡片
            result = _send_feishu_card(title, message)
        else:
            result = _send_feishu_text(message)
        return json.dumps({"success": True, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
