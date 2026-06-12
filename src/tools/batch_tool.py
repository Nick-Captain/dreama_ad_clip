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

                # 搜索框/BGM：优先附件列，兼容旧URL列，最后用「素材URL」备用列补位
                search_box_url = ""
                bgm_url = ""
                if fields.get("搜索框图片"):
                    try:
                        search_box_url = attachment_to_download_url(client, fields.get("搜索框图片"))
                    except Exception as att_err:
                        logger.warning(f"[批量处理] record_id={record_id} 读取附件「搜索框图片」失败: {att_err}")
                if fields.get("BGM"):
                    try:
                        bgm_url = attachment_to_download_url(client, fields.get("BGM"))
                    except Exception as att_err:
                        logger.warning(f"[批量处理] record_id={record_id} 读取附件「BGM」失败: {att_err}")
                if not search_box_url:
                    search_box_url = field_to_text(fields.get("搜索框图片URL")).strip()
                if not bgm_url:
                    bgm_url = field_to_text(fields.get("BGM URL")).strip()
                for material_url in filter(None, re.split(r"[\s,，;；]+", field_to_text(fields.get("素材URL")).strip())):
                    kind = _classify_material_url(material_url)
                    if kind == "image" and not search_box_url:
                        search_box_url = material_url
                    elif kind == "audio" and not bgm_url:
                        bgm_url = material_url
                    elif kind == "unknown":
                        logger.warning(f"[批量处理] record_id={record_id} 素材URL无法识别类型，已忽略: {material_url[:100]}")

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
                        style_layers=_layer_doc,
                        layer_context=_layer_ctx,
                    )
                    # process_video_pipeline 返回 dict，直接使用
                    result_data = result

                    if result_data.get("success"):
                        # 成功：写回输出URL；如有旧链接则备份说明，否则清空过期的错误信息
                        if old_output_url:
                            note = f"提示：本条为重新处理，旧输出视频已被覆盖。旧链接备份：{old_output_url}"
                        else:
                            note = ""
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {
                                    "处理状态": "成功",
                                    "输出视频URL": result_data.get("final_video_url", ""),
                                    "错误信息": note,
                                },
                            }],
                        )
                        logger.info(f"[批量处理] 成功: record_id={record_id}")
                        return {"record_id": record_id, "status": "success", "url": result_data.get("final_video_url")}
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
