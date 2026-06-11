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
import asyncio
import traceback
from typing import Optional

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
# 批量处理工具
# ============================================================

@tool
def batch_process_from_bitable(
    app_token: str,
    table_id: str,
    max_concurrency: int = 3,
    send_notification: bool = True,
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

        summary["total"] = len(all_items)
        logger.info(f"[批量处理] 共 {len(all_items)} 条待处理记录")

        if not all_items:
            return json.dumps({
                "success": True,
                "message": "没有待处理的记录",
                "summary": summary,
            }, ensure_ascii=False)

        # Step 2: 并发处理
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_one(item: dict) -> dict:
            async with semaphore:
                record_id = item.get("record_id")
                fields = item.get("fields", {})

                video_url = fields.get("视频URL", "")
                tail_name = fields.get("广告尾帧", "短剧推广尾帧")
                voice_name = fields.get("配音音色", "米仔（视频配音女声）")
                guide_text = fields.get("引导语", "") or "后续剧情该如何选择？快来左下角造梦次元"
                subtitle_text = fields.get("字幕", "") or guide_text
                search_box_url = fields.get("搜索框图片URL", "")
                bgm_url = fields.get("BGM URL", "")
                bgm_volume = fields.get("BGM音量", None)
                transition1 = fields.get("转场1", "硬切（无转场）")
                transition2 = fields.get("转场2", "硬切（无转场）")

                logger.info(f"[批量处理] 开始处理: record_id={record_id}")

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
                    _guide_text = guide_text.strip() if guide_text and guide_text.strip() else "后续剧情该如何选择？快来左下角造梦次元"
                    _subtitle_text = subtitle_text.strip() if subtitle_text and subtitle_text.strip() else _guide_text
                    _voice_name = voice_name.strip() if voice_name and voice_name.strip() else "米仔（视频配音女声）"
                    _tail_name = tail_name.strip() if tail_name and tail_name.strip() else "短剧推广尾帧"
                    _tail_custom_url = ""
                    if _tail_name == "自定义":
                        _tail_custom_url = (fields.get("自定义尾帧URL") or "").strip()
                    _transition1 = transition1.strip() if transition1 and transition1.strip() else "硬切（无转场）"
                    _transition2 = transition2.strip() if transition2 and transition2.strip() else "硬切（无转场）"
                    _search_box_url = search_box_url.strip() if search_box_url and search_box_url.strip() else ""
                    _bgm_url = bgm_url.strip() if bgm_url and bgm_url.strip() else ""
                    _bgm_volume = float(bgm_volume) if bgm_volume else 0.6

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
                    )
                    # process_video_pipeline 返回 dict，直接使用
                    result_data = result

                    if result_data.get("success"):
                        # 成功：写回输出URL
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {
                                    "处理状态": "成功",
                                    "输出视频URL": result_data.get("final_video_url", ""),
                                },
                            }],
                        )
                        logger.info(f"[批量处理] 成功: record_id={record_id}")
                        return {"record_id": record_id, "status": "success", "url": result_data.get("final_video_url")}
                    else:
                        raise Exception(result_data.get("error", "未知错误"))

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"[批量处理] 失败: record_id={record_id}, error={error_msg}")
                    # 失败：写回错误信息
                    try:
                        client.update_records(
                            app_token=app_token,
                            table_id=table_id,
                            records=[{
                                "record_id": record_id,
                                "fields": {
                                    "处理状态": "失败",
                                    "错误信息": error_msg[:500],
                                },
                            }],
                        )
                    except Exception as update_err:
                        logger.error(f"[批量处理] 更新失败状态出错: {update_err}")
                    return {"record_id": record_id, "status": "failed", "error": error_msg}

        # 并发执行
        async def _run_all():
            return await asyncio.gather(*[process_one(item) for item in all_items], return_exceptions=True)

        results = asyncio.run(_run_all())

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
