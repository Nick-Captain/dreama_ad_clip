"""
飞书多维表格工具

功能：
1. 自动创建「广告尾帧批量处理」多维表格模板
2. 读取表格记录
3. 更新记录状态和结果
"""

import json
import logging
import uuid
from typing import Optional

import requests
from functools import wraps
from cozeloop.decorator import observe
from coze_workload_identity import Client

from langchain.tools import tool

logger = logging.getLogger(__name__)

# ============================================================
# 表格模板字段定义
# ============================================================
TEMPLATE_FIELDS = [
    {
        "field_name": "视频URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "广告尾帧",
        "type": 3,  # 单选
        "property": {
            "options": [
                {"name": "派对接引尾帧"},
                {"name": "短剧推广尾帧"},
                {"name": "自定义"},
            ]
        },
    },
    {
        "field_name": "自定义尾帧URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "配音音色",
        "type": 3,  # 单选
        "property": {
            "options": [
                {"name": "小荷（通用女声）"},
                {"name": "米仔（视频配音女声）"},
                {"name": "大奕（视频配音男声）"},
                {"name": "可爱女生"},
            ]
        },
    },
    {
        "field_name": "引导语",
        "type": 1,  # 文本
    },
    {
        "field_name": "字幕",
        "type": 1,  # 文本
    },
    {
        "field_name": "搜索框图片URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "BGM URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "BGM音量",
        "type": 2,  # 数字
    },
    {
        "field_name": "转场1",
        "type": 1,  # 文本
    },
    {
        "field_name": "转场2",
        "type": 1,  # 文本
    },
    {
        "field_name": "处理状态",
        "type": 3,  # 单选
        "property": {
            "options": [
                {"name": "待处理"},
                {"name": "处理中"},
                {"name": "成功"},
                {"name": "失败"},
            ]
        },
    },
    {
        "field_name": "输出视频URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "预览图URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "错误信息",
        "type": 1,  # 文本
    },
]

# ============================================================
# 飞书多维表格客户端
# ============================================================

_client = Client()


def _get_access_token() -> str:
    return _client.get_integration_credential("integration-feishu-base")


def _require_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = _get_access_token()
        if not token:
            raise ValueError("FEISHU_TENANT_ACCESS_TOKEN is not set")
        return func(*args, **kwargs)
    return wrapper


class BitableClient:
    """飞书多维表格 HTTP 客户端（精简版，仅包含需要的接口）"""

    BASE_URL = "https://open.larkoffice.com/open-apis"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {_get_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }

    @observe
    def _request(self, method: str, path: str, params: dict | None = None, json_body: dict | None = None) -> dict:
        try:
            url = f"{self.BASE_URL}{path}"
            resp = requests.request(
                method, url, headers=self._headers(),
                params=params, json=json_body, timeout=self.timeout
            )
            resp_data = resp.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"FeishuBitable API request error: {e}")
        if resp_data.get("code") != 0:
            raise Exception(f"FeishuBitable API error: code={resp_data.get('code')}, msg={resp_data.get('msg')}")
        return resp_data

    def create_base(self, name: str) -> dict:
        """创建多维表格 Base"""
        return self._request("POST", "/bitable/v1/apps", json_body={"name": name})

    def list_tables(self, app_token: str) -> dict:
        """列出 Base 下所有数据表"""
        return self._request("GET", f"/bitable/v1/apps/{app_token}/tables")

    def create_table(self, app_token: str, table_name: str, fields: list | None = None) -> dict:
        """创建数据表"""
        body: dict = {"table_name": table_name}
        if fields:
            body["fields"] = fields
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables", json_body=body)

    def search_records(
        self,
        app_token: str,
        table_id: str,
        filter_dict: dict | None = None,
        page_size: int = 500,
        page_token: str | None = None,
    ) -> dict:
        """查询记录"""
        body: dict = {"page_size": page_size}
        if filter_dict:
            body["filter"] = filter_dict
        if page_token:
            body["page_token"] = page_token
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/search", json_body=body)

    def update_records(self, app_token: str, table_id: str, records: list) -> dict:
        """批量更新记录"""
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update", json_body={
            "records": records,
        })

    def list_fields(self, app_token: str, table_id: str) -> dict:
        """列出字段"""
        return self._request("GET", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields")

    def add_field(self, app_token: str, table_id: str, field: dict) -> dict:
        """新增字段"""
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields", json_body=field)


# ============================================================
# LangChain 工具
# ============================================================

@tool
def create_bitable_template(table_name: str = "广告尾帧批量处理") -> str:
    """
    自动创建「广告尾帧批量处理」飞书多维表格模板。

    参数说明：
    - table_name: 表格名称，默认「广告尾帧批量处理」

    返回：包含 app_token 和 table_id 的 JSON 字符串，用户可将此信息告诉机器人用于后续批量处理。
    """
    client = BitableClient()
    try:
        # 步骤1：创建 Base（会自动带一个默认数据表）
        base_resp = client.create_base(name=table_name)
        app_token = base_resp["data"]["app"]["app_token"]
        logger.info(f"创建 Base 成功: app_token={app_token}")

        # 步骤2：获取默认数据表
        tables_resp = client.list_tables(app_token=app_token)
        tables = tables_resp.get("data", {}).get("items", [])
        if not tables:
            raise Exception("Base 创建后未找到默认数据表")
        table_id = tables[0]["table_id"]
        logger.info(f"使用默认数据表: table_id={table_id}")

        # 步骤3：逐个添加字段
        added_fields = []
        for field_def in TEMPLATE_FIELDS:
            try:
                client.add_field(app_token=app_token, table_id=table_id, field=field_def)
                added_fields.append(field_def["field_name"])
                logger.info(f"添加字段成功: {field_def['field_name']}")
            except Exception as field_err:
                logger.warning(f"添加字段失败 {field_def['field_name']}: {field_err}")

        return json.dumps({
            "success": True,
            "app_token": app_token,
            "table_id": table_id,
            "fields_added": added_fields,
            "total_fields": len(TEMPLATE_FIELDS),
            "message": f"多维表格「{table_name}」创建成功！已添加 {len(added_fields)}/{len(TEMPLATE_FIELDS)} 个字段。",
            "usage": f"批量处理时告诉机器人：开始批量处理 app_token={app_token} table_id={table_id}",
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"创建表格模板失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"创建表格模板失败: {str(e)}",
        }, ensure_ascii=False)


@tool
def get_bitable_records(
    app_token: str,
    table_id: str,
    status_filter: str = "待处理",
) -> str:
    """
    从飞书多维表格中获取指定状态的记录。

    参数说明：
    - app_token: 多维表格 Base 的 app_token（必填）
    - table_id: 数据表的 table_id（必填）
    - status_filter: 按处理状态筛选，默认「待处理」。传空字符串获取全部记录。

    返回：记录列表的 JSON 字符串
    """
    client = BitableClient()
    try:
        filter_dict = None
        if status_filter:
            filter_dict = {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "处理状态",
                        "operator": "is",
                        "value": [status_filter],
                    }
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

        # 精简输出
        records = []
        for item in all_items:
            fields = item.get("fields", {})
            records.append({
                "record_id": item.get("record_id"),
                "视频URL": fields.get("视频URL", ""),
                "广告尾帧": fields.get("广告尾帧", ""),
                "配音音色": fields.get("配音音色", ""),
                "引导语": fields.get("引导语", ""),
                "字幕": fields.get("字幕", ""),
                "搜索框图片URL": fields.get("搜索框图片URL", ""),
                "BGM URL": fields.get("BGM URL", ""),
                "BGM音量": fields.get("BGM音量", ""),
                "转场1": fields.get("转场1", ""),
                "转场2": fields.get("转场2", ""),
                "处理状态": fields.get("处理状态", ""),
                "输出视频URL": fields.get("输出视频URL", ""),
                "预览图URL": fields.get("预览图URL", ""),
                "错误信息": fields.get("错误信息", ""),
            })

        return json.dumps({
            "success": True,
            "count": len(records),
            "records": records,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"获取表格记录失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"获取表格记录失败: {str(e)}",
        }, ensure_ascii=False)


@tool
def update_bitable_record(
    app_token: str,
    table_id: str,
    record_id: str,
    status: str = "",
    output_video_url: str = "",
    preview_url: str = "",
    error_message: str = "",
) -> str:
    """
    更新飞书多维表格中的单条记录。

    参数说明：
    - app_token: 多维表格 Base 的 app_token（必填）
    - table_id: 数据表的 table_id（必填）
    - record_id: 记录 ID（必填）
    - status: 处理状态，可选：待处理/处理中/成功/失败
    - output_video_url: 输出视频URL
    - preview_url: 预览图URL
    - error_message: 错误信息

    返回：更新结果的 JSON 字符串
    """
    client = BitableClient()
    try:
        fields = {}
        if status:
            fields["处理状态"] = status
        if output_video_url:
            fields["输出视频URL"] = output_video_url
        if preview_url:
            fields["预览图URL"] = preview_url
        if error_message:
            fields["错误信息"] = error_message

        if not fields:
            return json.dumps({"success": False, "error": "没有需要更新的字段"}, ensure_ascii=False)

        client.update_records(
            app_token=app_token,
            table_id=table_id,
            records=[{"record_id": record_id, "fields": fields}],
        )

        return json.dumps({
            "success": True,
            "record_id": record_id,
            "updated_fields": list(fields.keys()),
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"更新表格记录失败: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"更新表格记录失败: {str(e)}",
        }, ensure_ascii=False)
