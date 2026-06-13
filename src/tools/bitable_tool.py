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

from tools.video_pipeline import TRANSITION_OPTIONS

logger = logging.getLogger(__name__)


def field_to_text(value) -> str:
    """把多维表格 records/search 接口返回的字段值统一转成字符串。

    该接口对文本字段返回富文本片段数组（如 [{"text": "...", "type": "text"}]），
    URL 字段返回 dict，单选返回字符串——直接当字符串 .strip() 会抛 AttributeError。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(field_to_text(seg) for seg in value)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("link") or "")
    return str(value)


def attachment_to_download_url(client: "BitableClient", value) -> str:
    """把附件字段值换成可直接下载的临时URL（多个附件时取第一个）。

    附件字段值形如 [{"file_token": "...", "name": "...", "size": ..., "type": "..."}]，
    其中自带的 url/tmp_url 下载时仍需鉴权头，处理管线用不了，
    必须经 batch_get_tmp_download_url 换成免鉴权的临时链接。
    """
    if not isinstance(value, list) or not value:
        return ""
    first = value[0]
    if not isinstance(first, dict):
        return ""
    file_token = first.get("file_token", "")
    if not file_token:
        return ""
    resp = client.get_tmp_download_urls([file_token])
    urls = resp.get("data", {}).get("tmp_download_urls", [])
    return urls[0].get("tmp_download_url", "") if urls else ""


# ============================================================
# 中间帧引导语文案池：下拉选项 + 留空时随机选取
# ============================================================
GUIDE_TEXT_OPTIONS = [
    "真相即将揭晓！快来左下角造梦次元",
    "剧情反转猜不到？快来造梦次元解锁",
    "接下来会发生什么？造梦次元告诉你",
    "结局由你来定！快来左下角造梦次元",
    "下一步怎么选？来造梦次元亲自决定",
    "你的选择决定结局！速来造梦次元",
    "故事远未结束！左下角造梦次元等你",
    "看得意难平？来造梦次元改写结局",
    "千万别走开！精彩续集就在造梦次元",
    "想知道后续剧情？左下角搜造梦次元",
]

# ============================================================
# 表格模板字段定义
# ============================================================
TEMPLATE_FIELDS = [
    {
        "field_name": "视频URL",
        "type": 1,  # 文本
    },
    {
        "field_name": "视频附件",
        "type": 17,  # 附件：与「视频URL」二选一，附件优先级低于URL
    },
    {
        "field_name": "视频名",
        "type": 1,  # 文本：H5 上传时写入的原始文件名，用于列表展示
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
        "type": 3,  # 单选：中间帧显示的字幕及配音文案，留空则从文案池随机
        "property": {
            "options": [{"name": t} for t in GUIDE_TEXT_OPTIONS]
        },
    },
    {
        "field_name": "搜索框图片",
        "type": 17,  # 附件
    },
    {
        "field_name": "搜索框图片URL",
        "type": 1,  # 文本：H5 编辑器内上传的搜索框图片直链；附件列优先，此列兜底。预览与成片均读取
    },
    {
        "field_name": "BGM",
        "type": 17,  # 附件
    },
    {
        "field_name": "BGM URL",
        "type": 1,  # 文本：H5 编辑器内上传的 BGM 直链；附件列优先，此列兜底
    },
    {
        "field_name": "素材URL",
        "type": 1,  # 文本：备用入口，可放多个URL，图片自动归为搜索框、音频归为BGM
    },
    {
        "field_name": "BGM音量",
        "type": 2,  # 数字
    },
    {
        "field_name": "BGM渐入",
        "type": 2,  # 数字：淡入时长（秒），0=不渐入
    },
    {
        "field_name": "BGM渐出",
        "type": 2,  # 数字：淡出时长（秒），0=不渐出
    },
    {
        "field_name": "转场1",
        "type": 3,  # 单选：选项与 video_pipeline.TRANSITION_OPTIONS 同源
        "property": {
            "options": [{"name": name} for name in TRANSITION_OPTIONS]
        },
    },
    {
        "field_name": "转场2",
        "type": 3,  # 单选：选项与 video_pipeline.TRANSITION_OPTIONS 同源
        "property": {
            "options": [{"name": name} for name in TRANSITION_OPTIONS]
        },
    },
    {
        "field_name": "样式参数",
        "type": 1,  # 文本：H5 编辑器托管的图层 JSON，用户不手填
    },
    {
        "field_name": "调整链接",
        "type": 15,  # 超链接：该记录专属的 H5 编辑器入口
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
        "field_name": "缩略图URL",
        "type": 1,  # 文本：视频原始最后一帧，H5 列表缩略图（与带样式的「预览图URL」区分）
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

    def get_record(self, app_token: str, table_id: str, record_id: str) -> dict:
        """读取单条记录"""
        return self._request("GET", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}")

    def create_record(self, app_token: str, table_id: str, fields: dict) -> dict:
        """新增单条记录"""
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records", json_body={"fields": fields})

    def delete_record(self, app_token: str, table_id: str, record_id: str) -> dict:
        """删除单条记录"""
        return self._request("DELETE", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}")

    def update_records(self, app_token: str, table_id: str, records: list) -> dict:
        """批量更新记录"""
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update", json_body={
            "records": records,
        })

    def list_fields(self, app_token: str, table_id: str) -> dict:
        """列出字段"""
        return self._request("GET", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields")

    def get_tmp_download_urls(self, file_tokens: list) -> dict:
        """获取素材（如多维表格附件）的临时下载链接，链接可免鉴权直接下载"""
        return self._request(
            "GET", "/drive/v1/medias/batch_get_tmp_download_url",
            params={"file_tokens": file_tokens},
        )

    def add_field(self, app_token: str, table_id: str, field: dict) -> dict:
        """新增字段"""
        return self._request("POST", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields", json_body=field)

    def update_field(self, app_token: str, table_id: str, field_id: str, field: dict) -> dict:
        """更新字段（可改名称、类型、选项）"""
        return self._request("PUT", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}", json_body=field)

    def delete_field(self, app_token: str, table_id: str, field_id: str) -> dict:
        """删除字段"""
        return self._request("DELETE", f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}")


# 历史遗留的冗余列：新建 Base 自带的示例列 + 旧版模板的重复/改版字段
LEGACY_FIELDS_TO_DELETE = ("文本", "单选", "日期", "字幕")


def _count_field_usage(client: "BitableClient", app_token: str, table_id: str, field_names: list) -> dict:
    """统计每个字段在多少条记录里有值（用于判断哪列是空列）"""
    counts = {n: 0 for n in field_names}
    page_token = None
    while True:
        resp = client.search_records(app_token=app_token, table_id=table_id, page_token=page_token)
        for item in resp.get("data", {}).get("items", []):
            fields = item.get("fields", {})
            for n in field_names:
                if fields.get(n):
                    counts[n] += 1
        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp.get("data", {}).get("page_token")
    return counts


def migrate_bitable_schema(app_token: str, table_id: str) -> dict:
    """
    将已有表格的结构对齐到当前 TEMPLATE_FIELDS：
    1. 索引列改造为「创作日期」（日期类型；索引列不可删除只能改造，
       历史上曾是默认「文本」、后为「素材URL」）
    2. 合并附件列：只保留「视频附件」（空列删除、有数据的列改名，不丢数据）
    3. 删除历史遗留的冗余列（仅删 LEGACY_FIELDS_TO_DELETE 中点名的）
    4. 对齐字段类型/下拉选项，补齐缺失列；模板外的用户自建列一律不碰

    幂等：重复执行无副作用。
    """
    client = BitableClient()
    actions = []

    fields_resp = client.list_fields(app_token=app_token, table_id=table_id)
    existing = {f["field_name"]: f for f in fields_resp.get("data", {}).get("items", [])}

    # 1. 索引列改造为「创作日期」（素材URL 功能由普通列承接，缺失时步骤4自动补建）
    primary = next((f for f in existing.values() if f.get("is_primary")), None)
    if primary and primary["field_name"] != "创作日期":
        old_name = primary["field_name"]
        try:
            client.update_field(
                app_token, table_id, primary["field_id"],
                {
                    "field_name": "创作日期",
                    "type": 5,  # 日期
                    "property": {"date_formatter": "yyyy/MM/dd"},
                },
            )
            existing["创作日期"] = existing.pop(old_name)
            actions.append(f"索引列「{old_name}」改造为「创作日期」（日期类型）")
        except Exception as e:
            actions.append(f"改造索引列失败: {e}")

    # 2. 合并附件列：只保留「视频附件」
    if "附件" in existing:
        try:
            if "视频附件" not in existing:
                client.update_field(
                    app_token, table_id, existing["附件"]["field_id"],
                    {"field_name": "视频附件", "type": 17},
                )
                existing["视频附件"] = existing.pop("附件")
                actions.append("「附件」更名为「视频附件」")
            else:
                usage = _count_field_usage(client, app_token, table_id, ["附件", "视频附件"])
                if usage["视频附件"] == 0:
                    client.delete_field(app_token, table_id, existing["视频附件"]["field_id"])
                    client.update_field(
                        app_token, table_id, existing["附件"]["field_id"],
                        {"field_name": "视频附件", "type": 17},
                    )
                    existing["视频附件"] = existing.pop("附件")
                    actions.append("删除空的「视频附件」，「附件」更名为「视频附件」（原数据保留）")
                elif usage["附件"] == 0:
                    client.delete_field(app_token, table_id, existing["附件"]["field_id"])
                    existing.pop("附件")
                    actions.append("删除空的「附件」列")
                else:
                    actions.append("「附件」与「视频附件」均有数据，为避免丢数据未自动合并，请手动搬运后删除「附件」")
        except Exception as e:
            actions.append(f"合并附件列失败: {e}")

    # 3. 删除冗余列
    for name in LEGACY_FIELDS_TO_DELETE:
        if name in existing:
            try:
                client.delete_field(app_token, table_id, existing[name]["field_id"])
                actions.append(f"删除冗余列「{name}」")
                existing.pop(name)
            except Exception as e:
                actions.append(f"删除「{name}」失败: {e}")

    # 4. 对齐字段类型/下拉选项，补齐缺失列
    for field_def in TEMPLATE_FIELDS:
        name = field_def["field_name"]
        if name in existing:
            current = existing[name]
            need_update = current.get("type") != field_def["type"]
            template_options = {o["name"] for o in field_def.get("property", {}).get("options", [])}
            if not need_update and template_options:
                current_options = {o.get("name") for o in (current.get("property") or {}).get("options", [])}
                need_update = current_options != template_options
            if need_update:
                try:
                    body = {"field_name": name, "type": field_def["type"]}
                    if "property" in field_def:
                        body["property"] = field_def["property"]
                    client.update_field(app_token, table_id, current["field_id"], body)
                    actions.append(f"「{name}」类型/选项已对齐模板")
                except Exception as e:
                    actions.append(f"更新「{name}」失败: {e}")
        else:
            try:
                client.add_field(app_token=app_token, table_id=table_id, field=field_def)
                actions.append(f"新增列「{name}」")
            except Exception as e:
                actions.append(f"新增「{name}」失败: {e}")

    final_fields = client.list_fields(app_token=app_token, table_id=table_id)
    field_names = [f["field_name"] for f in final_fields.get("data", {}).get("items", [])]
    return {"success": True, "actions": actions, "current_fields": field_names}


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

        # 步骤3：对齐模板结构（清理默认示例列 + 添加业务字段）
        migrate_result = migrate_bitable_schema(app_token, table_id)

        return json.dumps({
            "success": True,
            "app_token": app_token,
            "table_id": table_id,
            "actions": migrate_result.get("actions", []),
            "fields": migrate_result.get("current_fields", []),
            "message": f"多维表格「{table_name}」创建成功！",
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
            def _attachment_count(name: str) -> int:
                value = fields.get(name)
                return len(value) if isinstance(value, list) else 0

            records.append({
                "record_id": item.get("record_id"),
                "视频URL": field_to_text(fields.get("视频URL")),
                "视频附件数": _attachment_count("视频附件") or _attachment_count("附件"),
                "广告尾帧": field_to_text(fields.get("广告尾帧")),
                "配音音色": field_to_text(fields.get("配音音色")),
                "引导语": field_to_text(fields.get("引导语")),
                "搜索框图片数": _attachment_count("搜索框图片"),
                "BGM附件数": _attachment_count("BGM"),
                "素材URL": field_to_text(fields.get("素材URL")),
                "BGM音量": fields.get("BGM音量", ""),
                "转场1": field_to_text(fields.get("转场1")),
                "转场2": field_to_text(fields.get("转场2")),
                "处理状态": field_to_text(fields.get("处理状态")),
                "输出视频URL": field_to_text(fields.get("输出视频URL")),
                "预览图URL": field_to_text(fields.get("预览图URL")),
                "错误信息": field_to_text(fields.get("错误信息")),
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
