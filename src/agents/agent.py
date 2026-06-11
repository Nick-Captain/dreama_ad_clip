import os
import json
import logging
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, ToolMessage
from coze_coding_utils.runtime_ctx.context import default_headers
from storage.memory.memory_saver import get_memory_saver

from tools.video_pipeline import process_ad_tail_video, list_available_options
from tools.preview_tool import preview_frame
from tools.bitable_tool import create_bitable_template, get_bitable_records
from tools.batch_tool import batch_process_from_bitable, send_feishu_notification

logger = logging.getLogger(__name__)

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old, new):
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]


@wrap_tool_call
def handle_tool_errors(request, handler):
    """处理工具执行错误，返回友好提示"""
    try:
        return handler(request)
    except Exception as e:
        logger.error(f"工具执行失败 [{request.tool_call.get('name', 'unknown')}]: {str(e)}")
        return ToolMessage(
            content=f"工具执行出错: {str(e)}。请检查参数是否正确，或稍后重试。",
            tool_call_id=request.tool_call["id"],
        )


def build_agent(ctx=None):
    """构建广告尾帧视频处理 Agent"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg["config"].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg["config"].get("temperature", 0.7),
        streaming=True,
        timeout=cfg["config"].get("timeout", 600),
        extra_body={
            "thinking": {
                "type": cfg["config"].get("thinking", "disabled"),
            }
        },
        default_headers=default_headers(ctx) if ctx else {},
    )

    tools = [
        process_ad_tail_video,
        list_available_options,
        preview_frame,
        create_bitable_template,
        get_bitable_records,
        batch_process_from_bitable,
        send_feishu_notification,
    ]

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
        middleware=[handle_tool_errors],
    )
