import argparse
import asyncio
import json
import os
import re
import threading
import traceback
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, AsyncIterable, AsyncGenerator, Optional
import cozeloop
import uvicorn
import time
import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from coze_coding_utils.runtime_ctx.context import new_context, Context
from coze_coding_utils.helper import graph_helper
from coze_coding_utils.log.node_log import LOG_FILE
from coze_coding_utils.log.write_log import setup_logging, request_context
from coze_coding_utils.log.config import LOG_LEVEL
from coze_coding_utils.error.classifier import ErrorClassifier, classify_error
from coze_coding_utils.helper.stream_runner import AgentStreamRunner, WorkflowStreamRunner,agent_stream_handler,workflow_stream_handler, RunOpt
from storage.database.db import get_session, get_engine
from storage.memory.memory_saver import get_memory_saver
from storage.database.shared.model import Base
from coze_coding_utils.async_tasks import (
    AsyncTaskRuntime,
    AsyncTaskStorageError,
    extract_biz_context,
    parse_deadline_sec,
)
from coze_coding_utils.async_tasks import config as async_task_config
from coze_coding_utils.async_tasks.headers import HEADER_X_RUN_ID as _ASYNC_HEADER_X_RUN_ID
from coze_coding_utils.runtime_ctx.context import new_context as _new_async_ctx
from sqlalchemy import event

setup_logging(
    log_file=LOG_FILE,
    max_bytes=100 * 1024 * 1024, # 100MB
    backup_count=5,
    log_level=LOG_LEVEL,
    use_json_format=True,
    console_output=True
)

logger = logging.getLogger(__name__)
from coze_coding_utils.helper.agent_helper import to_stream_input
from coze_coding_utils.openai.handler import OpenAIChatHandler
from coze_coding_utils.log.parser import LangGraphParser
from coze_coding_utils.log.err_trace import extract_core_stack
from coze_coding_utils.log.loop_trace import init_run_config, init_agent_config


# 超时配置常量
TIMEOUT_SECONDS = 900  # 15分钟

class GraphService:
    def __init__(self):
        # 用于跟踪正在运行的任务（使用asyncio.Task）
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # 错误分类器
        self.error_classifier = ErrorClassifier()
        # stream runner
        self._agent_stream_runner = AgentStreamRunner()
        self._workflow_stream_runner = WorkflowStreamRunner()
        self._graph = None
        self._graph_lock = threading.Lock()

    def set_graph(self, graph) -> None:
        """Inject the compiled graph used by sync endpoints. Called once from
        lifespan with a no-checkpointer build, so /run /stream_run /node_run
        never hit the checkpoint DB."""
        self._graph = graph

    def _get_graph(self, ctx=Context):
        if self._graph is not None:
            return self._graph
        with self._graph_lock:
            if self._graph is not None:
                return self._graph
            if graph_helper.is_agent_proj():
                self._graph = graph_helper.get_agent_instance("agents.agent", ctx)
            else:
                self._graph = graph_helper.get_graph_instance("graphs.graph")
            return self._graph

    @staticmethod
    def _sse_event(data: Any, event_id: Any = None) -> str:
        id_line = f"id: {event_id}\n" if event_id else ""
        return f"{id_line}event: message\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def _get_stream_runner(self):
        if graph_helper.is_agent_proj():
            return self._agent_stream_runner
        else:
            return self._workflow_stream_runner

    # 流式运行（原始迭代器）：本地调用使用
    def stream(self, payload: Dict[str, Any], run_config: RunnableConfig, ctx=Context) -> Iterable[Any]:
        graph = self._get_graph(ctx)
        stream_runner = self._get_stream_runner()
        for chunk in stream_runner.stream(payload, graph, run_config, ctx):
            yield chunk

    # 同步运行：本地/HTTP 通用
    async def run(self, payload: Dict[str, Any], ctx=None) -> Dict[str, Any]:
        if ctx is None:
            ctx = new_context("run")

        run_id = ctx.run_id
        logger.info(f"Starting run with run_id: {run_id}")

        try:
            graph = self._get_graph(ctx)
            # custom tracer
            run_config = init_run_config(graph, ctx)
            run_config.setdefault("configurable", {})["thread_id"] = ctx.run_id

            # 直接调用，LangGraph会在当前任务上下文中执行
            # 如果当前任务被取消，LangGraph的执行也会被取消
            return await graph.ainvoke(payload, config=run_config, context=ctx)

        except asyncio.CancelledError:
            logger.info(f"Run {run_id} was cancelled")
            return {"status": "cancelled", "run_id": run_id, "message": "Execution was cancelled"}
        except Exception as e:
            # 使用错误分类器分类错误
            err = self.error_classifier.classify(e, {"node_name": "run", "run_id": run_id})
            # 记录详细的错误信息和堆栈跟踪
            logger.error(
                f"Error in GraphService.run: [{err.code}] {err.message}\n"
                f"Category: {err.category.name}\n"
                f"Traceback:\n{extract_core_stack()}"
            )
            # 保留原始异常堆栈，便于上层返回真正的报错位置
            raise
        finally:
            # 清理任务记录
            self.running_tasks.pop(run_id, None)

    # 流式运行（SSE 格式化）：HTTP 路由使用
    async def stream_sse(self, payload: Dict[str, Any], ctx=None, run_opt: Optional[RunOpt] = None) -> AsyncGenerator[str, None]:
        if ctx is None:
            ctx = new_context(method="stream_sse")
        if run_opt is None:
            run_opt = RunOpt()

        run_id = ctx.run_id
        logger.info(f"Starting stream with run_id: {run_id}")
        graph = self._get_graph(ctx)
        if graph_helper.is_agent_proj():
            run_config = init_agent_config(graph, ctx)
        else:
            run_config = init_run_config(graph, ctx)  # vibeflow

        is_workflow = not graph_helper.is_agent_proj()

        try:
            async for chunk in self.astream(payload, graph, run_config=run_config, ctx=ctx, run_opt=run_opt):
                if is_workflow and isinstance(chunk, tuple):
                    event_id, data = chunk
                    yield self._sse_event(data, event_id)
                else:
                    yield self._sse_event(chunk)
        finally:
            # 清理任务记录
            self.running_tasks.pop(run_id, None)
            cozeloop.flush()

    # 取消执行 - 使用asyncio的标准方式
    def cancel_run(self, run_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """
        取消指定run_id的执行

        使用asyncio.Task.cancel()来取消任务,这是标准的Python异步取消机制。
        LangGraph会在节点之间检查CancelledError,实现优雅的取消。
        """
        logger.info(f"Attempting to cancel run_id: {run_id}")

        # 查找对应的任务
        if run_id in self.running_tasks:
            task = self.running_tasks[run_id]
            if not task.done():
                # 使用asyncio的标准取消机制
                # 这会在下一个await点抛出CancelledError
                task.cancel()
                logger.info(f"Cancellation requested for run_id: {run_id}")
                return {
                    "status": "success",
                    "run_id": run_id,
                    "message": "Cancellation signal sent, task will be cancelled at next await point"
                }
            else:
                logger.info(f"Task already completed for run_id: {run_id}")
                return {
                    "status": "already_completed",
                    "run_id": run_id,
                    "message": "Task has already completed"
                }
        else:
            logger.warning(f"No active task found for run_id: {run_id}")
            return {
                "status": "not_found",
                "run_id": run_id,
                "message": "No active task found with this run_id. Task may have already completed or run_id is invalid."
            }

    # 运行指定节点：本地/HTTP 通用
    async def run_node(self, node_id: str, payload: Dict[str, Any], ctx=None) -> Any:
        if ctx is None or Context.run_id == "":
            ctx = new_context(method="node_run")

        _graph = self._get_graph()
        node_func, input_cls, output_cls = graph_helper.get_graph_node_func_with_inout(_graph.get_graph(), node_id)
        if node_func is None or input_cls is None:
            raise KeyError(f"node_id '{node_id}' not found")

        parser = LangGraphParser(_graph)
        metadata = parser.get_node_metadata(node_id) or {}

        _g = StateGraph(input_cls, input_schema=input_cls, output_schema=output_cls)
        _g.add_node("sn", node_func, metadata=metadata)
        _g.set_entry_point("sn")
        _g.add_edge("sn", END)
        _graph = _g.compile()

        run_config = init_run_config(_graph, ctx)
        return await _graph.ainvoke(payload, config=run_config)

    def graph_inout_schema(self) -> Any:
        if graph_helper.is_agent_proj():
            return {"input_schema": {}, "output_schema": {}}
        builder = getattr(self._get_graph(), 'builder', None)
        if builder is not None:
            input_cls = getattr(builder, 'input_schema', None) or self.graph.get_input_schema()
            output_cls = getattr(builder, 'output_schema', None) or self.graph.get_output_schema()
        else:
            logger.warning(f"No builder input schema found for graph_inout_schema, using graph input schema instead")
            input_cls = self.graph.get_input_schema()
            output_cls = self.graph.get_output_schema()

        return {
            "input_schema": input_cls.model_json_schema(), 
            "output_schema": output_cls.model_json_schema(),
            "code":0,
            "msg":""
        }

    async def astream(self, payload: Dict[str, Any], graph: CompiledStateGraph, run_config: RunnableConfig, ctx=Context, run_opt: Optional[RunOpt] = None) -> AsyncIterable[Any]:
        stream_runner = self._get_stream_runner()
        async for chunk in stream_runner.astream(payload, graph, run_config, ctx, run_opt):
            yield chunk


service = GraphService()

async_runtime: Optional[AsyncTaskRuntime] = None
async_graph: Optional[CompiledStateGraph] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    @event.listens_for(engine, "connect")
    def _set_utc(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")
    try:
        # H5 编辑器的全局默认/素材库表，幂等建表
        Base.metadata.create_all(engine)
    except Exception as e:
        logger.warning(f"H5 数据表初始化失败（不影响主流程）: {e}")
    checkpointer = get_memory_saver()
    if graph_helper.is_agent_proj():
        base = graph_helper.get_agent_instance("agents.agent", None)
        sync_graph = base.builder.compile(checkpointer=checkpointer)
    else:
        base = graph_helper.get_graph_instance("graphs.graph")
        sync_graph = base.builder.compile()
    global async_graph, async_runtime
    async_graph = base.builder.compile(checkpointer=checkpointer)
    service.set_graph(sync_graph)
    async_runtime = AsyncTaskRuntime(
        session_factory=get_session, engine=engine,
        graph=async_graph, checkpointer=checkpointer,
    )
    yield
    if async_runtime is not None:
        await async_runtime.shutdown()

app = FastAPI(lifespan=lifespan)

# OpenAI 兼容接口处理器
openai_handler = OpenAIChatHandler(service)


@app.post("/async_run")
async def http_async_run(request: Request) -> dict:
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_async_run: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {extract_core_stack()}")
    try:
        deadline_sec = parse_deadline_sec(request.headers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 一个 ID 走到底：task_id == run_id == thread_id == ctx.run_id == coze_run_id。
    # 优先用上游 x-run-id；没传就生成 UUID。
    run_id = request.headers.get(_ASYNC_HEADER_X_RUN_ID) or uuid.uuid4().hex

    # ctx 在 handler scope 构造，与同步 /run 路径一致；后面 new_context 默认会
    # 给 run_id 一个新 UUID，同步路径也是显式覆盖（main.py /run 处），这里同理。
    ctx = _new_async_ctx(method="async_run", headers=request.headers)
    ctx.run_id = run_id
    request_context.set(ctx)  # 与其他 HTTP endpoint 一致：让日志组件拿到 run_id 等信息
    run_config = init_run_config(async_graph, ctx)
    run_config["recursion_limit"] = async_task_config.RECURSION_LIMIT
    run_config.setdefault("configurable", {})["thread_id"] = run_id

    biz_context = extract_biz_context(request.headers) or {}
    biz_context[_ASYNC_HEADER_X_RUN_ID] = run_id  # 也留 DB 一份方便审计/排查

    try:
        return await async_runtime.submit(
            task_id=run_id,
            payload=payload,
            biz_context=biz_context,
            deadline_sec=deadline_sec,
            run_config=run_config,
            ctx=ctx,
        )
    except AsyncTaskStorageError as e:
        raise HTTPException(status_code=503,
                            detail=f"async-task storage unavailable: {e}")


@app.get("/task/{task_id}")
async def http_get_task(task_id: str) -> dict:
    try:
        row = await async_runtime.get(task_id)
    except AsyncTaskStorageError as e:
        raise HTTPException(status_code=503,
                            detail=f"async-task storage unavailable: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return row


HEADER_X_RUN_ID = "x-run-id"
@app.post("/run")
async def http_run(request: Request) -> Dict[str, Any]:
    global result
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except Exception as e:
        body_text = str(raw_body)
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON format: {body_text}, traceback: {traceback.format_exc()}, error: {e}")

    ctx = new_context(method="run", headers=request.headers)
    # 优先使用上游指定的 run_id，保证 cancel 能精确匹配
    upstream_run_id = request.headers.get(HEADER_X_RUN_ID)
    if upstream_run_id:
        ctx.run_id = upstream_run_id
    run_id = ctx.run_id
    request_context.set(ctx)

    logger.info(
        f"Received request for /run: "
        f"run_id={run_id}, "
        f"query={dict(request.query_params)}, "
        f"body={body_text}"
    )

    try:
        payload = await request.json()

        # 创建任务并记录 - 这是关键，让我们可以通过run_id取消任务
        task = asyncio.create_task(service.run(payload, ctx))
        service.running_tasks[run_id] = task

        try:
            result = await asyncio.wait_for(task, timeout=float(TIMEOUT_SECONDS))
        except asyncio.TimeoutError:
            logger.error(f"Run execution timeout after {TIMEOUT_SECONDS}s for run_id: {run_id}")
            task.cancel()
            try:
                result = await task
            except asyncio.CancelledError:
                return {
                    "status": "timeout",
                    "run_id": run_id,
                    "message": f"Execution timeout: exceeded {TIMEOUT_SECONDS} seconds"
                }

        if not result:
            result = {}
        if isinstance(result, dict):
            result["run_id"] = run_id
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format, {extract_core_stack()}")

    except asyncio.CancelledError:
        logger.info(f"Request cancelled for run_id: {run_id}")
        result = {"status": "cancelled", "run_id": run_id, "message": "Execution was cancelled"}
        return result

    except Exception as e:
        # 使用错误分类器获取错误信息
        error_response = service.error_classifier.get_error_response(e, {"node_name": "http_run", "run_id": run_id})
        logger.error(
            f"Unexpected error in http_run: [{error_response['error_code']}] {error_response['error_message']}, "
            f"traceback: {traceback.format_exc()}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": error_response["error_code"],
                "error_message": error_response["error_message"],
                "stack_trace": extract_core_stack(),
            }
        )
    finally:
        cozeloop.flush()


HEADER_X_WORKFLOW_STREAM_MODE = "x-workflow-stream-mode"


def _register_task(run_id: str, task: asyncio.Task):
    service.running_tasks[run_id] = task


@app.post("/stream_run")
async def http_stream_run(request: Request):
    ctx = new_context(method="stream_run", headers=request.headers)
    # 优先使用上游指定的 run_id，保证 cancel 能精确匹配
    upstream_run_id = request.headers.get(HEADER_X_RUN_ID)
    if upstream_run_id:
        ctx.run_id = upstream_run_id
    workflow_stream_mode = request.headers.get(HEADER_X_WORKFLOW_STREAM_MODE, "").lower()
    workflow_debug = workflow_stream_mode == "debug"
    request_context.set(ctx)
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except Exception as e:
        body_text = str(raw_body)
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON format: {body_text}, traceback: {extract_core_stack()}, error: {e}")
    run_id = ctx.run_id
    is_agent = graph_helper.is_agent_proj()
    logger.info(
        f"Received request for /stream_run: "
        f"run_id={run_id}, "
        f"is_agent_project={is_agent}, "
        f"query={dict(request.query_params)}, "
        f"body={body_text}"
    )
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_stream_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{extract_core_stack()}")

    if is_agent:
        stream_generator = agent_stream_handler(
            payload=payload,
            ctx=ctx,
            run_id=run_id,
            stream_sse_func=service.stream_sse,
            sse_event_func=service._sse_event,
            error_classifier=service.error_classifier,
            register_task_func=_register_task,
        )
    else:
        stream_generator = workflow_stream_handler(
            payload=payload,
            ctx=ctx,
            run_id=run_id,
            stream_sse_func=service.stream_sse,
            sse_event_func=service._sse_event,
            error_classifier=service.error_classifier,
            register_task_func=_register_task,
            run_opt=RunOpt(workflow_debug=workflow_debug),
        )

    response = StreamingResponse(stream_generator, media_type="text/event-stream")
    return response

@app.post("/cancel/{run_id}")
async def http_cancel(run_id: str, request: Request):
    """
    取消指定run_id的执行

    使用asyncio.Task.cancel()实现取消,这是Python标准的异步任务取消机制。
    LangGraph会在节点之间的await点检查CancelledError,实现优雅取消。
    """
    ctx = new_context(method="cancel", headers=request.headers)
    request_context.set(ctx)
    logger.info(f"Received cancel request for run_id: {run_id}")
    result = service.cancel_run(run_id, ctx)
    return result


@app.post(path="/node_run/{node_id}")
async def http_node_run(node_id: str, request: Request):
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        body_text = str(raw_body)
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {body_text}")
    ctx = new_context(method="node_run", headers=request.headers)
    request_context.set(ctx)
    logger.info(
        f"Received request for /node_run/{node_id}: "
        f"query={dict(request.query_params)}, "
        f"body={body_text}",
    )

    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in http_node_run: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format:{extract_core_stack()}")
    try:
        return await service.run_node(node_id, payload, ctx)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"node_id '{node_id}' not found or input miss required fields, traceback: {extract_core_stack()}")
    except Exception as e:
        # 使用错误分类器获取错误信息
        error_response = service.error_classifier.get_error_response(e, {"node_name": node_id})
        logger.error(
            f"Unexpected error in http_node_run: [{error_response['error_code']}] {error_response['error_message']}, "
            f"traceback: {traceback.format_exc()}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": error_response["error_code"],
                "error_message": error_response["error_message"],
                "stack_trace": extract_core_stack(),
            }
        )
    finally:
        cozeloop.flush()


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI Chat Completions API 兼容接口"""
    ctx = new_context(method="openai_chat", headers=request.headers)
    request_context.set(ctx)

    logger.info(f"Received request for /v1/chat/completions: run_id={ctx.run_id}")

    try:
        payload = await request.json()
        return await openai_handler.handle(payload, ctx)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in openai_chat_completions: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    finally:
        cozeloop.flush()


@app.get("/health")
async def health_check():
    try:
        # 这里可以添加更多的健康检查逻辑
        return {
            "status": "ok",
            "message": "Service is running",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(path="/graph_parameter")
async def http_graph_inout_parameter(request: Request):
    return service.graph_inout_schema()

# ============================================================
# 广告尾帧视频处理 REST API 端点
# ============================================================

@app.post("/api/v1/preview-frame")
async def api_preview_frame(request: Request):
    """生成定格帧预览图"""
    ctx = new_context(method="api_preview", headers=request.headers)
    request_context.set(ctx)
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    from tools.preview_tool import preview_frame
    result = preview_frame.invoke(payload)
    return json.loads(result)


@app.post("/api/v1/process-video")
async def api_process_video(request: Request):
    """单条视频处理"""
    ctx = new_context(method="api_process", headers=request.headers)
    request_context.set(ctx)
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    from tools.video_pipeline import process_ad_tail_video
    result = process_ad_tail_video.invoke(payload)
    return json.loads(result)


@app.post("/api/v1/batch-process")
async def api_batch_process(request: Request):
    """批量处理（从多维表格读取）"""
    ctx = new_context(method="api_batch", headers=request.headers)
    request_context.set(ctx)
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    from tools.batch_tool import batch_process_from_bitable
    result = batch_process_from_bitable.invoke(payload)
    return json.loads(result)


@app.post("/api/v1/create-bitable")
async def api_create_bitable(request: Request):
    """创建多维表格模板"""
    ctx = new_context(method="api_create_bitable", headers=request.headers)
    request_context.set(ctx)
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    from tools.bitable_tool import create_bitable_template
    result = create_bitable_template.invoke(payload)
    return json.loads(result)


@app.post("/api/v1/migrate-bitable")
async def api_migrate_bitable(request: Request):
    """将已有多维表格结构对齐到当前模板（清理冗余列、引导语改下拉、补齐新列）"""
    ctx = new_context(method="api_migrate_bitable", headers=request.headers)
    request_context.set(ctx)
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    app_token = payload.get("app_token", "")
    table_id = payload.get("table_id", "")
    if not app_token or not table_id:
        raise HTTPException(status_code=400, detail="app_token 和 table_id 必填")

    from tools.bitable_tool import migrate_bitable_schema
    return await asyncio.to_thread(migrate_bitable_schema, app_token, table_id)


@app.post("/api/v1/debug-table")
async def api_debug_table(request: Request):
    """临时诊断：返回表格记录的原始字段值（排查转场链路，定位后移除）"""
    ctx = new_context(method="api_debug_table", headers=request.headers)
    request_context.set(ctx)
    payload = await request.json()

    def _read():
        from tools.bitable_tool import BitableClient
        client = BitableClient()
        resp = client.search_records(
            app_token=payload.get("app_token", ""),
            table_id=payload.get("table_id", ""),
        )
        items = resp.get("data", {}).get("items", [])
        return {
            "count": len(items),
            "records": [
                {
                    "record_id": item.get("record_id"),
                    "fields": item.get("fields", {}),
                }
                for item in items
            ],
        }

    return await asyncio.to_thread(_read)


@app.post("/api/v1/debug-concat")
async def api_debug_concat(request: Request):
    """临时诊断：直接调用云端视频拼接，验证 transitions 参数行为（定位后移除）"""
    ctx = new_context(method="api_debug_concat", headers=request.headers)
    request_context.set(ctx)
    payload = await request.json()
    videos = payload.get("videos") or []
    transitions = payload.get("transitions") or None
    if len(videos) < 2:
        raise HTTPException(status_code=400, detail="至少提供2个视频URL")

    def _concat():
        from coze_coding_dev_sdk.video_edit import VideoEditClient
        client = VideoEditClient(ctx=request_context.get())
        resp = client.concat_videos(videos=videos, transitions=transitions)
        return {"url": resp.url, "videos_count": len(videos), "transitions": transitions}

    return await asyncio.to_thread(_concat)


@app.get("/api/v1/options")
async def api_list_options():
    """列出所有可用选项"""
    from tools.video_pipeline import list_available_options
    result = list_available_options.invoke({})
    return json.loads(result)


# ============================================================
# Git 自动同步端点（GitHub Webhook）
# ============================================================

@app.post("/api/v1/git-sync")
async def api_git_sync(request: Request):
    """GitHub Webhook：收到 push 事件后把代码强制同步到 origin/main。

    GitHub 是唯一真相源，Coze 侧不保留本地修改（在 Coze IDE 里直接改的代码
    会在下次同步时被丢弃）。用 fetch + reset --hard 而非 pull，
    避免 Coze 侧历史分叉导致 divergent branches 同步失败。
    """
    import subprocess
    repo_dir = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    logger.info("[git-sync] 收到 webhook 请求，开始同步 origin/main ...")

    def _sync() -> dict:
        steps = []
        for cmd in (
            ["git", "-C", repo_dir, "fetch", "origin", "main"],
            ["git", "-C", repo_dir, "reset", "--hard", "origin/main"],
        ):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            step = {
                "cmd": " ".join(cmd[3:]),
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            steps.append(step)
            logger.info(f"[git-sync] {step['cmd']}: rc={result.returncode}, "
                        f"stdout={step['stdout']}, stderr={step['stderr']}")
            if result.returncode != 0:
                return {"status": "error", "steps": steps}
        return {"status": "ok", "steps": steps}

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error(f"[git-sync] 同步失败: {str(e)}")
        return {"status": "error", "message": str(e)}


# ============================================================
# 飞书事件回调端点（接收 @机器人 消息）
# ============================================================

# 凭据从配置文件读取（Coze 部署环境变量注入不可靠，改用文件方案）
# config/feishu_secrets.json 已加入 .gitignore，不会被推送到公开仓库

_FEISHU_SECRETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "feishu_secrets.json")


def _load_feishu_secrets() -> dict:
    """加载飞书凭据配置文件。"""
    try:
        with open(_FEISHU_SECRETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _get_feishu_config(key: str, default: str = "") -> str:
    """读取飞书配置项，优先环境变量，其次配置文件。"""
    # 先尝试环境变量（如果 Coze 后续修复了注入问题）
    val = os.getenv(key, "")
    if val:
        return val
    # 兜底：从配置文件读取
    secrets = _load_feishu_secrets()
    return secrets.get(key, default)


FEISHU_APP_ID = _get_feishu_config("FEISHU_APP_ID", "cli_a92bc422db799bd2")
FEISHU_VERIFICATION_TOKEN = _get_feishu_config("FEISHU_VERIFICATION_TOKEN", "")


def _get_feishu_app_secret() -> str:
    """读取飞书 App Secret（优先环境变量，其次配置文件）。"""
    return _get_feishu_config("FEISHU_APP_SECRET", "")


# 已处理事件去重表：飞书在回调响应慢/失败时会重试推送同一事件，不去重会导致视频被重复处理
_PROCESSED_EVENTS: Dict[str, float] = {}
_EVENT_DEDUP_TTL = 600  # 秒
_event_dedup_lock = threading.Lock()


def _is_duplicate_event(event_id: str) -> bool:
    """记录并判断事件是否已处理过（带 TTL 的内存去重，单 worker 足够）"""
    if not event_id:
        return False
    now = time.time()
    with _event_dedup_lock:
        for eid, ts in list(_PROCESSED_EVENTS.items()):
            if now - ts > _EVENT_DEDUP_TTL:
                _PROCESSED_EVENTS.pop(eid, None)
        if event_id in _PROCESSED_EVENTS:
            return True
        _PROCESSED_EVENTS[event_id] = now
        return False


def _get_tenant_access_token() -> str:
    """获取飞书 tenant_access_token"""
    secret = _get_feishu_app_secret()
    if not secret:
        raise Exception(
            "FEISHU_APP_SECRET 未配置：请设置环境变量，"
            "或在扣子沙箱创建 config/feishu_secrets.json（该文件不入库，沙箱重建后需重建）"
        )
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": FEISHU_APP_ID, "app_secret": secret},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"获取 tenant_access_token 失败: {data}")
    return data["tenant_access_token"]


def _reply_feishu_message(message_id: str, content: str) -> dict:
    """回复飞书消息"""
    token = _get_tenant_access_token()
    resp = requests.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json={
            "content": json.dumps({"text": content}),
            "msg_type": "text",
        },
        timeout=10,
    )
    return resp.json()


def _extract_text_from_event(event: dict) -> str:
    """从飞书事件中提取用户发送的文本"""
    try:
        # 消息内容在 event.message.content 中，是 JSON 字符串
        content_str = event.get("message", {}).get("content", "{}")
        content = json.loads(content_str)
        return content.get("text", "")
    except (json.JSONDecodeError, KeyError):
        return ""


# 默认绑定的多维表格：零参数「批量处理」直接用它，可被环境变量覆盖
DEFAULT_BITABLE_APP_TOKEN = os.getenv("DEFAULT_BITABLE_APP_TOKEN", "JOMibWw3wa6TzYsaHSIcAG27n2f")
DEFAULT_BITABLE_TABLE_ID = os.getenv("DEFAULT_BITABLE_TABLE_ID", "tblWNUywhvrkJ54u")


def _parse_batch_command(text: str) -> dict | None:
    """
    解析批量处理命令，按优先级支持三种形式：
    1. 显式参数：批量处理 app_token=xxx table_id=xxx
    2. 表格分享链接：批量处理 https://xxx.feishu.cn/base/<app_token>?table=<table_id>
    3. 零参数：「批量处理」→ 使用默认绑定的表格
    """
    app_token_match = re.search(r"app_token[=:：]\s*([a-zA-Z0-9_]+)", text)
    table_id_match = re.search(r"table_id[=:：]\s*([a-zA-Z0-9_-]+)", text)
    if app_token_match and table_id_match:
        return {
            "app_token": app_token_match.group(1),
            "table_id": table_id_match.group(1),
        }

    url_match = re.search(r"/base/([a-zA-Z0-9]+)\?\S*?table=(tbl[a-zA-Z0-9]+)", text)
    if url_match:
        return {
            "app_token": url_match.group(1),
            "table_id": url_match.group(2),
        }

    if "批量处理" in text:
        return {
            "app_token": DEFAULT_BITABLE_APP_TOKEN,
            "table_id": DEFAULT_BITABLE_TABLE_ID,
        }
    return None


def _parse_create_bitable_command(text: str) -> str | None:
    """解析创建表格命令，返回表格名称"""
    patterns = [
        r"创建.*表格[：:\s]*[「「]?(.+?)[」」]?$",
        r"新建.*表格[：:\s]*[「「]?(.+?)[」」]?$",
        r"帮我创建.*[「「](.+?)[」」]",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    # 如果只说了"创建表格"没有指定名称，用默认名
    if any(kw in text for kw in ["创建表格", "新建表格", "创建多维表格", "新建多维表格"]):
        return "广告尾帧批量处理"
    return None


def _parse_single_process_command(text: str) -> dict | None:
    """解析单条视频处理命令"""
    video_url_match = re.search(r"视频[URL链接地址:：]*\s*(https?://\S+)", text)
    if video_url_match:
        params = {"video_url": video_url_match.group(1)}

        tail_match = re.search(r"尾帧[：:]*\s*(派对接引尾帧|短剧推广尾帧|自定义)", text)
        if tail_match:
            params["tail_name"] = tail_match.group(1)

        voice_match = re.search(r"音色[：:]*\s*(小荷|米仔|大奕|可爱女生)", text)
        if voice_match:
            params["voice_name"] = voice_match.group(1)

        guide_match = re.search(r"引导语[：:]*\s*(.+?)(?:字幕|转场|搜索框|BGM|音色|尾帧|$)", text)
        if guide_match:
            params["guide_text"] = guide_match.group(1).strip()

        return params
    return None


@app.post("/api/v1/feishu/event")
async def feishu_event_callback(request: Request):
    """
    飞书事件订阅回调端点。

    接收飞书推送的消息事件，解析用户指令并执行相应操作：
    - @机器人 创建表格 → 自动创建多维表格模板
    - @机器人 批量处理 app_token=xxx table_id=xxx → 启动批量处理
    - @机器人 处理视频 URL → 单条视频处理
    - @机器人 预览 → 生成预览图
    - @机器人 帮助 → 返回使用说明
    """
    ctx = new_context(method="feishu_event", headers=request.headers)
    request_context.set(ctx)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"code": 400, "msg": "Invalid JSON"}, status_code=400)

    logger.info(f"[飞书回调] 收到事件: {json.dumps(body, ensure_ascii=False)[:500]}")

    # 来源校验：配置了 FEISHU_VERIFICATION_TOKEN 时，拒绝 token 不匹配的伪造请求
    if FEISHU_VERIFICATION_TOKEN:
        req_token = body.get("token") or body.get("header", {}).get("token", "")
        if req_token != FEISHU_VERIFICATION_TOKEN:
            logger.warning("[飞书回调] Verification Token 不匹配，已拒绝")
            return JSONResponse({"code": 403, "msg": "invalid token"}, status_code=403)

    # 飞书 URL 验证（首次配置事件订阅时）
    challenge = body.get("challenge")
    if challenge:
        return JSONResponse({"challenge": challenge})

    # 处理消息事件
    event = body.get("event", {})
    header = body.get("header", {})
    event_type = header.get("event_type", "")

    if event_type != "im.message.receive_v1":
        return JSONResponse({"code": 0, "msg": "ok"})

    # 事件去重：飞书超时重试会重复推送同一 event_id
    if _is_duplicate_event(header.get("event_id", "")):
        logger.info(f"[飞书回调] 重复事件，忽略: {header.get('event_id', '')}")
        return JSONResponse({"code": 0, "msg": "ok"})

    message_id = event.get("message", {}).get("message_id", "")
    chat_type = event.get("message", {}).get("chat_type", "")

    # 提取文本（去除 @ 机器人部分）
    raw_text = _extract_text_from_event(event)
    # 去掉 @ 机器人的 at 标记
    text = re.sub(r"@_user_\d+\s*", "", raw_text).strip()
    # 也去掉 @所有人 等
    text = re.sub(r"@\S+\s*", "", text).strip()

    logger.info(f"[飞书回调] 用户消息: raw={raw_text[:200]}, cleaned={text[:200]}")

    if not text:
        # 空消息（含直接发文件/视频等非文本消息），返回帮助
        asyncio.create_task(asyncio.to_thread(_reply_feishu_message, message_id, _get_help_text()))
        return JSONResponse({"code": 0, "msg": "ok"})

    # 后台线程处理（先回 200，再处理业务）。
    # 必须放线程：视频处理是分钟级阻塞调用，放在事件循环里会卡死整个服务，
    # 飞书等不到响应会重试推送，导致同一视频被重复处理。
    def handle_message():
        try:
            # 1. 创建表格
            table_name = _parse_create_bitable_command(text)
            if table_name:
                logger.info(f"[飞书回调] 创建表格: {table_name}")
                from tools.bitable_tool import create_bitable_template
                result_str = create_bitable_template.invoke({"table_name": table_name})
                result = json.loads(result_str)
                if result.get("success"):
                    reply = (
                        f"✅ 多维表格「{table_name}」创建成功！\n\n"
                        f"📋 app_token: {result['app_token']}\n"
                        f"📋 table_id: {result['table_id']}\n\n"
                        f"请在表格中填入视频信息，然后 @我 说「批量处理 app_token={result['app_token']} table_id={result['table_id']}」即可开始。"
                    )
                else:
                    reply = f"❌ 创建失败：{result.get('error', '未知错误')}"
                _reply_feishu_message(message_id, reply)
                return

            # 2. 批量处理
            batch_params = _parse_batch_command(text)
            if batch_params:
                logger.info(f"[飞书回调] 批量处理: {batch_params}")
                _reply_feishu_message(message_id, f"⏳ 开始批量处理，请稍候...\n\napp_token: {batch_params['app_token']}\ntable_id: {batch_params['table_id']}")

                from tools.batch_tool import batch_process_from_bitable
                result_str = batch_process_from_bitable.invoke({
                    "app_token": batch_params["app_token"],
                    "table_id": batch_params["table_id"],
                    "max_concurrency": 3,
                    "send_notification": True,
                })
                result = json.loads(result_str)
                if result.get("success"):
                    s = result.get("summary", {})
                    reply = (
                        f"✅ 批量处理完成！\n\n"
                        f"📊 总计：{s.get('total', 0)} 条\n"
                        f"✅ 成功：{s.get('success', 0)} 条\n"
                        f"❌ 失败：{s.get('failed', 0)} 条\n\n"
                        f"请查看多维表格获取详细结果。"
                    )
                else:
                    reply = f"❌ 批量处理失败：{result.get('error', '未知错误')}"
                _reply_feishu_message(message_id, reply)
                return

            # 3. 单条视频处理
            single_params = _parse_single_process_command(text)
            if single_params:
                logger.info(f"[飞书回调] 单条处理: {single_params}")
                _reply_feishu_message(message_id, "⏳ 正在处理视频，请稍候...")

                from tools.video_pipeline import process_ad_tail_video
                result_str = process_ad_tail_video.invoke(single_params)
                result = json.loads(result_str)
                if result.get("success"):
                    reply = f"✅ 视频处理完成！\n\n📹 输出视频：{result.get('final_video_url', '')}"
                else:
                    reply = f"❌ 处理失败：{result.get('error', '未知错误')}"
                _reply_feishu_message(message_id, reply)
                return

            # 4. 预览
            if "预览" in text:
                video_url_match = re.search(r"(https?://\S+)", text)
                if video_url_match:
                    from tools.preview_tool import preview_frame
                    params = {"video_url": video_url_match.group(1)}
                    # 检查是否有自定义字幕
                    subtitle_match = re.search(r"字幕[：:]*\s*(.+?)(?:搜索框|位置|颜色|字体|$)", text)
                    if subtitle_match:
                        params["subtitle_text"] = subtitle_match.group(1).strip()
                    result_str = preview_frame.invoke(params)
                    result = json.loads(result_str)
                    if result.get("success"):
                        reply = f"✅ 预览图生成成功！\n\n🖼️ {result.get('preview_url', '')}"
                    else:
                        reply = f"❌ 预览失败：{result.get('error', '未知错误')}"
                else:
                    reply = "请提供视频URL，例如：预览 https://example.com/video.mp4"
                _reply_feishu_message(message_id, reply)
                return

            # 5. 帮助 / 默认
            if any(kw in text for kw in ["帮助", "help", "怎么用", "使用说明"]):
                _reply_feishu_message(message_id, _get_help_text())
                return

            # 无法识别
            _reply_feishu_message(message_id, f"抱歉，我没有理解你的指令。\n\n{_get_help_text()}")

        except Exception as e:
            logger.error(f"[飞书回调] 处理消息异常: {e}", exc_info=True)
            try:
                _reply_feishu_message(message_id, f"❌ 处理出错：{str(e)[:200]}")
            except Exception:
                pass

    asyncio.create_task(asyncio.to_thread(handle_message))
    return JSONResponse({"code": 0, "msg": "ok"})


def _get_help_text() -> str:
    return (
        "🎬 **广告尾帧视频处理机器人**\n\n"
        "支持以下指令：\n\n"
        "1️⃣ **创建表格**\n"
        "「创建表格」或「帮我创建「广告尾帧批量处理」」\n\n"
        "2️⃣ **批量处理**\n"
        "「批量处理」→ 处理默认表格中所有「待处理」记录\n"
        "也可以带上表格链接或 app_token=xxx table_id=xxx 指定其他表格\n\n"
        "3️⃣ **单条处理**\n"
        "「处理视频 https://xxx.mp4 尾帧：派对接引尾帧 音色：可爱女生」\n\n"
        "4️⃣ **预览**\n"
        "「预览 https://xxx.mp4 字幕：后续剧情该如何选择」\n\n"
        "5️⃣ **帮助**\n"
        "「帮助」或「怎么用」"
    )


# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Start FastAPI server")
    parser.add_argument("-m", type=str, default="http", help="Run mode, support http,flow,node")
    parser.add_argument("-n", type=str, default="", help="Node ID for single node run")
    parser.add_argument("-p", type=int, default=5000, help="HTTP server port")
    parser.add_argument("-i", type=str, default="", help="Input JSON string for flow/node mode")
    return parser.parse_args()


def parse_input(input_str: str) -> Dict[str, Any]:
    """Parse input string, support both JSON string and plain text"""
    if not input_str:
        return {"text": "你好"}

    # Try to parse as JSON first
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        # If not valid JSON, treat as plain text
        return {"text": input_str}

def start_http_server(port):
    workers = 1
    reload = False
    if graph_helper.is_dev_env():
        reload = True

    logger.info(f"Start HTTP Server, Port: {port}, Workers: {workers}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, workers=workers)

if __name__ == "__main__":
    args = parse_args()
    if args.m == "http":
        start_http_server(args.p)
    elif args.m == "flow":
        payload = parse_input(args.i)
        result = asyncio.run(service.run(payload))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.m == "node" and args.n:
        payload = parse_input(args.i)
        result = asyncio.run(service.run_node(args.n, payload))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.m == "agent":
        agent_ctx = new_context(method="agent")
        for chunk in service.stream(
                {
                    "type": "query",
                    "session_id": "1",
                    "message": "你好",
                    "content": {
                        "query": {
                            "prompt": [
                                {
                                    "type": "text",
                                    "content": {"text": "现在几点了？请调用工具获取当前时间"},
                                }
                            ]
                        }
                    },
                },
                run_config={"configurable": {"session_id": "1"}},
                ctx=agent_ctx,
        ):
            print(chunk)
