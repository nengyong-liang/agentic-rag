"""聊天/问答端点"""
import logging
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agentic_rag.api.state import state
from agentic_rag.rag.llm_router import llm_router
from agentic_rag.rag.retriever import hybrid_retriever
from agentic_rag.rag.query_rewriter import query_rewriter
from agentic_rag.utils.sensitive_filter import sensitive_filter
from agentic_rag.utils.tenant_filter import tenant_permission_filter, TenantContext
from agentic_rag.execution.dispatcher import dispatch

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    history: Optional[List[Dict]] = None
    top_k: int = 5
    route: Optional[str] = "rag"
    domain: Optional[str] = "general"
    use_finetuned: Optional[bool] = None
    use_agentic_rag: Optional[bool] = None
    user_id: str = "default"
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]
    usage: Dict
    chain_log: Optional[List[Dict]] = None
    expansion_info: Optional[Dict] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能问答"""
    try:
        logger.info(f"收到问答请求：{request.query}")
        logger.info(f"路由类型：{request.route}，领域：{request.domain}，使用微调：{request.use_finetuned}")

        # 敏感词检查
        sensitive_result = sensitive_filter.check_sensitive(request.query)
        if sensitive_result['is_sensitive']:
            logger.warning(f"检测到敏感词：{sensitive_result['sensitive_words']}")
            return ChatResponse(
                answer="您的问题包含敏感内容，请修改后重新提问",
                citations=[],
                usage={}
            )

        # 使用 Java 后端传来的路由判断（如果有），统一经调度网关分派
        route = request.route if request.route else "rag"

        # 记忆系统：按 user_id 取（或创建）跨会话记忆管理器
        # 短期记忆按 session_id 隔离（本次对话独立窗口），长期记忆按 user_id 共享（跨对话互通）
        memory_manager = state.get_memory_manager(request.user_id, request.session_id)
        augmented_history = request.history
        if memory_manager is not None:
            try:
                memory_context = memory_manager.get_context_for_prompt(
                    request.query,
                    max_tokens=500,
                    user_id=request.user_id,
                    session_id=request.session_id
                )
                if memory_context:
                    # 将记忆上下文作为 history 的首条 system 消息注入
                    memory_msg = {"role": "system", "content": f"【相关记忆】\n{memory_context}"}
                    augmented_history = [memory_msg] + (request.history or [])
                    logger.debug(f"[chat] 注入记忆上下文：{len(memory_context)} 字符")
            except Exception as e:
                logger.warning(f"[chat] 记忆上下文生成失败，降级为无记忆：{e}")

        # 统一调度网关收口（rag / tool / agentic / multi 分派 + 参数贯穿 + Harness 审计）
        result = await dispatch(
            query=request.query,
            route=route,
            domain=request.domain,
            use_finetuned=request.use_finetuned,
            use_crag=True,
            history=augmented_history,
            user_id=request.user_id,
            session_id=request.session_id,
            use_agentic_rag=bool(request.use_agentic_rag),
        )

        filtered_answer = sensitive_filter.filter_sensitive(result["answer"])

        # 记忆系统：记录本次对话经验
        if memory_manager is not None and filtered_answer:
            try:
                memory_manager.add_observation("user", request.query)
                memory_manager.add_observation("assistant", filtered_answer)
            except Exception as e:
                logger.warning(f"[chat] 记忆记录失败，忽略：{e}")

        logger.info(
            f"[chat] 调度完成：mode={result['mode']}, route={result['route_decision']}, "
            f"requires_approval={result['requires_approval']}"
        )

        return ChatResponse(
            answer=filtered_answer,
            citations=result.get("citations", []),
            usage=result.get("usage", {}),
            chain_log=result.get("chain_log") or [],
            expansion_info=(query_rewriter.get_expansion_info(request.query)
                            if (result["mode"] == "rag" and query_rewriter is not None) else None)
        )
    except Exception as e:
        logger.error(f"问答失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route")
async def route_query(request: ChatRequest):
    """增强路由判断接口（多意图检测 + 三层路由 + 模型路由）

    route_query() 内部已包含多意图检测，外部无需重复调用。
    """
    try:
        route = llm_router.route_query(request.query)
        domain = llm_router.detect_domain(request.query)

        # 模型路由
        from agentic_rag.utils.model_router import select_model
        model_info = select_model(route, request.route or "default", domain)

        result = {
            "route": route,
            "domain": domain,
            "query": request.query,
            "model": model_info["model"],
            "reason": model_info["reason"],
            "description": "增强路由：多意图检测 + 三层路由(L1/L2/L3) + 模型路由"
        }

        if route == "multi":
            result["description"] = "多意图复合查询，建议并行走 RAG + Agent"

        return result
    except Exception as e:
        logger.error(f"路由判断失败：{e}")
        return {
            "route": "rag",
            "domain": "general",
            "query": request.query,
            "model": "qwen-plus",
            "reason": "路由失败，默认模型",
            "description": "路由失败，默认走 RAG"
        }


@router.post("/retriever/search")
async def search(request: ChatRequest):
    """检索接口（仅返回检索结果，不调用大模型）"""
    try:
        logger.info(f"收到检索请求：{request.query}")

        # 构建租户过滤上下文（默认 admin，可后续从请求中提取）
        tenant_ctx = TenantContext(
            tenant_id="default",
            user_id="system",
            roles=["admin"]
        )
        filter_expr = tenant_permission_filter.build_milvus_filter(tenant_ctx)

        results = hybrid_retriever.retrieve(
            query=request.query,
            filter_dict=filter_expr
        )

        logger.info(f"检索完成，返回 {len(results)} 条结果（租户过滤：{tenant_ctx.tenant_id}）")

        return {
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"检索失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agentic/chat")
async def agentic_chat(request: Dict[str, Any]):
    """Agentic RAG 2.0 多智能体协作问答"""
    try:
        query = request.get("query", "")
        history = request.get("history", [])
        config = request.get("config", {})
        if not query:
            raise HTTPException(status_code=400, detail="query参数不能为空")
        from agentic_rag.agent.agentic_rag import AgentConfig
        agentic_config = AgentConfig(
            enabled=True,
            max_iterations=config.get("max_iterations", 3),
            evidence_threshold=config.get("evidence_threshold", 0.7),
            enable_reflection=config.get("enable_reflection", True),
            enable_tool_agent=config.get("enable_tool_agent", True),
            enable_knowledge_fusion=config.get("enable_knowledge_fusion", True)
        )
        if state.agentic_rag_orchestrator is None:
            raise HTTPException(status_code=503, detail="Agentic RAG 2.0 服务不可用")
        # 更新全局单例的配置
        state.agentic_rag_orchestrator.config = agentic_config
        # 经统一调度网关（参数贯穿 + Harness 审计）
        result = await dispatch(
            query=query,
            route="agentic",
            domain=config.get("domain", "general"),
            use_finetuned=config.get("use_finetuned"),
            use_crag=config.get("use_crag", True),
            history=history,
            use_agentic_rag=True,
            session_id=config.get("session_id", "default"),
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agentic RAG 2.0 问答失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agentic/config")
async def get_agentic_config():
    """获取 Agentic RAG 2.0 当前配置"""
    try:
        from agentic_rag.agent.agentic_rag import AgentConfig
        config = AgentConfig()
        return {
            "success": True,
            "data": {
                "enabled": config.enabled,
                "max_iterations": config.max_iterations,
                "evidence_threshold": config.evidence_threshold,
                "enable_reflection": config.enable_reflection,
                "enable_tool_agent": config.enable_tool_agent,
                "enable_knowledge_fusion": config.enable_knowledge_fusion,
                "agents": [
                    {"name": "supervisor", "role": "元控制器，路由决策与迭代控制"},
                    {"name": "planner", "role": "任务分解与查询规划"},
                    {"name": "retriever", "role": "混合检索与证据收集"},
                    {"name": "validator", "role": "证据校验与质量把关"},
                    {"name": "tool_agent", "role": "MCP工具调用"},
                    {"name": "synthesizer", "role": "知识融合与答案生成"}
                ]
            }
        }
    except Exception as e:
        logger.error(f"获取配置失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/route/config")
async def route_config():
    """返回 Layer 1 关键词配置 — Java 端启动时调用此接口拉取"""
    return get_keywords_config()
