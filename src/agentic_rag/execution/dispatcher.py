"""
统一调度网关 (Unified Dispatch Gateway)

把 /chat、/agentic/chat 的请求统一收口，按 kind 分派：
  rag      → run_crag（模型路由 / query_type 贯穿 + 生成后 NLI(可配置)）
  tool     → state.agent.run_agent()（Agent 工具问答；工具执行经 mcp_bridge 单点，
                          RAGAgent 为 Harness 优先、harness 不可用/空答时安全回退）
  agentic  → AgenticRAGOrchestrator（知识检索 + 并行 CRAG；工具分支复用 mcp_bridge）
  multi    → asyncio.gather 并行 RAG + Agentic，合并 answer/citations/trace

设计原则（对应计划 DEC-3/DEC-4/D9）：
- 模型路由结果在入口 computed 并贯穿到 run_crag / orchestrator（D6 修复）。
- query_type 由 query_rewriter 计算并写入 CRAG state，使 MoA 可按 ENABLE_MOA 配置触发（D4）。
- NLI 仅当 ENABLE_NLI 开启且存在检索上下文时执行，规则式、不强依赖 DeBERTa/外网（D5）。
- Harness 统一运行层以审计/事件方式参与业务调度（D1 去闲置），工具执行仍收敛到 mcp_bridge 单点。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from agentic_rag.config import settings

logger = logging.getLogger(__name__)

_ENABLE_NLI = getattr(settings, "ENABLE_NLI", False)
_ENABLE_MOA = getattr(settings, "ENABLE_MOA", False)


def compute_query_type(query: str) -> str:
    """由 query_rewriter 分类当前查询类型（simple / fuzzy / complex）"""
    try:
        from agentic_rag.rag.query_rewriter import query_rewriter
        return (query_rewriter.get_expansion_info(query) or {}).get("query_type", "simple")
    except Exception as e:  # 分类器不可用时保守回退简单查询
        logger.warning(f"[dispatcher] query_type 计算失败，回退 simple: {e}")
        return "simple"


def _harness_audit(session_id: str, node: str, operation: str, meta: Dict) -> None:
    """把调度决策写入 Harness 审计（若可用），使统一运行层参与业务可观测。"""
    try:
        from agentic_rag.api.state import state
        auditor = getattr(state, "harness_auditor", None)
        if auditor is not None:
            auditor.record(
                session_id=session_id or "dispatcher",
                node=node,
                operation=operation,
                input_summary=str(meta)[:300],
                output_summary="ok",
                duration_ms=0,
                metadata=dict(meta),
            )
    except Exception as e:
        logger.debug(f"[dispatcher] Harness 审计记录跳过: {e}")


def _model_info(route: str, query_type: str, domain: str) -> Dict[str, Any]:
    """计算模型路由（单模型或 MoA，受 ENABLE_MOA 门控）。"""
    try:
        from agentic_rag.utils.model_router import select_model_or_moa
        return select_model_or_moa(route or "rag", query_type, domain)
    except Exception as e:
        return {"mode": "single", "model": "qwen-plus", "reason": f"模型路由失败，默认: {e}"}


def _run_nli(answer: str, contexts: List[str]) -> Dict:
    """生成后 NLI 校验（规则式）。默认关闭；开启且无上下文时直接返回未启用。"""
    result = {"enabled": False, "verdict": "neutral", "is_hallucination": False}
    if not _ENABLE_NLI or not answer or not contexts:
        return result
    try:
        from agentic_rag.utils.nli_checker import NLIConsistencyChecker
        checker = NLIConsistencyChecker(use_deberta=False)  # 规则式，稳定不联网
        r = checker.check_consistency(answer, contexts)
        result.update({
            "enabled": True,
            "verdict": str(r.get("verdict", "neutral")),
            "consistency_score": r.get("consistency_score", 0.5),
            "is_hallucination": r.get("is_hallucination", False),
            "suggestion": r.get("suggestion", ""),
        })
    except Exception as e:
        logger.warning(f"[dispatcher] NLI 校验失败，忽略: {e}")
    return result


class Dispatcher:
    """统一调度网关（实例化即用，也提供模块级单例 dispatch）"""

    async def dispatch(
        self,
        *,
        query: str,
        route: str = "rag",
        domain: str = "general",
        use_finetuned: Optional[bool] = None,
        use_crag: bool = True,
        history: Optional[List[Dict]] = None,
        user_id: str = "default",
        session_id: str = "default",
        use_agentic_rag: bool = False,
    ) -> Dict[str, Any]:
        """分派请求，返回规范化结果字典。"""
        key = f"{session_id}:{query[:50]}"
        route = route or "rag"
        query_type = compute_query_type(query)
        model = _model_info(route, query_type, domain)
        _harness_audit(key, "dispatcher", "dispatch", {
            "query": query[:120],
            "route": route,
            "domain": domain,
            "query_type": query_type,
            "model": model.get("mode", "single"),
            "use_agentic_rag": use_agentic_rag,
        })

        if use_agentic_rag or route in ("agentic", "multi", "react", "plan", "retrieve", "tool_call"):
            # Agentic / 复合：走 AgenticRAGOrchestrator（或 multi 并行）
            return await self._dispatch_agentic(
                query=query,
                route=route,
                history=history,
                use_crag=use_crag,
                domain=domain,
                use_finetuned=use_finetuned,
                user_id=user_id,
                session_id=session_id,
                model=model,
                query_type=query_type,
            )

        if route == "tool":
            return await self._dispatch_tool(
                query, history, session_id, model, domain, query_type)

        # 默认 rag
        return await self._dispatch_rag(
            query=query,
            domain=domain,
            use_finetuned=use_finetuned,
            history=history,
            user_id=user_id,
            session_id=session_id,
            model=model,
            query_type=query_type,
        )

    # ------------------------------------------------------------------
    async def _dispatch_rag(
        self, query, domain, use_finetuned, history, user_id, session_id, model, query_type
    ) -> Dict[str, Any]:
        from agentic_rag.api.state import state
        from agentic_rag.rag.crag import run_crag

        if use_finetuned is None:
            use_finetuned = domain != "general"

        try:
            crag_result = await run_crag(
                query=query,
                domain=domain,
                use_finetuned=use_finetuned,
                history=history,
                request_id=session_id,
            )
        except Exception as e:
            logger.error(f"[dispatcher] CRAG 失败: {e}")
            return self._error("rag", str(e))

        answer = crag_result.get("answer", "")
        contexts = crag_result.get("context_snippets", []) or []
        nli = _run_nli(answer, contexts)
        if nli.get("is_hallucination"):
            logger.warning(f"[dispatcher] NLI 判定疑似幻觉，标记风险: {nli.get('verdict')}")

        _harness_audit(session_id, "dispatcher", "rag_done", {
            "grade": crag_result.get("relevance_grade", ""),
            "rewrite": crag_result.get("rewrite_count", 0),
            "docs": len(contexts),
        })

        return {
            "mode": "rag",
            "answer": answer,
            "citations": crag_result.get("citations", []),
            "route_decision": "rag",
            "query_type": query_type,
            "model": model.get("model", "qwen-plus"),
            "model_mode": model.get("mode", "single"),
            "model_reason": model.get("reason", ""),
            "confidence": float(crag_result.get("relevance_score", 0.0)),
            "validation": {"grade": crag_result.get("relevance_grade", "")},
            "iteration_count": int(crag_result.get("rewrite_count", 0)),
            "chain_log": crag_result.get("chain_log", []),
            "expansion_info": None,
            "tool_used": None,
            "requires_approval": False,
            "approval_request_id": None,
            "nli": nli,
            "usage": {"mode": "rag", "model": model.get("model", "qwen-plus")},
        }

    async def _dispatch_tool(self, query, history, session_id, model,
                              domain: str = "general",
                              query_type: str = "simple") -> Dict[str, Any]:
        from agentic_rag.api.state import state

        if state.agent is None:
            return self._error("tool", "Agent 服务不可用")
        try:
            result = state.agent.run_agent(
                query, history, domain=domain, query_type=query_type)
        except Exception as e:
            logger.error(f"[dispatcher] Agent 工具问答失败: {e}")
            return self._error("tool", str(e))

        ra = result.get("requires_approval", False)
        approval_id = None
        if result.get("success"):
            data = result.get("data", {}) or {}
            answer = data.get("response", "")
            tool_used = data.get("tool_used")
            ra = bool(data.get("requires_approval", False))
            approval_id = data.get("approval_request_id")
        else:
            answer = result.get("message", "Agent 处理失败")
            data = result.get("data") or {}
            ra = bool(data.get("requires_approval", False))
            approval_id = data.get("approval_request_id")

        _harness_audit(session_id, "dispatcher", "tool_done", {
            "success": result.get("success"),
            "requires_approval": ra,
            "tool_used": result.get("data", {}).get("tool_used"),
        })

        return {
            "mode": "tool",
            "answer": answer,
            "citations": [],
            "route_decision": "tool",
            "model": model.get("model", "doubao"),
            "model_mode": model.get("mode", "single"),
            "model_reason": model.get("reason", ""),
            "confidence": 0.0,
            "validation": {},
            "iteration_count": 0,
            "chain_log": [],
            "expansion_info": None,
            "tool_used": result.get("data", {}).get("tool_used"),
            "tool_result": result.get("data", {}).get("tool_result"),
            "requires_approval": ra,
            "approval_request_id": approval_id,
            "nli": {"enabled": False, "verdict": "neutral", "is_hallucination": False},
            "usage": {"mode": "tool", "success": result.get("success")},
        }

    async def _dispatch_agentic(
        self, query, route, history, use_crag, domain, use_finetuned, user_id, session_id, model, query_type
    ) -> Dict[str, Any]:
        from agentic_rag.api.state import state

        if state.agentic_rag_orchestrator is None:
            return self._error("agentic", "Agentic RAG 2.0 服务不可用")

        if route == "multi":
            return await self._dispatch_multi(
                query, history, use_crag, domain, use_finetuned, user_id, session_id)

        try:
            result = await state.agentic_rag_orchestrator.run(
                query=query,
                history=history,
                use_crag=use_crag,
                route="rag" if not route or route in ("agentic",) else route,
                domain=domain,
                use_finetuned=use_finetuned,
            )
        except Exception as e:
            logger.error(f"[dispatcher] Agentic 编排失败: {e}")
            return self._error("agentic", str(e))

        _harness_audit(session_id, "dispatcher", "agentic_done", {
            "route_decision": result.get("route_decision", ""),
            "confidence": result.get("confidence", 0.0),
        })

        return {
            "mode": "agentic",
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "route_decision": result.get("route_decision", ""),
            "query_type": query_type,
            "model": model.get("model", "qwen-plus"),
            "model_mode": model.get("mode", "single"),
            "model_reason": model.get("reason", ""),
            "confidence": result.get("confidence", 0.0),
            "validation": result.get("validation", {}),
            "iteration_count": result.get("iteration_count", 0),
            "chain_log": result.get("agent_trace", []),
            "expansion_info": None,
            "tool_used": None,
            "requires_approval": False,
            "approval_request_id": None,
            "nli": {"enabled": False, "verdict": "neutral", "is_hallucination": False},
            "usage": {
                "mode": "agentic_rag_2.0",
                "route_decision": result.get("route_decision", ""),
                "iteration_count": result.get("iteration_count", 0),
                "confidence": result.get("confidence", 0.0),
                "validation": result.get("validation", {}),
            },
        }

    async def _dispatch_multi(
        self, query, history, use_crag, domain, use_finetuned, user_id, session_id
    ) -> Dict[str, Any]:
        """route==multi：并行 RAG + Agentic，合并 answer/citations/trace。"""
        from agentic_rag.api.state import state

        if state.agentic_rag_orchestrator is None:
            return await self._dispatch_rag(
                query=query, domain=domain, use_finetuned=use_finetuned,
                history=history, user_id=user_id, session_id=session_id,
                model=_model_info("rag", "complex", domain), query_type="complex")

        rag_f, agentic_f = await asyncio.gather(
            self._dispatch_rag(
                query=query, domain=domain, use_finetuned=use_finetuned,
                history=history, user_id=user_id, session_id=session_id,
                model=_model_info("rag", "complex", domain), query_type="complex"),
            state.agentic_rag_orchestrator.run(
                query=query, history=history, use_crag=use_crag,
                route=None, domain=domain, use_finetuned=use_finetuned),
            return_exceptions=True,
        )

        if isinstance(rag_f, Exception):
            logger.warning(f"[dispatcher] multi RAG 分支失败: {rag_f}")
            rag_f = self._error("rag", str(rag_f))
        if isinstance(agentic_f, Exception):
            logger.warning(f"[dispatcher] multi Agentic 分支失败: {agentic_f}")
            agentic_f = self._error("agentic", str(agentic_f))

        answer = agentic_f.get("answer") or rag_f.get("answer", "")
        merged = (rag_f.get("citations") or []) + (agentic_f.get("citations") or [])
        # citation 为 dict，无法 dict.fromkeys 去重；按 doc/引用键去重
        seen, citations = set(), []
        for c in merged:
            key = str(c.get("doc_id") if isinstance(c, dict) else c)
            if key not in seen:
                seen.add(key)
                citations.append(c)

        return {
            "mode": "multi",
            "answer": answer,
            "citations": citations,
            "route_decision": "multi",
            "model": "qwen-plus",
            "model_mode": "parallel",
            "model_reason": "multi 并行：RAG 检索 + Agentic 编排",
            "confidence": agentic_f.get("confidence", 0.0),
            "validation": agentic_f.get("validation", {}),
            "iteration_count": agentic_f.get("iteration_count", 0),
            "chain_log": (rag_f.get("chain_log") or []) + (agentic_f.get("chain_log") or []),
            "expansion_info": None,
            "tool_used": None,
            "requires_approval": False,
            "approval_request_id": None,
            "usage": {"mode": "multi", "rag": rag_f.get("usage", {}), "agentic": agentic_f.get("usage", {})},
        }

    @staticmethod
    def _error(mode: str, detail: str) -> Dict[str, Any]:
        return {
            "mode": mode,
            "answer": "系统繁忙，请稍后重试",
            "citations": [],
            "route_decision": mode,
            "model": "",
            "model_mode": "single",
            "model_reason": "",
            "confidence": 0.0,
            "validation": {"error": detail[:200]},
            "iteration_count": 0,
            "chain_log": [],
            "expansion_info": None,
            "tool_used": None,
            "requires_approval": False,
            "approval_request_id": None,
            "nli": {"enabled": False, "verdict": "neutral", "is_hallucination": False},
            "usage": {"mode": mode, "error": detail[:200]},
        }


# 模块级单例，便于路由直接调用
_dispatcher_instance = Dispatcher()


def dispatch(**kwargs) -> "asyncio.coroutine":
    """便捷入口，等价于 dispatcher.dispatch(...)"""
    return _dispatcher_instance.dispatch(**kwargs)