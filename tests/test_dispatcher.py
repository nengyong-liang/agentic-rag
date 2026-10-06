"""
统一调度网关 Dispatcher 单元测试（可复用）
=========================================
验证 execution/dispatcher.py 的分派逻辑：路由分类、RAG/Agent/Agentic/Multi
各分支、错误降级与 NLI 门控。全部用例通过注入桩（monkeypatch）规避外部
网络/模型调用，保证离线稳定。

用法：
    python -m pytest test_dispatcher.py -v
    python -m pytest test_dispatcher.py -v --fast   （跳过慢分支）
"""

import asyncio
import sys
import os
from unittest import mock

import pytest

from agentic_rag.execution.dispatcher import Dispatcher, compute_query_type, _run_nli


def _run(coro):
    """同步包装协程（避免依赖 pytest-asyncio 插件）"""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _dispatcher_patches(monkeypatch):
    """把调度网关依赖的模型路由/分类/审计都替换为桩，规避外部依赖。"""
    monkeypatch.setattr("agentic_rag.execution.dispatcher.compute_query_type", lambda q: "simple")
    monkeypatch.setattr("agentic_rag.execution.dispatcher._model_info",
                        lambda route, qt, domain: {"mode": "single", "model": "qwen-plus", "reason": ""})
    # _harness_audit 保持真实（state.harness_auditor 默认 None 时安全跳过），
    # 但为确定性，这里直接替换为空函数。
    monkeypatch.setattr("agentic_rag.execution.dispatcher._harness_audit", lambda *a, **k: None)
    yield


class _FakeAgent:
    """模拟 state.agent（工具问答分支）"""
    def __init__(self, result=None):
        self.result = result or {"success": True,
                                 "data": {"response": "库存：ENG-001 剩余 15 件",
                                          "tool_used": "inventory_query"},
                                 "message": "ok"}

    def run_agent(self, query, history=None, domain="general", query_type="simple"):
        return self.result


class _FakeOrchestrator:
    """模拟 state.agentic_rag_orchestrator"""
    def __init__(self, answer="编排答案", route="rag", confidence=0.9):
        self._answer = answer
        self._route = route
        self._confidence = confidence

    async def run(self, **kwargs):
        return {
            "answer": self._answer,
            "citations": [{"doc_id": "d1", "content": "c1"}],
            "route_decision": self._route,
            "confidence": self._confidence,
            "validation": {"grade": "relevant"},
            "iteration_count": 2,
            "agent_trace": [{"step": "retrieve"}],
        }


class _FakeCrag:
    """模拟 rag.crag.run_crag 返回"""
    @staticmethod
    async def __call__(**kwargs):
        return {
            "answer": "RAG 答案",
            "citations": [{"doc_id": "d1", "content": "c1"}],
            "relevance_grade": "relevant",
            "relevance_score": 0.85,
            "rewrite_count": 1,
            "chain_log": [{"step": "retrieve_start"}],
            "context_snippets": ["片段A", "片段B"],
        }


# ------------------------------------------------------------------
def test_compute_query_type_fallback(monkeypatch):
    """分类器不可用时返回 simple（保守降级）"""
    import agentic_rag.rag.query_rewriter

    class _Boom:
        def get_expansion_info(self, q):
            raise RuntimeError("classifier down")

    monkeypatch.setattr(agentic_rag.rag.query_rewriter, "query_rewriter", _Boom())
    assert compute_query_type("任意无法分类的查询") == "simple"


def test_nli_disabled_by_default():
    """/chat 主链路不开启时 NLI 保持未启用"""
    r = _run_nli("答案", ["上下文"])
    assert r["enabled"] is False
    assert r["is_hallucination"] is False


def test_nli_enabled_but_no_context():
    """开启 NLI 但无检索上下文 → 返回未启用（避免误判）"""
    from unittest import mock
    with mock.patch("agentic_rag.execution.dispatcher._ENABLE_NLI", True):
        r = _run_nli("答案", [])
    assert r["enabled"] is False


# ------------------------------------------------------------------
def test_dispatch_default_rag(monkeypatch):
    """默认 route=rag 走 CRAG 分支，返回规范化结果"""
    from agentic_rag.api.state import state
    state.agent = None
    state.agentic_rag_orchestrator = None
    monkeypatch.setattr("agentic_rag.rag.crag.run_crag", _FakeCrag())

    d = Dispatcher()
    res = _run(d.dispatch(query="年假政策是什么"))
    assert res["mode"] == "rag"
    assert res["route_decision"] == "rag"
    assert res["answer"] == "RAG 答案"
    assert res["citations"]
    assert res["model_mode"] == "single"
    assert res["query_type"] == "simple"
    assert res["nli"]["enabled"] is False


def test_dispatch_tool_branch(monkeypatch):
    """route=tool 走 Agent 工具问答分支"""
    from agentic_rag.api.state import state
    state.agent = _FakeAgent()
    state.agentic_rag_orchestrator = None

    d = Dispatcher()
    res = _run(d.dispatch(query="查询配件 ENG-001 库存", route="tool"))
    assert res["mode"] == "tool"
    assert res["answer"] == "库存：ENG-001 剩余 15 件"
    assert res["tool_used"] == "inventory_query"
    assert res["requires_approval"] is False


def test_dispatch_tool_requires_approval(monkeypatch):
    """工具分支命中 HITL → requires_approval=True 且带上 approval_request_id"""
    from agentic_rag.api.state import state
    state.agent = _FakeAgent({
        "success": False,
        "data": {"response": "需人工审批", "requires_approval": True,
                 "approval_request_id": "req-001", "tool_used": "repair_order_create"},
        "message": "需人工审批",
    })
    d = Dispatcher()
    res = _run(d.dispatch(query="创建维修工单", route="tool"))
    assert res["mode"] == "tool"
    assert res["requires_approval"] is True
    assert res["approval_request_id"] == "req-001"


def test_dispatch_agentic_branch(monkeypatch):
    """route=agentic 走 AgenticRAGOrchestrator 编排分支"""
    from agentic_rag.api.state import state
    state.agent = None
    state.agentic_rag_orchestrator = _FakeOrchestrator(answer="多智能体答案", route="reAct")

    d = Dispatcher()
    res = _run(d.dispatch(query="分析库存与工单关系", route="agentic"))
    assert res["mode"] == "agentic"
    assert res["answer"] == "多智能体答案"
    assert res["route_decision"] == "reAct"
    assert res["usage"]["mode"] == "agentic_rag_2.0"


def test_dispatch_multi_branch(monkeypatch):
    """route=multi 并行 RAG + Agentic，合并 answer 与去重 citations"""
    from agentic_rag.api.state import state
    state.agent = None
    state.agentic_rag_orchestrator = _FakeOrchestrator(answer="编排答案")
    monkeypatch.setattr("agentic_rag.rag.crag.run_crag", _FakeCrag())

    d = Dispatcher()
    res = _run(d.dispatch(query="综合对比", route="multi"))
    assert res["mode"] == "multi"
    assert res["route_decision"] == "multi"
    # 优先采用编排分支答案
    assert res["answer"] == "编排答案"
    assert len(res["citations"]) >= 1
    assert res["model_mode"] == "parallel"


def test_dispatch_error_degrade(monkeypatch):
    """CRAG 抛异常 → 返回友好降级结果而非崩溃"""
    from agentic_rag.api.state import state
    state.agent = None
    state.agentic_rag_orchestrator = None

    async def boom(**kwargs):
        raise RuntimeError("milvus down")
    monkeypatch.setattr("agentic_rag.rag.crag.run_crag", boom)

    d = Dispatcher()
    res = _run(d.dispatch(query="测试降级"))
    assert res["mode"] == "rag"
    assert res["answer"] == "系统繁忙，请稍后重试"
    assert res["confidence"] == 0.0


def test_dispatch_tool_agent_absent(monkeypatch):
    """route=tool 但 state.agent 未装配 → 降级提示"""
    from agentic_rag.api.state import state
    old = state.agent
    state.agent = None
    try:
        d = Dispatcher()
        res = _run(d.dispatch(query="查询库存", route="tool"))
        assert res["answer"] == "系统繁忙，请稍后重试"
        assert "error" in res["validation"]
    finally:
        state.agent = old


def test_module_level_dispatch_entry():
    """execution.__init__ 导出单例 dispatch 便捷入口"""
    from agentic_rag.execution import dispatch, Dispatcher
    assert callable(dispatch)
    assert callable(Dispatcher)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))