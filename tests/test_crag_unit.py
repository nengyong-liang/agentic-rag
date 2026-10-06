"""
CRAG LangGraph 流程逻辑测试
仅验证 LangGraph StateGraph 的流程控制逻辑，不调用外部服务
"""

import sys
import os
import pytest

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_langgraph_import():
    """测试 LangGraph 是否可以正常导入"""
    try:
        from langgraph.graph import StateGraph, END
        logger.info("✅ LangGraph 导入成功")
    except ImportError as e:
        logger.error(f"❌ LangGraph 导入失败：{e}")
        pytest.fail(f"LangGraph 导入失败：{e}")


def test_state_graph_construction():
    """测试 StateGraph 是否可以正常构建"""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Dict

    class TestState(TypedDict):
        query: str
        step: str
        result: str

    def node_a(state: TestState) -> Dict:
        logger.info(f"  节点A执行：query={state['query']}")
        return {"step": "a_done"}

    def node_b(state: TestState) -> Dict:
        logger.info(f"  节点B执行：step={state['step']}")
        return {"result": "success"}

    def router(state: TestState) -> str:
        if state["query"] == "go_b":
            return "node_b"
        return "end_early"

    workflow = StateGraph(TestState)
    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)
    workflow.set_entry_point("node_a")
    workflow.add_conditional_edges("node_a", router, {"node_b": "node_b", "end_early": END})
    workflow.add_edge("node_b", END)

    graph = workflow.compile()
    logger.info("✅ StateGraph 构建成功")

    result = graph.invoke({"query": "go_b", "step": "", "result": ""})
    logger.info(f"  执行结果：{result}")
    assert result["result"] == "success", f"预期 result=success，实际 result={result['result']}"
    logger.info("✅ StateGraph 执行成功")


def test_crag_graph_construction():
    """测试 CRAG 图是否可以正常构建"""
    try:
        from agentic_rag.rag.crag import build_crag_graph, CragState
        graph = build_crag_graph()
        logger.info("✅ CRAG 图构建成功")

        nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
        logger.info(f"  CRAG 节点列表：{nodes}")
    except Exception as e:
        logger.error(f"❌ CRAG 图构建失败：{e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"CRAG 图构建失败：{e}")


@pytest.mark.asyncio
async def test_crag_flow_with_mock():
    """用 mock 覆盖 CRAG 全链路（检索/Judge/生成均打桩，不产生真实调用）"""
    from unittest.mock import patch
    from agentic_rag.rag.crag import run_crag

    mock_docs = [
        {
            "content": "公司年假政策：员工入职满1年可享受5天年假，满5年可享受10天年假。",
            "document_id": 1,
            "chunk_index": 0,
            "metadata": {"section": "年假政策", "page_number": "12"},
            "rerank_score": 0.92
        },
        {
            "content": "年假申请流程：员工需提前3个工作日在OA系统提交年假申请。",
            "document_id": 1,
            "chunk_index": 1,
            "metadata": {"section": "年假申请", "page_number": "13"},
            "rerank_score": 0.85
        }
    ]

    # 相关性节点用的是 DeepSeek + Qwen 双 Judge，且解析严格 JSON 输出；
    # 桩必须命中真实契约，否则全员评估失败会静默回退规则打分，用例即失去意义。
    judge_payload = ('{"grade": "relevant", "score": 0.92,'
                     ' "contradictions": [], "reason": "命中年假政策条款"}')

    with patch('agentic_rag.rag.crag.hybrid_retriever') as mock_retriever, \
         patch('agentic_rag.rag.crag.DoubaoLLMService'), \
         patch('agentic_rag.rag.crag.DeepSeekLLMService') as mock_deepseek_class, \
         patch('agent_harness.providers.qwen.QwenLLMService') as mock_qwen_class, \
         patch('agentic_rag.rag.crag.hybrid_llm') as mock_hybrid_llm, \
         patch('agentic_rag.rag.crag.sensitive_filter') as mock_filter:

        mock_retriever.retrieve.return_value = mock_docs

        mock_deepseek_class.return_value._call_llm.return_value = {"text": judge_payload}
        mock_qwen_class.return_value._call_llm.return_value = {"text": judge_payload}

        mock_hybrid_llm.generate_answer.return_value = {
            "answer": "根据公司政策，员工入职满1年可享受5天年假 [员工手册 - 年假政策/12页]。",
            "model": "doubao",
            "usage": {}
        }

        mock_filter.filter_sensitive.return_value = "根据公司政策，员工入职满1年可享受5天年假 [员工手册 - 年假政策/12页]。"
        
        result = await run_crag("公司年假政策是什么？", domain="hr")

        logger.info(f"  CRAG 结果：answer={result['answer'][:50]}...")
        logger.info(f"  citations 数量：{len(result['citations'])}")
        logger.info(f"  relevance_grade：{result['relevance_grade']}")
        logger.info(f"  rewrite_count：{result['rewrite_count']}")

        assert result["answer"], "回答不应为空"
        assert result["relevance_grade"] == "relevant", f"预期 relevant，实际 {result['relevance_grade']}"
        logger.info("✅ CRAG 流程 mock 测试通过")


@pytest.mark.asyncio
async def test_crag_rewrite_flow():
    """测试 CRAG 改写重试流程"""
    from unittest.mock import patch, MagicMock
    from agentic_rag.rag.crag import run_crag

    call_count = {"retrieve": 0}

    def mock_retrieve_side_effect(query, filter_dict=None):
        call_count["retrieve"] += 1
        if call_count["retrieve"] == 1:
            return []
        return [
            {
                "content": "员工请假制度：事假需提前申请，病假需提供医院证明。",
                "document_id": 2,
                "chunk_index": 0,
                "metadata": {"section": "请假制度", "page_number": "5"},
                "rerank_score": 0.78
            }
        ]

    with patch('agentic_rag.rag.crag.hybrid_retriever') as mock_retriever, \
         patch('agentic_rag.rag.crag.DoubaoLLMService') as mock_llm_class, \
         patch('agentic_rag.rag.crag.hybrid_llm') as mock_hybrid_llm, \
         patch('agentic_rag.rag.crag.sensitive_filter') as mock_filter:

        mock_retriever.retrieve.side_effect = mock_retrieve_side_effect

        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        grade_call_count = {"count": 0}
        def grade_side_effect(messages, tools=None, tool_choice=None):
            grade_call_count["count"] += 1
            if grade_call_count["count"] == 1:
                return {"text": "irrelevant", "tool_calls": None}
            return {"text": "relevant", "tool_calls": None}

        mock_llm_instance._call_llm.side_effect = grade_side_effect

        mock_hybrid_llm.generate_answer.return_value = {
            "answer": "员工请假需要提前申请 [员工手册 - 请假制度/5页]。",
            "model": "doubao",
            "usage": {}
        }

        mock_filter.filter_sensitive.return_value = "员工请假需要提前申请 [员工手册 - 请假制度/5页]。"
        
        result = await run_crag("怎么请假？", domain="hr")

        logger.info(f"  检索调用次数：{call_count['retrieve']}")
        logger.info(f"  评估调用次数：{grade_call_count['count']}")
        logger.info(f"  rewrite_count：{result['rewrite_count']}")

        assert call_count["retrieve"] >= 2, f"预期至少2次检索，实际 {call_count['retrieve']}"
        assert result["rewrite_count"] >= 1, f"预期至少1次改写，实际 {result['rewrite_count']}"
        logger.info("✅ CRAG 改写重试流程测试通过")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CRAG LangGraph 流程测试")
    logger.info("=" * 60)

    tests = [
        ("LangGraph 导入测试", test_langgraph_import),
        ("StateGraph 构建测试", test_state_graph_construction),
        ("CRAG 图构建测试", test_crag_graph_construction),
        ("CRAG 流程 mock 测试", test_crag_flow_with_mock),
        ("CRAG 改写重试测试", test_crag_rewrite_flow),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        logger.info(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {name} 异常：{e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"测试结果：{passed} 通过，{failed} 失败")
    logger.info(f"{'=' * 60}")
