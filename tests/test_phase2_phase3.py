"""
Phase 2 + Phase 3 全面测试
RAGAS评估 / PDF版面分析 / 监控可观测性 / 知识图谱
"""

import asyncio
import sys
import os


def test_rag_evaluator():
    """测试1：RAGAS质量评估"""
    print("=" * 60)
    print("测试 1：RAGAS 质量评估")

    from agentic_rag.rag.rag_evaluator import rag_evaluator

    result = rag_evaluator.evaluate_single(
        query="什么是混合检索？",
        answer="混合检索是结合向量检索和关键词检索的方法，向量检索擅长语义匹配，关键词检索擅长精确匹配，两者互补可以提升召回率。",
        contexts=[
            "混合检索（Hybrid Search）是将向量检索和关键词检索结合的检索方式。",
            "向量检索基于语义相似度，关键词检索基于精确匹配。"
        ]
    )

    metrics = result.get("metrics", {})
    print(f"  忠实度(Faithfulness)：{metrics.get('faithfulness', 'N/A')}")
    print(f"  答案相关性(Answer Relevance)：{metrics.get('answer_relevance', 'N/A')}")
    print(f"  上下文精确度(Context Precision)：{metrics.get('context_precision', 'N/A')}")
    print(f"  综合评分：{result.get('overall_score', 'N/A')}")

    has_scores = all(isinstance(v, (int, float)) for v in metrics.values())
    print(f"  所有指标都有数值：{has_scores}")
    print("=" * 60)
    assert has_scores, "部分指标缺失或非数值"


def test_rag_evaluator_batch():
    """测试2：RAGAS批量评估"""
    print("=" * 60)
    print("测试 2：RAGAS 批量评估")

    from agentic_rag.rag.rag_evaluator import rag_evaluator

    test_cases = [
        {
            "query": "BGE-M3是什么？",
            "answer": "BGE-M3是BAAI发布的嵌入模型，支持多语言和多粒度。",
            "contexts": ["BGE-M3是北京智源研究院(BAAI)发布的嵌入模型。"]
        },
        {
            "query": "CRAG的工作流程是什么？",
            "answer": "CRAG通过检索-评估-重试循环工作，先检索文档，评估相关性，不相关则改写查询重试。",
            "contexts": ["CRAG(Corrective RAG)通过检索-评估-重试循环提升检索质量。"]
        }
    ]

    result = rag_evaluator.evaluate_batch(test_cases)

    print(f"  总测试用例数：{result['total_cases']}")
    print(f"  平均综合评分：{result['overall_avg']:.2f}")
    avg_metrics = result.get("avg_metrics", {})
    for name, value in avg_metrics.items():
        print(f"  平均 {name}：{value:.2f}")

    print("=" * 60)
    assert result["total_cases"] == 2, f"预期2个用例，实际{result['total_cases']}"


def test_pdf_layout_analyzer():
    """测试3：PDF版面分析"""
    print("=" * 60)
    print("测试 3：PDF 版面分析")

    from agentic_rag.rag.pdf_layout_analyzer import pdf_layout_analyzer

    print(f"  docling 可用：{pdf_layout_analyzer.use_docling}")
    print(f"  解析器类型：{'docling' if pdf_layout_analyzer.use_docling else 'PyMuPDF'}")

    test_pdf = os.path.join(os.path.dirname(__file__), "test_assets", "test.pdf")
    if os.path.exists(test_pdf):
        result = pdf_layout_analyzer.analyze(test_pdf)
        print(f"  解析元素数：{result.get('total_elements', 0)}")
        print(f"  解析器：{result.get('parser', 'unknown')}")
    else:
        print("  测试PDF不存在，验证接口可用性")
        result = pdf_layout_analyzer._classify_text_element("测试标题", 18, True)
        print(f"  标题分类：{result}")
        result2 = pdf_layout_analyzer._classify_text_element("正文内容", 10, False)
        print(f"  正文分类：{result2}")

    table_md = pdf_layout_analyzer._table_to_markdown(
        [["姓名", "年龄"], ["张三", "25"], ["李四", "30"]]
    )
    print(f"  表格Markdown转换：\n  {table_md[:80]}...")

    print("=" * 60)


def test_monitor():
    """测试4：监控可观测性"""
    print("=" * 60)
    print("测试 4：监控可观测性")

    from agent_harness.observability.monitor import rag_monitor

    print(f"  LangSmith 已启用：{rag_monitor.langsmith_enabled}")
    print(f"  LangFuse 已启用：{rag_monitor.langfuse_enabled}")

    with rag_monitor.trace(query="测试查询", domain="test") as t:
        t.model = "qwen-plus"
        t.retrieval_count = 5
        t.prompt_tokens = 100
        t.completion_tokens = 50
        t.total_tokens = 150
        t.rewrite_count = 1

    stats = rag_monitor.get_stats()
    print(f"  总追踪数：{stats['total_traces']}")
    print(f"  完成数：{stats['completed']}")
    print(f"  平均耗时：{stats['latency']['avg_ms']}ms")
    print(f"  总Token：{stats['total_tokens']}")

    recent = rag_monitor.get_recent_traces(limit=5)
    print(f"  最近追踪数：{len(recent)}")
    if recent:
        latest = recent[-1]
        print(f"  最新追踪ID：{latest['trace_id']}")
        print(f"  最新追踪状态：{latest['status']}")

    domain_stats = rag_monitor.get_domain_stats()
    print(f"  领域统计：{domain_stats}")

    print("=" * 60)
    assert stats["total_traces"] > 0, f"预期至少1条追踪，实际{stats['total_traces']}"


def test_knowledge_graph():
    """测试5：知识图谱实体关系抽取（桩掉 LLM，只验解析与溯源契约，零真实调用）"""
    from unittest.mock import patch

    print("=" * 60)
    print("测试 5：知识图谱（实体关系提取）")

    from agentic_rag.rag.knowledge_graph import knowledge_graph

    test_text = """
    长安汽车是中国知名的汽车制造商，总部位于重庆。长安汽车生产了多款车型，
    包括长安CS75 PLUS和长安逸动。CS75 PLUS搭载了蓝鲸发动机，
    该发动机由长安汽车自主研发。长安汽车与华为在智能驾驶领域有合作关系。
    """

    llm_output = """{
      "entities": [
        {"name": "长安汽车", "type": "组织", "description": "中国知名汽车制造商"},
        {"name": "蓝鲸发动机", "type": "产品", "description": "长安汽车自主研发"}
      ],
      "relations": [
        {"subject": "长安汽车", "predicate": "生产", "object": "长安CS75 PLUS"},
        {"subject": "长安汽车", "predicate": "合作", "object": "华为"}
      ]
    }"""

    with patch.object(knowledge_graph, "_call_llm", return_value=llm_output):
        result = knowledge_graph.extract_entities_and_relations(test_text, "长安汽车介绍")

    entities = result.get("entities", [])
    relations = result.get("relations", [])

    print(f"  提取实体数：{len(entities)}")
    print(f"  提取关系数：{len(relations)}")

    for e in entities[:3]:
        print(f"  实体：{e.get('name', '')} ({e.get('type', '')})")

    for r in relations[:3]:
        print(f"  关系：{r.get('subject', '')} -[{r.get('predicate', '')}]-> {r.get('object', '')}")

    print("=" * 60)
    assert len(entities) == 2, f"预期解析出 2 个实体，实际 {len(entities)}"
    assert len(relations) == 2, f"预期解析出 2 条关系，实际 {len(relations)}"
    assert all(e.get("source") == "长安汽车介绍" for e in entities), "实体应带回源文档名"
    assert all(r.get("source") == "长安汽车介绍" for r in relations), "关系应带回源文档名"


def test_knowledge_graph_multi_hop():
    """测试6：知识图谱多跳推理"""
    print("=" * 60)
    print("测试 6：知识图谱多跳推理")

    from agentic_rag.rag.knowledge_graph import knowledge_graph

    sub_queries = knowledge_graph.multi_hop_reason(
        "长安汽车自主研发的发动机搭载在哪些车型上？"
    )

    print(f"  子问题数：{len(sub_queries)}")
    for i, q in enumerate(sub_queries, 1):
        print(f"  子问题{i}：{q}")

    print("=" * 60)
    assert len(sub_queries) > 0, f"预期至少1个子问题，实际{len(sub_queries)}"


def run_all_tests():
    """运行所有测试"""
    print("\n🧪 Phase 2 + Phase 3 全面测试\n")

    test_cases = [
        ("RAGAS质量评估", test_rag_evaluator),
        ("RAGAS批量评估", test_rag_evaluator_batch),
        ("PDF版面分析", test_pdf_layout_analyzer),
        ("监控可观测性", test_monitor),
        ("知识图谱实体提取", test_knowledge_graph),
        ("知识图谱多跳推理", test_knowledge_graph_multi_hop),
    ]

    print("\n" + "=" * 60)
    print("📊 Phase 2 + Phase 3 测试结果汇总")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_func in test_cases:
        try:
            test_func()
            status = "✅ PASS"
            passed += 1
        except Exception as e:
            status = "❌ FAIL"
            failed += 1
            print(f"     原因：{e}")
        print(f"  {status} - {name}")

    print(f"\n总计：{passed} 通过，{failed} 失败")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
