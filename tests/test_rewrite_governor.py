"""查询改写治理测试（回退原 Query + 命中率/意图偏移归因）

@module rag/rewrite_governor.py, rag/query_rewriter.py, rag/crag.py, rag/retriever.py
@test_type unit

覆盖缺口（docs/KNOWN-ISSUES.md §12.3 缺口 10「Query 改写回退治理」）：
- 改写不保证变好：试跑择优、无收益时回退原 Query
- 意图偏移度量与超限剪枝（HyDE 长文本不因篇幅误判）
- 归因统计：各策略计划数/检索数/命中率/采纳数/平均偏移 + 整体回退率
- CRAG 改写护栏：空改写 / 原地打转 / 漂移过大 → 回退原查询并终止改写循环
"""

import os
import sys

import pytest


from agentic_rag.rag.rewrite_governor import (  # noqa: E402
    QueryVariant, RewriteGovernor, STRATEGY_ORIGINAL, STRATEGY_SYNONYM,
    STRATEGY_HYDE, STRATEGY_MULTI_QUERY, STRATEGY_STEPBACK, estimate_intent_drift,
)


# ---------------------------------------------------------------- 意图偏移
def test_drift_is_zero_for_identical_text():
    assert estimate_intent_drift("公司年假多少天", "公司年假多少天") == 0.0


def test_drift_measures_lost_intent_not_length():
    """HyDE 长假设答案覆盖原查询要点 → 偏移应很低"""
    original = "员工报销流程"
    hyde = "员工报销流程说明：员工需在系统提交报销申请，附发票原件，由部门负责人审批后转财务复核。"
    assert estimate_intent_drift(original, hyde) < 0.2


def test_drift_high_when_key_terms_dropped():
    assert estimate_intent_drift("员工报销流程", "食堂菜单公告") > 0.7


def test_drift_handles_empty_input():
    assert estimate_intent_drift("", "任意文本") == 0.0


# ---------------------------------------------------------------- 试跑择优 / 回退
def _retriever_table(table):
    def fn(query):
        return table.get(query, [])
    return fn


def test_measure_falls_back_to_original_when_rewrites_are_empty():
    gov = RewriteGovernor()
    table = {
        "公司年假多少天": [{"score": 0.81}],
        "法定年休假天数": [],
        "休假制度": [],
    }
    report = gov.measure(
        "公司年假多少天",
        [QueryVariant("法定年休假天数", STRATEGY_SYNONYM),
         QueryVariant("休假制度", STRATEGY_MULTI_QUERY)],
        _retriever_table(table))
    assert report.decision == "fallback_original"
    assert report.adopted_query == "公司年假多少天"
    assert gov.stats()["fallbacks"] == 1


def test_measure_adopts_rewrite_that_beats_original():
    gov = RewriteGovernor()
    table = {
        "报销": [{"score": 0.4}],
        "费用报销申请流程": [{"score": 0.9}, {"score": 0.7}],
    }
    report = gov.measure("报销", [QueryVariant("费用报销申请流程", STRATEGY_SYNONYM)],
                         _retriever_table(table))
    assert report.decision == "adopt_rewrites"
    assert report.winning_strategy == STRATEGY_SYNONYM
    winner = [o for o in report.outcomes if o.adopted][0]
    assert winner.query == "费用报销申请流程"


def test_measure_prunes_high_drift_variant():
    gov = RewriteGovernor(max_drift=0.6)
    calls = []

    def fn(query):
        calls.append(query)
        return [{"score": 0.9}]

    report = gov.measure("年假天数规定", [QueryVariant("食堂菜单公告", STRATEGY_HYDE)], fn)
    pruned = [o for o in report.outcomes if o.pruned]
    assert pruned and pruned[0].strategy == STRATEGY_HYDE
    assert "食堂菜单公告" not in calls, "超限变体不得触达检索"
    assert gov.stats()["by_strategy"][STRATEGY_HYDE]["pruned"] == 1


def test_measure_keeps_original_when_no_variants():
    gov = RewriteGovernor()
    report = gov.measure("ERP 版本号", [], lambda q: [{"score": 0.9}])
    assert report.decision == "no_rewrite"
    assert report.adopted_query == "ERP 版本号"


def test_measure_survives_retrieval_failure():
    gov = RewriteGovernor()

    def boom(query):
        raise RuntimeError("Milvus 不可用")

    report = gov.measure("年假", [QueryVariant("年休假", STRATEGY_SYNONYM)], boom)
    assert report.decision == "fallback_original"
    assert all(o.hit_count == 0 for o in report.outcomes)


def test_margin_requires_real_improvement():
    gov = RewriteGovernor(margin=0.1)
    table = {"A 查询": [{"score": 0.8}], "B 查询": [{"score": 0.85}]}
    report = gov.measure("A 查询", [QueryVariant("B 查询", STRATEGY_SYNONYM)],
                         _retriever_table(table))
    assert report.decision == "fallback_original"


# ---------------------------------------------------------------- 归因统计
def test_stats_aggregate_hit_rate_per_strategy():
    gov = RewriteGovernor()
    gov.measure("报销", [QueryVariant("费用报销", STRATEGY_SYNONYM)],
                 lambda q: [{"score": 0.8}] if q == "费用报销" else [{"score": 0.5}])
    gov.measure("报销", [QueryVariant("差旅费", STRATEGY_MULTI_QUERY)],
                 lambda q: [] if q == "差旅费" else [{"score": 0.5}])
    stats = gov.stats()
    assert stats["queries"] == 2
    assert stats["by_strategy"][STRATEGY_SYNONYM]["hit"] == 1
    assert stats["by_strategy"][STRATEGY_SYNONYM]["hit_rate"] == 1.0
    assert stats["by_strategy"][STRATEGY_MULTI_QUERY]["hit"] == 0
    assert 0 < stats["fallback_rate"] <= 1.0


def test_attribution_returns_per_query_reports():
    gov = RewriteGovernor()
    gov.measure("报销", [QueryVariant("费用报销", STRATEGY_SYNONYM)],
                 lambda q: [{"score": 0.9}])
    reports = gov.attribution()
    assert len(reports) == 1
    assert reports[0]["variants"][1]["strategy"] == STRATEGY_SYNONYM
    assert "intent_drift" in reports[0]["variants"][1]


def test_reset_clears_counters():
    gov = RewriteGovernor()
    gov.measure("报销", [], lambda q: [{"score": 0.9}])
    gov.reset()
    assert gov.stats() == {"queries": 0, "fallbacks": 0, "fallback_rate": 0.0,
                           "by_strategy": {}}


# ---------------------------------------------------------------- 真实结果归因
def test_observe_attributes_contributions_by_strategy():
    gov = RewriteGovernor()
    variants = [QueryVariant("报销", STRATEGY_ORIGINAL),
                QueryVariant("费用报销", STRATEGY_SYNONYM)]
    results = [
        {"_source_query": "费用报销", "vector_score": 0.88},
        {"_source_query": "费用报销", "vector_score": 0.7},
        {"_source_query": "报销", "vector_score": 0.6},
    ]
    report = gov.observe("报销", variants, results)
    assert report["decision"] == "adopt_rewrites"
    assert report["winning_strategy"] == STRATEGY_SYNONYM
    syn = [v for v in report["variants"] if v["strategy"] == STRATEGY_SYNONYM][0]
    assert syn["hit_count"] == 2 and syn["top_score"] == 0.88


def test_observe_flags_rewrite_that_contributed_nothing():
    gov = RewriteGovernor()
    variants = [QueryVariant("年假", STRATEGY_ORIGINAL),
                QueryVariant("带薪假期制度", STRATEGY_HYDE)]
    results = [{"_source_query": "年假", "vector_score": 0.9}]
    report = gov.observe("年假", variants, results)
    assert report["decision"] == "fallback_original"
    assert report["adopted_query"] == "年假"
    assert gov.stats()["fallback_rate"] == 1.0


# ---------------------------------------------------------------- CRAG 护栏
def test_guard_accepts_good_rewrite():
    gov = RewriteGovernor()
    final, info = gov.guard_rewrite("员工报销流程", "员工报销流程", "费用报销申请流程")
    assert final == "费用报销申请流程"
    assert info["action"] == "accept_rewrite"


def test_guard_falls_back_on_empty_rewrite():
    gov = RewriteGovernor()
    final, info = gov.guard_rewrite("员工报销流程", "员工报销流程", "  ")
    assert final == "员工报销流程"
    assert info["action"] == "fallback_original"
    assert info["reason"] == "改写结果为空"


def test_guard_falls_back_when_rewrite_is_no_progress():
    gov = RewriteGovernor()
    final, info = gov.guard_rewrite("员工报销流程", "报销流程怎么走", "报销流程怎么走")
    assert final == "员工报销流程"
    assert "无收益" in info["reason"]


def test_guard_falls_back_on_intent_drift():
    gov = RewriteGovernor(max_drift=0.5)
    final, info = gov.guard_rewrite("员工报销流程", "员工报销流程", "食堂营业时间公告")
    assert final == "员工报销流程"
    assert info["action"] == "fallback_original"
    assert "意图偏移" in info["reason"]


def test_crag_rewrite_query_uses_governor_fallback(monkeypatch):
    """LLM 改写出漂移串时，CRAG 改写节点必须回退原查询"""
    import agentic_rag.rag.crag as crag

    class _FakeLLM:
        def _call_llm(self, messages):
            return {"text": "食堂营业时间公告"}

    monkeypatch.setattr(crag, "DeepSeekLLMService", lambda: _FakeLLM())
    state = {"query": "员工报销流程", "original_query": "员工报销流程",
             "rewrite_count": 0, "chain_log": []}
    result = crag.rewrite_query(state)
    assert result["query"] == "员工报销流程"
    assert result["rewrite_count"] == 1
    actions = [log["step"] for log in result["chain_log"]]
    assert "rewrite_fallback" in actions


def test_crag_decide_stops_rewriting_after_fallback():
    import agentic_rag.rag.crag as crag
    state = {"relevance_grade": "partial", "rewrite_count": 1,
             "query": "员工报销流程", "original_query": "员工报销流程"}
    assert crag.decide_to_generate(state) == "generate"

    state["relevance_grade"] = "no_info"
    assert crag.decide_to_generate(state) == "no_info"


# ---------------------------------------------------------------- 变体标签
class _FakeClassifier:
    def __init__(self, query_type):
        self._type = query_type

    def classify(self, query):
        return self._type


def _make_rewriter(query_type):
    """构造不触网的 QueryRewriter（分类器打桩，LLM 策略由 use_llm 关闭）"""
    from agentic_rag.rag.query_rewriter import (
        QueryRewriter, SynonymExpander, HyDEExpander, StepBackExpander,
    )
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter.classifier = _FakeClassifier(query_type)
    rewriter.synonym_expander = SynonymExpander()
    rewriter.hyde_expander = HyDEExpander()
    rewriter.stepback_expander = StepBackExpander()
    rewriter._llm = None
    return rewriter


def test_plan_variants_tags_strategies_without_llm():
    rewriter = _make_rewriter("fuzzy")
    variants = rewriter.plan_variants("员工报销流程怎么走", use_llm=False)
    assert variants[0].strategy == STRATEGY_ORIGINAL
    assert any(v.strategy == STRATEGY_SYNONYM for v in variants)
    assert all(v.strategy != STRATEGY_MULTI_QUERY for v in variants), "use_llm=False 不应调 LLM"
    assert len(variants) <= 4


def test_simple_query_yields_only_original():
    rewriter = _make_rewriter("simple")
    variants = rewriter.plan_variants("ERP 系统版本号", use_llm=False)
    assert [v.strategy for v in variants] == [STRATEGY_ORIGINAL]


def test_rewrite_multi_delegates_to_plan_variants():
    rewriter = _make_rewriter("fuzzy")
    flat = rewriter.rewrite_multi("员工报销流程怎么走")
    tagged = rewriter.plan_variants("员工报销流程怎么走", use_llm=False)
    assert flat == [v.query for v in tagged]
    assert flat[0] == "员工报销流程怎么走"


def test_expansion_info_carries_drift_and_strategy():
    rewriter = _make_rewriter("fuzzy")
    info = rewriter.get_expansion_info("员工报销流程怎么走")
    assert info["variants"][0]["strategy"] == STRATEGY_ORIGINAL
    assert info["variants"][0]["intent_drift"] == 0.0
    assert set(info["variants"][1]) == {"query", "strategy", "intent_drift"}


def test_stepback_strategy_label_for_complex_query():
    class _FakeStepBack:
        def decompose(self, query, use_kg=True):
            return ["报销单据要求", "报销审批层级"]

    rewriter = _make_rewriter("complex")
    rewriter.stepback_expander = _FakeStepBack()
    variants = rewriter.plan_variants("对比分析报销制度与差旅制度的区别并说明审批流程",
                                      use_llm=False)
    assert [v.strategy for v in variants] == [STRATEGY_ORIGINAL, STRATEGY_STEPBACK,
                                              STRATEGY_STEPBACK]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
