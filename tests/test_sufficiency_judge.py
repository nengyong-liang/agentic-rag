"""证据充分性三态判定测试

@module agent/sufficiency_judge.py, agent/agentic_rag.py
@test_type unit

覆盖缺口（docs/KNOWN-ISSUES.md §12.3 缺口 9「证据充分性三态判定」）：
- SUFFICIENT / INSUFFICIENT / CONFLICTING 三态判定与结构化依据
- 规则优先 + LLM 只可加严（fail-closed）、引用不可核查即作废
- INSUFFICIENT 到轮次上限 → 显式弃答/澄清，不硬答
- ValidatorAgent / SupervisorAgent / SynthesizerAgent 全链路接入
- CRAG 评级归一到同一套 evidence_state 契约
"""

import json
import os
import sys

import pytest


from agentic_rag.agent.sufficiency_judge import (  # noqa: E402
    EvidenceState, SufficiencyJudge,
)
from agentic_rag.agent.agentic_rag import (  # noqa: E402
    AgentConfig, SupervisorAgent, SynthesizerAgent, ValidatorAgent,
)


def _ev(eid, score, content):
    return {"id": eid, "score": score, "content": content}


SUFFICIENT_EVIDENCE = [
    _ev("c1", 0.92, "年假规定：入职满一年享 5 天年假，由员工在系统提交申请。"),
    _ev("c2", 0.85, "年假申请流程说明：主管审批后计入考勤。"),
]

CONFLICT_EVIDENCE = [
    _ev("c1", 0.9, "A型配件库存12件，属正常水位。"),
    _ev("c2", 0.88, "A型配件库存5件，已触发补货。"),
]


# ---------------------------------------------------------------- 规则判定
def test_no_evidence_is_insufficient():
    verdict = SufficiencyJudge().judge("年假规定", [])
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.missing_info == ["年假规定"]
    assert verdict.abstain is False


def test_zero_evidence_at_iteration_limit_abstains():
    verdict = SufficiencyJudge().judge("年假规定", [], iteration=3, max_iterations=3)
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.abstain is True
    assert verdict.needs_clarification is True


def test_enough_evidence_is_sufficient_with_ids():
    verdict = SufficiencyJudge().judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.SUFFICIENT
    assert set(verdict.supported_by) == {"c1", "c2"}
    assert verdict.decided_by == "rule"
    assert verdict.confidence > 0.6


def test_single_evidence_is_insufficient():
    verdict = SufficiencyJudge().judge("年假规定", SUFFICIENT_EVIDENCE[:1])
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.supported_by == ["c1"]


def test_low_score_evidence_is_not_usable():
    verdict = SufficiencyJudge().judge("年假规定", [_ev("c1", 0.2, "无关内容")])
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.supported_by == []


def test_required_info_gap_is_reported():
    verdict = SufficiencyJudge().judge(
        "年假规定", SUFFICIENT_EVIDENCE,
        required_info=["年假规定", "加班补贴标准"])
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.missing_info == ["加班补贴标准"]


def test_numeric_conflict_is_detected():
    verdict = SufficiencyJudge().judge("A型配件库存", CONFLICT_EVIDENCE)
    assert verdict.state is EvidenceState.CONFLICTING
    assert len(verdict.conflicts) == 1
    conflict = verdict.conflicts[0]
    assert conflict.fact == "A型配件库存"
    assert set(conflict.values) == {"12件", "5件"}
    assert set(conflict.evidence_ids) == {"c1", "c2"}


def test_verdict_dict_uses_evidence_state_key():
    payload = SufficiencyJudge().judge("年假规定", SUFFICIENT_EVIDENCE).to_dict()
    assert payload["evidence_state"] == "sufficient"
    assert {"supported_by", "missing_info", "conflicts", "abstain",
            "needs_clarification", "decided_by"} <= set(payload)


# ---------------------------------------------------------------- LLM 叠加
def test_llm_can_only_tighten_not_lift():
    """规则已判充分时，LLM 说充分不改口径（decided_by 仍为 rule）"""
    judge = SufficiencyJudge(
        llm_callable=lambda p: json.dumps({"state": "SUFFICIENT",
                                           "supported_by": ["c1", "c2"]}))
    verdict = judge.judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.SUFFICIENT
    assert verdict.decided_by == "rule"


def test_llm_can_downgrade_to_insufficient():
    judge = SufficiencyJudge(
        llm_callable=lambda p: json.dumps({
            "state": "INSUFFICIENT",
            "supported_by": ["c1"],
            "missing_info": ["审批时限"],
            "reasoning": "证据未覆盖审批时限",
        }))
    verdict = judge.judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.INSUFFICIENT
    assert verdict.decided_by == "llm"
    assert verdict.missing_info == ["审批时限"]


def test_llm_citation_to_unknown_evidence_is_discarded():
    judge = SufficiencyJudge(
        llm_callable=lambda p: json.dumps({
            "state": "CONFLICTING",
            "supported_by": ["ghost-99"],
            "conflicts": [{"fact": "库存", "values": ["1", "2"], "evidence_ids": ["ghost-99"]}],
        }))
    verdict = judge.judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.SUFFICIENT
    assert verdict.decided_by == "rule"


def test_llm_failure_falls_back_to_rule():
    def boom(_prompt):
        raise RuntimeError("judge 服务不可用")

    verdict = SufficiencyJudge(llm_callable=boom).judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.SUFFICIENT
    assert verdict.decided_by == "rule"


def test_llm_garbage_output_is_ignored():
    verdict = SufficiencyJudge(
        llm_callable=lambda p: "我认为证据大概够了吧"
    ).judge("年假规定", SUFFICIENT_EVIDENCE)
    assert verdict.state is EvidenceState.SUFFICIENT


# ---------------------------------------------------------------- Validator 接入
def test_validator_emits_tristate_fields():
    state = {"query": "年假规定", "evidence_chunks": SUFFICIENT_EVIDENCE,
             "tool_results": [], "iteration_count": 1}
    validation = ValidatorAgent().validate(state)["validation_result"]
    assert validation["evidence_state"] == "sufficient"
    assert validation["passed"] is True
    assert validation["sufficiency"] is True
    assert validation["needs_retrieval"] is False
    assert validation["supported_by"] == ["c1", "c2"]


def test_validator_marks_conflict_and_targeted_queries():
    state = {"query": "A 型配件库存", "evidence_chunks": CONFLICT_EVIDENCE,
             "tool_results": [], "iteration_count": 1}
    validation = ValidatorAgent().validate(state)["validation_result"]
    assert validation["evidence_state"] == "conflicting"
    assert validation["needs_conflict_resolution"] is True
    assert validation["needs_retrieval"] is True
    assert validation["passed"] is False
    assert validation["targeted_queries"]


def test_validator_abstain_stops_retrieval():
    config = AgentConfig(max_iterations=2)
    state = {"query": "不存在的知识", "evidence_chunks": [], "tool_results": [],
             "iteration_count": 2}
    validation = ValidatorAgent(config=config).validate(state)["validation_result"]
    assert validation["abstain"] is True
    assert validation["needs_retrieval"] is False
    assert validation["needs_clarification"] is True


def test_validator_keeps_tool_requirement():
    state = {"query": "查询配件库存", "evidence_chunks": [], "tool_results": [],
             "iteration_count": 0}
    validation = ValidatorAgent().validate(state)["validation_result"]
    assert validation["needs_tool"] is True


def test_tristate_from_crag_mapping():
    sufficient = ValidatorAgent.tristate_from_crag("年假规定", "relevant", 0.85, [])
    assert sufficient["evidence_state"] == "sufficient"
    assert sufficient["sufficiency"] is True
    assert sufficient["decided_by"] == "crag"

    partial = ValidatorAgent.tristate_from_crag("年假规定", "partial", 0.4, [])
    assert partial["evidence_state"] == "insufficient"
    assert partial["targeted_queries"] == []

    conflicting = ValidatorAgent.tristate_from_crag(
        "配件库存", "relevant", 0.9,
        [{"fact": "库存", "values": ["12", "5"], "evidence_ids": ["c1", "c2"]}])
    assert conflicting["evidence_state"] == "conflicting"
    assert conflicting["needs_conflict_resolution"] is True
    assert any("库存" in q for q in conflicting["targeted_queries"])


# ---------------------------------------------------------------- 路由 / 生成
def test_supervisor_routes_conflict_to_retrieve():
    supervisor = SupervisorAgent(AgentConfig())
    state = {"query": "A 型配件库存", "iteration_count": 1,
             "validation_result": {"passed": False, "evidence_state": "conflicting",
                                   "needs_retrieval": True, "conflicts": [
                                       {"fact": "库存", "values": ["12", "5"]}]}}
    assert supervisor.route(state) == "retrieve"


def test_supervisor_stops_on_abstain():
    supervisor = SupervisorAgent(AgentConfig())
    state = {"query": "不存在的知识", "iteration_count": 1,
             "validation_result": {"passed": False, "abstain": True,
                                   "needs_retrieval": False, "missing_info": ["X"]}}
    assert supervisor.route(state) == "synthesize"


def test_synthesizer_abstains_instead_of_hallucinating():
    synthesizer = SynthesizerAgent()
    fused = {"sources": [{"type": "retrieval", "content": "沾边但不够的证据",
                          "score": 0.3, "document_name": "旧手册"}],
             "conflicts": []}
    answer = synthesizer._generate_answer(
        "加班补贴标准", fused,
        {"abstain": True, "missing_info": ["加班补贴计算口径"]})
    assert "不足以支撑可靠回答" in answer
    assert "加班补贴计算口径" in answer
    assert "旧手册" not in answer


def test_synthesizer_flags_unresolved_conflict():
    synthesizer = SynthesizerAgent()
    fused = {"sources": [{"type": "retrieval", "content": "库存 12 件", "score": 0.9,
                          "document_name": "台账A"}],
             "conflicts": [{"source_a": "台账A", "source_b": "台账B", "similarity": 0.5}]}
    answer = synthesizer._generate_answer(
        "A 型配件库存", fused,
        {"evidence_state": "conflicting",
         "conflicts": [{"fact": "A型配件库存", "values": ["12件", "5件"]}],
         "hallucination_risk": "low"})
    assert "互相矛盾" in answer
    assert "A型配件库存" in answer


# ---------------------------------------------------------------- 定向补检
def test_orchestrator_merges_targeted_queries():
    from agentic_rag.agent.agentic_rag import AgenticRAGOrchestrator
    orchestrator = AgenticRAGOrchestrator.__new__(AgenticRAGOrchestrator)
    state = {"rewritten_queries": ["A 型配件库存"]}
    orchestrator._apply_targeted_queries(state, {
        "targeted_queries": ["A型配件库存 最新 规定 权威", "A 型配件库存 补货阈值"]})
    assert state["rewritten_queries"] == [
        "A 型配件库存", "A型配件库存 最新 规定 权威", "A 型配件库存 补货阈值"]


def test_apply_targeted_queries_is_idempotent():
    from agentic_rag.agent.agentic_rag import AgenticRAGOrchestrator
    orchestrator = AgenticRAGOrchestrator.__new__(AgenticRAGOrchestrator)
    state = {"rewritten_queries": ["Q 补货阈值"]}
    orchestrator._apply_targeted_queries(state, {"targeted_queries": ["Q 补货阈值"]})
    assert state["rewritten_queries"] == ["Q 补货阈值"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
