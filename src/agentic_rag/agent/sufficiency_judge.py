"""证据充分性三态判定（Sufficiency Judge）

面试考点：「Agent 怎么决定继续检索还是已经可以停止并给出答案？」
生产级答法不是"让 LLM 判断"，而是给它一个**有结构的判定输出**：

    SUFFICIENT   证据够 → 停止检索，进入作答
    INSUFFICIENT 证据不够 → 带着 missing_info 定向补检
    CONFLICTING  多来源对同一事实矛盾 → 触发冲突消解（不直接作答）

判定依据必须显式列出（哪几条证据支持、缺什么信息），并遵守两条纪律：
1. **规则优先**：无证据 / 覆盖缺失 / 数值冲突这三类由规则直接定论，
   LLM 只能在规则之上"更严"，不能把结论"抬升"为 SUFFICIENT（fail-closed）。
2. **降级有出口**：INSUFFICIENT 且已到轮次上限 → 不硬答，转 ABSTAIN/向用户澄清
   （与"无证据不答"是同一套哲学的两个出口）。

可量化产出：`SUFFICIENT` 判定准确率（人工标注对齐）、平均检索轮次分布。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 数值抽取：`库存 12 件` / `剩余量: 12` / `12天`
_NUMBER_RE = re.compile(r"(?P<label>[\u4e00-\u9fa5A-Za-z_]{2,12})[^\d%]{0,6}(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[%\u4e00-\u9fa5]{0,4})")


class EvidenceState(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


@dataclass
class Conflict:
    """同一事实上的多来源矛盾"""

    fact: str
    values: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"fact": self.fact, "values": self.values, "evidence_ids": self.evidence_ids}


@dataclass
class SufficiencyVerdict:
    """三态判定结果（结构化、可审计、可标注对齐）"""

    state: EvidenceState
    supported_by: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    confidence: float = 0.0
    abstain: bool = False                 # 无证据 → 显式降级不答
    needs_clarification: bool = False      # 低置信度/补检到上限 → 二次确认用户
    decided_by: str = "rule"               # rule / llm
    reasoning: str = ""

    @property
    def is_sufficient(self) -> bool:
        return self.state is EvidenceState.SUFFICIENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_state": self.state.value,
            "supported_by": self.supported_by,
            "missing_info": self.missing_info,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "confidence": round(self.confidence, 4),
            "abstain": self.abstain,
            "needs_clarification": self.needs_clarification,
            "decided_by": self.decided_by,
            "reasoning": self.reasoning,
        }


class SufficiencyJudge:
    """证据充分性判定器（规则骨架 + 可选 LLM 语义增强）

    Args:
        llm_callable: `(prompt: str) -> str`，返回三态 JSON；None 时纯规则判定。
        min_evidence: 判定充分所需的最少可用证据条数
        relevance_threshold: 单条证据的最小可用相关性
        coverage: 需要被覆盖的子问题/关键信息列表（多跳场景由 Planner 供给）
    """

    PROMPT_TEMPLATE = """你是检索质量审计员。判断当前证据是否足以回答用户问题。

【用户问题】
{query}

【需要覆盖的关键信息/子问题】
{required}

【已有证据（含 ID 与相关性）】
{evidence}

只输出 JSON：
{{"state": "SUFFICIENT|INSUFFICIENT|CONFLICTING",
 "supported_by": ["证据ID"],
 "missing_info": ["仍缺什么"],
 "conflicts": [{{"fact": "哪个事实", "values": ["取值A","取值B"], "evidence_ids": ["ID"]}}],
 "reasoning": "简述判定依据"}}

规则：证据缺失或明显对不上问题必须 INSUFFICIENT；
同一事实出现不同取值必须 CONFLICTING；不要因为"已经检索了几轮"而判 SUFFICIENT。
"""

    def __init__(self, llm_callable=None, min_evidence: int = 2,
                 relevance_threshold: float = 0.6):
        self._call = llm_callable
        self.min_evidence = int(min_evidence)
        self.relevance_threshold = float(relevance_threshold)

    # ------------------------------------------------------------------
    def judge(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        required_info: Optional[List[str]] = None,
        iteration: int = 0,
        max_iterations: int = 3,
    ) -> SufficiencyVerdict:
        required = list(required_info or [])
        usable = [e for e in (evidence or []) if self._relevance(e) >= self.relevance_threshold]
        evidence_ids = [self._eid(e, i) for i, e in enumerate(evidence or [])]

        # 规则 1：零可用证据 → 不足，且到上限直接弃答（不编造）
        if not usable:
            return SufficiencyVerdict(
                state=EvidenceState.INSUFFICIENT,
                missing_info=required or [query[:60]],
                abstain=iteration >= max_iterations,
                needs_clarification=iteration >= max_iterations,
                decided_by="rule",
                reasoning=f"可用证据为 0（阈值 {self.relevance_threshold}）",
            )

        # 规则 2：关键信息覆盖缺口
        missing = [r for r in required if not self._covered(r, usable)]
        conflicts = self._detect_numeric_conflicts(usable)

        if conflicts:
            return SufficiencyVerdict(
                state=EvidenceState.CONFLICTING,
                supported_by=[self._eid(e, i) for i, e in enumerate(evidence or []) if e in usable],
                missing_info=missing,
                conflicts=conflicts,
                confidence=0.4,
                decided_by="rule",
                reasoning=f"同一事实存在矛盾取值：{[c.fact for c in conflicts]}",
            )

        enough_count = len(usable) >= self.min_evidence
        if not enough_count or missing:
            at_limit = iteration >= max_iterations
            return SufficiencyVerdict(
                state=EvidenceState.INSUFFICIENT,
                supported_by=[self._eid(e, i) for i, e in enumerate(evidence or []) if e in usable],
                missing_info=missing or ["更多可交叉验证的证据"],
                confidence=self._confidence(usable, missing),
                abstain=at_limit and bool(missing),
                needs_clarification=at_limit,
                decided_by="rule",
                reasoning=(f"可用证据 {len(usable)}/{self.min_evidence}，"
                           f"未覆盖 {missing}" if missing else
                           f"可用证据仅 {len(usable)} 条，不足 {self.min_evidence} 条"),
            )

        # 规则已判充分：LLM 只可加严，不可反向抬升（fail-closed）
        llm_verdict = self._judge_with_llm(query, evidence, usable, required, evidence_ids)
        if llm_verdict is None:
            return SufficiencyVerdict(
                state=EvidenceState.SUFFICIENT,
                supported_by=[self._eid(e, i) for i, e in enumerate(evidence or []) if e in usable],
                confidence=self._confidence(usable, []),
                decided_by="rule",
                reasoning="规则判定：证据数量与关键信息覆盖均达标",
            )
        return llm_verdict

    # ------------------------------------------------------------------
    def _judge_with_llm(self, query, evidence, usable, required, evidence_ids):
        if self._call is None:
            return None
        prompt = self.PROMPT_TEMPLATE.format(
            query=query,
            required=json.dumps(required, ensure_ascii=False),
            evidence=json.dumps(
                [{"id": self._eid(e, i), "relevance": round(self._relevance(e), 3),
                  "content": str(e.get("content", ""))[:300]}
                 for i, e in enumerate(usable)],
                ensure_ascii=False),
        )
        try:
            raw = self._call(prompt)
        except Exception as e:
            logger.warning(f"[SufficiencyJudge] LLM 判定失败，保留规则结论: {e}")
            return None
        data = _loads(raw)
        if not data:
            return None
        try:
            state = EvidenceState(str(data.get("state", "")).strip().lower())
        except ValueError:
            return None
        supported = [str(x) for x in (data.get("supported_by") or [])]
        # 依据必须可核查：引用了不存在的证据 ID → 作废 LLM 结论
        if state is not EvidenceState.INSUFFICIENT and supported and \
                not set(supported) <= set(evidence_ids):
            logger.info("[SufficiencyJudge] LLM 引用的证据 ID 不存在，作废其结论")
            return None
        conflicts = [
            Conflict(fact=str(c.get("fact", "")), values=[str(v) for v in (c.get("values") or [])],
                     evidence_ids=[str(x) for x in (c.get("evidence_ids") or [])])
            for c in (data.get("conflicts") or []) if isinstance(c, dict)
        ]
        missing = [str(x) for x in (data.get("missing_info") or [])]
        # 规则已判充分时，LLM 只能"更严"（INSUFFICIENT / CONFLICTING）
        if state is EvidenceState.SUFFICIENT:
            return None
        return SufficiencyVerdict(
            state=state,
            supported_by=supported or [self._eid(e, i) for i, e in enumerate(evidence) if e in usable],
            missing_info=missing,
            conflicts=conflicts,
            confidence=0.5,
            decided_by="llm",
            reasoning=str(data.get("reasoning", ""))[:300],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _eid(e: Dict[str, Any], idx: int) -> str:
        return str(e.get("id") or e.get("chunk_id") or e.get("document_id") or f"ev_{idx}")

    @staticmethod
    def _relevance(e: Dict[str, Any]) -> float:
        score = e.get("score", e.get("relevance", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            return 0.0
        return score if score <= 1.0 else score / 100.0

    @staticmethod
    def _covered(info: str, usable: List[Dict[str, Any]]) -> bool:
        key = re.sub(r"\s+", "", str(info))
        if not key:
            return True
        for e in usable:
            content = re.sub(r"\s+", "", str(e.get("content", "")))
            if key in content:
                return True
            # 字符重叠兜底（与 ValidatorAgent 的轻量口径一致）
            if key and content and len(set(key) & set(content)) / len(set(key)) >= 0.8:
                return True
        return False

    @staticmethod
    def _detect_numeric_conflicts(usable: List[Dict[str, Any]]) -> List[Conflict]:
        """同一标签事实出现不同数值 → 冲突（如 A 文档"库存 12 件"、B 文档"库存 5 件"）"""
        buckets: Dict[str, Dict[str, List[str]]] = {}
        for i, e in enumerate(usable):
            eid = SufficiencyJudge._eid(e, i)
            for m in _NUMBER_RE.finditer(str(e.get("content", ""))):
                label = m.group("label")
                value = f"{m.group('num')}{(m.group('unit') or '').strip()}"
                b = buckets.setdefault(label, {"values": {}, "ids": {}})
                b["values"].setdefault(value, []).append(eid)
        conflicts: List[Conflict] = []
        for label, b in buckets.items():
            if len(b["values"]) > 1:
                conflicts.append(Conflict(
                    fact=label,
                    values=sorted(b["values"].keys()),
                    evidence_ids=sorted({x for v in b["values"].values() for x in v}),
                ))
        return conflicts

    @staticmethod
    def _confidence(usable: List[Dict[str, Any]], missing: List[str]) -> float:
        avg = sum(SufficiencyJudge._relevance(e) for e in usable) / max(1, len(usable))
        penalty = min(0.4, 0.15 * len(missing))
        return max(0.0, min(1.0, avg - penalty))


def _loads(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None
