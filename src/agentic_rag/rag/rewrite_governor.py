"""查询改写治理（Rewrite Governor）：改写回退 + 命中率/意图偏移归因

面试考点：「Query 改写一定能变好吗？」——不能。改写是有代价的赌博：
- 同义词/HyDE/分解都可能把意图带偏（drift），召回一堆语义相邻但答非所问的文档；
- 改写失败时如果继续用改写串检索，等于用更差的 query 污染了整条 RRF 融合结果。

本模块提供两件生产级机制：

1. **回退原 Query（fail-back）**：`RewriteGovernor.measure()` 会用同一套检索函数
   分别试跑原始查询与各改写变体，只有当某个变体的命中质量确实优于原查询时才采纳；
   否则判定 `fallback_original`，调用方退回原查询结果，避免无收益的改写稀释排序。
   CRAG 的改写节点则用 `guard_rewrite()` 做单跳护栏：LLM 改写若意图偏移过大，
   或改写后与当前查询等价（原地打转），直接回退原查询并终止改写循环。

2. **归因（attribution）**：每条变体带上策略标签（synonym / multi_query / stepback /
   hyde），累计「计划数 / 检索数 / 命中数 / 被采纳数 / 平均偏移 / 命中率」，
   可直接回答"哪类改写真正提升了召回、哪类在帮倒忙"。

纪律：判定只依赖可核查的检索信号（命中条数、top score），不依赖 LLM 自评。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(r"[一-龥]")

STRATEGY_ORIGINAL = "original"
STRATEGY_SYNONYM = "synonym"
STRATEGY_MULTI_QUERY = "multi_query"
STRATEGY_STEPBACK = "stepback"
STRATEGY_HYDE = "hyde"


def _tokens(text: str) -> set:
    """中英混排 token 集合：ASCII 词 + 单字（含 CJK 字符级 bigram 兜底）"""
    text = text or ""
    tokens = {w.lower() for w in _ASCII_WORD_RE.findall(text)}
    tokens.update(_CJK_RE.findall(text))
    return tokens


def estimate_intent_drift(original: str, variant: str) -> float:
    """意图偏移度 ∈ [0,1]：原查询关键信息在变体中丢失的比例

    以「原查询被变体覆盖」为口径（而非对称 Jaccard），这样 HyDE 这类
    长文本改写不会因为篇幅而被误判为高偏移。
    """
    src = _tokens(original)
    if not src:
        return 0.0
    dst = _tokens(variant)
    covered = len(src & dst) / len(src)
    return round(1.0 - covered, 4)


@dataclass
class QueryVariant:
    """带策略标签的改写变体"""

    query: str
    strategy: str = STRATEGY_ORIGINAL

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "strategy": self.strategy}


@dataclass
class VariantOutcome:
    """单个变体的试跑结果（可核查、可归因）"""

    query: str
    strategy: str
    drift: float = 0.0
    hit_count: int = 0
    top_score: float = 0.0
    pruned: bool = False
    pruned_reason: str = ""
    adopted: bool = False

    @property
    def quality(self) -> Tuple[int, float]:
        return (self.hit_count, round(self.top_score, 4))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "intent_drift": self.drift,
            "hit_count": self.hit_count,
            "top_score": round(self.top_score, 4),
            "pruned": self.pruned,
            "pruned_reason": self.pruned_reason,
            "adopted": self.adopted,
        }


@dataclass
class RewriteReport:
    """一次改写的完整归因记录"""

    original: str
    outcomes: List[VariantOutcome] = field(default_factory=list)
    decision: str = "no_rewrite"      # no_rewrite / adopt_rewrites / fallback_original
    adopted_query: str = ""
    winning_strategy: str = STRATEGY_ORIGINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "decision": self.decision,
            "adopted_query": self.adopted_query,
            "winning_strategy": self.winning_strategy,
            "variants": [o.to_dict() for o in self.outcomes],
        }


class RewriteGovernor:
    """改写治理器：试跑 → 择优 → 回退 → 归因

    Args:
        max_drift: 意图偏移上限，超过则该变体直接剪枝（不参与检索）
        min_hits: 采纳一个变体所需的最少命中条数
        min_top_score: 采纳一个变体所需的最小 top 相关性分
        margin: 变体需比原查询高出的分数余量（避免持平即改写）
    """

    def __init__(self, max_drift: float = 0.72, min_hits: int = 1,
                 min_top_score: float = 0.0, margin: float = 0.0):
        self.max_drift = float(max_drift)
        self.min_hits = int(min_hits)
        self.min_top_score = float(min_top_score)
        self.margin = float(margin)
        self._counters: Dict[str, Dict[str, float]] = {}
        self._queries = 0
        self._fallbacks = 0
        self._reports: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def measure(self, original: str, variants: List[QueryVariant],
                retrieval_fn: Callable[[str], List[Dict[str, Any]]]) -> RewriteReport:
        """用同一检索函数试跑原查询与变体，择优或回退"""
        self._queries += 1
        outcomes: List[VariantOutcome] = [self._probe(original, STRATEGY_ORIGINAL, retrieval_fn)]
        self._bump(STRATEGY_ORIGINAL, "planned")
        self._bump(STRATEGY_ORIGINAL, "retrieved")

        candidates = [v for v in variants or [] if v.query and v.query != original]
        for variant in candidates:
            drift = estimate_intent_drift(original, variant.query)
            outcome = VariantOutcome(query=variant.query, strategy=variant.strategy,
                                     drift=drift)
            self._bump(outcome.strategy, "planned")
            self._bump(outcome.strategy, "drift", drift)
            if drift > self.max_drift:
                outcome.pruned = True
                outcome.pruned_reason = f"意图偏移 {drift} 超过上限 {self.max_drift}"
                outcomes.append(outcome)
                self._bump(outcome.strategy, "pruned")
                logger.info(f"[RewriteGovernor] 剪枝变体: {variant.query[:40]} ({outcome.pruned_reason})")
                continue
            probed = self._probe(variant.query, variant.strategy, retrieval_fn)
            probed.drift = drift
            outcomes.append(probed)
            self._bump(outcome.strategy, "retrieved")
            self._bump(outcome.strategy, "hits", probed.hit_count)
            if probed.hit_count >= self.min_hits:
                self._bump(outcome.strategy, "hit")

        base = outcomes[0]
        winners = [o for o in outcomes[1:] if self._beats(o, base)]
        report = RewriteReport(original=original, outcomes=outcomes)

        if not candidates:
            report.decision = "no_rewrite"
            report.adopted_query = original
        elif winners:
            best = max(winners, key=lambda o: (o.top_score, o.hit_count))
            best.adopted = True
            report.decision = "adopt_rewrites"
            report.adopted_query = best.query
            report.winning_strategy = best.strategy
            self._bump(best.strategy, "adopted")
        else:
            report.decision = "fallback_original"
            report.adopted_query = original
            base.adopted = True
            self._fallbacks += 1
            self._bump(STRATEGY_ORIGINAL, "adopted")

        self._reports.append(report.to_dict())
        logger.info(f"[RewriteGovernor] {original[:30]} -> {report.decision} "
                    f"(winning={report.winning_strategy})")
        return report

    # ------------------------------------------------------------------
    def observe(self, original: str, variants: List[QueryVariant],
                results: List[Dict[str, Any]], planned: Optional[List[QueryVariant]] = None,
                source_key: str = "_source_query",
                score_keys: Tuple[str, ...] = ("combined_score", "vector_score", "score")
                ) -> Dict[str, Any]:
        """从真实检索结果做归因：谁的改写最终进了结果集

        与 `measure()`（试跑择优）不同，本方法不额外发起检索，只读取检索器在每条
        结果上打的 `_source_query` 标签，统计各策略的实际贡献与命中率。
        若所有改写变体对最终结果 0 贡献，则判定 `fallback_original`（改写无收益）。
        """
        self._queries += 1
        strategy_of = {v.query: v.strategy for v in variants}
        contributions: Dict[str, int] = {v.strategy: 0 for v in variants}
        top_scores: Dict[str, float] = {v.strategy: 0.0 for v in variants}

        for result in results or []:
            strategy = strategy_of.get(result.get(source_key, ""), STRATEGY_ORIGINAL)
            contributions[strategy] = contributions.get(strategy, 0) + 1
            score = 0.0
            for key in score_keys:
                value = result.get(key)
                if value:
                    try:
                        score = float(value)
                    except (TypeError, ValueError):
                        score = 0.0
                    break
            top_scores[strategy] = max(top_scores.get(strategy, 0.0), score)

        for variant in (planned or variants):
            self._bump(variant.strategy, "planned")
        for variant in variants:
            self._bump(variant.strategy, "retrieved")
            self._bump(variant.strategy, "drift",
                       estimate_intent_drift(original, variant.query))
            if contributions.get(variant.strategy, 0) >= self.min_hits:
                self._bump(variant.strategy, "hit")
                self._bump(variant.strategy, "hits", contributions[variant.strategy])

        rewrite_contributed = sum(v for s, v in contributions.items()
                                  if s != STRATEGY_ORIGINAL)
        best_strategy = max(contributions.items(), key=lambda kv: kv[1])[0] \
            if contributions else STRATEGY_ORIGINAL
        decision = "adopt_rewrites" if rewrite_contributed else "fallback_original"
        if decision == "fallback_original":
            self._fallbacks += 1
        self._bump(best_strategy, "adopted")
        adopted_query = original
        for variant in variants:
            if variant.strategy == best_strategy:
                adopted_query = variant.query
                break
        if decision == "fallback_original":
            adopted_query = original

        outcomes = [
            VariantOutcome(
                query=v.query, strategy=v.strategy,
                drift=estimate_intent_drift(original, v.query),
                hit_count=contributions.get(v.strategy, 0),
                top_score=top_scores.get(v.strategy, 0.0),
                adopted=(v.strategy == best_strategy and rewrite_contributed > 0),
            )
            for v in variants
        ]
        report = RewriteReport(original=original, outcomes=outcomes, decision=decision,
                               adopted_query=adopted_query,
                               winning_strategy=best_strategy).to_dict()
        self._reports.append(report)
        logger.info(f"[RewriteGovernor] 归因 {original[:24]}: {decision} "
                    f"(贡献分布={contributions})")
        return report

    # ------------------------------------------------------------------
    def guard_rewrite(self, original: str, current: str,
                      rewritten: str) -> Tuple[str, Dict[str, Any]]:
        """单跳改写护栏（CRAG 用）：偏移过大或原地打转则回退原查询"""
        text = (rewritten or "").strip()
        drift = estimate_intent_drift(original, text)

        if not text:
            return original, {"action": "fallback_original", "reason": "改写结果为空",
                              "intent_drift": drift}
        if text == (current or "").strip():
            return original, {"action": "fallback_original",
                              "reason": "改写与当前查询等价，继续改写无收益",
                              "intent_drift": drift}
        if drift > self.max_drift:
            return original, {"action": "fallback_original",
                              "reason": f"意图偏移 {drift} 超过上限 {self.max_drift}",
                              "intent_drift": drift}
        return text, {"action": "accept_rewrite", "intent_drift": drift}

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """累计归因指标：各策略命中率 / 采纳数 / 平均偏移 + 整体回退率"""
        by_strategy: Dict[str, Dict[str, Any]] = {}
        for strategy, c in self._counters.items():
            retrieved = c.get("retrieved", 0)
            planned = c.get("planned", 0)
            by_strategy[strategy] = {
                "planned": planned,
                "retrieved": retrieved,
                "pruned": c.get("pruned", 0),
                "hit": c.get("hit", 0),
                "adopted": c.get("adopted", 0),
                "total_hits": c.get("hits", 0),
                "hit_rate": round(c.get("hit", 0) / retrieved, 4) if retrieved else 0.0,
                "avg_hits": round(c.get("hits", 0) / retrieved, 2) if retrieved else 0.0,
                "avg_drift": round(c.get("drift", 0) / retrieved, 4) if retrieved else 0.0,
            }
        return {
            "queries": self._queries,
            "fallbacks": self._fallbacks,
            "fallback_rate": round(self._fallbacks / self._queries, 4) if self._queries else 0.0,
            "by_strategy": by_strategy,
        }

    def attribution(self) -> List[Dict[str, Any]]:
        """逐次改写的归因明细（供离线分析「哪类改写在帮倒忙」）"""
        return list(self._reports)

    def reset(self):
        self._counters = {}
        self._queries = 0
        self._fallbacks = 0
        self._reports = []

    # ------------------------------------------------------------------
    def _probe(self, query: str, strategy: str,
               retrieval_fn: Callable[[str], List[Dict[str, Any]]]) -> VariantOutcome:
        outcome = VariantOutcome(query=query, strategy=strategy)
        try:
            results = retrieval_fn(query) or []
        except Exception as e:  # 检索失败不能拖垮主链路，记为 0 命中
            logger.warning(f"[RewriteGovernor] 试跑检索失败({query[:30]}): {e}")
            return outcome
        outcome.hit_count = len(results)
        scores = []
        for r in results:
            try:
                scores.append(float(r.get("score", 0) or 0))
            except (TypeError, ValueError):
                continue
        outcome.top_score = max(scores) if scores else 0.0
        return outcome

    def _beats(self, variant: VariantOutcome, base: VariantOutcome) -> bool:
        if variant.hit_count < self.min_hits:
            return False
        if variant.top_score < self.min_top_score:
            return False
        if variant.hit_count == 0 and base.hit_count == 0:
            return False
        if variant.hit_count < base.hit_count:
            return False
        return variant.top_score >= base.top_score + self.margin

    def _bump(self, strategy: str, key: str, amount: float = 1):
        c = self._counters.setdefault(strategy, {})
        c[key] = c.get(key, 0) + amount


# 进程级归因聚合器（纯内存计数，无外部依赖）
rewrite_governor = RewriteGovernor()
