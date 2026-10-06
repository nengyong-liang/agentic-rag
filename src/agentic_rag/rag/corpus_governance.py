"""知识库保鲜 / Corpus 侧治理（docs/KNOWN-ISSUES.md §12.3 缺口 7）

`rag/drift_controller.py` 只管 Query 侧一致性与分数阈值，`utils/quality_gate.py`
只管**入仓那一刻**的单切片门禁（空/过短/特殊字符/SHA 精确重复）。两者都不回答
"知识库跑半年后变脏了怎么办"——本模块补 Corpus 侧的三件事：

1. **增量索引 diff**：用 `chunk_id + 内容哈希` 对比"索引现状"与"本轮解析结果"，
   产出 新增/变更/删除/未变 四类清单，重导入只碰变化部分（全量重建既慢又会让
   向量与 ES 短暂不一致），并显式列出需要下线的孤儿 chunk。
2. **过期 / 冲突 / 错误 chunk 发现**：
   - `expired`：有效期到期（`effective_until`）、被新版本取代（`version` /
     `is_latest`）、超过 TTL 未更新；
   - `conflict`：同一主题（`topic_key`/章节）下抽到的数值型事实不一致
     （如"年假 5 天" vs "年假 10 天"）——检索层无法自证谁对，必须人工裁决；
   - `duplicate`：SHA 精确重复之外的**近重复**（token Jaccard ≥ 阈值）；
   - `defect`：空文、缺标识、疑似截断（以逗号/顿号/冒号结尾的硬切）。
3. **处置建议 + 检索侧剔除**：治理结果落成可审计的 finding（规则号 + 口径 +
   关联 chunk），检索链路按 `suppress/drop_duplicate` 剔除不再召回的 chunk。

安全边界：本模块**只读、只标记、只出建议**，不删任何数据；真正下线的动作由
调用方（导入路由 / 离线任务）执行。所有输出按 chunk_id 稳定排序，保证可重跑。
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 口径常量
# ------------------------------------------------------------------

CATEGORY_EXPIRED = "expired"
CATEGORY_SUPERSEDED = "superseded"
CATEGORY_CONFLICT = "conflict"
CATEGORY_DUPLICATE = "duplicate"
CATEGORY_DEFECT = "defect"
GOVERNANCE_CATEGORIES = (CATEGORY_EXPIRED, CATEGORY_SUPERSEDED, CATEGORY_CONFLICT,
                         CATEGORY_DUPLICATE, CATEGORY_DEFECT)

ACTION_REVIEW = "review"                  # 只标记，人工裁决（冲突类默认走这条）
ACTION_SUPPRESS = "suppress"              # 检索层不再召回，索引里保留
ACTION_DROP_DUPLICATE = "drop_duplicate"  # 近重复的落后副本，建议下线（保留正本）
ACTION_REINDEX = "reindex"                # 切片本身有问题，需要重新切分/重新入库
GOVERNANCE_ACTIONS = (ACTION_REVIEW, ACTION_SUPPRESS, ACTION_DROP_DUPLICATE,
                      ACTION_REINDEX)

#: 动作 -> 计算口径（写进报告，不靠口头解释）
ACTION_RULES: Dict[str, str] = {
    ACTION_REVIEW: "仅标记不改数据：同主题数值/日期冲突需人工裁决谁为准",
    ACTION_SUPPRESS: "检索层剔除，索引保留：有效期到期或已被新版本取代",
    ACTION_DROP_DUPLICATE: "下线落后副本：内容近重复且更新时间更晚，正本继续可用",
    ACTION_REINDEX: "需要重切/重入库：空文、缺标识或疑似硬截断",
}

_ID_KEYS = ("chunk_id", "id")
_CONTENT_KEYS = ("content", "text")
_TOPIC_KEYS = ("topic_key", "policy_key", "section", "title")
_VERSION_KEYS = ("version", "version_no")
_UPDATED_KEYS = ("updated_at", "updated", "indexed_at", "processed_at", "published_at")
_EXPIRE_KEYS = ("effective_until", "valid_until", "expire_at", "expiry_date")

#: 数值型事实：数字 + 单位（政策类冲突绝大多数落在这里）
_NUMERIC_FACT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|％|个工作日|个小时|小时|分钟|个月|月|年|天|日|元|次|位|条|款|人)")
#: 疑似硬截断的结尾标点
_TRUNCATION_TAIL = tuple("，,、；;：:")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_SPECIAL_RE = re.compile(r"[\w\u4e00-\u9fff\s]", re.UNICODE)


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _first_present(source: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def as_date(value: Any) -> Optional[date]:
    """把 date/datetime/'2026-09-30'/'2026-09-30T10:00:00'/1700000000 归一成 date"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).date()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    for candidate in (text, text.replace("/", "-")[:10]):
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            continue
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def tokenize(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def special_char_ratio(text: str) -> float:
    """特殊字符占比（与入仓门禁同口径，用于复扫已入库语料）"""
    if not text:
        return 1.0
    return len(_SPECIAL_RE.sub("", text)) / len(text)


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """治理视图：把检索结果/切片/索引导出统一成同一种记录"""

    chunk_id: str
    content: str
    kb_id: str = ""
    topic_key: str = ""
    content_hash: str = ""
    version: Optional[int] = None
    updated_at: Optional[date] = None
    effective_until: Optional[date] = None
    is_latest: Optional[bool] = None
    tokens: Set[str] = field(default_factory=set, repr=False, compare=False)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def numeric_facts(self) -> Dict[str, Set[str]]:
        """unit -> 该 chunk 内出现的数值集合（同主题冲突判定用）"""
        facts: Dict[str, Set[str]] = {}
        for value, unit in _NUMERIC_FACT.findall(self.content or ""):
            facts.setdefault(unit.replace("％", "%"), set()).add(value)
        return facts


@dataclass
class GovernanceFinding:
    chunk_id: str
    category: str
    rule: str
    action: str
    detail: str
    related: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "category": self.category,
                "rule": self.rule, "action": self.action, "detail": self.detail,
                "related": list(self.related)}


@dataclass
class IncrementalPlan:
    """增量索引计划：重导入只处理这几类，不做全量重建"""

    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)

    @property
    def needs_work(self) -> bool:
        return self.changed_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"added": list(self.added), "updated": list(self.updated),
                "deleted": list(self.deleted), "unchanged": len(self.unchanged),
                "changed_count": self.changed_count,
                "full_rebuild_required": False}


# ------------------------------------------------------------------
# 治理器
# ------------------------------------------------------------------

class CorpusGovernor:
    """Corpus 侧治理器（只读扫描 + 处置建议）"""

    def __init__(self, as_of: Optional[Any] = None, ttl_days: int = 180,
                 duplicate_similarity: float = 0.9, min_chars: int = 8,
                 max_special_ratio: float = 0.6, max_pairwise_chunks: int = 4000):
        #: 判定基准日：所有时间判断都以它为锚，离线复算结果可重跑
        self.today = as_date(as_of) or date.today()
        self.ttl_days = ttl_days
        self.duplicate_similarity = duplicate_similarity
        self.min_chars = min_chars
        self.max_special_ratio = max_special_ratio
        #: 近重复是 O(n^2) 成对比较，超过就如实跳过（不静默降精度）
        self.max_pairwise_chunks = max_pairwise_chunks

    # -- 归一化 -------------------------------------------------
    def to_record(self, chunk: Dict[str, Any]) -> ChunkRecord:
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        lookup = {**chunk, **meta}

        chunk_id = str(_first_present(lookup, _ID_KEYS) or "")
        if not chunk_id:
            document_id = lookup.get("document_id")
            index = lookup.get("chunk_index")
            if document_id is not None:
                chunk_id = f"{document_id}_{index if index is not None else 0}"
        content = str(_first_present(lookup, _CONTENT_KEYS) or "")

        digest = str(meta.get("chunk_hash") or chunk.get("chunk_hash") or "")
        if digest.startswith("sha256:"):
            content_hash = digest
        else:
            content_hash = f"sha256:{_digest(content)}"

        version_raw = _first_present(lookup, _VERSION_KEYS)
        version = None
        if version_raw is not None:
            try:
                version = int(str(version_raw).lstrip("vV"))
            except ValueError:
                version = None

        latest_raw = lookup.get("is_latest")
        is_latest = latest_raw if isinstance(latest_raw, bool) else None

        topic = _first_present(lookup, _TOPIC_KEYS)
        return ChunkRecord(
            chunk_id=chunk_id,
            content=content,
            kb_id=str(lookup.get("kb_id") or lookup.get("tenant_id") or ""),
            topic_key=str(topic) if topic else "",
            content_hash=content_hash,
            version=version,
            updated_at=as_date(_first_present(lookup, _UPDATED_KEYS)),
            effective_until=as_date(_first_present(lookup, _EXPIRE_KEYS)),
            is_latest=is_latest,
            tokens=tokenize(content),
            raw=chunk,
        )

    def to_records(self, chunks: Iterable[Dict[str, Any]]) -> List[ChunkRecord]:
        return [self.to_record(c) for c in chunks]

    # -- ① 增量索引 diff ---------------------------------------
    def plan_incremental_update(
        self, indexed: Iterable[Dict[str, Any]], current: Iterable[Dict[str, Any]]
    ) -> IncrementalPlan:
        """对比索引现状与本轮解析结果

        `indexed` 为空表示首次建库，此时全部按 added 处理；未提供 `indexed`
        的场景（调用方拿不到索引清单）应改用 `scan()` 而不是假装做过 diff。
        """
        before = {r.chunk_id: r.content_hash for r in self.to_records(indexed)
                  if r.chunk_id}
        after = {r.chunk_id: r.content_hash for r in self.to_records(current)
                 if r.chunk_id}

        plan = IncrementalPlan(
            added=sorted(set(after) - set(before)),
            updated=sorted(i for i in set(after) & set(before) if after[i] != before[i]),
            deleted=sorted(set(before) - set(after)),
            unchanged=sorted(i for i in set(after) & set(before) if after[i] == before[i]),
        )
        logger.info(
            f"[保鲜] 增量计划：新增={len(plan.added)} 变更={len(plan.updated)} "
            f"下线={len(plan.deleted)} 未变={len(plan.unchanged)}"
        )
        return plan

    # -- ② 发现：过期 / 取代 / 冲突 / 近重复 / 缺陷 ------------
    def scan(self, chunks: Iterable[Dict[str, Any]]) -> "GovernanceReport":
        records = [r for r in self.to_records(chunks) if r.chunk_id]
        findings: List[GovernanceFinding] = []
        findings.extend(self._find_defects(records))
        findings.extend(self._find_expiry(records))
        findings.extend(self._find_superseded(records))
        conflicts = self._find_conflicts(records)
        findings.extend(conflicts)
        # 冲突对文本几乎同形，若按近重复下线其一会把矛盾证据悄悄抹掉——
        # 冲突必须留在 review 通道里由人裁决，判重阶段显式避让。
        protected = {tuple(sorted({f.chunk_id, c})) for f in conflicts for c in f.related}
        duplicates, skipped = self._find_near_duplicates(records, protected)
        findings.extend(duplicates)

        report = GovernanceReport(
            as_of=self.today.isoformat(),
            chunk_count=len(records),
            findings=sorted(findings, key=lambda f: (f.category, f.chunk_id, f.rule)),
            near_duplicate_scan_skipped=skipped,
        )
        logger.info(f"[保鲜] 扫描 {len(records)} 个 chunk，命中 {len(findings)} 条："
                    f"{report.counts_by_category()}")
        return report

    def _find_defects(self, records: List[ChunkRecord]) -> List[GovernanceFinding]:
        out = []
        for r in records:
            text = (r.content or "").strip()
            if not text:
                out.append(GovernanceFinding(r.chunk_id, CATEGORY_DEFECT, "D1/empty",
                                             ACTION_REINDEX, "内容为空，无法被检索命中"))
                continue
            if len(text) < self.min_chars:
                out.append(GovernanceFinding(r.chunk_id, CATEGORY_DEFECT, "D2/too_short",
                                             ACTION_REINDEX,
                                             f"内容仅 {len(text)} 字（阈值 {self.min_chars}）"))
            if special_char_ratio(text) > self.max_special_ratio:
                out.append(GovernanceFinding(r.chunk_id, CATEGORY_DEFECT, "D3/garbled",
                                             ACTION_REINDEX,
                                             "特殊字符占比过高，疑似解析乱码"))
            if text.endswith(_TRUNCATION_TAIL):
                out.append(GovernanceFinding(r.chunk_id, CATEGORY_DEFECT, "D4/truncated",
                                             ACTION_REINDEX,
                                             f"以「{text[-1]}」结尾，疑似切片硬截断"))
        return out

    def _find_expiry(self, records: List[ChunkRecord]) -> List[GovernanceFinding]:
        out = []
        deadline = self.today - timedelta(days=self.ttl_days)
        for r in records:
            if r.effective_until and r.effective_until < self.today:
                overdue = (self.today - r.effective_until).days
                out.append(GovernanceFinding(
                    r.chunk_id, CATEGORY_EXPIRED, "E1/effective_until", ACTION_SUPPRESS,
                    f"有效期 {r.effective_until.isoformat()} 已过 {overdue} 天"))
            if r.is_latest is False:
                out.append(GovernanceFinding(
                    r.chunk_id, CATEGORY_EXPIRED, "E2/is_latest_false", ACTION_SUPPRESS,
                    "文档标记为非最新版"))
            if r.updated_at and r.updated_at < deadline and r.effective_until is None:
                out.append(GovernanceFinding(
                    r.chunk_id, CATEGORY_EXPIRED, "E3/ttl", ACTION_REVIEW,
                    f"最后更新 {r.updated_at.isoformat()} 距今超过 {self.ttl_days} 天，"
                    f"需确认是否仍有效"))
        return out

    def _topic_groups(self, records: List[ChunkRecord]) -> Dict[str, List[ChunkRecord]]:
        groups: Dict[str, List[ChunkRecord]] = {}
        for r in records:
            if r.topic_key:
                groups.setdefault(f"{r.kb_id}\x1f{r.topic_key}", []).append(r)
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _find_superseded(self, records: List[ChunkRecord]) -> List[GovernanceFinding]:
        """同一主题内按 (version, updated_at, is_latest) 仲裁出正本，其余判为已被取代"""
        out = []
        for group in self._topic_groups(records).values():
            ranked = sorted(group, key=self._authority, reverse=True)
            winner, losers = ranked[0], ranked[1:]
            for loser in losers:
                if loser.version is not None and winner.version is not None \
                        and loser.version < winner.version:
                    rule, extra = "S1/version", f"版本 {loser.version} < {winner.version}"
                elif loser.is_latest is False:
                    rule, extra = "S2/is_latest", "自身标记非最新"
                else:
                    rule, extra = "S3/recency", "同主题存在更新的版本"
                out.append(GovernanceFinding(
                    loser.chunk_id, CATEGORY_SUPERSEDED, rule, ACTION_SUPPRESS,
                    f"{extra}，正本为 {winner.chunk_id}", related=[winner.chunk_id]))
        return out

    @staticmethod
    def _authority(r: ChunkRecord) -> Tuple[int, int, int, str]:
        return (1 if r.is_latest else 0,
                r.version if r.version is not None else -1,
                r.updated_at.toordinal() if r.updated_at else 0,
                r.chunk_id)

    def _find_conflicts(self, records: List[ChunkRecord]) -> List[GovernanceFinding]:
        """同主题下同一单位的数值不一致 → 冲突（只标记，交人工裁决）"""
        out = []
        for group in self._topic_groups(records).values():
            by_unit: Dict[str, Dict[str, List[str]]] = {}
            for r in group:
                for unit, values in r.numeric_facts().items():
                    for value in values:
                        by_unit.setdefault(unit, {}).setdefault(value, []).append(r.chunk_id)
            for unit, by_value in sorted(by_unit.items()):
                if len(by_value) < 2:
                    continue
                candidates = "; ".join(f"{value}←{sorted(v)}"
                                       for value, v in sorted(by_value.items()))
                for value, owners in sorted(by_value.items()):
                    others = sorted({c for c, v in by_value.items() for c in v
                                     if c not in owners})
                    out.append(GovernanceFinding(
                        owners[0], CATEGORY_CONFLICT, f"C1/{unit}", ACTION_REVIEW,
                        f"同主题「{unit}」取值冲突 -> {candidates}",
                        related=others))
                    # 同取值的其余 chunk 不重复开单，但要在关联里可见
                    for extra_owner in owners[1:]:
                        out.append(GovernanceFinding(
                            extra_owner, CATEGORY_CONFLICT, f"C1/{unit}", ACTION_REVIEW,
                            f"与 {owners[0]} 同取值 {value}（冲突组同上）",
                            related=sorted(set(owners) | set(others))))
        return _dedupe_findings(out)

    def _find_near_duplicates(
        self, records: List[ChunkRecord],
        protected: Optional[Set[Tuple[str, str]]] = None,
    ) -> Tuple[List[GovernanceFinding], bool]:
        """近重复：token 集合 Jaccard ≥ 阈值；正本 = 更新时间最早/ID 最小"""
        protected = protected or set()
        if len(records) > self.max_pairwise_chunks:
            logger.warning(f"[保鲜] chunk 数 {len(records)} 超过成对比较上限 "
                           f"{self.max_pairwise_chunks}，本轮跳过近重复扫描")
            return [], True

        by_hash: Dict[str, List[ChunkRecord]] = {}
        for r in records:
            if r.tokens:
                by_hash.setdefault(r.kb_id + "\x1f" + r.content_hash, []).append(r)
        out: List[GovernanceFinding] = []

        # 精确重复（同哈希）
        for group in by_hash.values():
            if len(group) < 2:
                continue
            canonical = _pick_canonical(group)
            for loser in group:
                if loser.chunk_id == canonical.chunk_id:
                    continue
                out.append(GovernanceFinding(
                    loser.chunk_id, CATEGORY_DUPLICATE, "P1/exact_hash",
                    ACTION_DROP_DUPLICATE,
                    f"内容与 {canonical.chunk_id} 完全相同", related=[canonical.chunk_id]))

        # 近重复（跨哈希成对比较）
        seen = {(f.chunk_id, f.related[0]) for f in out if f.related}
        pool = [r for r in records if r.tokens]
        for i, left in enumerate(pool):
            for right in pool[i + 1:]:
                if left.kb_id != right.kb_id or left.content_hash == right.content_hash:
                    continue
                if tuple(sorted((left.chunk_id, right.chunk_id))) in protected:
                    continue        # 冲突对不判重，矛盾证据必须留在人工裁决通道
                union = len(left.tokens | right.tokens)
                if union == 0:
                    continue
                similarity = len(left.tokens & right.tokens) / union
                if similarity < self.duplicate_similarity:
                    continue
                canonical, loser = ((left, right)
                                    if _rank_key(left) <= _rank_key(right)
                                    else (right, left))
                if (loser.chunk_id, canonical.chunk_id) in seen:
                    continue
                out.append(GovernanceFinding(
                    loser.chunk_id, CATEGORY_DUPLICATE, "P2/near_duplicate",
                    ACTION_DROP_DUPLICATE,
                    f"与 {canonical.chunk_id} 相似度 {similarity:.3f} ≥ "
                    f"{self.duplicate_similarity}", related=[canonical.chunk_id]))
        return _dedupe_findings(out), False


def _rank_key(r: ChunkRecord) -> Tuple[int, str]:
    return (r.updated_at.toordinal() if r.updated_at else 0, r.chunk_id)


def _pick_canonical(group: List[ChunkRecord]) -> ChunkRecord:
    return sorted(group, key=_rank_key)[0]


def _dedupe_findings(findings: List[GovernanceFinding]) -> List[GovernanceFinding]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[GovernanceFinding] = []
    for f in findings:
        key = (f.chunk_id, f.rule, f.action)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


@dataclass
class GovernanceReport:
    as_of: str
    chunk_count: int
    findings: List[GovernanceFinding] = field(default_factory=list)
    near_duplicate_scan_skipped: bool = False

    def counts_by_category(self) -> Dict[str, int]:
        counts = {c: 0 for c in GOVERNANCE_CATEGORIES}
        for f in self.findings:
            counts[f.category] = counts.get(f.category, 0) + 1
        return counts

    def counts_by_action(self) -> Dict[str, int]:
        counts = {a: 0 for a in GOVERNANCE_ACTIONS}
        for f in self.findings:
            counts[f.action] = counts.get(f.action, 0) + 1
        return counts

    def suppressed_ids(self) -> Set[str]:
        """检索层应剔除的 chunk：明确过期/被取代/重复副本"""
        return {f.chunk_id for f in self.findings
                if f.action in (ACTION_SUPPRESS, ACTION_DROP_DUPLICATE)}

    def review_ids(self) -> Set[str]:
        return {f.chunk_id for f in self.findings if f.action == ACTION_REVIEW}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "chunk_count": self.chunk_count,
            "finding_count": len(self.findings),
            "by_category": self.counts_by_category(),
            "by_action": self.counts_by_action(),
            "suppressed": sorted(self.suppressed_ids()),
            "needs_review": sorted(self.review_ids()),
            "action_rules": dict(ACTION_RULES),
            "near_duplicate_scan_skipped": self.near_duplicate_scan_skipped,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


# ------------------------------------------------------------------
# 检索链路接入点
# ------------------------------------------------------------------

_GOVERNOR = CorpusGovernor()


def chunk_id_of(result: Dict[str, Any]) -> str:
    """与治理器同口径地取 chunk 标识（检索结果 / 切片 / 索引导出通用）"""
    return _GOVERNOR.to_record(result).chunk_id


def suppress_stale(results: List[Dict[str, Any]], stale_ids: Set[str]
                   ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """剔除下线清单里的 chunk，返回 (保留结果, 被剔除的 chunk_id)

    全剔除时如实返回空列表——交给上层证据充分性判定走 CRAG 补救，
    而不是把过期证据硬塞进上下文。
    """
    if not stale_ids:
        return list(results), []
    kept: List[Dict[str, Any]] = []
    removed: List[str] = []
    for result in results:
        chunk_id = chunk_id_of(result)
        if chunk_id in stale_ids:
            removed.append(chunk_id)
        else:
            kept.append(result)
    if removed:
        logger.info(f"[保鲜] 检索结果剔除 {len(removed)} 条过期/重复证据：{sorted(set(removed))}")
    return kept, sorted(set(removed))


def filter_fresh(results: List[Dict[str, Any]],
                 report: GovernanceReport) -> Tuple[List[Dict[str, Any]], List[str]]:
    """按治理扫描报告剔除不再召回的 chunk（报告 -> 下线清单的便捷入口）"""
    return suppress_stale(results, report.suppressed_ids())


corpus_governor = CorpusGovernor()
