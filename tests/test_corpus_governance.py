"""知识库保鲜 / Corpus 治理测试（rag/corpus_governance.py）

覆盖 docs/KNOWN-ISSUES.md §12.3 缺口 7 的四件事：增量索引 diff、过期发现、冲突发现、错误/重复发现，
以及"治理结论如何落到检索链路"。所有时间判定都传 `as_of`，结果可重跑。
"""

from datetime import date

import hashlib
import json
import os

import pytest

from agentic_rag.rag.corpus_governance import (
    ACTION_DROP_DUPLICATE, ACTION_REINDEX, ACTION_REVIEW, ACTION_SUPPRESS,
    CATEGORY_CONFLICT, CATEGORY_DEFECT, CATEGORY_DUPLICATE, CATEGORY_EXPIRED,
    CATEGORY_SUPERSEDED, GOVERNANCE_CATEGORIES, CorpusGovernor, GovernanceReport,
    as_date, chunk_id_of, filter_fresh, suppress_stale, tokenize,
)

TODAY = date(2026, 9, 30)


def gov(**kwargs) -> CorpusGovernor:
    kwargs.setdefault("as_of", TODAY)
    return CorpusGovernor(**kwargs)


def _read_source(*parts: str) -> str:
    # 源码在 src/agentic_rag/ 下，测试按包内路径读取实现做接线断言
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "agentic_rag")
    with open(os.path.join(root, *parts), encoding="utf-8") as f:
        return f.read()


def chunk(chunk_id, content, kb="kb1", **meta):
    metadata = {k: v for k, v in meta.items() if k in
                ("topic_key", "version", "updated_at", "effective_until", "is_latest",
                 "chunk_hash", "section", "document_id", "chunk_index")}
    record = {"chunk_id": chunk_id, "content": content, "kb_id": kb}
    if metadata:
        record["metadata"] = metadata
    return record


# ============================================================
# 一、字段归一与时间解析
# ============================================================

def test_as_date_accepts_common_shapes_and_rejects_junk():
    assert as_date("2026-09-30") == date(2026, 9, 30)
    assert as_date("2026/09/30") == date(2026, 9, 30)
    assert as_date("2026-09-30T23:59:59") == date(2026, 9, 30)
    assert as_date(date(2026, 1, 2)) == date(2026, 1, 2)
    assert as_date(None) is None and as_date("") is None
    assert as_date("未标注") is None
    assert as_date("2026-13-45") is None           # 非法日期不崩


def test_to_record_falls_back_to_document_and_chunk_index():
    record = gov().to_record({"document_id": 7, "chunk_index": 3,
                              "content": "库存 12 件", "metadata": {"section": "库存"}})
    assert record.chunk_id == "7_3"
    assert record.topic_key == "库存"
    assert record.content_hash == "sha256:" + hashlib.sha256("库存 12 件".encode()).hexdigest()
    assert record.tokens == tokenize("库存 12 件")


def test_to_record_reuses_existing_chunk_hash_and_casts_version():
    record = gov().to_record(chunk("c1", "正文", chunk_hash="sha256:abc", version="v2"))
    assert record.content_hash == "sha256:abc"     # 门禁已写好的哈希不重算
    assert record.version == 2
    assert gov().to_record(chunk("c2", "正文", version="草稿")).version is None


def test_numeric_facts_groups_by_unit():
    record = gov().to_record(chunk(
        "c1", "入职满 1 年的员工年假 5 天，加班超过 3 小时按 1 个工作日计。"))
    facts = record.numeric_facts()
    assert facts["年"] == {"1"} and facts["天"] == {"5"} and facts["个工作日"] == {"1"}


# ============================================================
# 二、增量索引 diff
# ============================================================

def test_incremental_plan_splits_four_categories():
    indexed = [chunk("c1", "不变的内容一二三", updated_at="2026-09-01"),
               chunk("c2", "旧版本内容哈哈哈", updated_at="2026-09-01"),
               chunk("c3", "将被删除的旧切片", updated_at="2026-09-01")]
    current = [chunk("c1", "不变的内容一二三", updated_at="2026-09-28"),   # 仅元数据变
               chunk("c2", "新版本内容哈哈哈", updated_at="2026-09-28"),   # 正文变
               chunk("c4", "全新加入的切片", updated_at="2026-09-28")]     # 新增

    plan = gov().plan_incremental_update(indexed, current)

    assert plan.added == ["c4"]
    assert plan.updated == ["c2"]
    assert plan.deleted == ["c3"]
    assert plan.unchanged == ["c1"]                 # 内容未变不重新向量化
    assert plan.changed_count == 3 and plan.needs_work is True
    assert plan.to_dict()["full_rebuild_required"] is False


def test_incremental_plan_on_first_build_is_all_added():
    plan = gov().plan_incremental_update([], [chunk("c1", "内容"), chunk("c2", "内容")])
    assert plan.added == ["c1", "c2"] and plan.updated == [] and plan.deleted == []


def test_identical_snapshots_produce_empty_plan():
    snapshot = [chunk("c1", "同样内容"), chunk("c2", "另一段内容")]
    plan = gov().plan_incremental_update(snapshot, list(snapshot))
    assert plan.needs_work is False
    assert sorted(plan.unchanged) == ["c1", "c2"]


# ============================================================
# 三、错误 chunk（defect）
# ============================================================

def test_defect_rules_fire_per_problem_type():
    report = gov(min_chars=8).scan([
        chunk("empty", ""),
        chunk("short", "只有一句"),
        chunk("garbled", "★☆◆■▲△◇◆▲■" * 3),
        chunk("truncated", "请假需要提交申请单，并由直属主管审批后，"),
        chunk("ok", "请假需提交申请单，直属主管审批通过后生效。"),
    ])
    rules = {f.rule for f in report.findings}

    assert {"D1/empty", "D2/too_short", "D3/garbled", "D4/truncated"} <= rules
    defect = {f.chunk_id for f in report.findings if f.category == CATEGORY_DEFECT}
    assert defect == {"empty", "short", "garbled", "truncated"}
    assert all(f.action == ACTION_REINDEX for f in report.findings
               if f.category == CATEGORY_DEFECT)
    assert "ok" not in defect                       # 正常切片不误伤


def test_records_without_id_are_dropped_from_scan_not_misreported():
    report = gov().scan([{"content": "没有标识的切片，长度足够。"}] * 2)
    assert report.chunk_count == 0
    assert report.findings == []


# ============================================================
# 四、过期 / 被取代
# ============================================================

def test_effective_until_and_is_latest_suppress_retrieval():
    report = gov().scan([
        chunk("expired", "2025 年考勤制度：打卡 9 点。", effective_until="2026-06-30"),
        chunk("old_copy", "同一制度的旧版说明，内容较长即可。", is_latest=False),
        chunk("fresh", "现行考勤制度：打卡 9 点半。", effective_until="2027-06-30"),
    ])
    rules = {f.chunk_id: f.rule for f in report.findings}

    assert rules["expired"] == "E1/effective_until"
    assert rules["old_copy"] == "E2/is_latest_false"
    assert all(f.category == CATEGORY_EXPIRED for f in report.findings
               if f.rule.startswith("E"))
    assert report.suppressed_ids() == {"expired", "old_copy"}
    assert "fresh" not in report.suppressed_ids()
    assert report.review_ids() == set()


def test_ttl_flags_review_not_silent_expiry():
    """超 TTL 只标待复核——老文档只要没声明失效就不能自动下线"""
    report = gov(ttl_days=180).scan([
        chunk("ancient", "这条制度写于两年前，内容仍在库里。", updated_at="2024-01-01"),
        chunk("recent", "这条制度上个月刚更新，内容仍在库里。", updated_at="2026-09-01"),
        chunk("bounded", "带有效期的制度，内容仍在库里可用。", updated_at="2024-01-01",
              effective_until="2027-01-01"),
    ])
    assert report.review_ids() == {"ancient"}        # 有有效期字段就不再按 TTL 猜
    assert report.suppressed_ids() == set()


def test_superseded_group_uses_version_then_recency():
    report = gov().scan([
        chunk("v1", "年假规定第一版，内容较长可用即可。", topic_key="annual_leave",
              version=1, updated_at="2025-01-01"),
        chunk("v2", "年假规定第二版，内容较长可用即可。", topic_key="annual_leave",
              version=2, updated_at="2026-01-01"),
        chunk("a", "无版本字段的一条规定说明文字。", topic_key="no_version_group",
              updated_at="2025-01-01"),
        chunk("b", "无版本字段的另一条规定说明。", topic_key="no_version_group",
              updated_at="2026-05-01"),
    ])
    by_id = {f.chunk_id: f for f in report.findings}

    assert by_id["v1"].category == CATEGORY_SUPERSEDED
    assert by_id["v1"].rule == "S1/version" and by_id["v1"].related == ["v2"]
    assert by_id["a"].rule == "S3/recency" and by_id["a"].related == ["b"]
    assert report.suppressed_ids() == {"v1", "a"}


def test_topics_only_compare_within_same_kb():
    """不同知识库的同名主题不做版本仲裁，避免跨租户误伤"""
    report = gov().scan([
        chunk("t1", "考勤规定：打卡 9 点上班。", kb="kb1", topic_key="attendance", version=1),
        chunk("t2", "考勤规定：打卡 9 点半上班。", kb="kb2", topic_key="attendance", version=2),
    ])
    assert report.counts_by_category()[CATEGORY_SUPERSEDED] == 0


# ============================================================
# 五、冲突（只标记，不下线）
# ============================================================

def test_same_topic_numeric_conflict_is_flagged_for_review():
    report = gov().scan([
        chunk("a", "入职满 1 年的员工，年假 5 天，加班按 2 小时起算。",
              topic_key="annual_leave", updated_at="2025-01-01"),
        chunk("b", "入职满 1 年的员工，年假 10 天，加班按 2 小时起算。",
              topic_key="annual_leave", updated_at="2026-01-01"),
    ])
    conflicts = [f for f in report.findings if f.category == CATEGORY_CONFLICT]

    assert {f.rule for f in conflicts} == {"C1/天"}
    assert {f.action for f in conflicts} == {ACTION_REVIEW}
    assert {f.chunk_id for f in conflicts} == {"a", "b"}
    assert next(f for f in conflicts if f.chunk_id == "a").related == ["b"]
    assert "5" in conflicts[0].detail and "10" in conflicts[0].detail
    # 冲突不自动下线：谁为准要人工裁决，检索层不能替业务决定
    assert report.suppressed_ids() == {"a"}          # 仅因 S3/recency 版本仲裁
    assert report.review_ids() == {"a", "b"}
    # 冲突对文本近乎同形，绝不能被"近重复"判重下线——否则矛盾证据会被悄悄抹掉
    assert report.counts_by_category()[CATEGORY_DUPLICATE] == 0


def test_identical_numbers_are_not_conflicts():
    report = gov().scan([
        chunk("a", "年假 5 天，按自然年计算。", topic_key="leave"),
        chunk("b", "年假 5 天，按入职周年计算。", topic_key="leave"),
    ])
    assert report.counts_by_category()[CATEGORY_CONFLICT] == 0


# ============================================================
# 六、重复（精确 + 近重复）
# ============================================================

def test_exact_duplicates_keep_earliest_copy():
    report = gov().scan([
        chunk("late", "同一段完全重复的制度正文。", updated_at="2026-05-01"),
        chunk("early", "同一段完全重复的制度正文。", updated_at="2025-01-01"),
    ])
    duplicates = [f for f in report.findings if f.category == CATEGORY_DUPLICATE]

    assert [f.chunk_id for f in duplicates] == ["late"]
    assert duplicates[0].rule == "P1/exact_hash"
    assert duplicates[0].action == ACTION_DROP_DUPLICATE
    assert duplicates[0].related == ["early"]


def test_near_duplicates_caught_by_similarity_threshold():
    base = "员工请假需提前一天提交申请单，并经直属主管审批同意后方可休假。"
    variant = "员工请假需提前两天提交申请单，并经直属主管审批同意后方可休假。"
    report = gov(duplicate_similarity=0.8).scan([
        chunk("d1", base, updated_at="2025-01-01"),
        chunk("d2", variant, updated_at="2026-01-01"),
    ])
    near = [f for f in report.findings if f.rule == "P2/near_duplicate"]

    assert [f.chunk_id for f in near] == ["d2"]      # 后出现的副本被判重
    assert "相似度" in near[0].detail and near[0].related == ["d1"]


def test_near_duplicate_scan_reports_skipped_instead_of_lying():
    report = gov(duplicate_similarity=0.8, max_pairwise_chunks=2).scan([
        chunk("x", "同一段完全重复的制度正文。"),
        chunk("y", "同一段完全重复的制度正文。"),
        chunk("z", "员工请假需提前一天提交申请单，并经直属主管审批同意后方可休假。"),
    ])
    assert report.near_duplicate_scan_skipped is True
    assert report.counts_by_category()[CATEGORY_DUPLICATE] == 0


# ============================================================
# 七、报告与检索链路接入
# ============================================================

def _dirty_report() -> GovernanceReport:
    return gov().scan([
        chunk("stale", "过期制度正文若干字。", effective_until="2026-01-01"),
        chunk("copy", "重复制度正文若干个字。", updated_at="2026-02-02"),
        chunk("orig", "重复制度正文若干个字。", updated_at="2025-02-02"),
        chunk("good", "现行制度正文若干个字啊。", effective_until="2027-01-01"),
    ])


def test_report_is_json_safe_and_covers_all_categories():
    payload = _dirty_report().to_dict()
    restored = json.loads(json.dumps(payload, ensure_ascii=False))

    assert set(restored["by_category"]) == set(GOVERNANCE_CATEGORIES)
    assert restored["as_of"] == "2026-09-30"
    assert restored["action_rules"][ACTION_SUPPRESS]
    assert restored["suppressed"] == ["copy", "stale"]
    assert len(restored["findings"]) == restored["finding_count"]


def test_suppress_stale_removes_only_flagged_chunks():
    report = _dirty_report()
    results = [{"chunk_id": "stale", "content": "x"},
               {"chunk_id": "good", "content": "y"},
               {"document_id": 3, "chunk_index": 0, "content": "z"}]

    kept, removed = filter_fresh(results, report)

    assert removed == ["stale"]
    assert [c.get("chunk_id") or chunk_id_of(c) for c in kept] == ["good", "3_0"]


def test_suppress_stale_is_noop_without_governance_and_honest_when_all_stale():
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert suppress_stale(results, set()) == (results, [])

    report = gov().scan([chunk("a", "正文", effective_until="2020-01-01"),
                         chunk("b", "另一些正文", effective_until="2020-01-01")])
    kept, removed = filter_fresh([{"chunk_id": "a"}, {"chunk_id": "b"}], report)
    assert kept == []                                # 证据全过期：如实交上层补救
    assert removed == ["a", "b"]


def test_retriever_is_wired_to_the_freshness_filter():
    """保鲜剔除点确实挂在检索链路上（漂移过滤之后、KG 增强之前）

    `rag.retriever` 一导入就要加载本地 embedding 模型（实测 >30s），单元门禁里
    不做该导入；接线位置用源码校验，改动顺序错位即报警。
    """
    source = _read_source("rag", "retriever.py")

    assert "corpus_governance import suppress_stale" in source
    assert "self.stale_chunk_ids" in source and "load_corpus_governance" in source
    assert (source.index('latency_breakdown["drift_filter"]')
            < source.index('latency_breakdown["corpus_freshness"]')
            < source.index('latency_breakdown["kg_enhance"]'))


def test_ingest_route_reports_governance_before_vectorising():
    """导入链路：治理扫描在双写 Milvus/ES 之前，结论随响应回传（不静默改数据）"""
    source = _read_source("api", "routes", "document.py")

    assert "corpus_governance import corpus_governor" in source
    assert source.index("corpus_governor.scan(passed_chunks)") < \
        source.index("milvus_service.insert_chunks(passed_chunks)")
    assert '"governance_report"' in source
