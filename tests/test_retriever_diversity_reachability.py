"""G1 用例：`RAG-01` 修复 —— 多样性重排在 `retrieve()` 主链路上**真正可达**

背景（docs/KNOWN-ISSUES.md §2.26-E / §2.27-A）：
  修复前 `rag/retriever.py::retrieve()` 的 `:360`/`:362` 两条分支产出的列表长度**恒等于**
  `dynamic_rerank_top_k`，而 `_diversity_rerank` 首判 `len(results) <= top_k` → **恒真早退**
  → 过滤逻辑**永不执行**。
  修复后：启用多样性重排时，重排/截断到候选池 `dynamic_top_k`，再由 `_diversity_rerank`
  收敛到 `dynamic_rerank_top_k`。

本文件两条用例：
  1. `test_diversity_filter_is_reachable`（**新行为**）—— 冗余候选下，最终结果必须**包含**
     候选池里排名第 6 位之后（旧代码视野之外）的文档 → **在改前码上必红**。
  2. `test_diversity_disabled_keeps_legacy_slice`（**非回归守卫**）——
     `enable_diversity_rerank=False` 时输出 == 候选前 `dynamic_rerank_top_k` 条 → **两态皆绿**。

零 API、零模型、零外部服务：服务层全部用假对象注入。
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTEST_COLLECT_ONLY", "1")

import pytest

import agentic_rag.rag.retriever as R


# ---------------------------------------------------------------- 假服务层

class _FakeDM:
    """降级管理器：不缓存、不跳 rerank、单路向量模式。"""

    def get_adjusted_top_k(self, top_k):
        return top_k

    def should_skip_cache(self):
        return True

    def get_retrieval_mode(self):
        return "vector_only"

    def record_success(self, *a, **k):
        pass

    def record_failure(self, *a, **k):
        pass

    def should_skip_rerank(self):
        return False


class _FakeMilvus:
    def __init__(self, docs):
        self._docs = docs

    def search_similar(self, query, top_k, filter_expr=None):
        return [dict(d) for d in self._docs]


class _FakeES:
    def search(self, query, top_k, filter_dict=None):
        return []


class _FakeReranker:
    """保序截断器：真实重排器会改变顺序，这里只要保持输入顺序即可让断言可判别。"""

    def rerank_results(self, query, results, top_k=None):
        results = list(results)
        return results[:top_k] if top_k else results


class _FakeRewriter:
    """固定返回 fuzzy：dynamic_top_k=10 / dynamic_rerank_top_k=5。"""

    def get_expansion_info(self, query):
        return {
            "variants": [{"query": query, "strategy": "original"}],
            "query_type": "fuzzy",
            "expanded_queries": [query],
            "strategy": "original",
        }


class _FakeGovernor:
    def observe(self, query, variants, final_results):
        return {"decision": "keep", "winning_strategy": "original"}


class _FakeDrift:
    def validate_expansion(self, original_query, queries):
        return list(queries)

    def filter_by_threshold(self, results, threshold):
        return list(results)


# ---------------------------------------------------------------- 候选数据

def _docs():
    """前 4 条互为近重复（字符级 Jaccard ≈ 1.0 > 阈值 0.85），第 5 条起互不相同。"""
    dup = "公司年假为5天入职满1年可享受带薪休假"
    distinct = [
        "报销需要填写申请单并附发票",
        "考勤打卡规则说明与迟到处理",
        "加班申请流程与审批节点",
        "社保缴纳基数与比例说明",
        "公积金提取条件与所需材料",
        "培训申请入口与课程目录",
    ]
    out = []
    for i in range(4):
        out.append({"content": dup, "document_id": i, "chunk_index": 0})
    for j, text in enumerate(distinct):
        out.append({"content": text, "document_id": 4 + j, "chunk_index": 0})
    return out  # 共 10 条


def _make_retriever(enable_diversity=True, diversity_threshold=0.85):
    obj = R.HybridRetriever.__new__(R.HybridRetriever)  # 跳过 __init__ 副作用（Redis/KG）
    obj.top_k = 10
    obj.rerank_top_k = 5
    obj.similarity_threshold = 0.5
    obj.enable_rewrite = True
    obj.enable_multi_query = True
    obj.enable_kg = False
    obj.enable_dynamic_topk = True
    obj.enable_diversity_rerank = enable_diversity
    obj.diversity_threshold = diversity_threshold
    obj.fusion_method = "rrf"
    obj.rrf_k = 60
    obj.vector_weight = 0.6
    obj.es_weight = 0.4
    obj.stale_chunk_ids = set()
    obj._kg = None
    obj.last_rewrite_report = {}
    obj.drift_controller = _FakeDrift()
    obj.redis_cache = None  # should_skip_cache()=True → 不会被调用
    obj._get_degradation_manager = lambda: _FakeDM()
    return obj


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(R, "milvus_service", _FakeMilvus(_docs()))
    monkeypatch.setattr(R, "es_service", _FakeES())
    monkeypatch.setattr(R, "reranker_service", _FakeReranker())
    monkeypatch.setattr(R, "query_rewriter", _FakeRewriter())
    monkeypatch.setattr(R, "rewrite_governor", _FakeGovernor())
    yield


# ---------------------------------------------------------------- 用例

def test_diversity_filter_is_reachable(patched):
    """🔴 新行为用例：冗余候选下，过滤必须真正发生。

    判别点：修复后候选池 = dynamic_top_k(10) → 排名第 6 位之后的不同主题文档
    （document_id ≥ 5）**应当进入最终结果**；修复前候选池 = 5，视野内只有
    document_id 0~4 → 该断言必红。
    """
    r = _make_retriever(enable_diversity=True)
    out = r.retrieve("年假怎么算")

    ids = [d["document_id"] for d in out]

    # 1) 最终条数不得超过 dynamic_rerank_top_k（fuzzy → 5）
    assert len(out) <= 5, f"最终条数应 ≤5，实际 {len(out)}：{ids}"

    # 2) 近重复只应保留 1 条
    dup_kept = [i for i in ids if i < 4]
    assert len(dup_kept) == 1, f"4 条近重复应只保留 1 条，实际保留 {dup_kept}"

    # 3) 🔴 判别点：必须看到候选池深处（旧代码视野之外）的文档
    assert any(i >= 5 for i in ids), (
        f"多样性过滤未真正执行：结果 {ids} 全部落在旧候选池（document_id 0~4）内"
    )


def test_diversity_disabled_keeps_legacy_slice(patched):
    """✅ 非回归守卫：关闭多样性重排时，输出 == 候选前 dynamic_rerank_top_k 条（两态皆绿）。"""
    r = _make_retriever(enable_diversity=False)
    out = r.retrieve("年假怎么算")

    ids = [d["document_id"] for d in out]
    assert ids == [0, 1, 2, 3, 4], f"关闭多样性重排时应退化为旧行为（前 5 条），实际 {ids}"


def test_no_diversity_when_candidates_not_redundant(patched, monkeypatch):
    """✅ 守卫：候选互不相似时，过滤不应误杀 —— 输出仍应填满到 top_k 条。"""
    texts = [
        "报销需要填写申请单并附发票",
        "考勤打卡规则说明与迟到处理",
        "加班申请流程与审批节点",
        "社保缴纳基数与比例说明",
        "公积金提取条件与所需材料",
        "培训申请入口与课程目录",
        "离职手续办理时间要求",
        "绩效评估周期与打分标准",
        "调岗申请需要哪些审批",
        "库存查询系统使用手册",
    ]
    docs = [{"content": t, "document_id": i, "chunk_index": 0} for i, t in enumerate(texts)]
    monkeypatch.setattr(R, "milvus_service", _FakeMilvus(docs))
    r = _make_retriever(enable_diversity=True)
    out = r.retrieve("随便问问")
    assert len(out) == 5, f"非冗余候选应填满 5 条，实际 {len(out)}"
