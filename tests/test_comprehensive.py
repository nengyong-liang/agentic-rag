"""
综合单元测试（轻量级，不依赖BGE-M3/Milvus/ES）
测试：查询分类器 + RRF融合 + 漂移控制器
"""
import sys
import os


from agentic_rag.rag.drift_controller import DriftController
from agentic_rag.rag.query_classifier import QueryClassifier
from agentic_rag.rag.fusion import reciprocal_rank_fusion, weighted_merge


def test_classifier_simple():
    classifier = QueryClassifier()
    simple_queries = [
        "公司年假多少天",
        "报销流程是什么",
        "加班费怎么算",
        "社保缴纳标准",
        "工资发放时间",
        "考勤打卡规定",
    ]
    for q in simple_queries:
        result = classifier._is_likely_simple(q)
        assert result == True, f"'{q}' 应被分类为 simple，但被分类为 {result}"
    print(f"[PASS] Simple分类测试（{len(simple_queries)} 条）")


def test_classifier_fuzzy():
    classifier = QueryClassifier()
    fuzzy_queries = [
        "怎么退单",
        "那个东西怎么用",
        "怎么那个什么",
    ]
    for q in fuzzy_queries:
        result = classifier._is_likely_simple(q)
        assert result == False, f"'{q}' 应被分类为非 simple，但被分类为 simple"
    print(f"[PASS] Fuzzy分类测试（{len(fuzzy_queries)} 条）")


def test_classifier_edge_cases():
    classifier = QueryClassifier()
    assert classifier._is_likely_simple("加班费怎么算") == True, "有具体关键词+1个模糊词应分类为simple"
    assert classifier._is_likely_simple("社保怎么交") == True, "有具体关键词+1个模糊词应分类为simple"
    assert classifier._is_likely_simple("怎么那个什么") == False, "多个模糊词应分类为fuzzy"
    print("[PASS] 分类器边界case测试")


def test_rrf_basic():
    vector_results = [
        {"document_id": 1, "chunk_index": 0, "content": "文档A", "vector_score": 0.9},
        {"document_id": 2, "chunk_index": 0, "content": "文档B", "vector_score": 0.8},
        {"document_id": 3, "chunk_index": 0, "content": "文档C", "vector_score": 0.7},
    ]
    es_results = [
        {"document_id": 2, "chunk_index": 0, "content": "文档B", "es_score": 0.95},
        {"document_id": 4, "chunk_index": 0, "content": "文档D", "es_score": 0.85},
        {"document_id": 1, "chunk_index": 0, "content": "文档A", "es_score": 0.75},
    ]

    fused = reciprocal_rank_fusion(vector_results, es_results)

    assert len(fused) == 4, f"融合后应有4条结果，实际{len(fused)}"
    assert fused[0]["document_id"] == 2, f"文档B在两路都排名靠前，应排第一，实际是文档{fused[0]['document_id']}"
    assert "rrf_score" in fused[0], "融合结果应包含rrf_score"
    assert "combined_score" in fused[0], "融合结果应包含combined_score"
    print("[PASS] RRF基本融合测试")


def test_rrf_weights():
    vector_results = [
        {"document_id": 1, "chunk_index": 0, "content": "仅向量", "vector_score": 0.9},
    ]
    es_results = [
        {"document_id": 2, "chunk_index": 0, "content": "仅ES", "es_score": 0.9},
    ]

    fused = reciprocal_rank_fusion(vector_results, es_results, vector_weight=0.6, es_weight=0.4)

    assert len(fused) == 2, f"融合后应有2条结果，实际{len(fused)}"
    assert fused[0]["document_id"] == 1, f"向量权重0.6>ES权重0.4，向量结果应排第一"
    print("[PASS] RRF权重测试")


def test_rrf_empty():
    fused = reciprocal_rank_fusion([], [])
    assert len(fused) == 0, "空输入应返回空结果"

    fused2 = reciprocal_rank_fusion(
        [{"document_id": 1, "chunk_index": 0, "content": "A", "vector_score": 0.9}],
        []
    )
    assert len(fused2) == 1, "单路结果应正常返回"
    print("[PASS] RRF空输入测试")


def test_rrf_vs_weighted():
    vector_results = [
        {"document_id": 1, "chunk_index": 0, "content": "高分向量", "vector_score": 0.99},
        {"document_id": 2, "chunk_index": 0, "content": "低分向量", "vector_score": 0.3},
    ]
    es_results = [
        {"document_id": 2, "chunk_index": 0, "content": "高分ES", "es_score": 0.99},
        {"document_id": 1, "chunk_index": 0, "content": "低分ES", "es_score": 0.3},
    ]

    fused = reciprocal_rank_fusion(vector_results, es_results)

    assert len(fused) == 2, f"融合后应有2条结果"
    for r in fused:
        assert "rrf_score" in r, "每条结果应有rrf_score"
        assert r["rrf_score"] > 0, "rrf_score应大于0"
    print("[PASS] RRF vs 加权求和对比测试")


def test_weighted_merge():
    vector_results = [
        {"document_id": 1, "chunk_index": 0, "content": "A", "vector_score": 0.9},
    ]
    es_results = [
        {"document_id": 1, "chunk_index": 0, "content": "A", "es_score": 0.8},
    ]

    merged = weighted_merge(vector_results, es_results, vector_weight=0.7, es_weight=0.3)
    assert len(merged) == 1
    expected = 0.9 * 0.7 + 0.8 * 0.3
    assert abs(merged[0]["combined_score"] - expected) < 0.001, f"加权分数计算错误"
    print("[PASS] 加权融合测试")


def test_drift_valid_synonyms():
    dc = DriftController()
    valid_pairs = [
        ("社保", "五险一金"),
        ("年假", "带薪年假"),
        ("加班费怎么算", "加班费计算标准"),
    ]
    for original, expanded in valid_pairs:
        assert dc._check_consistency(original, expanded) == True, f"有效同义词被误过滤：'{original}' vs '{expanded}'"
    print("[PASS] 漂移控制-有效同义词保留")


def test_drift_invalid_drift():
    dc = DriftController()
    invalid_pairs = [
        ("年假", "年度假期旅游"),
        ("社保", "社会新闻"),
        ("加班", "加入班级"),
    ]
    for original, expanded in invalid_pairs:
        assert dc._check_consistency(original, expanded) == False, f"漂移扩展未被过滤：'{original}' vs '{expanded}'"
    print("[PASS] 漂移控制-无效扩展过滤")


if __name__ == "__main__":
    print("=" * 60)
    print("综合单元测试（分类器 + RRF + 漂移控制）")
    print("=" * 60)

    print("\n--- 查询分类器测试 ---")
    test_classifier_simple()
    test_classifier_fuzzy()
    test_classifier_edge_cases()

    print("\n--- RRF融合测试 ---")
    test_rrf_basic()
    test_rrf_weights()
    test_rrf_empty()
    test_rrf_vs_weighted()
    test_weighted_merge()

    print("\n--- 漂移控制测试 ---")
    test_drift_valid_synonyms()
    test_drift_invalid_drift()

    print("\n" + "=" * 60)
    print("全部测试通过！")
    print("=" * 60)
