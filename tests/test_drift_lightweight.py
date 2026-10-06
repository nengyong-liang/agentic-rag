"""
漂移控制器单元测试（轻量级，不依赖BGE-M3/Milvus/ES）
直接从独立模块导入 DriftController，避免加载重模型
"""
import sys
import os


from agentic_rag.rag.drift_controller import DriftController


def test_char_overlap():
    dc = DriftController()
    assert dc._char_overlap_ratio("社保", "社会保险") >= 0.5
    assert dc._char_overlap_ratio("年假", "带薪年假") >= 0.5
    assert dc._char_overlap_ratio("加班费", "加班费计算") >= 0.6
    print("[PASS] 字符重叠率测试")


def test_bigram_jaccard():
    dc = DriftController()
    sim = dc._bigram_jaccard("加班费怎么算", "加班费计算标准")
    assert sim > 0.1, f"bigram jaccard should > 0.1 for similar queries, got {sim}"
    sim2 = dc._bigram_jaccard("年假", "年度假期旅游")
    assert sim2 < 0.3, f"bigram jaccard should < 0.3 for '年假' vs '年度假期旅游', got {sim2}"
    sim3 = dc._bigram_jaccard("退货流程", "退款流程")
    assert sim3 > 0, f"bigram jaccard should > 0 for '退货流程' vs '退款流程', got {sim3}"
    print("[PASS] Bigram Jaccard测试")


def test_synonym_trust():
    dc = DriftController()
    assert dc._keyword_preserved("社保", "五险一金") == True
    assert dc._keyword_preserved("社保", "社会保险") == True
    assert dc._keyword_preserved("年假", "带薪年假") == True
    assert dc._keyword_preserved("加班费", "加班申请") == True
    assert dc._keyword_preserved("退货", "退款") == True
    print("[PASS] 同义词词典信任测试")


def test_check_consistency_valid_synonyms():
    dc = DriftController()
    valid_pairs = [
        ("社保", "五险一金"),
        ("社保", "社会保险"),
        ("年假", "带薪年假"),
        ("年假", "年休假"),
        ("加班费怎么算", "加班费计算标准"),
        ("退货流程", "退款流程"),
        ("工资发放", "薪资发放"),
    ]
    for original, expanded in valid_pairs:
        result = dc._check_consistency(original, expanded)
        assert result == True, f"有效同义词被误过滤：'{original}' vs '{expanded}'"
    print(f"[PASS] 有效同义词保留测试（{len(valid_pairs)} 组）")


def test_check_consistency_invalid_drift():
    dc = DriftController()
    invalid_pairs = [
        ("年假", "年度假期旅游"),
        ("社保", "社会新闻"),
        ("加班", "加入班级"),
        ("退货", "退回老家"),
        ("报销", "报告销售"),
    ]
    for original, expanded in invalid_pairs:
        result = dc._check_consistency(original, expanded)
        assert result == False, f"漂移扩展未被过滤：'{original}' vs '{expanded}'"
    print(f"[PASS] 漂移扩展过滤测试（{len(invalid_pairs)} 组）")


def test_validate_expansion():
    dc = DriftController()
    original = "社保怎么交"
    expanded = ["社保怎么交", "五险一金缴纳", "社会保险缴费", "社会新闻热点"]
    validated = dc.validate_expansion(original, expanded)
    assert "五险一金缴纳" in validated, "有效同义词扩展被误过滤"
    assert "社会保险缴费" in validated, "有效同义词扩展被误过滤"
    assert "社会新闻热点" not in validated, "漂移扩展未被过滤"
    print("[PASS] 扩展校验集成测试")


def test_filter_by_threshold():
    dc = DriftController()
    results = [
        {"document_id": 1, "chunk_index": 0, "vector_score": 0.8, "es_score": 0.7},
        {"document_id": 2, "chunk_index": 0, "vector_score": 0.3, "es_score": 0.2},
        {"document_id": 3, "chunk_index": 0, "vector_score": 0.6, "es_score": 0.5},
    ]
    filtered = dc.filter_by_threshold(results, min_score=0.5)
    assert len(filtered) >= 2, f"阈值过滤后应至少2条，实际{len(filtered)}"
    print("[PASS] 阈值过滤测试")


if __name__ == "__main__":
    print("=" * 60)
    print("漂移控制器单元测试（轻量级）")
    print("=" * 60)

    test_char_overlap()
    test_bigram_jaccard()
    test_synonym_trust()
    test_check_consistency_valid_synonyms()
    test_check_consistency_invalid_drift()
    test_validate_expansion()
    test_filter_by_threshold()

    print("=" * 60)
    print("全部测试通过！")
    print("=" * 60)
