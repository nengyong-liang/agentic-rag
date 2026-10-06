"""
检索结果融合模块
支持 RRF (Reciprocal Rank Fusion) 和加权求和两种融合策略

RRF 相比固定权重加权的优势：
1. 不依赖原始分数的绝对值，只依赖排名，更稳健
2. 不同Query的最佳权重差异大，RRF自动适应
3. 生产环境更稳定，避免分数尺度不一致问题
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    es_results: List[Dict],
    k: int = RRF_K,
    vector_weight: float = 0.6,
    es_weight: float = 0.4
) -> List[Dict]:
    """
    Reciprocal Rank Fusion（RRF）融合向量检索和关键词检索结果

    RRF公式：score(d) = Σ weight_i / (k + rank_i)

    Args:
        vector_results: 向量检索结果
        es_results: ES关键词检索结果
        k: RRF常数（默认60，业界标准值）
        vector_weight: 向量检索权重（默认0.6）
        es_weight: 关键词检索权重（默认0.4）
    """
    rrf_scores: Dict[str, Dict] = {}

    for rank, result in enumerate(vector_results, 1):
        doc_id = f"{result.get('document_id', 0)}_{result.get('chunk_index', 0)}"
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"result": dict(result), "rrf_score": 0.0}
        rrf_scores[doc_id]["rrf_score"] += vector_weight / (k + rank)
        rrf_scores[doc_id]["result"]["vector_rank"] = rank

    for rank, result in enumerate(es_results, 1):
        doc_id = f"{result.get('document_id', 0)}_{result.get('chunk_index', 0)}"
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"result": dict(result), "rrf_score": 0.0}
        rrf_scores[doc_id]["rrf_score"] += es_weight / (k + rank)
        rrf_scores[doc_id]["result"]["es_rank"] = rank

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    fused = []
    for item in sorted_results:
        result = item["result"]
        result["rrf_score"] = round(item["rrf_score"], 6)
        result["combined_score"] = result["rrf_score"]
        fused.append(result)

    logger.info(
        f"[RRF融合] 向量{len(vector_results)}条 + 关键词{len(es_results)}条 "
        f"→ 融合{len(fused)}条（k={k}, 权重={vector_weight}:{es_weight}）"
    )
    return fused


def weighted_merge(
    vector_results: List[Dict],
    es_results: List[Dict],
    vector_weight: float = 0.7,
    es_weight: float = 0.3
) -> List[Dict]:
    """
    固定权重加权融合（传统方式，作为RRF的对比基线）

    Args:
        vector_results: 向量检索结果
        es_results: ES关键词检索结果
        vector_weight: 向量检索权重（默认0.7）
        es_weight: 关键词检索权重（默认0.3）
    """
    merged_map: Dict[str, Dict] = {}

    for result in vector_results:
        doc_id = f"{result.get('document_id', 0)}_{result.get('chunk_index', 0)}"
        if doc_id not in merged_map:
            merged_map[doc_id] = dict(result)
            merged_map[doc_id]["combined_score"] = result.get("vector_score", 0) * vector_weight
        else:
            merged_map[doc_id]["vector_score"] = result.get("vector_score", 0)
            merged_map[doc_id]["combined_score"] += result.get("vector_score", 0) * vector_weight

    for result in es_results:
        doc_id = f"{result.get('document_id', 0)}_{result.get('chunk_index', 0)}"
        if doc_id not in merged_map:
            merged_map[doc_id] = dict(result)
            merged_map[doc_id]["combined_score"] = (result.get("es_score") or 0) * es_weight
        else:
            merged_map[doc_id]["es_score"] = result.get("es_score", 0)
            merged_map[doc_id]["combined_score"] += (result.get("es_score") or 0) * es_weight

    sorted_results = sorted(merged_map.values(), key=lambda x: x.get("combined_score", 0), reverse=True)
    return sorted_results
