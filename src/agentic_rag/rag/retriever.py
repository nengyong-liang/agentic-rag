"""
混合检索器（工业级分级扩展版 + 降级策略 + 动态TopK + 多样性重排）

支持：
1. 查询分类 → 分级扩展（simple/fuzzy/complex）
2. 语义漂移控制：相似度阈值过滤 + 扩展结果一致性校验
3. RRF融合：替代固定权重加权，更稳健的混合检索结果合并
4. 降级策略：Milvus/ES单路宕机自动切换，Rerank超时跳过
5. 动态TopK：simple→5, fuzzy→10, complex→20
6. 多样性重排：相似度惩罚避免冗余
7. 全链路日志 + 延迟追踪
8. 知识库保鲜：按 Corpus 治理扫描的下线清单剔除过期/被取代/重复副本证据
"""

from typing import List, Dict, Optional
import logging
import time
import os

from agentic_rag.utils.milvus import milvus_service
from agentic_rag.utils.elasticsearch import es_service
from agentic_rag.rag.chunker import semantic_chunker
from agentic_rag.utils.reranker import reranker_service
from agentic_rag.utils.redis_cache import get_redis_cache
from agentic_rag.rag.query_rewriter import query_rewriter
from agentic_rag.rag.rewrite_governor import (
    QueryVariant, STRATEGY_ORIGINAL, rewrite_governor,
)
from agentic_rag.rag.drift_controller import DriftController, SIMILARITY_THRESHOLD
from agentic_rag.rag.corpus_governance import suppress_stale
from agentic_rag.rag.fusion import reciprocal_rank_fusion, weighted_merge, RRF_K
from agent_harness.observability.tracing import get_current_span, record_attributes, traced

logger = logging.getLogger(__name__)

DYNAMIC_TOP_K = {
    "simple": 5,
    "fuzzy": 10,
    "complex": 20,
}

DYNAMIC_RERANK_TOP_K = {
    "simple": 3,
    "fuzzy": 5,
    "complex": 10,
}


class HybridRetriever:
    """混合检索器：分级扩展 + 降级策略 + 动态TopK + 多样性重排"""

    def __init__(self, top_k: int = 10, rerank_top_k: int = 5,
                 enable_rewrite: bool = True, enable_multi_query: bool = True,
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 enable_kg: bool = True,
                 fusion_method: str = "rrf",
                 rrf_k: int = RRF_K,
                 vector_weight: float = 0.6,
                 es_weight: float = 0.4,
                 enable_dynamic_topk: bool = True,
                 enable_diversity_rerank: bool = True,
                 diversity_threshold: float = 0.85):
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold
        self.redis_cache = get_redis_cache()
        self.enable_rewrite = enable_rewrite
        self.enable_multi_query = enable_multi_query
        self.enable_kg = enable_kg
        self.drift_controller = DriftController()
        self.stale_chunk_ids: set = set()  # 知识库保鲜下线清单（由 load_corpus_governance 载入）
        self._kg = None
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.es_weight = es_weight
        self.enable_dynamic_topk = enable_dynamic_topk
        self.enable_diversity_rerank = enable_diversity_rerank
        self.diversity_threshold = diversity_threshold
        self.last_rewrite_report: Dict = {}  # 最近一次检索的改写命中归因报告

    def _get_degradation_manager(self):
        from agentic_rag.utils.degradation import degradation_manager
        return degradation_manager

    def _get_kg(self):
        from agentic_rag.rag.knowledge_graph import knowledge_graph
        return knowledge_graph

    def load_corpus_governance(self, report) -> int:
        """载入 Corpus 治理扫描结论，返回下线条目数（清单为空即不启用保鲜）"""
        self.stale_chunk_ids = set(report.suppressed_ids())
        logger.info(f"[检索] 知识库保鲜清单载入：{len(self.stale_chunk_ids)} 个 chunk 停止召回")
        return len(self.stale_chunk_ids)

    def _suppress_stale(self, results: List[Dict]) -> List[Dict]:
        kept, _ = suppress_stale(results, self.stale_chunk_ids)
        return kept

    def _kg_enhance(self, query: str, results: List[Dict]) -> List[Dict]:
        if not self.enable_kg:
            return results

        try:
            if self._kg is None:
                self._kg = self._get_kg()

            entities = self._extract_query_entities(query)
            if not entities:
                return results

            kg_contexts = []
            for entity in entities[:3]:
                related = self._kg.query_related_entities(entity, depth=1)
                related_entities = related.get("related_entities", [])
                relations = related.get("relations", [])

                if related_entities or relations:
                    context_str = f"[KG] {entity}的关联信息："
                    for re_item in related_entities[:3]:
                        if isinstance(re_item, dict):
                            context_str += f" {re_item.get('name', '')}({re_item.get('type', '')})"
                    for rel in relations[:3]:
                        if isinstance(rel, dict):
                            context_str += f"; {rel.get('subject', '')}-{rel.get('predicate', '')}-{rel.get('object', '')}"
                    kg_contexts.append(context_str)

            if kg_contexts:
                kg_doc = {
                    "content": " ".join(kg_contexts),
                    "document_id": -1,
                    "chunk_index": -1,
                    "metadata": {"source": "knowledge_graph"},
                    "rerank_score": 0.3,
                    "vector_score": 0.3,
                    "es_score": 0.0
                }
                results.append(kg_doc)
                logger.info(f"[KG增强] 补充 {len(kg_contexts)} 条实体关系上下文")

        except Exception as e:
            logger.warning(f"KG增强失败，跳过：{e}")

        return results

    def _extract_query_entities(self, query: str) -> List[str]:
        entity_keywords = [
            "年假", "社保", "公积金", "工资", "考勤", "加班", "请假",
            "合同", "绩效", "培训", "调岗", "入职", "离职", "报销",
            "库存", "配件", "维修", "工单", "退货", "退款"
        ]
        found = []
        for kw in entity_keywords:
            if kw in query:
                found.append(kw)
        return found

    def _merge_results(self, vector_results: List[Dict], es_results: List[Dict]) -> List[Dict]:
        if self.fusion_method == "rrf":
            return reciprocal_rank_fusion(
                vector_results, es_results,
                k=self.rrf_k,
                vector_weight=self.vector_weight,
                es_weight=self.es_weight
            )
        return weighted_merge(vector_results, es_results, self.vector_weight, self.es_weight)

    def _get_dynamic_top_k(self, query_type: str) -> tuple:
        """动态TopK：simple→5/3, fuzzy→10/5, complex→20/10"""
        if not self.enable_dynamic_topk: # 如果未启用动态TopK，直接返回默认值
            return self.top_k, self.rerank_top_k

        top_k = DYNAMIC_TOP_K.get(query_type, self.top_k) # 获取动态TopK
        rerank_top_k = DYNAMIC_RERANK_TOP_K.get(query_type, self.rerank_top_k) # 获取动态重排TopK

        dm = self._get_degradation_manager() # 获取降级管理器
        adjusted_k = dm.get_adjusted_top_k(top_k) # 获取调整后的TopK

        logger.info(f"[动态TopK] query_type={query_type}, top_k={top_k}, rerank_top_k={rerank_top_k}, adjusted={adjusted_k}")
        return adjusted_k, rerank_top_k

    def _diversity_rerank(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        多样性重排：基于内容相似度惩罚，避免冗余

        面试核心：后重排阶段，对相似度>阈值的结果降权
        避免返回5条内容几乎一样的结果
        """
        if not self.enable_diversity_rerank or len(results) <= top_k:
            return results[:top_k]

        selected = [results[0]]
        for candidate in results[1:]:
            if len(selected) >= top_k:
                break

            is_redundant = False
            for existing in selected:
                sim = self._content_similarity(
                    candidate.get("content", ""),
                    existing.get("content", "")
                )
                if sim > self.diversity_threshold:
                    is_redundant = True
                    logger.debug(f"[多样性重排] 过滤冗余结果：sim={sim:.2f}")
                    break

            if not is_redundant:
                selected.append(candidate)

        if len(selected) < min(3, len(results)):
            for r in results:
                if r not in selected:
                    selected.append(r)
                if len(selected) >= top_k:
                    break

        logger.info(f"[多样性重排] {len(results)} → {len(selected)} 条（阈值={self.diversity_threshold}）")
        return selected

    def _content_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的字符级Jaccard相似度（轻量版，不依赖模型）"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1)
        set2 = set(text2)
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    @traced("rag.retrieve",
            attributes_fn=lambda self, query, filter_dict=None: {
                "rag.query": str(query)[:60],
                "rag.tenant_id": (filter_dict or {}).get("tenant_id")
                if isinstance(filter_dict, dict) else None,
            })
    def retrieve(self, query: str, filter_dict: Dict = None) -> List[Dict]:
        """执行混合检索（分级扩展 + 降级策略 + 动态TopK + 多样性重排）"""
        start_time = time.time() # 记录检索开始时间
        original_query = query # 记录原始查询语句
        latency_breakdown = {} # 记录检索时间分段

        dm = self._get_degradation_manager()# 获取降级管理器

        expansion_info = query_rewriter.get_expansion_info(query) # 获取查询扩展信息
        variants = [QueryVariant(v.get("query", ""), v.get("strategy", STRATEGY_ORIGINAL))
                    for v in expansion_info.get("variants", []) if v.get("query")]
        queries = [v.query for v in variants] or [query] # 生成扩展查询
        query_type = expansion_info.get("query_type", "fuzzy") # 获取查询类型，默认fuzzy

        
        queries = self.drift_controller.validate_expansion(original_query, queries) # 校验扩展查询与原查询的一致性，过滤漂移严重的扩展
        variants = [v for v in variants if v.query in queries] # 归因口径与实际检索口径一致
        
        expansion_info["validated_queries"] = queries
        expansion_info["drift_filtered"] = len(expansion_info.get("expanded_queries", [])) - len(queries) # 记录过滤后的扩展查询数

        dynamic_top_k, dynamic_rerank_top_k = self._get_dynamic_top_k(query_type) # 获取动态TopK和重排TopK

        logger.info(f"[检索] 扩展策略={expansion_info['strategy']}, 查询数={len(queries)}, top_k={dynamic_top_k}")
        record_attributes(get_current_span(), {
            "rag.query_type": query_type,
            "rag.top_k": dynamic_top_k,
            "rag.variant_count": len(variants),
            "rag.drift_filtered": expansion_info.get("drift_filtered", 0),
        })

        if not dm.should_skip_cache(): # 如果未跳过缓存
            cached_results = self.redis_cache.get_cached_retrieval(original_query, filter_dict.get("tenant_id") if isinstance(filter_dict, dict) else None) # 从缓存获取检索结果（带租户维度）
            if cached_results: # 如果缓存命中，检索到缓存的文档片段
                logger.info("[检索] 从缓存获取检索结果")
                return cached_results # 返回缓存结果

        all_results = []
        seen_ids = set()

        retrieval_mode = dm.get_retrieval_mode() # 获取检索模式

        for i, q in enumerate(queries): # 迭代每个扩展查询
            t0 = time.time()
            logger.info(f"[检索] 查询 [{i + 1}/{len(queries)}]：{q[:50]}...")

            vector_results = [] # 初始化向量检索结果
            es_results = [] # 初始化ES检索结果

            if retrieval_mode != "es_only": # 如果不是ES只检索模式
                try:
                    
                    # milvus向量检索
                    vector_results = milvus_service.search_similar(
                        query=q,
                        top_k=dynamic_top_k,
                        filter_expr=self._build_filter_expr(filter_dict) if filter_dict else None
                    )
                    dm.record_success("milvus")
                except Exception as e:
                    logger.error(f"[检索] Milvus检索失败：{e}")
                    dm.record_failure("milvus", str(e))
                    if retrieval_mode == "hybrid":
                        retrieval_mode = dm.get_retrieval_mode()

            if retrieval_mode != "vector_only": # 如果不是向量只检索模式
                try:
                    es_results = es_service.search(
                        query=q,
                        top_k=dynamic_top_k,
                        filter_dict=filter_dict
                    )
                    dm.record_success("elasticsearch")
                except Exception as e:
                    logger.error(f"[检索] ES检索失败：{e}")
                    dm.record_failure("elasticsearch", str(e))
                    if retrieval_mode == "hybrid":
                        retrieval_mode = dm.get_retrieval_mode()

            if retrieval_mode == "es_only":
                merged = es_results
            elif retrieval_mode == "vector_only":
                merged = vector_results
            else:
                merged = self._merge_results(vector_results, es_results)

            for result in merged:
                
                # 去重
                # 构建结果ID，包含文档ID和片段索引
                result_id = f"{result.get('document_id', 0)}_{result.get('chunk_index', 0)}"
                
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    result["_source_query"] = q  # 改写命中率归因用，返回前剥离
                    all_results.append(result)

            latency_breakdown[f"query_{i+1}"] = round(time.time() - t0, 3)

        logger.info(f"[检索] 合并后共 {len(all_results)} 条去重结果（模式={retrieval_mode}）")

        t_filter = time.time()
        # 过滤漂移严重的文档片段
        filtered_results = self.drift_controller.filter_by_threshold(
            all_results, self.similarity_threshold
        )
        latency_breakdown["drift_filter"] = round(time.time() - t_filter, 3) # 记录漂移过滤耗时

        # 知识库保鲜：剔除治理扫描判定过期/被取代/重复副本的证据
        t_fresh = time.time()
        filtered_results = self._suppress_stale(filtered_results)
        latency_breakdown["corpus_freshness"] = round(time.time() - t_fresh, 3)

        # 知识图谱增强
        t_kg = time.time()
        kg_enhanced_results = self._kg_enhance(original_query, filtered_results) # 知识图谱增强
        latency_breakdown["kg_enhance"] = round(time.time() - t_kg, 3) # 记录知识图谱增强耗时

        # 重排
        t_rerank = time.time()
        # 🔧 RAG-01 修复：多样性重排需要「候选池 > 最终条数」，否则 `_diversity_rerank` 的
        #    首判 `len(results) <= top_k` 恒真 → 过滤逻辑永不执行（死代码，见 docs/KNOWN-ISSUES.md §2.26-E）。
        #    故启用多样性重排时，重排/截断到候选池 `dynamic_top_k`（5/10/20），
        #    再由 `_diversity_rerank` 收敛到最终条数 `dynamic_rerank_top_k`（3/5/10）。
        rerank_pool = dynamic_top_k if self.enable_diversity_rerank else dynamic_rerank_top_k
        if dm.should_skip_rerank():
            logger.info("[检索] 降级：跳过Rerank，直接使用RRF结果")
            reranked_results = kg_enhanced_results[:rerank_pool]
        else:
            reranked_results = self._rerank(original_query, kg_enhanced_results, rerank_pool) # 重排
        latency_breakdown["rerank"] = round(time.time() - t_rerank, 3) # 记录重排耗时

        # 多样性重排（RAG-01：候选池 rerank_pool > top_k → 过滤逻辑真正可达）
        t_diversity = time.time()
        final_results = self._diversity_rerank(reranked_results, dynamic_rerank_top_k)
        latency_breakdown["diversity_rerank"] = round(time.time() - t_diversity, 3) # 记录多样性重排耗时

        # 改写归因：哪一路改写真正进入了最终结果（0 贡献即判定改写无收益）
        self.last_rewrite_report = rewrite_governor.observe(
            original_query, variants, final_results)
        for result in final_results:
            result.pop("_source_query", None)
        latency_breakdown["rewrite_decision"] = self.last_rewrite_report.get("decision", "")
        record_attributes(get_current_span(), {
            "rag.rewrite.decision": self.last_rewrite_report.get("decision", ""),
            "rag.rewrite.winning_strategy": self.last_rewrite_report.get("winning_strategy", ""),
            "rag.result_count": len(final_results),
        })

        if not dm.should_skip_cache():
            self.redis_cache.cache_retrieval(original_query, final_results, filter_dict.get("tenant_id") if isinstance(filter_dict, dict) else None)

        elapsed = time.time() - start_time
        latency_breakdown["total"] = round(elapsed, 3)
        latency_breakdown["fusion_method"] = self.fusion_method
        latency_breakdown["retrieval_mode"] = retrieval_mode
        latency_breakdown["dynamic_top_k"] = dynamic_top_k

        logger.info(
            f"[检索] 完成，返回 {len(final_results)} 条结果，"
            f"耗时 {elapsed:.2f}s，模式={retrieval_mode}，"
            f"top_k={dynamic_top_k}，融合={self.fusion_method}"
        )

        return final_results

    def _build_filter_expr(self, filter_dict: Dict) -> str:
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, int):
                conditions.append(f"{key} == {value}")
            elif isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
        return " && ".join(conditions) if conditions else None

    def _rerank(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        top_k = top_k or self.rerank_top_k
        try:
            reranked_results = reranker_service.rerank_results(
                query=query,
                results=results,
                top_k=top_k
            )
            return reranked_results
        except Exception as e:
            logger.error(f"重排序失败，使用默认排序：{e}")
            dm = self._get_degradation_manager()
            dm.record_failure("reranker", str(e))
            return self._default_rerank(results, top_k)

    def _default_rerank(self, results: List[Dict], top_k: int = None) -> List[Dict]:
        top_k = top_k or self.rerank_top_k
        for result in results:
            combined_score = result.get("vector_score", 0) * 0.7 + (result.get("es_score") or 0) * 0.3
            result["combined_score"] = combined_score

        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return results[:top_k]


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    hybrid_retriever = HybridRetriever()
else:
    hybrid_retriever = None
