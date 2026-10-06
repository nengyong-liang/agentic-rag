"""
语义去重模块

面试核心考点：
1. 查询扩展后，多个变体Query可能语义高度相似
2. 余弦相似度 > 0.92 的Query合并，减少无效检索
3. 基于Sentence-Transformer计算语义相似度
4. 降级方案：字符级Jaccard相似度（不依赖模型）

BGE-M3 多向量输出支持：
1. Dense向量（1024维）：语义相似度检索
2. Sparse向量：关键词匹配（BM25替代）
3. ColBERT向量：细粒度交互（可选，显存消耗大）
"""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

SEMANTIC_DEDUP_THRESHOLD = 0.92
CHAR_JACCARD_THRESHOLD = 0.75


class SemanticDeduplicator:
    """语义去重器"""

    def __init__(self, use_model: bool = True, threshold: float = SEMANTIC_DEDUP_THRESHOLD):
        self._model = None
        self._use_model = use_model
        self.threshold = threshold
        self._model_loaded = False

        if use_model:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "BAAI/bge-m3"
            logger.info(f"[语义去重] 尝试加载模型：{model_name}")
            self._model = SentenceTransformer(model_name)
            self._model_loaded = True
            logger.info("[语义去重] 模型加载成功")
        except ImportError:
            logger.warning("[语义去重] sentence-transformers未安装，回退字符级去重")
            self._use_model = False
        except Exception as e:
            logger.warning(f"[语义去重] 模型加载失败：{e}，回退字符级去重")
            self._use_model = False

    def deduplicate_queries(self, queries: List[str]) -> List[str]:
        """
        对查询列表进行语义去重

        Args:
            queries: 查询列表

        Returns:
            去重后的查询列表
        """
        if len(queries) <= 1:
            return queries

        if self._use_model and self._model_loaded:
            return self._model_based_dedup(queries)
        else:
            return self._char_based_dedup(queries)

    def _model_based_dedup(self, queries: List[str]) -> List[str]:
        """基于Sentence-Transformer的语义去重"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        embeddings = self._model.encode(queries, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)

        keep = [True] * len(queries)
        for i in range(len(queries)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(queries)):
                if not keep[j]:
                    continue
                if sim_matrix[i][j] > self.threshold:
                    keep[j] = False
                    logger.info(
                        f"[语义去重] 合并相似Query：'{queries[j]}' "
                        f"→ '{queries[i]}' (sim={sim_matrix[i][j]:.3f})"
                    )

        result = [q for q, k in zip(queries, keep) if k]
        logger.info(f"[语义去重] {len(queries)} → {len(result)} 条（模型版，阈值={self.threshold}）")
        return result

    def _char_based_dedup(self, queries: List[str]) -> List[str]:
        """基于字符级Jaccard相似度的去重（降级方案）"""
        keep = [True] * len(queries)

        for i in range(len(queries)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(queries)):
                if not keep[j]:
                    continue
                sim = self._char_jaccard(queries[i], queries[j])
                if sim > CHAR_JACCARD_THRESHOLD:
                    keep[j] = False
                    logger.info(
                        f"[语义去重] 合并相似Query：'{queries[j]}' "
                        f"→ '{queries[i]}' (jaccard={sim:.3f})"
                    )

        result = [q for q, k in zip(queries, keep) if k]
        logger.info(f"[语义去重] {len(queries)} → {len(result)} 条（字符版，阈值={CHAR_JACCARD_THRESHOLD}）")
        return result

    def _char_jaccard(self, s1: str, s2: str) -> float:
        """字符级Jaccard相似度"""
        set1 = set(s1)
        set2 = set(s2)
        if not set1 or not set2:
            return 0.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)


class BGE_M3_Encoder:
    """
    BGE-M3 多向量编码器

    支持三种输出：
    1. Dense向量（1024维）：语义相似度检索
    2. Sparse向量：关键词匹配（BM25替代）
    3. ColBERT向量：细粒度交互（可选）

    面试核心：
    - Dense适合语义相似，Sparse适合关键词精确匹配
    - 混合检索时Dense走Milvus，Sparse走ES
    - ColBERT精度最高但显存消耗大，适合小规模精排
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model = None
        self._model_name = model_name
        self._model_loaded = False
        self._try_load_model()

    def _try_load_model(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"[BGE-M3] 尝试加载模型：{self._model_name}")
            self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
            self._model_loaded = True
            logger.info("[BGE-M3] 模型加载成功（支持Dense+Sparse+ColBERT）")
        except ImportError:
            logger.warning("[BGE-M3] FlagEmbedding未安装，仅支持Dense向量")
            self._try_load_sentence_transformer()
        except Exception as e:
            logger.warning(f"[BGE-M3] FlagEmbedding加载失败：{e}，回退SentenceTransformer")
            self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._model_loaded = True
            self._sparse_available = False
            logger.info("[BGE-M3] SentenceTransformer加载成功（仅Dense向量）")
        except Exception as e:
            logger.error(f"[BGE-M3] 所有模型加载失败：{e}")
            self._model_loaded = False

    def encode_dense(self, texts: List[str]) -> List[List[float]]:
        """编码Dense向量（语义检索）"""
        if not self._model_loaded:
            return []

        try:
            if hasattr(self._model, 'encode'):
                result = self._model.encode(texts, return_dense=True)
                if isinstance(result, dict):
                    return result.get("dense_vecs", []).tolist()
                return result.tolist()
        except Exception as e:
            logger.error(f"[BGE-M3] Dense编码失败：{e}")

        return []

    def encode_sparse(self, texts: List[str]) -> List[Dict]:
        """编码Sparse向量（关键词匹配）"""
        if not self._model_loaded:
            return []

        try:
            if hasattr(self._model, 'encode'):
                result = self._model.encode(texts, return_sparse=True)
                if isinstance(result, dict):
                    return result.get("lexical_weights", [])
        except Exception as e:
            logger.error(f"[BGE-M3] Sparse编码失败：{e}")

        return []

    def encode_hybrid(self, texts: List[str]) -> Dict:
        """
        混合编码：同时输出Dense和Sparse向量

        Returns:
            {
                'dense': List[List[float]],
                'sparse': List[Dict],
                'colbert_vecs': Optional[List]
            }
        """
        if not self._model_loaded:
            return {'dense': [], 'sparse': [], 'colbert_vecs': None}

        try:
            if hasattr(self._model, 'encode'):
                result = self._model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
                if isinstance(result, dict):
                    dense = result.get("dense_vecs", [])
                    sparse = result.get("lexical_weights", [])
                    return {
                        'dense': dense.tolist() if hasattr(dense, 'tolist') else dense,
                        'sparse': sparse,
                        'colbert_vecs': None
                    }
        except Exception as e:
            logger.error(f"[BGE-M3] 混合编码失败：{e}")

        return {'dense': [], 'sparse': [], 'colbert_vecs': None}


semantic_deduplicator = SemanticDeduplicator(use_model=False)
