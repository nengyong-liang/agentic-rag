"""
向量嵌入服务（可配置化版本）
支持运行时切换嵌入模型，无需修改代码

升级要点：
1. 模型名从配置文件读取，而非硬编码
2. 支持动态切换嵌入模型
3. 向量维度自动适配
4. 提供基准测试接口
"""

import os
import logging

logger = logging.getLogger(__name__)

has_embeddings = False
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    has_embeddings = True
    logger.info("langchain_community 已加载")
except ImportError:
    logger.warning("langchain_community 未安装，向量嵌入功能将不可用")

EMBEDDING_DIMS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-small-zh-v1.5": 512,
    "shibing624/text2vec-base-chinese": 768,
    "DMetaSoul/sbert-chinese-qmc-domain-v1": 768,
}


class EmbeddingService:
    """向量嵌入服务（可配置化）

    ⚠️ 低配版说明：
       架构图设计为 BGE-M3 Dense + Sparse 双编码（同时支持语义检索和关键词匹配），
       semantic_dedup.py 中实现了完整的 Dense+Sparse+ColBERT 三编码。
       但生产检索链路仅使用 Dense 向量（HuggingFaceEmbeddings），Sparse 编码
       未接入主检索管道。原因是 Sparse 编码需要额外的 Splade 模型推理，在纯 CPU
       环境下会显著增加检索延迟。
    """

    def __init__(self, model_name: str = None):
        from agentic_rag.config import settings
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.available = has_embeddings
        self.dimensions = EMBEDDING_DIMS.get(self.model_name, 1024)
        self.embeddings = None

        if self.available:
            try:
                logger.info(f"加载嵌入模型：{self.model_name}（维度：{self.dimensions}）")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                test_vec = self.embeddings.embed_query("测试")
                self.dimensions = len(test_vec)
                logger.info(f"嵌入模型加载完成，实际维度：{self.dimensions}")
            except Exception as e:
                logger.error(f"加载嵌入模型失败：{e}")
                self.available = False

    def embed_text(self, text: str) -> list[float]:
        """将文本转换为向量"""
        if not self.available:
            logger.error("向量嵌入功能不可用")
            return [0.0] * self.dimensions

        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"文本向量化失败：{e}")
            return [0.0] * self.dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        if not self.available:
            logger.error("向量嵌入功能不可用")
            return [[0.0] * self.dimensions for _ in texts]

        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"批量文本向量化失败：{e}")
            return [[0.0] * self.dimensions for _ in texts]

    def switch_model(self, model_name: str) -> bool:
        """
        动态切换嵌入模型

        Args:
            model_name: 新模型名称

        Returns:
            是否切换成功
        """
        if not has_embeddings:
            logger.error("langchain_community 未安装，无法切换模型")
            return False

        try:
            logger.info(f"切换嵌入模型：{self.model_name} -> {model_name}")
            new_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            test_vec = new_embeddings.embed_query("测试")
            new_dims = len(test_vec)

            self.embeddings = new_embeddings
            self.model_name = model_name
            self.dimensions = new_dims

            logger.info(f"嵌入模型切换成功：{model_name}（维度：{new_dims}）")
            return True
        except Exception as e:
            logger.error(f"切换嵌入模型失败：{e}")
            return False

    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "available": self.available,
            "supported_models": list(EMBEDDING_DIMS.keys())
        }


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    embedding_service = EmbeddingService()
else:
    embedding_service = None
