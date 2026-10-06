import os
import logging

from agentic_rag.config import settings

logger = logging.getLogger(__name__)

# 尝试导入 CrossEncoder
has_reranker = False
try:
    from sentence_transformers import CrossEncoder
    has_reranker = True
    logger.info("sentence_transformers 已加载")
except ImportError:
    logger.warning("sentence_transformers 未安装，重排序功能将不可用")


class RerankerService:
    """BGE-Reranker 重排序服务

    ⚠️ 低配版说明：
       架构图设计为 BGE-Reranker-v2-m3（更强语义理解，模型体积 ~2GB）。
       实际使用 bge-reranker-base（~440MB），在 16GB 内存无 GPU 机器上
       可正常加载运行，精度略低但足够满足原型需求。
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.available = has_reranker
        
        if self.available:
            try:
                logger.info(f"加载重排序模型：{model_name}")
                self.model = CrossEncoder(model_name, max_length=512)
                logger.info("重排序模型加载完成")
            except Exception as e:
                logger.error(f"加载重排序模型失败：{e}")
                self.available = False
    
    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[int]:
        """重排序段落列表"""
        if not self.available:
            logger.error("重排序功能不可用，因为 sentence_transformers 未安装")
            # 返回原始顺序
            return list(range(min(top_k, len(passages))))
        
        try:
            # 构建 (query, passage) 对
            pairs = [[query, passage] for passage in passages]
            
            # 计算得分
            scores = self.model.predict(pairs)
            
            # 按得分排序
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # 返回前 top_k 个索引
            return sorted_indices[:top_k]
        except Exception as e:
            logger.error(f"重排序失败：{e}")
            # 发生错误时返回原始顺序
            return list(range(min(top_k, len(passages))))
    
    def rerank_results(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        """重排序检索结果"""
        if not self.available:
            logger.error("重排序功能不可用，因为 sentence_transformers 未安装")
            # 返回原始结果
            return results[:top_k]
        
        try:
            # 提取段落内容
            passages = [result.get("content", "") for result in results]
            
            # 重排序
            sorted_indices = self.rerank(query, passages, top_k)
            
            # 按排序结果重新组织
            sorted_results = [results[i] for i in sorted_indices]
            
            # 添加重排序得分
            for i, result in enumerate(sorted_results):
                result["rerank_score"] = top_k - i  # 简单的得分
                result["rank"] = i + 1
            
            return sorted_results
        except Exception as e:
            logger.error(f"重排序结果失败：{e}")
            # 发生错误时返回原始结果
            return results[:top_k]


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    reranker_service = RerankerService()
else:
    reranker_service = None
