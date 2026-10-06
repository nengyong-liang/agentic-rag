"""
意图分类器（轻量级，第2层路由）

三层路由架构中的第2层：
  Layer 1: 关键词规则（Java RouteServiceImpl + RAGAgent.route_query）
  Layer 2: 本模块 - Embedding 相似度分类（CPU 可运行，10-50ms）
  Layer 3: LLM Prompt 分类（LLMRouter / QueryClassifier）

⚠️ 低配版说明：
    工业界 Layer 2 通常使用 BERT-base 微调模型（精度更高，但需要标注数据和 GPU 微调）。
    本模块使用 sentence-transformers 做 Embedding + 预定义种子示例的余弦相似度分类，
    无需训练数据、无需 GPU，在 16GB CPU 机器上即可运行。
    分类精度不如微调 BERT，但对原型验证已足够。
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 各意图类别的种子示例查询（尽可能覆盖项目业务场景）
SEED_EXAMPLES: Dict[str, List[str]] = {
    "knowledge_query": [
        "公司年假政策是什么",
        "社保缴纳比例是多少",
        "报销流程怎么走",
        "劳动合同怎么签",
        "工资发放时间",
        "加班的调休规定",
        "制度条款有哪些",
        "培训管理制度",
        "绩效考评标准",
        "员工福利有哪些",
    ],
    "tool_call": [
        "查询配件 ENG-001 的库存",
        "创建维修工单",
        "查询员工张三的信息",
        "低库存的配件有哪些",
        "生成一个采购订单",
        "查一下这个配件的价格",
        "搜索制度条款",
        "批量导入员工信息",
        "核验工单状态",
        "下单采购配件",
    ],
    "complex_query": [
        "刚入职五险一金怎么交",
        "离职补偿金怎么算",
        "加班费和绩效工资的关系",
        "对比不同岗位的薪资结构",
        "为什么社保基数每年会变",
        "分析近三个月的维修数据",
        "如何优化配件库存管理",
        "影响员工满意度的因素有哪些",
    ],
    "chitchat": [
        "你好",
        "谢谢",
        "你是谁",
        "你叫什么名字",
        "今天天气怎么样",
        "你能做什么",
        "再见",
        "好的",
    ],
}

# 类别名称映射（用于路由决策）
CLASS_TO_ROUTE = {
    "tool_call": "agent",
    "knowledge_query": "rag",
    "complex_query": "rag",
    "chitchat": "rag",
}

# 分类置信度阈值：低于此值则降级到 LLM（Layer 3）
CONFIDENCE_THRESHOLD = 0.55


class IntentClassifier:
    """Embedding 相似度意图分类器（第2层路由）"""

    def __init__(self, model_name: str = None):
        self._model = None
        self._model_name = model_name or "BAAI/bge-small-zh-v1.5"
        self._available = False
        self._seed_embeddings: Dict[str, List] = {}  # class_name -> [embeddings]
        self._try_load_model() # 尝试加载模型

    def _try_load_model(self):
        """尝试加载 embedding 模型"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"[IntentClassifier] 加载模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._available = True
            self._precompute_seed_embeddings() # 预计算种子示例的 embedding
            logger.info(f"[IntentClassifier] 模型加载成功，维度: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            logger.warning("[IntentClassifier] sentence-transformers 未安装，仅能使用前序关键词匹配")
        except Exception as e:
            logger.warning(f"[IntentClassifier] 模型加载失败: {e}，降级为纯关键词+LLM 路由")

    def _precompute_seed_embeddings(self):
        """预计算所有种子示例的 embedding"""
        if not self._available or self._model is None:
            return

        for intent, examples in SEED_EXAMPLES.items():
            embeddings = self._model.encode(examples, normalize_embeddings=True)
            self._seed_embeddings[intent] = embeddings
            logger.info(f"[IntentClassifier] 预计算 {intent} 的 {len(examples)} 个种子示例")

    def classify(self, query: str) -> Tuple[str, str, float]:
        """对查询进行意图分类

        Args:
            query: 用户输入

        Returns:
            (intent_class, route, confidence)
            - intent_class: knowledge_query / tool_call / complex_query / chitchat
            - route: rag / agent
            - confidence: 0-1 置信度
        """
        if not self._available:
            return ("unknown", "rag", 0.0)

        try:
            query_emb = self._model.encode([query], normalize_embeddings=True)[0]

            best_intent = "knowledge_query"
            best_similarity = 0.0

            for intent, seed_embs in self._seed_embeddings.items():
                # 计算与所有种子示例的余弦相似度，取最大值
                similarities = seed_embs @ query_emb  # 矩阵乘法，已归一化
                max_sim = float(similarities.max())

                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_intent = intent

            route = CLASS_TO_ROUTE.get(best_intent, "rag") # 路由决策
            logger.info(
                f"[IntentClassifier] {query} -> {best_intent} (route={route}, "
                f"confidence={best_similarity:.3f})"
            )
            return (best_intent, route, best_similarity)

        except Exception as e:
            logger.error(f"[IntentClassifier] 分类失败: {e}")
            return ("unknown", "rag", 0.0)

    def is_available(self) -> bool:
        return self._available


# 全局单例
if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    intent_classifier = IntentClassifier()
else:
    intent_classifier = None
