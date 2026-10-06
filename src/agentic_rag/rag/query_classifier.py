"""
查询分类器
判断query类型(simple/fuzzy/complex)，决定扩展策略

⚠️ 低配版说明：
   架构图设计为 BERT-base 微调模型做意图识别（分类精度更高，但需要标注数据和 GPU 训练）。
   实际使用 LLM Prompt（DeepSeek）实现意图分类，成本低、零训练数据，适合原型验证阶段。

分级策略（面试核心考点）：
- 明确查询(simple)：仅BM25+向量混合检索，不扩展，节省成本
- 模糊查询(fuzzy)：同义词 + Multi-Query，低成本提升召回
- 复杂查询(complex)：结构化分解 + HyDE，深度覆盖
"""

from typing import List
import logging
import re
import os

logger = logging.getLogger(__name__)


def _get_llm():
    from agent_harness.providers.deepseek import DeepSeekLLMService
    return DeepSeekLLMService()


class QueryClassifier:
    """查询分类器：判断query类型，决定扩展策略"""

    def __init__(self):
        self._llm = None
        self.classify_prompt = """分析用户查询的复杂度，判断属于哪种类型。

用户查询："{query}"

分类标准：
- "simple"：查询清晰明确，关键词具体，直接检索即可（如"公司年假多少天"、"报销流程是什么"）
- "fuzzy"：查询模糊、口语化、表述不精确，需要改写才能检索（如"怎么退单"、"那个东西怎么用"）
- "complex"：查询包含多个维度/多步推理，需要拆分子问题（如"刚入职五险一金怎么交"、"办信用卡需要什么条件和材料"）

只输出一个分类词：simple、fuzzy 或 complex，不要解释。"""

    def classify(self, query: str) -> str:
        if len(query) <= 4:
            return "fuzzy"

        if self._is_likely_simple(query):
            return "simple"

        if self._llm is None:
            self._llm = _get_llm()
        try:
            prompt = self.classify_prompt.format(query=query)
            messages = [
                {"role": "system", "content": "你是一个查询分类器，只输出分类结果。"},
                {"role": "user", "content": prompt}
            ]

            response = self._llm._call_llm(messages)
            result = response.get("text", "").strip().lower()

            if "simple" in result:
                query_type = "simple"
            elif "complex" in result:
                query_type = "complex"
            else:
                query_type = "fuzzy"

            logger.info(f"查询分类：{query} -> {query_type}")
            return query_type

        except Exception as e:
            logger.error(f"查询分类失败：{e}，默认fuzzy")
            return "fuzzy"

    def _is_likely_simple(self, query: str) -> bool:
        """基于规则的快速判断，避免不必要的LLM调用"""
        specific_patterns = [
            r"多少天", r"多少钱", r"什么时间", r"在哪里", r"谁的",
            r"流程", r"规定", r"标准", r"制度", r"政策",
            r"\d+天", r"\d+元", r"\d+%",
        ]

        specific_keywords = [
            "年假", "社保", "公积金", "工资", "考勤", "加班费", "请假",
            "合同", "绩效", "培训", "调岗", "入职", "离职", "报销",
            "库存", "配件", "维修", "工单", "退货", "退款",
            "五险一金", "带薪", "休假", "薪资", "薪酬",
        ]

        vague_patterns = [
            r"怎么", r"如何", r"那个", r"这个", r"能不能",
            r"可不可以", r"有没有", r"啥", r"啥子",
        ]

        specific_count = sum(1 for p in specific_patterns if re.search(p, query))
        has_specific_keyword = any(kw in query for kw in specific_keywords)
        vague_count = sum(1 for p in vague_patterns if re.search(p, query))

        if (specific_count > 0 or has_specific_keyword) and vague_count == 0:
            return True
        if has_specific_keyword and vague_count <= 1:
            return True
        if vague_count >= 2:
            return False

        return False


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    query_classifier = QueryClassifier()
else:
    query_classifier = None
