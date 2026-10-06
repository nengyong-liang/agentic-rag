"""
查询改写器（工业级分级扩展版）

支持4种扩展策略 + 前置分类判断：
1. 同义词扩展（Synonym）：领域词典/LLM生成近义词，低成本精准扩展
2. Multi-Query扩展：LLM生成多个不同角度搜索词，覆盖广
3. HyDE假设文档嵌入：LLM生成假设答案→用答案向量检索，跨越语义鸿沟
4. 结构化分解（Step-back）：拆分子问题分别检索，解决多维度需求

分级策略（面试核心考点）：
- 明确查询(simple)：仅BM25+向量混合检索，不扩展，节省成本
- 模糊查询(fuzzy)：同义词 + Multi-Query，低成本提升召回
- 复杂查询(complex)：结构化分解 + HyDE，深度覆盖
"""

from typing import List, Dict, Optional
import logging
import re
import os

logger = logging.getLogger(__name__)


def _get_llm():
    """延迟导入LLM，避免循环依赖"""
    from agent_harness.providers.deepseek import DeepSeekLLMService
    return DeepSeekLLMService()


from agentic_rag.rag.query_classifier import QueryClassifier, query_classifier as _classifier
from agentic_rag.rag.rewrite_governor import (
    QueryVariant, STRATEGY_ORIGINAL, STRATEGY_SYNONYM, STRATEGY_MULTI_QUERY,
    STRATEGY_STEPBACK, STRATEGY_HYDE, estimate_intent_drift,
)


class SynonymExpander:
    """同义词扩展器：领域词典 + LLM生成"""

    ENTERPRISE_SYNONYMS = {
        "退货": ["退款", "退单", "售后申请", "退换货"],
        "报销": ["费用报销", "报销申请", "费用申请"],
        "入职": ["新员工入职", "入职办理", "入职手续", "新入职"],
        "离职": ["辞职", "离职手续", "离职办理", "退职"],
        "年假": ["带薪年假", "年休假", "法定年假"],
        "社保": ["社会保险", "五险", "五险一金"],
        "公积金": ["住房公积金", "公积金缴纳"],
        "工资": ["薪资", "薪酬", "工资发放", "工资条"],
        "考勤": ["打卡", "出勤", "考勤记录", "签到"],
        "加班": ["加班申请", "加班费", "加班审批"],
        "请假": ["假期申请", "休假", "请假审批", "调休"],
        "合同": ["劳动合同", "用工合同", "聘用合同"],
        "绩效": ["绩效考核", "KPI", "绩效评估", "业绩考核"],
        "培训": ["员工培训", "培训申请", "技能培训"],
        "调岗": ["岗位调整", "岗位调动", "内部调动"],
        "库存": ["库存查询", "库存数量", "库存余量", "存货"],
        "配件": ["零部件", "零件", "备件"],
        "维修": ["维修工单", "维修申请", "修理"],
        "工单": ["维修工单", "服务工单", "工单生成"],
    }

    def __init__(self):
        self._llm = None
        self.synonym_prompt = """为以下查询生成2-3个同义词或近义表述，用于扩展检索范围。

查询："{query}"

要求：
1. 只生成同义词或近义表述，不要改变核心含义
2. 每行一个，不要编号，不要解释
3. 适合企业知识库检索场景
4. 如果查询已经足够精确，可以不生成"""

    def expand(self, query: str) -> List[str]:
        """同义词扩展：先查词典，再LLM补充"""
        synonyms = [query] # 初始化同义词列表，包含原始查询

        #ENTERPRISE_SYNONYMS 企业词典，包含常用同义词和近义词
        for key, values in self.ENTERPRISE_SYNONYMS.items():
            if key in query: # 如果查询包含关键词
                for syn in values:
                    expanded = query.replace(key, syn) # 替换关键词为同义词
                    if expanded not in synonyms:
                        synonyms.append(expanded)

        if len(synonyms) > 3:# 如果同义词数量超过3个
            logger.info(f"同义词扩展（词典）：{query} -> {synonyms}")
            return synonyms[:4] # 返回前4个同义词

        if self._llm is None: # 如果LLM服务未初始化，初始化LLM服务
            self._llm = _get_llm()
        try:
            prompt = self.synonym_prompt.format(query=query) # 格式化提示词，包含查询内容
            messages = [
                {"role": "system", "content": "你是同义词生成专家，只输出同义词列表。"},
                {"role": "user", "content": prompt}
            ]

            response = self._llm._call_llm(messages) # 调用LLM服务，生成同义词
            text = response.get("text", "").strip() # 提取文本内容，移除首尾空格

            if text: # 如果文本内容不为空
                # 提取同义词列表，移除编号和空格
                lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
                for line in lines:
                    if line not in synonyms and len(line) > 2: # 如果同义词长度大于2，且未重复添加
                        synonyms.append(line) # 添加同义词到列表

        except Exception as e:
            logger.error(f"LLM同义词扩展失败：{e}")

        if len(synonyms) > 4:
            synonyms = synonyms[:4]

        logger.info(f"同义词扩展：{query} -> {synonyms}")
        return synonyms


class HyDEExpander:
    """HyDE假设文档嵌入：LLM生成假设答案，用答案向量检索"""

    def __init__(self):
        self._llm = None
        self.hyde_prompt = """请针对以下问题，生成一段假设性的答案文档。这段文档将用于在知识库中检索相关内容。

        问题："{query}"

        要求：
        1. 生成一段100-200字的假设答案
        2. 使用专业、正式的文档表述风格
        3. 包含可能出现在真实文档中的关键词和术语
        4. 不需要完全正确，但要覆盖问题可能涉及的要点
        5. 只输出假设答案文本，不要解释"""

    def generate_hypothetical_answer(self, query: str) -> str:
        """生成假设答案用于向量检索"""
        if self._llm is None:
            self._llm = _get_llm() # 初始化LLM服务
        try:
            prompt = self.hyde_prompt.format(query=query) # 格式化提示词，包含查询内容
            messages = [
                {"role": "system", "content": "你是一个企业知识库文档生成专家，生成假设性答案文档用于检索。"},
                {"role": "user", "content": prompt}
            ]

            response = self._llm._call_llm(messages)
            hypothetical = response.get("text", "").strip() # 提取文本内容，移除首尾空格

            if not hypothetical: # 如果假设答案为空ss
                logger.warning(f"HyDE生成空结果，使用原查询：{query}")
                return query

            logger.info(f"HyDE假设答案：{query} -> {hypothetical[:80]}...")
            return hypothetical

        except Exception as e:
            logger.error(f"HyDE生成失败：{e}，使用原查询")
            return query


class StepBackExpander:
    """结构化分解：将复杂查询拆分为多个子问题（支持知识图谱增强）"""

    def __init__(self):
        self._llm = None
        self._kg = None
        self.decompose_prompt = """将以下复杂查询拆分为2-3个独立的子问题，每个子问题覆盖一个维度。
                                复杂查询："{query}"

                                要求：
                                1. 每个子问题独立可检索
                                2. 覆盖原查询的不同维度（如流程、条件、材料等）
                                3. 每个子问题不超过30字
                                4. 每行一个子问题，不要编号，不要解释
                                5. 子问题之间要有差异性，不要重复"""

    def _get_kg(self):
        """延迟导入知识图谱，避免循环依赖"""
        from agentic_rag.rag.knowledge_graph import knowledge_graph
        return knowledge_graph

    def decompose(self, query: str, use_kg: bool = True) -> List[str]:
        """结构化分解复杂查询（优先使用知识图谱引导分解）"""
        sub_queries = [query]

        if use_kg: # 如果使用知识图谱增强分解
            kg_sub_queries = self._kg_decompose(query) # 调用知识图谱分解函数
            if len(kg_sub_queries) > 1:
                for sq in kg_sub_queries:
                    if sq not in sub_queries and len(sq) > 2:
                        sub_queries.append(sq)
                logger.info(f"[KG增强分解] {query} -> {sub_queries}")
                if len(sub_queries) > 4:
                    return sub_queries[:4]
                return sub_queries
        
        # 如果知识图谱分解失败，使用LLM分解
        llm_sub_queries = self._llm_decompose(query) # 调用LLM分解函数  
        for sq in llm_sub_queries:
            if sq not in sub_queries and len(sq) > 2:
                sub_queries.append(sq)

        if len(sub_queries) > 4:
            sub_queries = sub_queries[:4]

        logger.info(f"结构化分解：{query} -> {sub_queries}")
        return sub_queries

    def _kg_decompose(self, query: str) -> List[str]:
        """基于知识图谱的多跳推理分解"""
        try:
            if self._kg is None:
                self._kg = self._get_kg() # 初始化知识图谱
            kg_result = self._kg.multi_hop_reason(query, max_hops=2) # 调用知识图谱多跳推理函数
            if isinstance(kg_result, list) and len(kg_result) > 1:
                return kg_result
        except Exception as e:
            logger.warning(f"KG多跳推理失败，回退到LLM分解：{e}")
        return [query]

    def _llm_decompose(self, query: str) -> List[str]:
        """基于LLM的结构化分解"""
        if self._llm is None:
            self._llm = _get_llm()
        try:
            prompt = self.decompose_prompt.format(query=query)
            messages = [
                {"role": "system", "content": "你是查询分解专家，将复杂问题拆分为可独立检索的子问题。"},
                {"role": "user", "content": prompt}
            ]

            response = self._llm._call_llm(messages)
            text = response.get("text", "").strip()

            if not text:
                return [query]

            lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
            sub_queries = []
            for line in lines:
                if line not in sub_queries and len(line) > 2:
                    sub_queries.append(line)

            return sub_queries[:3] # 限制子问题数量为3个

        except Exception as e:
            logger.error(f"结构化分解失败：{e}，使用原查询")
            return [query]


class QueryRewriter:
    """查询改写器：分级扩展策略（面试核心）"""

    def __init__(self):
        self.classifier = _classifier
        self.synonym_expander = SynonymExpander()
        self.hyde_expander = HyDEExpander()
        self.stepback_expander = StepBackExpander()
        self._llm = None

        self.rewrite_prompt = """你是一个企业知识库搜索优化专家。用户输入了模糊表述，请生成多个不同角度的搜索词。

                                原始查询："{query}"

                                要求：
                                1. 分析用户意图，从不同角度生成搜索词
                                2. 生成1-3个更精确的搜索词
                                3. 每个搜索词不超过30字
                                4. 每行一个搜索词，不要编号，不要解释
                                5. 如果原始查询已经清晰准确，直接返回原查询即可
                                6. 搜索词之间要有差异性，覆盖不同的表述方式
                                7. 仅生成与企业知识库相关的内容"""

    def rewrite(self, query: str) -> str:
        """单查询改写（兼容旧接口）"""
        queries = self.rewrite_multi(query)
        if queries:
            return queries[0]
        return query

    def rewrite_multi(self, query: str) -> List[str]:
        """
        分级扩展：根据query类型选择不同扩展策略

        策略映射：
        - simple：不扩展，直接检索 [query]
        - fuzzy：同义词扩展 + Multi-Query
        - complex：结构化分解 + HyDE
        """
        return [v.query for v in self.plan_variants(query)]

    def plan_variants(self, query: str, use_llm: bool = True) -> List[QueryVariant]:
        """产出带策略标签的改写变体（供 RewriteGovernor 做命中率归因与回退判定）

        Args:
            query: 原始查询
            use_llm: False 时只走零成本词典/图谱路径，不调用 LLM 类策略
                     （Multi-Query / HyDE 依赖 LLM，用于离线评测与降级）
        """
        query_type = self.classifier.classify(query)
        logger.info(f"[分级扩展] 查询类型={query_type}, query={query}")

        variants: List[QueryVariant] = [QueryVariant(query, STRATEGY_ORIGINAL)]
        seen = {query}
        limit = 4 if query_type == "fuzzy" else 5

        def safe(fn, fallback, label: str):
            try:
                return fn()
            except Exception as e:
                logger.warning(f"[分级扩展] {label} 策略失败，跳过: {e}")
                return fallback

        def add(text: str, strategy: str):
            text = (text or "").strip()
            if text and text not in seen:
                seen.add(text)
                variants.append(QueryVariant(text, strategy))

        if query_type == "fuzzy":
            for q in safe(lambda: self.synonym_expander.expand(query), [], "synonym"):
                add(q, STRATEGY_SYNONYM)
            if use_llm:
                for q in safe(lambda: self._multi_query_expand(query), [], "multi_query"):
                    add(q, STRATEGY_MULTI_QUERY)
        elif query_type == "complex":
            for q in safe(lambda: self.stepback_expander.decompose(query), [], "stepback"):
                add(q, STRATEGY_STEPBACK)
            if use_llm:
                hyde = safe(lambda: self.hyde_expander.generate_hypothetical_answer(query),
                            "", "hyde")
                add(hyde, STRATEGY_HYDE)

        if len(variants) > limit:
            variants = variants[:limit]

        logger.info(f"[分级扩展-{query_type}] {query} -> "
                    f"{[(v.strategy, v.query[:24]) for v in variants]}")
        return variants

    def _multi_query_expand(self, query: str) -> List[str]:
        """Multi-Query扩展：LLM生成多个变体"""
        if self._llm is None:
            self._llm = _get_llm()
        try:
            prompt = self.rewrite_prompt.format(query=query)
            messages = [
                {"role": "system", "content": "你是一个企业知识库搜索优化专家，擅长从多个角度生成搜索词以提高检索召回率。"},
                {"role": "user", "content": prompt}
            ]

            response = self._llm._call_llm(messages) # 调用LLM服务，生成Multi-Query
            rewritten = response.get("text", "").strip() # 提取文本内容，移除首尾空格

            if not rewritten:
                return [query]

            lines = [line.strip() for line in rewritten.split("\n") if line.strip()]
            lines = [line.lstrip("0123456789.-) ") for line in lines]

            seen = {query}
            unique = []
            for line in lines:
                if line not in seen and len(line) > 2:
                    unique.append(line)
                    seen.add(line)

            return unique[:3]

        except Exception as e:
            logger.error(f"Multi-Query扩展失败：{e}")
            return [query]

    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """批量改写查询（兼容旧接口）"""
        return [self.rewrite(q) for q in queries]

    def get_expansion_info(self, query: str) -> Dict:
        """获取扩展详情（用于全链路日志和可解释性）"""
        query_type = self.classifier.classify(query) # 获取查询类型
        variants = self.plan_variants(query) # 生成带策略标签的扩展查询
        expanded = [v.query for v in variants]

        return {
            "original_query": query, # 原始查询语句
            "query_type": query_type, # 查询类型
            "expanded_queries": expanded, # 扩展查询列表
            "variants": [
                {"query": v.query, "strategy": v.strategy,
                 "intent_drift": estimate_intent_drift(query, v.query)}
                for v in variants
            ],
            "expansion_count": len(expanded) - 1, # 扩展查询数量
            "strategy": {
                "simple": "无扩展，直接检索",
                "fuzzy": "同义词扩展 + Multi-Query",
                "complex": "结构化分解 + HyDE"
            }.get(query_type, "未知") # 获取扩展策略
        }


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    query_rewriter = QueryRewriter()
else:
    query_rewriter = None
