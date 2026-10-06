"""
三层路由判断服务
  Layer 1: 关键词规则（毫秒级，零成本）
  Layer 2: Embedding 相似度分类（10-50ms，CPU 可跑）
  Layer 3: LLM Prompt（数百毫秒，有成本）

附加能力：
  Layer 4: 模型路由（ModelRouter）— 路由后选择最优 LLM
  多意图检测 — 检测同时需 RAG+Agent 的复合查询

当 use_llm_route=false（默认）时，Java 端只走 Layer 1 关键词匹配。
本模块仅在 Java 端 use_llm_route=true 时被调用，此时完整走三层路由。
"""

import logging
import os
from typing import Dict, Optional

from agent_harness.providers.deepseek import DeepSeekLLMService

logger = logging.getLogger(__name__)

# ============================================================
# Layer 1 关键词配置 — 单⼀真相源（Single Source of Truth）
# ⚠️ Java 端 RouteServiceImpl 启动时通过 /route/config 拉取此配置
# ============================================================

# Layer 1：强信号关键词 —— 命中直接判定，不走后续层
STRONG_AGENT_KEYWORDS = [
    "创建工单", "生成工单", "批量导入", "下单采购",
    "核验员工", "核验工单", "查询库存", "配件库存",
]
STRONG_RAG_KEYWORDS = [
    "政策", "制度", "规定", "流程", "标准", "指南",
    "是什么", "什么是", "什么意思", "如何理解",
]

# 多意图检测：复合连接词
# 当查询包含这些词时，可能同时包含 RAG 和 Agent 意图
COMPOUND_CONNECTORS = ["和", "并且", "同时", "以及", "与", "并", "还要", "还有"]
# 多意图检测：各意图类别的独立关键词
MULTI_RAG_INDICATORS = ["制度", "政策", "规定", "流程", "是什么", "如何", "什么意思"]
MULTI_AGENT_INDICATORS = ["查询", "库存", "工单", "创建", "生成", "核验", "配件"]


def get_keywords_config() -> dict:
    """导出关键词配置供 Java 端/API 使用

    Java RouteServiceImpl 启动时调用 GET /route/config 获取此配置，
    用于 useLlmRoute=false 模式下的 Layer 1 关键词匹配。
    """
    return {
        "agent_keywords": list(STRONG_AGENT_KEYWORDS),
        "rag_keywords": list(STRONG_RAG_KEYWORDS),
        "multi_rag_indicators": list(MULTI_RAG_INDICATORS),
        "multi_agent_indicators": list(MULTI_AGENT_INDICATORS),
        "compound_connectors": list(COMPOUND_CONNECTORS),
        "description": "Layer 1 关键词路由配置 — Python 为单真相源",
    }


class LLMRouter:
    """三层路由判断服务

    ⚠️ 低配版说明：
       架构图设计为 BERT-base 微调模型做意图识别。
       实际使用 Embedding 相似度（Layer 2）+ LLM Prompt（Layer 3）的混合方案，
       Layer 2 无需 GPU / 无需训练数据，在 CPU 16GB 机器上即可运行。
       精度不如微调 BERT，但相比纯关键词已大幅提升。
    """

    def __init__(self):
        self.llm = DeepSeekLLMService()
        self._classifier = None

        self.route_prompt = """你是一个意图识别专家。请判断用户的问题应该走哪条处理路径。

        用户问题：{query}

        可选路径：
        - "rag"：用户在查询知识信息（如公司政策、规章制度、操作流程、技术文档等），需要从知识库检索回答
        - "agent"：用户需要执行操作或查询业务数据（如查询库存、创建工单、核验员工、搜索制度条款等），需要调用工具

        判断标准：
        1. 如果用户只是"了解/查询/知道"某个信息 → rag
        2. 如果用户需要"查询具体数据/执行操作/调用系统" → agent
        3. 如果不确定，默认走 rag

        只输出一个词："rag" 或 "agent"，不要解释。"""

        self.domain_prompt = """你是一个领域识别专家。请判断用户问题属于哪个业务领域。

        用户问题：{query}

        可选领域：
        - "hr"：人事管理（入职、离职、工资、社保、劳动合同、考勤等）
        - "it"：IT技术（电脑、网络、邮箱、VPN、服务器等）
        - "finance"：财务管理（报销、发票、预算、采购等）
        - "general"：通用领域（不属于以上任何领域）

        只输出一个领域词，不要解释。"""

    @property
    def classifier(self):
        """延迟加载意图分类器（避免 import 循环）"""
        if self._classifier is None:
            try:
                from agentic_rag.utils.intent_classifier import intent_classifier
                self._classifier = intent_classifier
            except ImportError:
                self._classifier = None
        return self._classifier

    def detect_multi_intent(self, query: str) -> Optional[Dict]:
        """检测多意图查询（同时需要 RAG + Agent）

        工业界实践：单查询可能同时包含知识查询和工具调用需求。
        ⚠️ 低配版：用关键词检测，工业界用多标签分类模型。

        Returns:
            {
                "is_multi": True,
                "sub_queries": {"rag": "...", "agent": "..."},
                "route": "multi"
            }
            或 None（非多意图）
        """
        has_rag_signal = any(kw in query for kw in MULTI_RAG_INDICATORS)
        has_agent_signal = any(kw in query for kw in MULTI_AGENT_INDICATORS)
        has_connector = any(c in query for c in COMPOUND_CONNECTORS) # 包含复合连接词

        if has_rag_signal and has_agent_signal and has_connector:
            logger.info(f"多意图检测: {query} -> 复合查询（RAG + Agent）")
            return {
                "is_multi": True,
                "sub_queries": {
                    "rag": query,
                    "agent": query,
                },
                "route": "multi",
            }

        return None

    def route_query(self, query: str) -> str:
        """三层路由判断 + 多意图检测

        优先级：
          多意图检测 → Layer 1（关键词）→ Layer 2（Embedding）→ Layer 3（LLM）
        """
        # ------ Step 0: 多意图检测 ------
        multi_intent = self.detect_multi_intent(query)
        if multi_intent:
            return "multi"

        # ------ Layer 1: 关键词强信号（毫秒级） ------
        query_lower = query.lower()
        for kw in STRONG_AGENT_KEYWORDS:
            if kw in query_lower:
                logger.info(f"三层路由 [L1-关键词]: {query} -> agent（命中强信号: {kw}）")
                return "agent"
        for kw in STRONG_RAG_KEYWORDS:
            if kw in query_lower:
                logger.info(f"三层路由 [L1-关键词]: {query} -> rag（命中强信号: {kw}）")
                return "rag"

        # ------ Layer 2: Embedding 相似度分类（CPU 可跑） ------
        clf = self.classifier
        if clf and clf.is_available():
            try:
                intent, route, confidence = clf.classify(query)
                if confidence >= 0.55:  # 置信度足够，直接返回
                    logger.info(
                        f"三层路由 [L2-Embedding]: {query} -> {route} "
                        f"(intent={intent}, confidence={confidence:.3f})"
                    )
                    return route
                else:
                    logger.info(
                        f"三层路由 [L2-Embedding]: {query} -> 置信度不足 "
                        f"({confidence:.3f} < 0.55)，降级 L3-LLM"
                    )
            except Exception as e:
                logger.warning(f"三层路由 [L2-Embedding]: 执行失败 {e}，降级 L3-LLM")

        # ------ Layer 3: LLM Prompt ------
        try:
            prompt = self.route_prompt.format(query=query)
            messages = [
                {"role": "system", "content": "你是一个意图识别专家，只输出路由结果。"},
                {"role": "user", "content": prompt}
            ]
            response = self.llm._call_llm(messages)
            result = response.get("text", "").strip().lower()

            if "agent" in result:
                route = "agent"
            else:
                route = "rag"

            logger.info(f"三层路由 [L3-LLM]: {query} -> {route}")
            return route

        except Exception as e:
            logger.error(f"三层路由 [L3-LLM]: 失败 {e}，默认走 rag")
            return "rag"

    def detect_domain(self, query: str) -> str:
        """LLM 领域检测"""
        try:
            prompt = self.domain_prompt.format(query=query)

            messages = [
                {"role": "system", "content": "你是一个领域识别专家，只输出领域名称。"},
                {"role": "user", "content": prompt}
            ]

            response = self.llm._call_llm(messages)
            result = response.get("text", "").strip().lower()

            valid_domains = ["hr", "it", "finance", "general"]
            for domain in valid_domains:
                if domain in result:
                    logger.info(f"LLM 领域检测：{query} -> {domain}")
                    return domain

            logger.info(f"LLM 领域检测：{query} -> general（未匹配）")
            return "general"

        except Exception as e:
            logger.error(f"LLM 领域检测失败：{e}，默认 general")
            return "general"


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    llm_router = LLMRouter()
else:
    llm_router = None
