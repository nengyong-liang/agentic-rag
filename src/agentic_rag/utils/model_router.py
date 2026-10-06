"""
模型级路由（Model Router）+ MoA 策略映射

根据查询类型和复杂度选择最合适的 LLM 模型或 MoA 策略，实现成本-性能平衡。

属于"三层路由"之外的第四层路由：
  意图路由决定 RAG/Agent 路径后，模型路由决定用哪个 LLM 生成。

工业界实践（MoMA / LangChain 2025 Report）：
  简单查询 → 小模型（低成本）
  复杂推理 → 大模型（高质量）
  工具调用 → 快速模型（低延迟）

⚠️ 低配版说明：
    工业界通常用独立的 Router 模型（如 MoMA 框架）动态选择 LLM。
    本项目使用规则 + 配置映射的静态方案，简单可靠，适合原型阶段。
    MoA 策略默认仅对 complex 查询启用，避免简单查询浪费多模型资源。
"""

import logging
from typing import Dict, Optional

from agentic_rag.config import settings

logger = logging.getLogger(__name__)


# 模型配置映射：route + query_type → (model_name, reason)
# 各模型特性：
#   doubao-seed: 快速、低成本、工具调用优化
#   qwen-plus:  平衡性能与成本，适合一般问答
#   deepseek:   强推理能力，适合复杂/分析类查询
MODEL_ROUTE_MAP: Dict[str, Dict[str, tuple]] = {
    "rag": {
        "simple":  ("qwen-plus", "简单知识查询，平衡性价比"),
        "fuzzy":   ("qwen-plus", "模糊查询，需一定推理能力"),
        "complex": ("deepseek", "复杂查询需强推理"),
    },
    "agent": {
        "tool_call": ("doubao", "工具调用，低延迟优先"),
        "plan":      ("deepseek", "任务规划需强推理"),
        "react":     ("deepseek", "ReAct 循环需持续推理"),
        "retrieve":  ("qwen-plus", "知识检索，平衡性能"),
    },
    "default": {
        "default": ("qwen-plus", "默认模型"),
    },
}

# MoA 策略配置映射：route + query_type → MoAConfig
# ⚠️ 低配版：仅对 complex 类型启用 MoA，Proposer 数量限制为 2
# 工业级可扩展为：所有查询类型均可配置 MoA，Proposer 4-5 个
MOA_ROUTE_MAP: Dict[str, Dict[str, dict]] = {
    "rag": {
        "complex": {
            "proposers": ["deepseek", "qwen"],
            "aggregator": "deepseek",
            "strategy": "aggregate",
            "reason": "复杂查询：多模型互补降低幻觉",
        },
    },
    "agent": {
        "plan": {
            "proposers": ["deepseek", "qwen"],
            "aggregator": "deepseek",
            "strategy": "debate",
            "reason": "任务规划：辩论模式提升规划质量",
        },
    },
}


def select_model(
    route: str,
    query_type: str = "default",
    domain: str = "general"
) -> Dict:
    """根据路由和查询类型选择最优模型

    Args:
        route: "rag" 或 "agent"
        query_type: simple/fuzzy/complex 或 tool_call/plan/react/retrieve
        domain: hr/it/finance/general（预留，可扩展为领域感知路由）

    Returns:
        {"model": "qwen-plus", "reason": "简单知识查询，平衡性价比"}
    """
    route_config = MODEL_ROUTE_MAP.get(route, MODEL_ROUTE_MAP["default"])
    model_info = route_config.get(query_type, MODEL_ROUTE_MAP["default"]["default"])

    model_name, reason = model_info

    logger.info(
        f"[ModelRouter] route={route}, type={query_type}, domain={domain} "
        f"-> model={model_name}, reason={reason}"
    )

    return {
        "model": model_name,
        "reason": reason,
    }


def select_model_or_moa(
    route: str,
    query_type: str = "default",
    domain: str = "general"
) -> Dict:
    """选择单模型或 MoA 策略

    复杂查询（complex/plan）走 MoA 多模型协作，其他走单模型。

    ⚠️ 低配版：仅 complex 和 plan 类型启用 MoA

    Args:
        route: "rag" 或 "agent"
        query_type: simple/fuzzy/complex 或 tool_call/plan/react/retrieve
        domain: hr/it/finance/general

    Returns:
        {"mode": "single", "model": "qwen-plus", "reason": "..."}
        或 {"mode": "moa", "config": MoAConfig, "reason": "..."}
    """
    # 检查是否有 MoA 配置
    moa_config_dict = MOA_ROUTE_MAP.get(route, {}).get(query_type)

    if moa_config_dict:
        from agent_harness.providers.moa import MoAConfig
        config = MoAConfig(
            proposers=moa_config_dict["proposers"],
            aggregator=moa_config_dict["aggregator"],
            strategy=moa_config_dict["strategy"],
        )
        reason = moa_config_dict.get("reason", "MoA 多模型协作")

        logger.info(
            f"[ModelRouter] route={route}, type={query_type} "
            f"-> mode=moa, proposers={config.proposers}, aggregator={config.aggregator}, "
            f"strategy={config.strategy}, reason={reason}"
        )

        return {
            "mode": "moa",
            "config": config,
            "reason": reason,
        }

    # 无 MoA 配置，走单模型
    result = select_model(route, query_type, domain)
    result["mode"] = "single"
    return result


def select_model_for_query(
    query: str,
    route: str,
    query_type: str = "default",
    domain: str = "general"
) -> Dict:
    """面向外部调用的模型选择接口

    可根据 query 内容进一步自定义（如涉及敏感领域用更强模型）。
    """
    return select_model(route, query_type, domain)


# 全局实例
model_router = select_model
