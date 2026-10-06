"""
降级策略矩阵 (Degradation Strategy Matrix)

面试核心考点：
- 关键组件失败时自动切换兜底策略，保障核心链路可用性
- 降级不是"不能用"，而是"用次优方案保证服务不中断"

降级场景：
1. Milvus/ES单路宕机 → 切换单路检索，k放宽至80
2. Rerank服务超时 → 跳过精排，直接使用RRF结果
3. LLM Judge不可用 → 规则打分(BM25×0.7 + Dense×0.3)
4. 72B模型排队/限流 → 路由至小模型或启用缓存
5. Redis不可用 → 跳过缓存，直接检索
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class DegradationLevel(Enum):
    NORMAL = "normal"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    name: str
    status: ComponentStatus = ComponentStatus.HEALTHY
    last_check_time: float = 0
    failure_count: int = 0
    last_error: str = ""
    response_time_ms: float = 0

    CHECK_INTERVAL = 30
    FAILURE_THRESHOLD = 3
    RECOVERY_THRESHOLD = 2

    def record_success(self, response_time_ms: float = 0):
        self.response_time_ms = response_time_ms
        if self.status == ComponentStatus.DEGRADED:
            self.failure_count = max(0, self.failure_count - 1)
            if self.failure_count < self.RECOVERY_THRESHOLD:
                self.status = ComponentStatus.HEALTHY
                logger.info(f"[降级] {self.name} 恢复正常")
        elif self.status == ComponentStatus.DOWN:
            self.failure_count = max(0, self.failure_count - 1)
            if self.failure_count < self.RECOVERY_THRESHOLD:
                self.status = ComponentStatus.DEGRADED
                logger.info(f"[降级] {self.name} 从DOWN恢复为DEGRADED")
        self.last_check_time = time.time()

    def record_failure(self, error: str = ""):
        self.failure_count += 1
        self.last_error = error
        self.last_check_time = time.time()

        if self.failure_count >= self.FAILURE_THRESHOLD:
            old_status = self.status
            self.status = ComponentStatus.DOWN
            if old_status != ComponentStatus.DOWN:
                logger.warning(f"[降级] {self.name} 状态变更为DOWN（连续失败{self.failure_count}次）")
        elif self.failure_count >= 1:
            if self.status == ComponentStatus.HEALTHY:
                self.status = ComponentStatus.DEGRADED
                logger.warning(f"[降级] {self.name} 状态变更为DEGRADED")

    def should_check(self) -> bool:
        return time.time() - self.last_check_time > self.CHECK_INTERVAL


@dataclass
class DegradationAction:
    component: str
    action: str
    impact: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)


# 降级策略矩阵
DEGRADATION_MATRIX = {
    # Milvus/ES单路宕机
    "milvus_down": DegradationAction(
        component="milvus",
        action="切换单路ES检索，k放宽至80",
        impact="召回率↓10%，延迟↑200ms",
        config_overrides={"retrieval_mode": "es_only", "top_k": 80}
    ),
        
    # ES单路宕机
    "es_down": DegradationAction(
        component="elasticsearch",
        action="切换单路Milvus检索，k放宽至80",
        impact="召回率↓10%，延迟↑200ms",
        config_overrides={"retrieval_mode": "vector_only", "top_k": 80}
    ),
        
    # Rerank服务超时
    "rerank_timeout": DegradationAction(
        component="reranker",
        action="跳过精排，直接使用RRF结果",
        impact="相关性↓15%，成本↓40%",
        config_overrides={"enable_rerank": False}
    ),
        
    # LLM Judge不可用   
    "llm_judge_unavailable": DegradationAction(
        component="llm_judge",
        action="规则打分(BM25×0.7 + Dense×0.3)",
        impact="CRAG精度下降，但流程不中断",
        config_overrides={"grade_method": "rule_based"}
    ),
        
    # 72B模型排队/限流
    "llm_72b_throttled": DegradationAction(
        component="llm_generate",
        action="路由至小模型或启用缓存",
        impact="生成速度↑30%，质量微降",
        config_overrides={"use_small_model": True}
    ),
        
    # Redis不可用
    "redis_down": DegradationAction(
        component="redis",
        action="跳过缓存，直接检索",
        impact="重复查询延迟↑，QPS承载能力↓",
        config_overrides={"enable_cache": False}
    ),
}


class DegradationManager:
    """降级策略管理器"""

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {
            "milvus": ComponentHealth(name="milvus"),
            "elasticsearch": ComponentHealth(name="elasticsearch"),
            "reranker": ComponentHealth(name="reranker"),
            "llm_judge": ComponentHealth(name="llm_judge"),
            "llm_generate": ComponentHealth(name="llm_generate"),
            "redis": ComponentHealth(name="redis"),
        }
        self._active_degradations: Dict[str, DegradationAction] = {}
        logger.info("[降级管理器] 初始化完成")

    def record_success(self, component: str, response_time_ms: float = 0):
        if component in self.components:
            self.components[component].record_success(response_time_ms)
            if component in self._active_degradations:
                health = self.components[component]
                if health.status == ComponentStatus.HEALTHY:
                    del self._active_degradations[component]
                    logger.info(f"[降级] {component} 恢复正常，取消降级策略")

    def record_failure(self, component: str, error: str = ""):
        if component in self.components:
            self.components[component].record_failure(error)
            self._apply_degradation(component)

    def _apply_degradation(self, component: str):
        health = self.components.get(component)
        if not health:
            return

        degradation_key = None
        if component == "milvus" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "milvus_down"
        elif component == "elasticsearch" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "es_down"
        elif component == "reranker" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "rerank_timeout"
        elif component == "llm_judge" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "llm_judge_unavailable"
        elif component == "llm_generate" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "llm_72b_throttled"
        elif component == "redis" and health.status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED]:
            degradation_key = "redis_down"

        if degradation_key and degradation_key not in self._active_degradations:
            action = DEGRADATION_MATRIX[degradation_key]
            self._active_degradations[component] = action
            logger.warning(
                f"[降级] 触发降级策略：{action.component} → {action.action}（影响：{action.impact}）"
            )

    def get_config_overrides(self) -> Dict[str, Any]:
        """获取当前所有降级策略的配置覆盖"""
        overrides = {} # 初始化空字典
        for action in self._active_degradations.values(): # 遍历所有降级策略
            overrides.update(action.config_overrides) # 合并配置覆盖
        return overrides

    def get_degradation_level(self) -> DegradationLevel:
        """获取当前整体降级级别"""
        down_count = sum(1 for h in self.components.values() if h.status == ComponentStatus.DOWN)
        degraded_count = sum(1 for h in self.components.values() if h.status == ComponentStatus.DEGRADED)

        if down_count >= 2:
            return DegradationLevel.CRITICAL
        elif down_count >= 1:
            return DegradationLevel.MAJOR
        elif degraded_count >= 2:
            return DegradationLevel.MINOR
        elif degraded_count >= 1:
            return DegradationLevel.MINOR
        else:
            return DegradationLevel.NORMAL

    def is_component_available(self, component: str) -> bool:
        health = self.components.get(component)
        if not health:
            return True
        return health.status != ComponentStatus.DOWN

    def should_skip_rerank(self) -> bool:
        return "reranker" in self._active_degradations

    def should_use_rule_grade(self) -> bool:
        return "llm_judge" in self._active_degradations

    def should_skip_cache(self) -> bool:
        return "redis" in self._active_degradations

    def get_retrieval_mode(self) -> str:
        """获取当前检索模式"""
        overrides = self.get_config_overrides() # 获取所有降级策略的配置覆盖
        return overrides.get("retrieval_mode", "hybrid") # 返回检索模式，默认hybrid

    def get_adjusted_top_k(self, default_k: int = 10) -> int:
        overrides = self.get_config_overrides() # 获取所有降级策略的配置覆盖
        return overrides.get("top_k", default_k)

    def get_status(self) -> Dict[str, Any]:
        """获取降级状态报告"""
        level = self.get_degradation_level()
        component_status = {
            name: {
                "status": health.status.value,
                "failure_count": health.failure_count,
                "response_time_ms": health.response_time_ms,
                "last_error": health.last_error
            }
            for name, health in self.components.items()
        }

        active_actions = {
            comp: {
                "action": action.action,
                "impact": action.impact,
                "config_overrides": action.config_overrides
            }
            for comp, action in self._active_degradations.items()
        }

        return {
            "degradation_level": level.value,
            "components": component_status,
            "active_degradations": active_actions,
            "config_overrides": self.get_config_overrides(),
            "timestamp": time.time()
        }


degradation_manager = DegradationManager()
