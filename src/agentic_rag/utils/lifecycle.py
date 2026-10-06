import logging
import math
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LifecycleState:
    """文档生命周期状态"""
    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class DecayConfig:
    """时间衰减配置"""
    decay_factor: float = 0.5
    grace_period_days: int = 30
    archive_after_days: int = 90


class DocumentLifecycleManager:
    """文档生命周期管理器
    
    工业级 RAG 系统的知识库维护核心组件，负责：
    1. 知识过期检测与自动降权
    2. 时间衰减因子计算
    3. 过期知识归档
    4. 版本管理（is_latest 标记）
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
    
    def compute_time_decay(
        self, 
        valid_until: Optional[str] = None,
        created_at: Optional[str] = None,
        is_latest: bool = True
    ) -> float:
        """计算时间衰减因子
        
        Args:
            valid_until: 知识有效期截止时间 (ISO格式)
            created_at: 知识创建时间 (ISO格式)
            is_latest: 是否为最新版本
            
        Returns:
            衰减因子 0.0~1.0，1.0表示无衰减
        """
        now = datetime.now()
        
        # 非最新版本直接降权
        if not is_latest:
            return 0.3
        
        def _parse_iso(dt_str: str) -> Optional[datetime]:
            """安全解析 ISO 时间字符串，统一返回 naive datetime"""
            try:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except (ValueError, TypeError):
                return None
        
        # 有 valid_until 的知识
        if valid_until:
            expire_time = _parse_iso(valid_until)
            if expire_time is None:
                logger.warning(f"解析 valid_until 失败，按过期处理（fail-closed）：{valid_until}")
                return 0.0  # fail-closed：解析失败按过期处理
            
            if now > expire_time:
                days_expired = (now - expire_time).days
                
                if days_expired > self.config.archive_after_days:
                    return 0.0
                
                grace_remaining = max(0, self.config.grace_period_days - days_expired)
                if grace_remaining > 0:
                    return self.config.decay_factor + (1 - self.config.decay_factor) * (grace_remaining / self.config.grace_period_days)
                
                decay_days = days_expired - self.config.grace_period_days
                return self.config.decay_factor * math.exp(-decay_days / 30)
            
            days_remaining = (expire_time - now).days
            if days_remaining < self.config.grace_period_days:
                return 0.5 + 0.5 * (days_remaining / self.config.grace_period_days)
            
            return 1.0
        
        # 有 created_at 的知识（无明确过期时间）
        if created_at:
            created_time = _parse_iso(created_at)
            if created_time is None:
                logger.warning(f"解析 created_at 失败: {created_at}，按无衰减处理")
                return 1.0
            
            age_days = (now - created_time).days
            
            if age_days > 365:
                return max(0.5, 1.0 - (age_days - 365) / 730)
            return 1.0
        
        return 1.0
    
    def check_lifecycle_state(
        self,
        valid_until: Optional[str] = None,
        created_at: Optional[str] = None,
        is_latest: bool = True
    ) -> str:
        """检查文档生命周期状态
        
        Returns:
            LifecycleState 中的状态值
        """
        decay = self.compute_time_decay(valid_until, created_at, is_latest)
        
        if not is_latest:
            return LifecycleState.DEPRECATED
        
        if decay == 0.0:
            return LifecycleState.ARCHIVED
        
        if valid_until:
            try:
                expire_time = datetime.fromisoformat(valid_until.replace('Z', '+00:00').replace('+08:00', ''))
                # 统一为 naive datetime 比较，避免 aware/naive 混用导致异常（P1-09）
                now = datetime.now()
                if expire_time.tzinfo is not None:
                    expire_time = expire_time.replace(tzinfo=None)
                if now > expire_time:
                    return LifecycleState.EXPIRED
            except (ValueError, TypeError) as e:
                logger.warning(f"解析 valid_until 失败，标记为过期（fail-closed）：{valid_until}, error: {e}")
                return LifecycleState.EXPIRED  # fail-closed：解析失败也标记为过期
        
        return LifecycleState.ACTIVE
    
    def apply_decay_to_chunks(
        self, 
        chunks: List[Dict],
        decay_field: str = "score"
    ) -> List[Dict]:
        """对检索结果应用时间衰减因子
        
        Args:
            chunks: 检索结果列表，每个结果需包含 metadata
            decay_field: 需要衰减的分数字段名
            
        Returns:
            应用衰减后的结果列表
        """
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            
            valid_until = metadata.get("valid_until")
            created_at = metadata.get("created_at")
            is_latest = metadata.get("is_latest", True)
            
            decay = self.compute_time_decay(valid_until, created_at, is_latest)
            
            original_score = chunk.get(decay_field, 1.0)
            chunk[decay_field] = original_score * decay
            chunk["time_decay_factor"] = decay
            
            state = self.check_lifecycle_state(valid_until, created_at, is_latest)
            chunk["lifecycle_state"] = state
        
        chunks.sort(key=lambda x: x.get(decay_field, 0), reverse=True)
        return chunks
    
    def filter_expired_chunks(
        self, 
        chunks: List[Dict],
        include_expired: bool = False
    ) -> List[Dict]:
        """过滤过期/归档的切片
        
        Args:
            chunks: 检索结果列表
            include_expired: 是否包含已过期但未归档的知识
            
        Returns:
            过滤后的结果列表
        """
        filtered = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            valid_until = metadata.get("valid_until")
            created_at = metadata.get("created_at")
            is_latest = metadata.get("is_latest", True)
            
            state = self.check_lifecycle_state(valid_until, created_at, is_latest)
            
            if state == LifecycleState.ARCHIVED:
                continue
            
            if state == LifecycleState.EXPIRED and not include_expired:
                continue
            
            chunk["lifecycle_state"] = state
            filtered.append(chunk)
        
        logger.info(
            f"生命周期过滤：输入={len(chunks)}, 输出={len(filtered)}, "
            f"过滤={len(chunks) - len(filtered)}"
        )
        return filtered
    
    def build_milvus_filter_with_lifecycle(
        self,
        base_filter: Optional[str] = None
    ) -> str:
        """构建包含生命周期过滤的 Milvus 查询条件
        
        Args:
            base_filter: 基础过滤条件
            
        Returns:
            增强后的过滤表达式
        """
        now_iso = datetime.now().isoformat()
        
        lifecycle_filter = (
            f'(metadata["is_latest"] == true or metadata["is_latest"] is null) '
            f'and (metadata["valid_until"] > "{now_iso}" or metadata["valid_until"] is null)'
        )
        
        if base_filter:
            return f'({base_filter}) and ({lifecycle_filter})'
        return lifecycle_filter
    
    def build_es_filter_with_lifecycle(
        self,
        base_filter: Optional[Dict] = None
    ) -> Dict:
        """构建包含生命周期过滤的 ES 查询条件
        
        Args:
            base_filter: 基础过滤条件
            
        Returns:
            增强后的 ES bool 查询
        """
        now_iso = datetime.now().isoformat()
        
        lifecycle_must = [
            {
                "bool": {
                    "should": [
                        {"term": {"metadata.is_latest": True}},
                        {"bool": {"must_not": {"exists": {"field": "metadata.is_latest"}}}}
                    ]
                }
            },
            {
                "bool": {
                    "should": [
                        {"range": {"metadata.valid_until": {"gt": now_iso}}},
                        {"bool": {"must_not": {"exists": {"field": "metadata.valid_until"}}}}
                    ]
                }
            }
        ]
        
        if base_filter:
            if "bool" in base_filter:
                existing_must = base_filter["bool"].get("must", [])
                if isinstance(existing_must, list):
                    existing_must.extend(lifecycle_must)
                else:
                    existing_must = [existing_must] + lifecycle_must
                base_filter["bool"]["must"] = existing_must
                return base_filter
            else:
                return {
                    "bool": {
                        "must": [base_filter] + lifecycle_must
                    }
                }
        
        return {"bool": {"must": lifecycle_must}}


document_lifecycle_manager = DocumentLifecycleManager()
