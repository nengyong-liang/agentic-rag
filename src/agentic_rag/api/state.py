"""全局单例状态容器，由 main.py 初始化"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class _State:
    """全局单例状态容器，所有路由模块通过 state.xxx 访问"""

    def __init__(self):
        # 全局单例引用，由 main.py 在初始化时设置
        self.agent = None
        self.mcp_bridge = None
        self.redis_cache = None
        self.memory_manager = None
        self._memory_managers: Dict[str, Any] = {}  # user_id -> MemoryManager（跨会话共享）
        self.task_manager = None
        self.agentic_rag_orchestrator = None
        self.hybrid_llm = None
        self.synthesis_pipeline = None
        self._synthesis_available = False

        # Harness 单例
        self.harness_loop = None
        self.harness_event_store = None
        self.harness_provider = None
        self.harness_projector = None
        self.harness_breaker = None
        self.harness_validator = None
        self.harness_governor = None
        self.harness_gateway = None
        self.harness_auditor = None

        # 配置
        self.DATA_DIR = None
        self.has_multipart = False


    def get_memory_manager(self, user_id: str = "default", session_id: str = "default"):
        """按 user_id 取（或创建）跨会话记忆管理器。

        - 同一 user_id 复用同一 MemoryManager：长期记忆按用户共享（跨对话互通）
        - session_id 仅用于短期记忆的会话级隔离，每次请求动态刷新
        - 创建失败返回 None（调用方降级为无记忆）
        """
        from agentic_rag.agent.memory import MemoryManager, MemoryConfig
        key = user_id or "default"
        mm = self._memory_managers.get(key)
        if mm is None:
            try:
                mm = MemoryManager(MemoryConfig(), user_id=key)
                self._memory_managers[key] = mm
            except Exception as e:
                logger.warning(f"[state] 记忆管理器创建失败，降级无记忆: {e}")
                return None
        # 短期记忆按会话隔离：每次请求刷新 session_id（即使复用同一用户管理器）
        mm.session_id = session_id or "default"
        return mm


# 全局唯一实例
state = _State()