"""调度执行层（表示层与 RAG/Agent 能力之间的统一调度网关）"""
from agentic_rag.execution.dispatcher import Dispatcher, dispatch

__all__ = ["Dispatcher", "dispatch"]