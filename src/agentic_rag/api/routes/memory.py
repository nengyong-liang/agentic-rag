"""记忆端点"""
import logging

from fastapi import APIRouter, HTTPException

from agentic_rag.api.state import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/memory/stats")
async def get_memory_stats():
    """获取Agent记忆统计"""
    try:
        if state.memory_manager is None:
            raise HTTPException(status_code=503, detail="Memory 服务不可用")
        return state.memory_manager.get_stats()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取记忆统计失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear")
async def clear_memory():
    """清空Agent记忆"""
    try:
        if state.memory_manager is None:
            raise HTTPException(status_code=503, detail="Memory 服务不可用")
        state.memory_manager.clear()
        return {"status": "success", "message": "记忆已清空"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空记忆失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
