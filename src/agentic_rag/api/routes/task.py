"""长程任务端点"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from agentic_rag.api.state import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/task/create")
async def create_long_task(request: dict):
    """创建长程任务"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        task_id = state.task_manager.create_task(
            name=request.get("name", ""),
            description=request.get("description", ""),
            steps_def=request.get("steps", [])
        )
        task_status = state.task_manager.get_task_status(task_id)
        return {
            "task_id": task_id,
            "status": task_status.get("data", {}).get("status", "pending")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建长程任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取长程任务状态"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        status = state.task_manager.get_task_status(task_id)
        if not status.get("success", False):
            raise HTTPException(status_code=404, detail="任务不存在")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """列出所有长程任务"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        return state.task_manager.list_tasks(status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task/{task_id}/pause")
async def pause_task(task_id: str):
    """暂停长程任务"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        result = state.task_manager.pause_task(task_id)
        return {"success": result.get("success", False), "message": result.get("message", "")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task/{task_id}/resume")
async def resume_task(task_id: str):
    """恢复长程任务"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        result = state.task_manager.mark_task_resumable(task_id)
        if not result.get("success", False):
            return {"success": False, "message": result.get("message", "恢复失败")}
        return {"success": True, "message": result.get("message", "")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消长程任务"""
    try:
        if state.task_manager is None:
            raise HTTPException(status_code=503, detail="长程任务服务不可用")
        result = state.task_manager.cancel_task(task_id)
        return {"success": result.get("success", False), "message": result.get("message", "")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
