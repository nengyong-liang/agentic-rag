"""Agent/MCP端点"""
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.api.state import state
from agentic_rag.api.routes.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# HITL 审批工作流端点（Phase 4）
# ------------------------------------------------------------------
class ApprovalSubmitRequest(BaseModel):
    """人工审批决策提交请求"""
    request_id: str = Field(..., description="审批请求 ID")
    decision: str = Field(..., description="决策类型：approve / edit / reject / respond")
    reviewer: str = Field("human", description="审批人标识")
    edited_args: Optional[Dict[str, Any]] = Field(None, description="EDIT 决策时的新参数")
    response_text: Optional[str] = Field(None, description="RESPOND 决策时的回应文本")
    reason: str = Field("", description="决策原因")


# HITL 端点（无条件注册：HITL 模块不依赖外部服务，仅依赖 SQLite）
try:
    from agent_harness.hitl import approval_node, DecisionType, RiskLevel
    _hitl_available = True
except Exception as e:
    logger.warning(f"HITL 模块不可用，跳过审批端点注册：{e}")
    _hitl_available = False

if _hitl_available:
    @router.get("/agent/hitl/pending")
    async def list_pending_approvals(session_id: Optional[str] = None):
        """列出未决审批请求（供审批面板轮询）"""
        try:
            pending = approval_node.list_pending(session_id)
            return {
                "success": True,
                "data": [r.to_dict() for r in pending],
                "total": len(pending),
                "message": "获取未决审批列表成功",
            }
        except Exception as e:
            logger.error(f"获取未决审批失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/agent/hitl/stats")
    async def get_hitl_stats():
        """获取审批统计数据（用于监控面板）"""
        try:
            return {
                "success": True,
                "data": approval_node.get_stats(),
                "message": "获取审批统计成功",
            }
        except Exception as e:
            logger.error(f"获取审批统计失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/agent/hitl/request/{request_id}")
    async def get_approval_request(request_id: str):
        """根据 request_id 获取审批请求详情"""
        from agent_harness.hitl.checkpointer import sqlite_checkpointer
        try:
            req = sqlite_checkpointer.get_request(request_id)
            if req is None:
                raise HTTPException(status_code=404, detail="审批请求不存在")
            decision = sqlite_checkpointer.get_decision(request_id)
            return {
                "success": True,
                "data": {
                    "request": req.to_dict(),
                    "decision": decision.to_dict() if decision else None,
                },
                "message": "获取审批请求详情成功",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取审批请求详情失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/agent/hitl/submit")
    async def submit_approval_decision(req: ApprovalSubmitRequest):
        """人工提交审批决策"""
        try:
            try:
                decision_type = DecisionType(req.decision)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的决策类型：{req.decision}（应为 approve/edit/reject/respond）",
                )
            ok, decision = approval_node.submit_decision(
                request_id=req.request_id,
                decision=decision_type,
                reviewer=req.reviewer,
                edited_args=req.edited_args,
                response_text=req.response_text,
                reason=req.reason,
            )
            if not ok:
                raise HTTPException(status_code=409, detail=f"决策提交失败：{decision.reason}")
            return {
                "success": True,
                "data": decision.to_dict(),
                "message": f"审批决策已提交：{decision.decision.value}",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"提交审批决策失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/agent/hitl/risk_map")
    async def get_tool_risk_map():
        """获取工具风险等级映射（供前端展示风险标识）"""
        try:
            risk_map = {
                name: level.value
                for name, level in approval_node.TOOL_RISK_MAP.items()
            }
            return {
                "success": True,
                "data": risk_map,
                "message": "获取工具风险等级映射成功",
            }
        except Exception as e:
            logger.error(f"获取风险映射失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))


# Agent 端点（条件注册：仅当 agent 可用时）
if state.agent:
    @router.get("/agent/tools")
    async def get_agent_tools():
        """获取 Agent 可用工具列表"""
        try:
            logger.info("获取 Agent 工具列表")
            result = state.agent.get_available_tools()
            return result
        except Exception as e:
            logger.error(f"获取工具列表失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/agent/run")
    async def run_agent(request: ChatRequest):
        """运行 Agent 处理查询"""
        try:
            logger.info(f"运行 Agent 处理请求：{request.query}")
            result = state.agent.run_agent(request.query, request.history)
            return result
        except Exception as e:
            logger.error(f"Agent 处理失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))


# MCP 端点（条件注册：仅当 mcp_bridge 可用时）
if state.mcp_bridge:
    @router.get("/mcp/tools")
    async def get_mcp_tools():
        """获取 MCP 工具 schema"""
        try:
            logger.info("获取 MCP 工具 schema")
            tools_schema = state.mcp_bridge.get_tools_schema()
            return {
                "success": True,
                "data": tools_schema,
                "message": "获取工具 schema 成功"
            }
        except Exception as e:
            logger.error(f"获取工具 schema 失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/mcp/tool_call")
    async def execute_mcp_tool_call(tool_call: Dict[str, Any]):
        """执行 MCP 工具调用"""
        try:
            tool_name = tool_call.get("tool")
            tool_params = tool_call.get("params", {})
            logger.info(f"执行 MCP 工具调用：{tool_name}")
            result = await state.mcp_bridge.execute_tool_async(tool_name, tool_params)
            return result
        except Exception as e:
            logger.error(f"执行工具调用失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))
