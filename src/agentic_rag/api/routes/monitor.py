"""监控/缓存/降级端点"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.api.state import state
from agent_harness.observability.monitor import rag_monitor
from agentic_rag.utils.degradation import degradation_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# 评测体系端点（Phase 5）
# ------------------------------------------------------------------
class EvalRequest(BaseModel):
    """单任务评测请求"""
    test_id: str = Field(..., description="测试任务 ID（来自 benchmark 任务集）")
    actual_tools: List[str] = Field(default_factory=list, description="实际工具调用序列")
    actual_route: Optional[str] = Field(None, description="实际路由决策")
    actual_answer: str = Field("", description="实际生成的答案")
    answer_relevance: float = Field(0.0, description="答案相关性评分 [0,1]")
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Agent 执行轨迹")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="引用来源")


class ReliabilityRequest(BaseModel):
    """可靠性评估请求（pass^k）"""
    task_id: str
    success_history: List[bool] = Field(..., description="每次独立尝试的成功状态")
    scores: Optional[List[float]] = Field(None, description="每次尝试的得分")


# 评测模块条件注册
try:
    from agent_harness.evaluation import (
        OutcomeEvaluator, TrajectoryEvaluator, ProcessJudge, ReliabilityMetrics
    )
    from agent_harness.evaluation.benchmarks import ALL_BENCHMARK_TASKS, get_benchmark_summary, get_task_by_id
    _eval_available = True
except Exception as e:
    logger.warning(f"评测模块不可用，跳过评测端点注册：{e}")
    _eval_available = False

if _eval_available:
    _outcome_evaluator = OutcomeEvaluator()
    _trajectory_evaluator = TrajectoryEvaluator()
    _process_judge = ProcessJudge()
    _reliability_metrics = ReliabilityMetrics()

    @router.get("/eval/benchmark/summary")
    async def get_eval_benchmark_summary():
        """获取基准任务集统计"""
        try:
            return {
                "success": True,
                "data": get_benchmark_summary(),
                "message": "获取基准任务集统计成功",
            }
        except Exception as e:
            logger.error(f"获取基准任务集统计失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/eval/benchmark/tasks")
    async def list_benchmark_tasks(
        category: Optional[str] = None, difficulty: Optional[str] = None
    ):
        """列出基准任务（可按场景/难度过滤）"""
        try:
            tasks = ALL_BENCHMARK_TASKS
            if category:
                tasks = [t for t in tasks if t.category == category]
            if difficulty:
                tasks = [t for t in tasks if t.difficulty == difficulty]
            return {
                "success": True,
                "data": [
                    {
                        "id": t.id, "category": t.category, "difficulty": t.difficulty,
                        "query": t.query, "expected_tools": t.expected_tools,
                        "expected_route": t.expected_route, "expected_keywords": t.expected_keywords,
                        "optimal_steps": t.optimal_steps, "description": t.description,
                        "metadata": t.metadata,
                    }
                    for t in tasks
                ],
                "total": len(tasks),
                "message": "获取基准任务列表成功",
            }
        except Exception as e:
            logger.error(f"获取基准任务列表失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/eval/single")
    async def evaluate_single_task(req: EvalRequest):
        """单任务三层评测（Outcome + Trajectory + Process）"""
        try:
            task = get_task_by_id(req.test_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"任务 {req.test_id} 不存在于基准任务集")

            # 三层评测
            outcome = _outcome_evaluator.evaluate(
                test_id=req.test_id,
                expected_tools=task.expected_tools,
                actual_tools=req.actual_tools,
                expected_route=task.expected_route,
                actual_route=req.actual_route,
                expected_keywords=task.expected_keywords,
                actual_answer=req.actual_answer,
                answer_relevance=req.answer_relevance,
            )
            trajectory = _trajectory_evaluator.evaluate(
                test_id=req.test_id,
                expected_trajectory=task.expected_tools,
                actual_trajectory=req.actual_tools,
            )
            process = _process_judge.evaluate(
                test_id=req.test_id,
                agent_trace=req.agent_trace,
                final_answer=req.actual_answer,
                citations=req.citations,
            )
            return {
                "success": True,
                "data": {
                    "outcome": outcome.to_dict(),
                    "trajectory": trajectory.to_dict(),
                    "process": process.to_dict(),
                    "overall_score": round(
                        (outcome.overall_score + trajectory.trajectory_score + process.process_score) / 3,
                        4
                    ),
                },
                "message": "三层评测完成",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"单任务评测失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/eval/reliability")
    async def evaluate_reliability(req: ReliabilityRequest):
        """可靠性评测（pass^k 系列）"""
        try:
            result = _reliability_metrics.pass_at_k(
                task_id=req.task_id,
                success_history=req.success_history,
                scores=req.scores,
            )
            return {
                "success": True,
                "data": result.to_dict(),
                "message": f"可靠性评测完成（k={result.k}, level={result.reliability_level}）",
            }
        except Exception as e:
            logger.error(f"可靠性评测失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/eval/reliability/levels")
    async def get_reliability_levels():
        """获取可靠性等级阈值说明"""
        try:
            return {
                "success": True,
                "data": {
                    "thresholds": _reliability_metrics.RELIABILITY_THRESHOLDS,
                    "description": {
                        "excellent": "一致性差距 ≤ 5%，Agent 行为高度稳定",
                        "good": "一致性差距 ≤ 15%，可靠性良好",
                        "acceptable": "一致性差距 ≤ 30%，可接受但需优化",
                        "poor": "一致性差距 > 30%，不稳定，不宜投产",
                    },
                },
                "message": "获取可靠性等级说明成功",
            }
        except Exception as e:
            logger.error(f"获取可靠性等级失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        if state.redis_cache:
            stats = state.redis_cache.get_cache_stats()
            return {"success": True, "data": stats}
        return {"success": False, "data": {"error": "Redis缓存不可用"}}
    except Exception as e:
        logger.error(f"获取缓存统计失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/stats")
async def get_monitor_stats():
    """获取工业级监控指标（QPS/P50/P99/命中率/延迟预算）"""
    try:
        stats = rag_monitor.get_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"获取监控统计失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/health")
async def get_monitor_health():
    """健康检查（延迟预算+错误率+缓存命中率+降级状态）"""
    try:
        health = rag_monitor.get_health_check()
        degradation_status = degradation_manager.get_status()
        return {
            "success": True,
            "data": {
                **health,
                "degradation": degradation_status
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/latency_budget")
async def get_latency_budget():
    """获取延迟预算状态"""
    try:
        budget = rag_monitor.get_latency_budget_status()
        return {"success": True, "data": budget}
    except Exception as e:
        logger.error(f"获取延迟预算失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/percentiles")
async def get_latency_percentiles():
    """获取延迟百分位数"""
    try:
        percentiles = rag_monitor.get_latency_percentiles()
        return {"success": True, "data": percentiles}
    except Exception as e:
        logger.error(f"获取延迟百分位数失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/degradation/status")
async def get_degradation_status():
    """获取降级策略状态"""
    try:
        status = degradation_manager.get_status()
        return {"success": True, "data": status}
    except Exception as e:
        logger.error(f"获取降级状态失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
