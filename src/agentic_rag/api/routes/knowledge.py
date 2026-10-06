"""知识图谱/评估端点"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from agentic_rag.rag.knowledge_graph import knowledge_graph
from agentic_rag.rag.rag_evaluator import RAGEvaluator
from agentic_rag.rag.query_rewriter import query_rewriter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/kg/entity")
async def query_kg_entity(name: str, depth: int = 1):
    """查询知识图谱实体及关联关系"""
    try:
        result = knowledge_graph.query_related_entities(name, depth=depth)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"KG查询失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kg/extract")
async def extract_kg_entities(request: Dict[str, Any]):
    """从文本中提取实体和关系"""
    try:
        text = request.get("text", "")
        doc_name = request.get("doc_name", "")
        if not text:
            raise HTTPException(status_code=400, detail="text参数不能为空")
        result = knowledge_graph.extract_entities_and_relations(text, doc_name)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KG提取失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kg/multi_hop")
async def kg_multi_hop(request: Dict[str, Any]):
    """知识图谱多跳推理"""
    try:
        query = request.get("query", "")
        max_hops = request.get("max_hops", 2)
        if not query:
            raise HTTPException(status_code=400, detail="query参数不能为空")
        result = knowledge_graph.multi_hop_reason(query, max_hops=max_hops)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KG多跳推理失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/run")
async def run_evaluation(request: Dict[str, Any] = None):
    """运行RAG评估（检索指标 + 生成指标）"""
    try:
        request = request or {}
        evaluator = RAGEvaluator()
        golden = evaluator.create_golden_testset(request.get("testset", "enterprise"))
        retrieval_result = evaluator.evaluate_retrieval_only(golden)
        return {
            "success": True,
            "data": {
                "testset_size": len(golden),
                "retrieval_metrics": retrieval_result
            }
        }
    except Exception as e:
        logger.error(f"评估失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/ablation")
async def run_ablation_study(request: Dict[str, Any] = None):
    """消融实验：逐个关闭组件，评估每个组件的贡献"""
    try:
        request = request or {}
        evaluator = RAGEvaluator()
        result = evaluator.run_ablation_study(
            testset=request.get("testset"),
            configs=request.get("configs")
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"消融实验失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/expansion/info")
async def get_expansion_info(query: str):
    """获取查询扩展详情（分级策略 + 扩展结果 + 漂移控制）"""
    try:
        info = query_rewriter.get_expansion_info(query)
        return {"success": True, "data": info}
    except Exception as e:
        logger.error(f"获取扩展信息失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
