"""安全相关端点"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from agentic_rag.utils.prompt_injection import prompt_injection_detector
from agentic_rag.utils.sensitive_filter import sensitive_filter
from agentic_rag.utils.nli_checker import nli_checker
from agentic_rag.utils.tenant_filter import tenant_permission_filter, TenantContext
from agentic_rag.utils.semantic_dedup import semantic_deduplicator
from agentic_rag.utils.quality_gate import chunk_quality_gate
from agentic_rag.utils.lifecycle import document_lifecycle_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/security/check_injection")
async def check_prompt_injection(request: Dict[str, Any]):
    """检测Prompt注入攻击"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="text参数不能为空")
        result = prompt_injection_detector.detect(text)
        return {
            "success": True,
            "data": {
                "is_injection": result["is_injection"],
                "risk_level": result["risk_level"].value,
                "score": result["score"],
                "detected_patterns": result["detected_patterns"],
                "suggestion": result["suggestion"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt注入检测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/sanitize")
async def sanitize_prompt(request: Dict[str, Any]):
    """净化Prompt注入内容"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="text参数不能为空")
        sanitized, detection = prompt_injection_detector.sanitize(text)
        return {
            "success": True,
            "data": {
                "original_length": len(text),
                "sanitized_length": len(sanitized),
                "sanitized_text": sanitized,
                "is_injection": detection["is_injection"],
                "risk_level": detection["risk_level"].value
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt净化失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/mask_pii")
async def mask_pii(request: Dict[str, Any]):
    """PII掩码（支持Redis映射+还原）"""
    try:
        text = request.get("text", "")
        request_id = request.get("request_id", None)
        if not text:
            raise HTTPException(status_code=400, detail="text参数不能为空")
        masked_text, mask_result = sensitive_filter.mask_pii_with_mapping(text, request_id)
        return {
            "success": True,
            "data": {
                "masked_text": masked_text,
                "request_id": mask_result.request_id,
                "pii_count": mask_result.pii_count,
                "pii_items": mask_result.pii_items
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PII掩码失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/restore_pii")
async def restore_pii(request: Dict[str, Any]):
    """PII还原"""
    try:
        text = request.get("text", "")
        request_id = request.get("request_id", "")
        if not text or not request_id:
            raise HTTPException(status_code=400, detail="text和request_id参数不能为空")
        restored_text = sensitive_filter.restore_pii(text, request_id)
        return {
            "success": True,
            "data": {
                "restored_text": restored_text
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PII还原失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/nli_check")
async def nli_consistency_check(request: Dict[str, Any]):
    """NLI一致性校验（检测幻觉）"""
    try:
        answer = request.get("answer", "")
        contexts = request.get("contexts", [])
        threshold = request.get("threshold", 0.5)
        if not answer or not contexts:
            raise HTTPException(status_code=400, detail="answer和contexts参数不能为空")
        result = nli_checker.check_consistency(answer, contexts, threshold)
        return {
            "success": True,
            "data": {
                "verdict": result["verdict"].value,
                "consistency_score": result["consistency_score"],
                "is_hallucination": result["is_hallucination"],
                "contradiction_details": result["contradiction_details"],
                "suggestion": result["suggestion"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NLI校验失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tenant/filter")
async def tenant_filter_query(request: Dict[str, Any]):
    """多租户权限过滤"""
    try:
        tenant_id = request.get("tenant_id", "")
        user_id = request.get("user_id", "")
        roles = request.get("roles", ["employee"])
        security_level = request.get("security_level", "public")
        if not tenant_id or not user_id:
            raise HTTPException(status_code=400, detail="tenant_id和user_id参数不能为空")
        tenant_ctx = TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=roles,
            security_level=security_level
        )
        milvus_filter = tenant_permission_filter.build_milvus_filter(tenant_ctx)
        es_filter = tenant_permission_filter.build_es_filter(tenant_ctx)
        return {
            "success": True,
            "data": {
                "milvus_filter": milvus_filter,
                "es_filter": es_filter,
                "allowed_security_levels": tenant_permission_filter._get_allowed_security_levels(tenant_ctx)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多租户过滤失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic/dedup")
async def semantic_dedup_queries(request: Dict[str, Any]):
    """语义去重"""
    try:
        queries = request.get("queries", [])
        if not queries:
            raise HTTPException(status_code=400, detail="queries参数不能为空")
        deduped = semantic_deduplicator.deduplicate_queries(queries)
        return {
            "success": True,
            "data": {
                "original_count": len(queries),
                "deduped_count": len(deduped),
                "deduped_queries": deduped
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语义去重失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/gate")
async def run_quality_gate(request: Dict[str, Any]):
    """对切片列表运行质量门禁（不入库，仅检查）"""
    try:
        chunks = request.get("chunks", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="chunks参数不能为空")
        chunk_quality_gate.reset_seen_hashes()
        passed, report = chunk_quality_gate.run_quality_gate(chunks)
        return {
            "success": True,
            "data": {
                "passed_chunks": passed,
                "quality_report": report.to_dict()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"质量门禁检查失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/decay")
async def compute_lifecycle_decay(request: Dict[str, Any]):
    """计算知识的时间衰减因子"""
    try:
        valid_until = request.get("valid_until")
        created_at = request.get("created_at")
        is_latest = request.get("is_latest", True)
        decay = document_lifecycle_manager.compute_time_decay(
            valid_until=valid_until,
            created_at=created_at,
            is_latest=is_latest
        )
        state = document_lifecycle_manager.check_lifecycle_state(
            valid_until=valid_until,
            created_at=created_at,
            is_latest=is_latest
        )
        return {
            "success": True,
            "data": {
                "decay_factor": decay,
                "lifecycle_state": state
            }
        }
    except Exception as e:
        logger.error(f"计算衰减因子失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/filter")
async def filter_expired_chunks(request: Dict[str, Any]):
    """过滤过期/归档的检索结果"""
    try:
        chunks = request.get("chunks", [])
        include_expired = request.get("include_expired", False)
        if not chunks:
            raise HTTPException(status_code=400, detail="chunks参数不能为空")
        filtered = document_lifecycle_manager.filter_expired_chunks(
            chunks, include_expired=include_expired
        )
        return {
            "success": True,
            "data": {
                "original_count": len(chunks),
                "filtered_count": len(filtered),
                "filtered_chunks": filtered
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生命周期过滤失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/apply_decay")
async def apply_decay_to_results(request: Dict[str, Any]):
    """对检索结果应用时间衰减并重新排序"""
    try:
        chunks = request.get("chunks", [])
        decay_field = request.get("decay_field", "score")
        if not chunks:
            raise HTTPException(status_code=400, detail="chunks参数不能为空")
        decayed = document_lifecycle_manager.apply_decay_to_chunks(
            chunks, decay_field=decay_field
        )
        return {
            "success": True,
            "data": {
                "decayed_chunks": decayed
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"应用衰减失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
