"""vLLM/数据合成端点"""
import logging

from fastapi import APIRouter, HTTPException

from agentic_rag.api.state import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/inference/status")
async def get_inference_status():
    """获取推理服务状态"""
    try:
        from agentic_rag.inference.vllm_service import VLLMService, InferenceConfig
        service = VLLMService(InferenceConfig())
        return {
            "vllm_available": service.is_running(),
            "model_info": service.get_model_info() if service.is_running() else None,
            "hybrid_llm_models": {
                "preferred": state.hybrid_llm.preferred_model if state.hybrid_llm else "unknown",
                "vllm_registered": state.hybrid_llm._vllm_service is not None if state.hybrid_llm else False,
            }
        }
    except Exception as e:
        return {"vllm_available": False, "error": str(e)}


@router.post("/synthesis/run")
async def run_synthesis(request: dict):
    """运行数据合成管线"""
    try:
        if not state._synthesis_available or state.synthesis_pipeline is None:
            raise HTTPException(status_code=503, detail="数据合成服务不可用")
        seeds = request.get("seed_instructions", [])
        result = state.synthesis_pipeline.run(seeds)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"运行数据合成失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesis/export_dpo")
async def export_dpo(request: dict):
    """导出DPO格式数据"""
    try:
        if not state._synthesis_available:
            raise HTTPException(status_code=503, detail="数据合成服务不可用")
        from agentic_rag.data_synthesis.synthesizer import PreferenceDataGenerator
        generator = PreferenceDataGenerator()
        prompts = request.get("prompts", [])
        pairs = generator.generate_batch(prompts)
        return {"data": [vars(p) for p in pairs]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出DPO数据失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
