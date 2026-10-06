"""文档处理端点"""
import os
import hashlib
import logging
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from agentic_rag.api.state import state
from agentic_rag.rag.pdf_parser import pdf_parser
from agentic_rag.rag.word_parser import word_parser
from agentic_rag.rag.chunker import semantic_chunker
from agentic_rag.utils.milvus import milvus_service
from agentic_rag.utils.elasticsearch import es_service
from agentic_rag.utils.quality_gate import chunk_quality_gate, document_deduplicator
from agentic_rag.rag.corpus_governance import corpus_governor

logger = logging.getLogger(__name__)

router = APIRouter()


class DocumentProcessRequest(BaseModel):
    document_id: int
    file_path: str
    file_type: str
    metadata: Optional[Dict] = None


@router.post("/document/process")
async def process_document(request: DocumentProcessRequest):
    """处理文档：解析 → 切片 → 质量治理 → 向量化

    工业级管道：
    1. 文档解析（PDF/Word）
    2. 语义切片
    3. 质量门禁（SHA-256去重 + 空切片过滤 + 低质拦截 + 元数据校验）
    4. 元数据增强（chunk_hash + 时间戳）
    5. 保鲜治理扫描（过期/冲突/重复/缺陷只标记，不改数据）
    6. 双写 Milvus + ES
    """
    try:
        logger.info(f"开始处理文档：{request.document_id}")

        # 文档级去重检查
        doc_hash = document_deduplicator.compute_document_hash(request.file_path) # 计算文档哈希
        if doc_hash: # 如果文档哈希存在
            # 检查是否重复
            is_dup, existing_id = document_deduplicator.is_duplicate(doc_hash)
            if is_dup: # 如果重复
                logger.warning(f"文档重复：{request.document_id}，已存在文档ID：{existing_id}")
                return {
                    "document_id": request.document_id,
                    "chunk_count": 0,
                    "status": "duplicate",
                    "message": f"文档与已存在文档ID={existing_id}内容完全相同，跳过处理",
                    "quality_report": {"total": 0, "passed": 0, "rejected": 0, "pass_rate": "0%"}
                }

        file_path = request.file_path
        file_type = request.file_type.upper() # 转换为大写

        if file_type == "PDF":
            parse_result = pdf_parser.parse(file_path) # 解析pdf
            sections = [{"title": "", "content": [parse_result["text"]]}] # 
        elif file_type == "WORD":
            parse_result = word_parser.parse(file_path) #   
            sections = parse_result.get("sections", [])
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{file_type}")

        # 文件的原数据
        metadata = request.metadata or {} 
        metadata["document_id"] = request.document_id
        metadata["document_name"] = metadata.get("original_name", "")
        metadata["is_latest"] = True

        if doc_hash:
            metadata["document_hash"] = f"sha256:{doc_hash}"

        # 按章节切片
        chunks = semantic_chunker.chunk_by_section(sections, metadata)

        for chunk in chunks:
            chunk["document_id"] = request.document_id

        # 质量门禁
        chunk_quality_gate.reset_seen_hashes()
        passed_chunks, quality_report = chunk_quality_gate.run_quality_gate(
            chunks, document_id=request.document_id
        )

        if not passed_chunks:
            logger.warning(f"文档 {request.document_id} 所有切片均未通过质量门禁")
            return {
                "document_id": request.document_id,
                "chunk_count": 0,
                "status": "quality_rejected",
                "quality_report": quality_report.to_dict()
            }

        # 知识库保鲜：入仓前治理扫描（只标记不删数据）
        governance = corpus_governor.scan(passed_chunks)
        if governance.findings:
            logger.warning(
                f"文档 {request.document_id} 保鲜扫描命中 {len(governance.findings)} 条："
                f"{governance.counts_by_category()}"
            )

        # 向量化
        milvus_service.insert_chunks(passed_chunks)
        es_service.insert_chunks(passed_chunks)

        # 注册文档哈希
        if doc_hash:
            document_deduplicator.register_document(doc_hash, request.document_id)

        # 缓存失效
        if state.redis_cache:
            try:
                state.redis_cache.invalidate_on_knowledge_update()
            except Exception as e:
                logger.warning(f"缓存失效失败：{e}")

        logger.info(
            f"文档处理完成：{request.document_id}, "
            f"原始切片={len(chunks)}, 通过={len(passed_chunks)}, "
            f"通过率={quality_report.pass_rate}%"
        )

        return {
            "document_id": request.document_id,
            "chunk_count": len(passed_chunks),
            "original_chunk_count": len(chunks),
            "status": "success",
            "quality_report": quality_report.to_dict(),
            "governance_report": {
                "as_of": governance.as_of,
                "finding_count": len(governance.findings),
                "by_category": governance.counts_by_category(),
                "needs_review": sorted(governance.review_ids()),
            }
        }
    except Exception as e:
        logger.error(f"处理文档失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# 处理文件上传接口
if state.has_multipart:
    @router.post("/document/upload")
    async def upload_document(file: UploadFile = File(...), document_id: int = Form(...)):
        """上传并处理文档"""
        try:
            file_content = await file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            file_extension = os.path.splitext(file.filename)[1].lower()
            file_path = os.path.join(state.DATA_DIR, f"{document_id}_{file_hash}{file_extension}")
            with open(file_path, "wb") as f:
                f.write(file_content)
            file_type = file_extension.replace(".", "").upper()
            process_request = DocumentProcessRequest(
                document_id=document_id,
                file_path=file_path,
                file_type=file_type,
                metadata={
                    "original_name": file.filename,
                    "page_number": 1,
                    "security_level": "public",
                    "tenant_id": "default"
                }
            )
            return await process_document(process_request)
        except Exception as e:
            logger.error(f"上传文档失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))
else:
    @router.post("/document/upload")
    async def upload_document():
        """上传并处理文档"""
        raise HTTPException(status_code=400, detail="文件上传功能不可用，因为缺少 python-multipart 依赖")


@router.delete("/document/{document_id}")
async def delete_document(document_id: int):
    """删除文档"""
    try:
        milvus_service.delete_by_document_id(document_id)
        es_service.delete_by_document_id(document_id)

        logger.info(f"删除文档成功：{document_id}")
        return {"document_id": document_id, "status": "success"}
    except Exception as e:
        logger.error(f"删除文档失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
