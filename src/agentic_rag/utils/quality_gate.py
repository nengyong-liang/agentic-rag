import hashlib
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """质量治理报告"""
    total_chunks: int = 0
    passed_chunks: int = 0
    rejected_chunks: int = 0
    duplicate_count: int = 0
    empty_count: int = 0
    too_short_count: int = 0
    special_char_ratio_high: int = 0
    rejection_reasons: List[Dict] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []
    
    @property
    def pass_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return round(self.passed_chunks / self.total_chunks * 100, 2)
    
    def to_dict(self) -> Dict:
        return {
            "total": self.total_chunks,
            "passed": self.passed_chunks,
            "rejected": self.rejected_chunks,
            "duplicate_count": self.duplicate_count,
            "empty_count": self.empty_count,
            "too_short_count": self.too_short_count,
            "special_char_ratio_high": self.special_char_ratio_high,
            "pass_rate": f"{self.pass_rate}%",
            "rejection_reasons": self.rejection_reasons[:20]
        }


class ChunkQualityGate:
    """切片级质量治理器
    
    工业级 RAG 数据处理管道中的质量门禁，负责：
    1. SHA-256 精确去重（完全重复检测）
    2. 空切片过滤
    3. 过短切片拦截
    4. 特殊字符比例过高拦截
    5. 元数据必填字段校验
    """
    
    def __init__(
        self,
        min_chunk_length: int = 20,
        max_special_char_ratio: float = 0.3,
        enable_dedup: bool = True,
        required_metadata_fields: Optional[List[str]] = None
    ):
        self.min_chunk_length = min_chunk_length
        self.max_special_char_ratio = max_special_char_ratio
        self.enable_dedup = enable_dedup
        self.required_metadata_fields = required_metadata_fields or [
            "document_id", "document_name", "page_number", 
            "security_level", "tenant_id"
        ]
        
        self._seen_hashes: set = set()
    
    def compute_sha256(self, content: str) -> str:
        """计算内容的 SHA-256 哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def check_special_char_ratio(self, content: str) -> float:
        """计算特殊字符比例"""
        if not content:
            return 1.0
        
        total_chars = len(content)
        if total_chars == 0:
            return 1.0
        
        special_chars = len(re.sub(r'[\w\u4e00-\u9fff\s]', '', content))
        return special_chars / total_chars
    
    def validate_metadata(
        self, 
        chunk: Dict, 
        required_fields: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """验证元数据必填字段"""
        fields = required_fields or self.required_metadata_fields
        metadata = chunk.get("metadata", {})
        missing = []
        
        for field in fields:
            value = metadata.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        
        return len(missing) == 0, missing
    
    def enrich_metadata(self, chunk: Dict) -> Dict:
        """增强元数据：添加 chunk_hash 和时间戳"""
        content = chunk.get("content", "")
        chunk_hash = self.compute_sha256(content)
        
        from datetime import datetime
        
        metadata = chunk.get("metadata", {})
        metadata["chunk_hash"] = f"sha256:{chunk_hash}"
        metadata["chunk_length"] = len(content)
        metadata["processed_at"] = datetime.now().isoformat()
        
        chunk["metadata"] = metadata
        return chunk
    
    def check_single_chunk(self, chunk: Dict) -> Tuple[bool, str]:
        """检查单个切片是否通过质量门禁
        
        Returns:
            (是否通过, 拒绝原因) 如果通过则拒绝原因为空字符串
        """
        content = chunk.get("content", "")
        
        # 1. 空切片检查
        if not content or not content.strip():
            return False, "empty"
        
        # 2. 过短切片检查
        if len(content.strip()) < self.min_chunk_length:
            return False, "too_short"
        
        # 3. 特殊字符比例检查
        ratio = self.check_special_char_ratio(content)
        if ratio > self.max_special_char_ratio:
            return False, "special_char_ratio_high"
        
        # 4. SHA-256 去重检查
        if self.enable_dedup:
            chunk_hash = self.compute_sha256(content)
            if chunk_hash in self._seen_hashes:
                return False, "duplicate"
            self._seen_hashes.add(chunk_hash)
        
        return True, ""
    
    def run_quality_gate(
        self, 
        chunks: List[Dict],
        document_id: Optional[int] = None
    ) -> Tuple[List[Dict], QualityReport]:
        """运行质量门禁，返回通过检查的切片和报告
        
        Args:
            chunks: 待检查的切片列表
            document_id: 文档ID，用于日志记录
            
        Returns:
            (通过的切片列表, 质量报告)
        """
        report = QualityReport(total_chunks=len(chunks))
        passed_chunks = []
        
        logger.info(f"开始质量门禁检查，文档ID={document_id}，总切片数={len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            passed, reason = self.check_single_chunk(chunk)
            
            if passed:
                enriched = self.enrich_metadata(chunk)
                
                meta_valid, missing_fields = self.validate_metadata(enriched)
                if not meta_valid:
                    report.rejected_chunks += 1
                    report.rejection_reasons.append({
                        "index": i,
                        "reason": f"missing_metadata:{','.join(missing_fields)}",
                        "content_preview": chunk.get("content", "")[:50]
                    })
                    continue
                
                passed_chunks.append(enriched)
                report.passed_chunks += 1
            else:
                report.rejected_chunks += 1
                
                reason_map = {
                    "empty": ("empty_count", "空切片"),
                    "too_short": ("too_short_count", f"过短(<{self.min_chunk_length})"),
                    "special_char_ratio_high": ("special_char_ratio_high", "特殊字符过多"),
                    "duplicate": ("duplicate_count", "SHA-256重复")
                }
                
                counter_name, desc = reason_map.get(reason, ("unknown", reason))
                setattr(report, counter_name, getattr(report, counter_name) + 1)
                
                report.rejection_reasons.append({
                    "index": i,
                    "reason": desc,
                    "content_preview": chunk.get("content", "")[:50]
                })
        
        logger.info(
            f"质量门禁完成：通过={report.passed_chunks}/{report.total_chunks} "
            f"({report.pass_rate}%)，拒绝={report.rejected_chunks} "
            f"[重复={report.duplicate_count}, 空={report.empty_count}, "
            f"过短={report.too_short_count}, 特殊字符={report.special_char_ratio_high}]"
        )
        
        return passed_chunks, report
    
    def reset_seen_hashes(self):
        """重置已见哈希集合（新批次处理时调用）"""
        self._seen_hashes.clear()
        logger.info("已重置去重哈希集合")


class DocumentDeduplicator:
    """文档级去重器（基于全文 SHA-256）"""
    
    def __init__(self):
        self._doc_hashes: Dict[str, int] = {}
    
    def compute_document_hash(self, file_path: str) -> str:
        """计算文件的 SHA-256 哈希"""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for block in iter(lambda: f.read(8192), b''):
                    sha256.update(block)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败：{e}")
            return ""
    
    def is_duplicate(self, doc_hash: str) -> Tuple[bool, Optional[int]]:
        """检查文档是否重复
        
        Returns:
            (是否重复, 已存在的文档ID)
        """
        if doc_hash in self._doc_hashes:
            return True, self._doc_hashes[doc_hash]
        return False, None
    
    def register_document(self, doc_hash: str, document_id: int):
        """注册文档哈希"""
        self._doc_hashes[doc_hash] = document_id


chunk_quality_gate = ChunkQualityGate()
document_deduplicator = DocumentDeduplicator()
