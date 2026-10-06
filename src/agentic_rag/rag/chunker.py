from typing import List, Dict
import re
import os
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """语义切片器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_section(self, sections: List[Dict], metadata: Dict = None) -> List[Dict]:
        """按章节切片"""
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_title = section.get("title", "")
            section_content = "\n".join(section.get("content", []))
            
            if not section_content:
                continue
            
            full_text = f"{section_title}\n{section_content}" if section_title else section_content
            
            section_chunks = self._split_text(full_text, chunk_index)
            
            for chunk in section_chunks:
                chunk["metadata"] = {
                    **(metadata or {}),
                    "section": section_title,
                    "level": section.get("level", 1)
                }
            
            chunks.extend(section_chunks)
            chunk_index = chunks[-1]["chunk_index"] + 1 if chunks else chunk_index
        
        logger.info(f"按章节切片完成，共 {len(chunks)} 个切片")
        return chunks
    
    def chunk_by_paragraph(self, text: str, metadata: Dict = None) -> List[Dict]:
        """按段落切片"""
        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_index": chunk_index,
                        "content": current_chunk.strip(),
                        "metadata": metadata or {}
                    })
                    chunk_index += 1
                current_chunk = para + "\n"
        
        if current_chunk:
            chunks.append({
                "chunk_index": chunk_index,
                "content": current_chunk.strip(),
                "metadata": metadata or {}
            })
        
        logger.info(f"按段落切片完成，共 {len(chunks)} 个切片")
        return chunks
    
    def _split_text(self, text: str, start_index: int = 0) -> List[Dict]:
        """将文本分割成固定大小的切片"""
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [{
                "chunk_index": start_index,
                "content": text,
                "metadata": {}
            }]
        
        start = 0
        chunk_index = start_index
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk_text = text[start:]
            else:
                chunk_text = text[start:end]
                last_newline = chunk_text.rfind("\n")
                if last_newline != -1:
                    chunk_text = chunk_text[:last_newline]
                    end = start + last_newline
            
            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_text.strip(),
                "metadata": {}
            })
            
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def merge_results(self, vector_results: List[Dict], es_results: List[Dict]) -> List[Dict]:
        """合并向量检索和关键词检索结果"""
        merged = {}
        
        for result in vector_results:
            chunk_id = f"{result['document_id']}_{result['chunk_index']}"
            merged[chunk_id] = {
                **result,
                "vector_score": result["score"],
                "es_score": 0
            }
        
        for result in es_results:
            chunk_id = f"{result['document_id']}_{result['chunk_index']}"
            if chunk_id in merged:
                merged[chunk_id]["es_score"] = result["score"]
            else:
                merged[chunk_id] = {
                    **result,
                    "vector_score": 0,
                    "es_score": result["score"]
                }
        
        merged_list = list(merged.values())
        
        merged_list.sort(key=lambda x: x["vector_score"] + x["es_score"] * 0.1, reverse=True)
        
        return merged_list


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    semantic_chunker = SemanticChunker()
else:
    semantic_chunker = None
