from typing import Dict
import logging

logger = logging.getLogger(__name__)

# 尝试导入 Document
Document = None
try:
    from docx import Document
    logger.info("python-docx 已加载")
except ImportError:
    logger.warning("python-docx 未安装，Word 解析功能将不可用")


class WordParser:
    """Word 文档解析器"""
    
    def parse(self, file_path: str) -> Dict:
        """解析 Word 文件"""
        if Document is None:
            logger.error("Word 解析功能不可用，因为 python-docx 未安装")
            return {
                "text": "",
                "sections": []
            }
        
        try:
            doc = Document(file_path)
            
            full_text = []
            sections = []
            current_section = {"title": "", "content": [], "level": 0}
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                style = para.style.name if para.style else ""
                
                if "Heading" in style:
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    level = int(style.replace("Heading", "")) if style.replace("Heading", "").isdigit() else 1
                    current_section = {
                        "title": text,
                        "content": [],
                        "level": level
                    }
                else:
                    current_section["content"].append(text)
                    full_text.append(text)
            
            if current_section["content"]:
                sections.append(current_section)
            
            logger.info(f"解析 Word 成功：{file_path}, 章节数：{len(sections)}")
            
            return {
                "text": "\n".join(full_text),
                "sections": sections
            }
        except Exception as e:
            logger.error(f"解析 Word 失败：{e}")
            raise


word_parser = WordParser()
