from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# 尝试导入 fitz
fitz = None
try:
    import fitz
    logger.info("PyMuPDF 已加载")
except ImportError:
    logger.warning("PyMuPDF 未安装，PDF 解析功能将不可用")


class PDFParser:
    """PDF 文档解析器"""
    
    def parse(self, file_path: str) -> Dict: # 输入字符的路径，返回字典
        """解析 PDF 文件"""
        if fitz is None:
            logger.error("PDF 解析功能不可用，因为 PyMuPDF 未安装")
            return {
                "text": "",
                "pages": [],
                "total_pages": 0
            }
        
        try:
            doc = fitz.open(file_path) # 打开文件，fitz = PyMuPDF
            
            full_text = ""
            pages = []
            total_pages = len(doc)
            
            for page_num in range(total_pages): # 迭代页面
                page = doc[page_num]
                text = page.get_text()
                full_text += text
                
                pages.append({
                    "page_number": page_num + 1,
                    "content": text
                })
            
            doc.close()
            
            logger.info(f"解析 PDF 成功：{file_path}, 页数：{len(pages)}")
            
            return {
                "text": full_text,
                "pages": pages,
                "total_pages": total_pages
            }
        except Exception as e:
            logger.error(f"解析 PDF 失败：{e}")
            raise
    
    def parse_tables(self, file_path: str) -> List[Dict]:
        """解析 PDF 中的表格"""
        if fitz is None:
            logger.error("PDF 表格解析功能不可用，因为 PyMuPDF 未安装")
            return []
        
        try:
            doc = fitz.open(file_path)
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_tables = page.find_tables()
                
                for table_idx, table in enumerate(page_tables):
                    table_data = table.extract()
                    tables.append({
                        "page_number": page_num + 1,
                        "table_index": table_idx,
                        "data": table_data,
                        "content": self.table_to_text(table_data)
                    })
            
            doc.close()
            logger.info(f"解析 PDF 表格成功：{file_path}, 表格数：{len(tables)}")
            return tables
        except Exception as e:
            logger.error(f"解析 PDF 表格失败：{e}")
            return []
    
    def table_to_text(self, table_data: List) -> str:
        """将表格数据转换为文本"""
        text_lines = []
        for row in table_data:
            row_text = " | ".join([str(cell) if cell else "" for cell in row])
            text_lines.append(row_text)
        return "\n".join(text_lines)


pdf_parser = PDFParser()
