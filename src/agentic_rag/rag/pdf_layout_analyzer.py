"""
PDF 版面分析器
升级版 PDF 解析，支持版面元素检测、表格结构化、图片提取、公式识别

实现方式：
- 优先使用 docling（IBM 开源，版面分析能力强）
- 备选使用 PyMuPDF（fitz）原生能力
- 视觉模型辅助分析复杂图表
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PDFLayoutAnalyzer:
    """PDF 版面分析器"""

    def __init__(self):
        self.use_docling = False
        self._check_docling()

    def _check_docling(self):
        """检查 docling 是否可用"""
        try:
            from docling.document_converter import DocumentConverter
            self.use_docling = True
            logger.info("[PDF Layout] docling 可用，使用高级版面分析")
        except ImportError:
            logger.info("[PDF Layout] docling 不可用，使用 PyMuPDF 基础版面分析")

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        分析 PDF 文件的版面结构

        Args:
            file_path: PDF 文件路径

        Returns:
            版面分析结果，包含文本、表格、图片、标题等
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在：{file_path}")
            return {"error": "文件不存在", "elements": []}

        if self.use_docling:
            return self._analyze_with_docling(file_path)
        else:
            return self._analyze_with_pymupdf(file_path)

    def _analyze_with_docling(self, file_path: str) -> Dict[str, Any]:
        """使用 docling 进行高级版面分析"""
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(file_path)

            elements = []
            for item in result.document.iterate_items():
                element = {
                    "type": self._map_docling_type(item),
                    "content": str(item.text) if hasattr(item, 'text') else "",
                    "page": getattr(item, 'page', 0),
                    "metadata": {}
                }
                elements.append(element)

            tables = []
            for table in result.document.tables:
                table_data = table.export_to_dataframe()
                tables.append({
                    "type": "table",
                    "content": table_data.to_markdown(),
                    "page": getattr(table, 'page', 0),
                    "metadata": {"format": "markdown"}
                })

            logger.info(f"[PDF Layout] docling 分析完成：{len(elements)} 个元素，{len(tables)} 个表格")

            return {
                "source": file_path,
                "parser": "docling",
                "elements": elements,
                "tables": tables,
                "total_elements": len(elements) + len(tables)
            }

        except Exception as e:
            logger.error(f"[PDF Layout] docling 分析失败：{e}，回退到 PyMuPDF")
            return self._analyze_with_pymupdf(file_path)

    def _analyze_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """使用 PyMuPDF 进行基础版面分析"""
        try:
            import fitz
        except ImportError:
            logger.error("[PDF Layout] PyMuPDF 未安装，无法解析 PDF")
            return {"error": "PyMuPDF 未安装", "elements": []}

        try:
            doc = fitz.open(file_path)
            elements = []
            tables = []
            images = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

                for block in blocks:
                    if block["type"] == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue

                                font_size = span.get("size", 12)
                                font_flags = span.get("flags", 0)
                                is_bold = bool(font_flags & 2 ** 4)

                                element_type = self._classify_text_element(
                                    text, font_size, is_bold
                                )

                                elements.append({
                                    "type": element_type,
                                    "content": text,
                                    "page": page_num + 1,
                                    "metadata": {
                                        "font_size": font_size,
                                        "is_bold": is_bold,
                                        "bbox": span.get("bbox", [])
                                    }
                                })
                    elif block["type"] == 1:
                        images.append({
                            "type": "image",
                            "content": f"[图片: 宽{block.get('width', 0):.0f}x高{block.get('height', 0):.0f}]",
                            "page": page_num + 1,
                            "metadata": {
                                "width": block.get("width", 0),
                                "height": block.get("height", 0),
                                "bbox": block.get("bbox", [])
                            }
                        })

                page_tables = page.find_tables()
                for table_idx, table in enumerate(page_tables):
                    table_data = table.extract()
                    table_md = self._table_to_markdown(table_data)
                    tables.append({
                        "type": "table",
                        "content": table_md,
                        "page": page_num + 1,
                        "metadata": {
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                            "table_index": table_idx
                        }
                    })

            doc.close()

            logger.info(
                f"[PDF Layout] PyMuPDF 分析完成："
                f"{len(elements)} 个文本元素，{len(tables)} 个表格，{len(images)} 张图片"
            )

            return {
                "source": file_path,
                "parser": "pymupdf",
                "elements": elements,
                "tables": tables,
                "images": images,
                "total_elements": len(elements) + len(tables) + len(images)
            }

        except Exception as e:
            logger.error(f"[PDF Layout] PyMuPDF 分析失败：{e}")
            return {"error": str(e), "elements": []}

    def _classify_text_element(
        self,
        text: str,
        font_size: float,
        is_bold: bool
    ) -> str:
        """根据字体大小和样式分类文本元素"""
        if font_size >= 18 and is_bold:
            return "title"
        elif font_size >= 14 and is_bold:
            return "heading"
        elif font_size >= 12 and is_bold:
            return "subheading"
        elif text.startswith(("•", "-", "*", "●", "○", "◆")):
            return "list_item"
        elif len(text) < 5 and text.endswith((":", "：")):
            return "label"
        else:
            return "paragraph"

    def _table_to_markdown(self, table_data: List) -> str:
        """将表格数据转换为 Markdown 格式"""
        if not table_data:
            return ""

        lines = []
        for i, row in enumerate(table_data):
            cells = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(lines)

    def _map_docling_type(self, item) -> str:
        """映射 docling 元素类型"""
        type_name = type(item).__name__.lower()
        type_map = {
            "textitem": "paragraph",
            "headingitem": "heading",
            "tableitem": "table",
            "pictureitem": "image",
            "listitem": "list_item",
            "codeitem": "code",
            "formulaitem": "formula",
        }
        return type_map.get(type_name, "paragraph")

    def extract_structured_content(self, file_path: str) -> Dict[str, Any]:
        """
        提取结构化内容：将版面分析结果按类型分组

        Returns:
            按类型分组的内容字典
        """
        result = self.analyze(file_path)

        if "error" in result:
            return result

        structured = {
            "source": file_path,
            "titles": [],
            "headings": [],
            "paragraphs": [],
            "tables": [],
            "images": [],
            "list_items": [],
            "other": []
        }

        for elem in result.get("elements", []):
            elem_type = elem.get("type", "paragraph")
            target_key = {
                "title": "titles",
                "heading": "headings",
                "subheading": "headings",
                "paragraph": "paragraphs",
                "list_item": "list_items",
                "image": "images",
                "table": "tables"
            }.get(elem_type, "other")

            structured[target_key].append(elem)

        for table in result.get("tables", []):
            structured["tables"].append(table)

        for image in result.get("images", []):
            structured["images"].append(image)

        structured["summary"] = {
            "titles": len(structured["titles"]),
            "headings": len(structured["headings"]),
            "paragraphs": len(structured["paragraphs"]),
            "tables": len(structured["tables"]),
            "images": len(structured["images"]),
            "list_items": len(structured["list_items"]),
        }

        return structured


pdf_layout_analyzer = PDFLayoutAnalyzer()
