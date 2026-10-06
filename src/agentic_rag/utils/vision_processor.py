"""
多模态文档处理器
基于用户的 vision_analyzer_mcp 项目适配

功能：
- PDF/图片上传时自动提取图片、表格、公式
- 使用 Qwen-VL 生成图片描述
- 将描述文本向量化存入 Milvus

调用方式：
- 优先使用 OpenAI 兼容接口（复用 ALIBABA_API_KEY）
- 备选 DashScope 原生 SDK
- 备选 Doubao Vision（火山引擎）
"""

import os
import re
import base64
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class VisionProcessor:
    """视觉处理器：解析文档中的图片、表格、公式"""

    def __init__(self):
        self.alibaba_api_key = os.getenv("ALIBABA_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
        self.ark_api_key = os.getenv("ARK_API_KEY", os.getenv("DOUBAO_API_KEY", ""))
        self.qwen_model = os.getenv("VISION_MODEL", "qwen-vl-plus")
        self.qwen_api_url = os.getenv(
            "QWEN_VL_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

    def is_available(self) -> bool:
        """检查视觉模型是否可用"""
        return bool(self.alibaba_api_key or self.ark_api_key)

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "请详细描述这张图片的内容，包括文字、图表、数据等信息。",
        provider: str = "qwen"
    ) -> str:
        """
        分析单张图片

        Args:
            image_path: 图片路径
            prompt: 分析指令
            provider: 模型提供商，qwen 或 doubao

        Returns:
            图片描述文本
        """
        if not os.path.exists(image_path):
            logger.error(f"图片不存在：{image_path}")
            return ""

        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")

            mime_type = self._get_mime_type(image_path)
            image_content = f"data:{mime_type};base64,{base64_str}"

            if provider == "qwen" and self.alibaba_api_key:
                return self._call_qwen_vl_openai(image_content, prompt)
            elif provider == "doubao" and self.ark_api_key:
                return self._call_doubao_vision(image_content, prompt)
            else:
                logger.warning("无可用的视觉模型")
                return ""
        except Exception as e:
            logger.error(f"图片分析失败：{e}")
            return ""

    def analyze_image_url(
        self,
        image_url: str,
        prompt: str = "请详细描述这张图片的内容，包括文字、图表、数据等信息。",
        provider: str = "qwen"
    ) -> str:
        """
        分析网络图片URL

        Args:
            image_url: 图片URL
            prompt: 分析指令
            provider: 模型提供商

        Returns:
            图片描述文本
        """
        try:
            if provider == "qwen" and self.alibaba_api_key:
                return self._call_qwen_vl_openai(image_url, prompt)
            elif provider == "doubao" and self.ark_api_key:
                return self._call_doubao_vision(image_url, prompt)
            else:
                logger.warning("无可用的视觉模型")
                return ""
        except Exception as e:
            logger.error(f"URL图片分析失败：{e}")
            return ""

    def _call_qwen_vl_openai(self, image_content: str, prompt: str) -> str:
        """通过 OpenAI 兼容接口调用 Qwen-VL"""
        headers = {
            "Authorization": f"Bearer {self.alibaba_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.qwen_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_content}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "stream": False
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.qwen_api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Qwen-VL 响应解析失败：{e}, 响应：{str(data)[:200]}")
            return ""

    def _call_doubao_vision(self, image_content: str, prompt: str) -> str:
        """通过火山引擎 Ark SDK 调用 Doubao Vision"""
        try:
            from volcenginesdkarkruntime import Ark

            client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.ark_api_key
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_content}},
                    {"type": "text", "text": prompt}
                ]
            }]

            response = client.responses.create(
                model=os.getenv("DOUBAO_VISION_MODEL", "doubao-seed-1-6-vision"),
                messages=messages
            )

            return response.choices[0].message.content
        except ImportError:
            logger.error("volcenginesdkarkruntime 未安装。运行：pip install volcengine-python-sdk[ark]")
            return ""
        except Exception as e:
            logger.error(f"Doubao Vision 调用失败：{e}")
            return ""

    def _get_mime_type(self, path: str) -> str:
        lower_path = path.lower()
        if lower_path.endswith(('.jpg', '.jpeg')):
            return "image/jpeg"
        elif lower_path.endswith('.png'):
            return "image/png"
        elif lower_path.endswith('.gif'):
            return "image/gif"
        elif lower_path.endswith('.webp'):
            return "image/webp"
        elif lower_path.endswith('.pdf'):
            return "application/pdf"
        else:
            return "image/jpeg"

    def process_document_images(
        self,
        doc_path: str,
        image_paths: List[str],
        doc_name: str
    ) -> List[Dict[str, Any]]:
        """
        处理文档中的所有图片

        Args:
            doc_path: 文档路径
            image_paths: 提取的图片路径列表
            doc_name: 文档名称

        Returns:
            图片描述列表
        """
        if not self.is_available():
            logger.info("视觉模型不可用，跳过图片处理")
            return []

        descriptions = []
        for i, img_path in enumerate(image_paths):
            logger.info(f"[Vision] 处理图片 {i+1}/{len(image_paths)}：{img_path}")

            prompt = f"这是文档《{doc_name}》中的第{i+1}张图片。请详细描述图片内容，包括图表类型、数据趋势、文字说明等关键信息。"
            desc = self.analyze_image(img_path, prompt)

            if desc:
                descriptions.append({
                    "type": "image",
                    "content": desc,
                    "metadata": {
                        "source": doc_name,
                        "image_path": img_path,
                        "image_index": i + 1,
                        "modality": "vision"
                    }
                })

        return descriptions

    def extract_images_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        从PDF中提取图片

        需要安装 PyMuPDF：pip install pymupdf
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            image_paths = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images(full=True)

                for img_idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    output_path = os.path.join(
                        output_dir,
                        f"{Path(pdf_path).stem}_page{page_num+1}_img{img_idx+1}.{image_ext}"
                    )

                    with open(output_path, "wb") as f:
                        f.write(image_bytes)

                    image_paths.append(output_path)
                    logger.info(f"[Vision] 提取图片：{output_path}")

            doc.close()
            return image_paths

        except ImportError:
            logger.warning("PyMuPDF 未安装，跳过图片提取。运行：pip install pymupdf")
            return []
        except Exception as e:
            logger.error(f"PDF图片提取失败：{e}")
            return []


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    vision_processor = VisionProcessor()
else:
    vision_processor = None
