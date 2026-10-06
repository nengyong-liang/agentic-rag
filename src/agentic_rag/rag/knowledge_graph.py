"""
轻量级知识图谱模块
基于 Milvus + MySQL 的实体关系存储，无需 Neo4j

功能：
- 从文档中提取实体和关系
- 存储实体关系到 MySQL
- 支持实体检索和关系查询
- 多跳推理辅助

设计原则：
- 不引入 Neo4j，复用现有基础设施
- 实体关系存储在 MySQL，向量检索用 Milvus
- LLM 驱动的实体关系提取
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """轻量级知识图谱"""

    def __init__(self):
        self.alibaba_api_key = os.getenv("ALIBABA_API_KEY", "")
        self.qwen_api_url = os.getenv(
            "QWEN_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
        self.qwen_model = os.getenv("QWEN_MODEL", "qwen-plus")
        self.java_api_base = "http://localhost:8080/api"

    def extract_entities_and_relations(self, text: str, doc_name: str = "") -> Dict[str, Any]:
        """
        从文本中提取实体和关系

        Args:
            text: 输入文本
            doc_name: 来源文档名称

        Returns:
            包含 entities 和 relations 的字典
        """
        if not text or len(text) < 20:
            return {"entities": [], "relations": []}

        prompt = f"""请从以下文本中提取实体和关系。

文本：
{text[:2000]}

请以JSON格式输出，格式如下：
{{
  "entities": [
    {{"name": "实体名", "type": "类型(人/组织/产品/技术/概念/地点)", "description": "简短描述"}}
  ],
  "relations": [
    {{"subject": "主体实体", "predicate": "关系类型", "object": "客体实体"}}
  ]
}}

要求：
1. 只提取文本中明确提到的实体和关系
2. 关系类型包括：属于/包含/依赖/影响/使用/生产/管理/合作等
3. 最多提取10个实体和10个关系
4. 只输出JSON，不要其他内容"""

        try:
            result_text = self._call_llm(prompt)

            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                relations = data.get("relations", [])

                for e in entities:
                    e["source"] = doc_name
                for r in relations:
                    r["source"] = doc_name

                logger.info(
                    f"[KG] 提取完成：{len(entities)} 个实体，{len(relations)} 条关系"
                )

                return {"entities": entities, "relations": relations}
            else:
                logger.warning(f"[KG] 无法解析JSON：{result_text[:100]}")
                return {"entities": [], "relations": []}

        except Exception as e:
            logger.error(f"[KG] 实体关系提取失败：{e}")
            return {"entities": [], "relations": []}

    def query_related_entities(
        self,
        entity_name: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        查询与指定实体相关的实体和关系

        Args:
            entity_name: 实体名称
            depth: 查询深度（1=直接关系，2=二跳关系）

        Returns:
            相关实体和关系
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.java_api_base}/knowledge-graph/entity/query",
                    params={"name": entity_name, "depth": depth}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass

        return {"entity": entity_name, "related_entities": [], "relations": []}

    def multi_hop_reason(
        self,
        query: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        多跳推理：基于知识图谱回答需要多步推理的问题

        Args:
            query: 用户问题
            max_hops: 最大推理跳数

        Returns:
            推理路径列表
        """
        prompt = f"""请分析以下问题是否需要多跳推理，如果是，请给出推理路径。

                    问题：{query}

                    请以JSON格式输出：
                    {{
                    "needs_multi_hop": true/false,
                    "reasoning_path": [
                        {{"step": 1, "query": "第一步需要查找的信息", "entity": "关键实体"}},
                        {{"step": 2, "query": "第二步需要查找的信息", "entity": "关键实体"}}
                    ],
                    "sub_queries": ["拆分后的子问题1", "拆分后的子问题2"]
                    }}

                    只输出JSON，不要其他内容。"""

        try:
            result_text = self._call_llm(prompt)

            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
                logger.info(
                    f"[KG] 多跳推理分析：needs_multi_hop={data.get('needs_multi_hop')}, "
                    f"子问题数={len(data.get('sub_queries', []))}"
                )
                return data.get("sub_queries", [])
        except Exception as e:
            logger.error(f"[KG] 多跳推理失败：{e}")

        return [query]

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        headers = {
            "Authorization": f"Bearer {self.alibaba_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.qwen_model,
            "messages": [
                {"role": "system", "content": "你是一个知识图谱构建专家，擅长从文本中提取实体和关系。只输出JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.1
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.qwen_api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


knowledge_graph = KnowledgeGraph()
