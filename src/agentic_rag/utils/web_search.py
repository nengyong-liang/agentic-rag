"""
Web Search 服务
基于用户 mcp_web_search 项目适配，为 RAG 系统提供联网搜索能力

用途：
- CRAG handle_no_info 时触发联网搜索
- 补充知识库缺失的实时信息
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class WebSearchService:
    """网络搜索服务，支持 SerpAPI 和 Bing API"""

    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY", "")
        self.bing_api_key = os.getenv("BING_API_KEY", "")
        self.default_engine = os.getenv("SEARCH_ENGINE", "google")
        self.default_results = 5
        self.timeout = 30

        if not self.serpapi_key and not self.bing_api_key:
            logger.warning("未配置任何搜索 API 密钥！请设置 SERPAPI_KEY 或 BING_API_KEY")

    def is_available(self) -> bool:
        """检查搜索服务是否可用"""
        return bool(self.serpapi_key or self.bing_api_key)

    async def search(
        self,
        query: str,
        num_results: int = 5,
        engine: str = None
    ) -> Dict[str, Any]:
        """
        执行网络搜索

        Args:
            query: 搜索查询词
            num_results: 返回结果数量
            engine: 搜索引擎，google 或 bing

        Returns:
            搜索结果字典
        """
        if not query or not query.strip():
            return {"error": "查询词不能为空", "query": query}

        num_results = max(1, min(10, num_results))
        engine = engine or self.default_engine

        try:
            if engine == "bing" and self.bing_api_key:
                return await self._search_with_bing(query, num_results)
            elif self.serpapi_key:
                return await self._search_with_serpapi(query, num_results, engine)
            else:
                return {"error": "无可用搜索引擎", "query": query}
        except Exception as e:
            logger.error(f"搜索失败：{e}")
            return {"error": str(e), "query": query}

    async def _search_with_serpapi(
        self,
        query: str,
        num_results: int,
        engine: str
    ) -> Dict[str, Any]:
        serpapi_engine = "google" if engine == "google" else engine
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
            "engine": serpapi_engine,
            "hl": "zh-cn",
            "gl": "cn"
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://serpapi.com/search",
                params=params
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_serpapi_results(data, query)

    def _parse_serpapi_results(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        results = []
        organic_results = data.get("organic_results", [])
        for item in organic_results[:self.default_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0)
            })

        knowledge_graph = None
        if data.get("knowledge_graph"):
            kg = data["knowledge_graph"]
            knowledge_graph = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", "")
            }

        return {
            "query": query,
            "search_engine": data.get("search_metadata", {}).get("engine", "unknown"),
            "results": results,
            "knowledge_graph": knowledge_graph,
            "web_search": True
        }

    async def _search_with_bing(self, query: str, num_results: int) -> Dict[str, Any]:
        params = {
            "q": query,
            "count": num_results,
            "mkt": "zh-CN"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

        results = []
        web_pages = data.get("webPages", {}).get("value", [])
        for item in web_pages[:self.default_results]:
            results.append({
                "title": item.get("name", ""),
                "link": item.get("url", ""),
                "snippet": item.get("snippet", "")
            })

        return {
            "query": query,
            "search_engine": "bing",
            "results": results,
            "knowledge_graph": None,
            "web_search": True
        }

    def format_results_for_prompt(self, results: Dict[str, Any]) -> str:
        """将搜索结果格式化为 LLM 可用的上下文格式"""
        if "error" in results:
            return f"搜索失败：{results['error']}"

        lines = [f"网络搜索结果（查询：{results.get('query', '')}）："]

        kg = results.get("knowledge_graph")
        if kg:
            lines.append(f"知识图谱：{kg.get('title', '')} - {kg.get('description', '')}")

        for i, item in enumerate(results.get("results", []), 1):
            lines.append(f"[{i}] {item.get('title', '')}")
            lines.append(f"    摘要：{item.get('snippet', '')[:200]}")
            lines.append(f"    链接：{item.get('link', '')}")

        return "\n".join(lines)


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    web_search_service = WebSearchService()
else:
    web_search_service = None
