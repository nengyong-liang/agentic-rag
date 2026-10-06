"""
MCP 客户端 SDK
提供简单易用的 MCP Server 调用接口
"""

import os

import httpx
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger


class MCPClient:
    """MCP Server 客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: str = os.getenv("MCP_API_KEY", ""),
        timeout: float = 30.0
    ):
        """
        初始化 MCP 客户端
        
        Args:
            base_url: MCP Server 地址
            api_key: API Key
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        logger.info(f"MCP 客户端初始化 | 地址：{base_url}")
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(method, url, headers=self.headers, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"HTTP 请求失败：{e}")
                raise
            except Exception as e:
                logger.error(f"请求异常：{e}")
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            return response.json()
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        result = await self._request("GET", "/tools")
        return result.get("result", [])
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """获取可用资源列表"""
        result = await self._request("GET", "/resources")
        return result.get("result", [])
    
    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具调用结果
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
            "id": 1
        }
        
        logger.debug(f"调用工具：{tool_name} | 参数：{arguments}")
        result = await self._request("POST", "/mcp", json=payload)
        return result.get("result", {})
    
    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """
        读取 MCP 资源
        
        Args:
            resource_uri: 资源 URI
        
        Returns:
            资源内容
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {
                "uri": resource_uri
            },
            "id": 1
        }
        
        logger.debug(f"读取资源：{resource_uri}")
        result = await self._request("POST", "/mcp", json=payload)
        return result.get("result", {})


class InventoryClient:
    """配件库存查询客户端（高级封装）"""
    
    def __init__(self, mcp_client: MCPClient):
        """
        初始化库存客户端
        
        Args:
            mcp_client: MCP 客户端实例
        """
        self.mcp = mcp_client
        logger.info("配件库存客户端初始化")
    
    async def query(
        self,
        part_number: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        查询配件库存
        
        Args:
            part_number: 配件编号
            vehicle_model: 适用车型
            category: 配件类别
        
        Returns:
            库存查询结果
        """
        args = {}
        if part_number:
            args["part_number"] = part_number
        if vehicle_model:
            args["vehicle_model"] = vehicle_model
        if category:
            args["category"] = category
        
        result = await self.mcp.call_tool("inventory.query", args)
        logger.info(f"库存查询 | 参数：{args} | 结果：{result.get('success')}")
        return result
    
    async def get_detail(self, part_number: str) -> Dict[str, Any]:
        """
        获取配件详情
        
        Args:
            part_number: 配件编号
        
        Returns:
            配件详情
        """
        result = await self.mcp.call_tool("inventory.detail", {"part_number": part_number})
        logger.info(f"配件详情查询 | 编号：{part_number}")
        return result
    
    async def get_low_stock_alert(self, threshold: int = 20) -> Dict[str, Any]:
        """
        获取低库存预警
        
        Args:
            threshold: 库存预警阈值
        
        Returns:
            低库存配件列表
        """
        result = await self.mcp.call_tool("inventory.low_stock_alert", {"threshold": threshold})
        logger.info(f"低库存预警 | 阈值：{threshold}")
        return result


# ============== 使用示例 ==============
async def example_usage():
    """使用示例"""
    # 初始化客户端
    client = MCPClient(
        base_url="http://localhost:8001",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    
    # 健康检查
    health = await client.health_check()
    print(f"服务状态：{health}")
    
    # 获取工具列表
    tools = await client.list_tools()
    print(f"可用工具：{[t['name'] for t in tools]}")
    
    # 调用工具
    result = await client.call_tool("inventory.query", {"part_number": "PN001"})
    print(f"查询结果：{result}")
    
    # 使用高级客户端
    inventory_client = InventoryClient(client)
    detail = await inventory_client.get_detail("PN001")
    print(f"配件详情：{detail}")


if __name__ == "__main__":
    asyncio.run(example_usage())
