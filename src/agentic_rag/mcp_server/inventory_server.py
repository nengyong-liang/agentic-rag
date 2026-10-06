"""
配件库存查询 MCP Server
提供标准的 MCP 协议接口，支持外部系统调用
"""

import requests
import httpx
from typing import Dict, Any, Optional
from loguru import logger
from agentic_rag.mcp_server.protocol import MCPServer, MCPErrorCode
from agentic_rag.mcp_server.config import settings

# 配置（与 MCP Server 共用仓库根 .env）
JAVA_API_BASE_URL = settings.JAVA_API_URL


class InventoryMCPServer(MCPServer):
    """配件库存查询 MCP Server"""
    
    def __init__(self):
        super().__init__(
            name="auto-inventory-mcp",
            version="1.0.0"
        )
        # super() 调用父类 MCPServer 的 __init__ 方法
        
        # 注册工具
        self._register_tools()
        
        # 注册资源
        self._register_resources()
    
    def _register_tools(self):
        """注册 MCP 工具"""
        
        @self.register_tool(
            name="inventory.query",
            description="查询汽车配件库存信息，包括库存数量、仓库位置、价格、供应商等",
            input_schema={
                "type": "object",
                "properties": {
                    "part_number": {
                        "type": "string",
                        "description": "配件编号，如 ENG-001（火花塞）、BRK-001（刹车片）"
                    },
                    "vehicle_model": {
                        "type": "string",
                        "description": "适用车型，如 长安 CS75 PLUS、长安逸动"
                    },
                    "category": {
                        "type": "string",
                        "description": "配件类别，如 发动机系统、制动系统"
                    }
                },
                "required": []
            }
        )
        async def query_inventory(
            part_number: Optional[str] = None,
            vehicle_model: Optional[str] = None,
            category: Optional[str] = None
        ) -> Dict[str, Any]:
            """查询配件库存"""
            try:
                # 构建查询参数
                params = {}
                if part_number:
                    params['partNumber'] = part_number
                if vehicle_model:
                    params['vehicleModel'] = vehicle_model
                if category:
                    params['category'] = category
                
                # 调用 Java API（使用免认证接口）
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/part-inventory/mcp/query",
                        params=params
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            if len(data) > 0:
                                return {
                                    "success": True,
                                    "data": {
                                        "items": data,
                                        "total": len(data)
                                    },
                                    "message": f"查询成功，找到 {len(data)} 个配件"
                                }
                            else:
                                return {
                                    "success": False,
                                    "data": None,
                                    "message": "未找到匹配的配件"
                                }
                        elif isinstance(data, dict):
                            return {
                                "success": True,
                                "data": data,
                                "message": "查询成功"
                            }
                        else:
                            return {
                                "success": False,
                                "data": None,
                                "message": "返回数据格式错误"
                            }
                    elif response.status_code == 403:
                        return {
                            "success": False,
                            "data": None,
                            "message": "无权访问，请检查权限配置"
                        }
                    elif response.status_code == 404:
                        return {
                            "success": False,
                            "data": None,
                            "message": "配件不存在"
                        }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"查询失败，HTTP 状态码：{response.status_code}"
                        }
            
            except httpx.TimeoutException:
                logger.error("请求超时")
                return {
                    "success": False,
                    "data": None,
                    "message": "请求超时，请稍后重试"
                }
            except httpx.ConnectError as e:
                logger.error(f"连接错误：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": "无法连接到库存服务，请检查服务是否运行"
                }
            except Exception as e:
                logger.error(f"查询出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"查询出错：{str(e)}"
                }
        
        @self.register_tool(
            name="inventory.detail",
            description="通过配件 ID 获取详细信息",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "配件 ID"
                    }
                },
                "required": ["id"]
            }
        )
        async def get_inventory_detail(id: int) -> Dict[str, Any]:
            """获取配件详情"""
            try:
                # 使用免认证接口查询
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/part-inventory/mcp/{id}"
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and isinstance(data, dict):
                            return {
                                "success": True,
                                "data": data,
                                "message": "查询成功"
                            }
                        else:
                            return {
                                "success": False,
                                "data": None,
                                "message": "配件不存在"
                            }
                    elif response.status_code == 404:
                        return {
                            "success": False,
                            "data": None,
                            "message": "配件不存在"
                        }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"查询失败，HTTP 状态码：{response.status_code}"
                        }
            
            except Exception as e:
                logger.error(f"查询详情出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"查询出错：{str(e)}"
                }
        
        @self.register_tool(
            name="inventory.low_stock_alert",
            description="查询库存预警，返回库存数量低于最低库存的配件列表",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        async def get_low_stock_alert() -> Dict[str, Any]:
            """查询库存预警"""
            try:
                # 使用免认证接口查询
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/part-inventory/mcp/alert/low-stock"
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            return {
                                "success": True,
                                "data": {
                                    "items": data,
                                    "total": len(data),
                                    "alert_level": "high" if len(data) > 10 else "medium" if len(data) > 5 else "low"
                                },
                                "message": f"查询成功，发现 {len(data)} 个配件库存不足"
                            }
                        else:
                            return {
                                "success": True,
                                "data": data,
                                "message": "查询成功"
                            }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"查询失败，HTTP 状态码：{response.status_code}"
                        }
            
            except Exception as e:
                logger.error(f"查询预警出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"查询出错：{str(e)}"
                }
    
    def _register_resources(self):
        """注册 MCP 资源"""
        
        @self.register_resource(
            uri="inventory://parts/{part_number}",
            name="Part Inventory Resource",
            description="通过配件编号获取配件库存信息",
            mime_type="application/json"
        )
        async def get_part_resource(uri: str) -> Dict[str, Any]:
            """获取配件资源"""
            try:
                # 从 URI 中提取配件编号
                part_number = uri.split("/")[-1]
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/part-inventory/query",
                        params={"partNumber": part_number}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            return {
                                "contents": [
                                    {
                                        "uri": uri,
                                        "mimeType": "application/json",
                                        "text": str(data)
                                    }
                                ]
                            }
                        else:
                            return {
                                "error": {
                                    "code": MCPErrorCode.RESOURCE_NOT_FOUND.value,
                                    "message": f"配件 {part_number} 不存在"
                                }
                            }
                    else:
                        return {
                            "error": {
                                "code": response.status_code,
                                "message": f"查询失败"
                            }
                        }
            
            except Exception as e:
                return {
                    "error": {
                        "code": MCPErrorCode.INTERNAL_ERROR.value,
                        "message": str(e)
                    }
                }
        
        @self.register_resource(
            uri="inventory://categories",
            name="Part Categories Resource",
            description="获取所有配件类别列表",
            mime_type="application/json"
        )
        async def get_categories_resource(uri: str) -> Dict[str, Any]:
            """获取配件类别资源"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{JAVA_API_BASE_URL}/part-inventory/list")
                    params = {"category": ""}
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            # 提取所有类别
                            categories = list(set([item.get('category') for item in data if item.get('category')]))
                            return {
                                "contents": [
                                    {
                                        "uri": uri,
                                        "mimeType": "application/json",
                                        "text": str({"categories": categories, "total": len(categories)})
                                    }
                                ]
                            }
                        else:
                            return {
                                "error": {
                                    "code": MCPErrorCode.INTERNAL_ERROR.value,
                                    "message": "数据格式错误"
                                }
                            }
                    else:
                        return {
                            "error": {
                                "code": response.status_code,
                                "message": "查询失败"
                            }
                        }
            
            except Exception as e:
                return {
                    "error": {
                        "code": MCPErrorCode.INTERNAL_ERROR.value,
                        "message": str(e)
                    }
                }
