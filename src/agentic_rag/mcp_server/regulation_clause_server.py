"""
制度条款核验 MCP Server
提供标准的 MCP 协议接口，支持外部系统调用
"""

import httpx
from typing import Dict, Any, Optional
from loguru import logger
from agentic_rag.mcp_server.protocol import MCPServer, MCPErrorCode
from agentic_rag.mcp_server.config import settings

# 配置
JAVA_API_BASE_URL = settings.JAVA_API_URL
# MCP Server 专用的 JWT Token（从环境变量读取，生产环境通过 MCP_JWT_TOKEN 环境变量设置）
MCP_JWT_TOKEN = settings.MCP_JWT_TOKEN


class RegulationClauseMCPServer(MCPServer):
    """制度条款核验 MCP Server"""
    
    def __init__(self):
        super().__init__(
            name="auto-regulation-mcp",
            version="1.0.0"
        )
        
        # 注册工具
        self._register_tools()
        
        # 注册资源
        self._register_resources()
    
    def _register_tools(self):
        """注册 MCP 工具"""
        
        @self.register_tool(
            name="regulation.search",
            description="搜索制度条款，支持关键词全文检索",
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如 考勤、报销、安全"
                    },
                    "category": {
                        "type": "string",
                        "description": "制度类别：人事/财务/行政/安全"
                    }
                },
                "required": ["keyword"]
            }
        )
        async def search_regulation(
            keyword: str,
            category: Optional[str] = None
        ) -> Dict[str, Any]:
            """搜索制度条款"""
            try:
                # 构建查询参数
                params = {"keyword": keyword}
                if category:
                    params['category'] = category
                
                # 调用 Java API（携带 JWT Token 进行认证）
                headers = {"Authorization": f"Bearer {MCP_JWT_TOKEN}"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/regulation-clause/mcp/search",
                        params=params,
                        headers=headers
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
                                    "message": f"搜索成功，找到 {len(data)} 条条款"
                                }
                            else:
                                return {
                                    "success": False,
                                    "data": None,
                                    "message": "未找到匹配的条款"
                                }
                        else:
                            return {
                                "success": True,
                                "data": data,
                                "message": "搜索成功"
                            }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"搜索失败，HTTP 状态码：{response.status_code}"
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
                    "message": "无法连接到制度条款服务"
                }
            except Exception as e:
                logger.error(f"搜索出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"搜索出错：{str(e)}"
                }
        
        @self.register_tool(
            name="regulation.query",
            description="查询制度条款列表，支持按类别、部门等条件查询",
            input_schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "制度类别：人事/财务/行政/安全"
                    },
                    "department": {
                        "type": "string",
                        "description": "发布部门，如 人力资源部、财务部"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "内容关键词"
                    }
                },
                "required": []
            }
        )
        async def query_regulation(
            category: Optional[str] = None,
            department: Optional[str] = None,
            keyword: Optional[str] = None
        ) -> Dict[str, Any]:
            """查询制度条款"""
            try:
                # 构建查询参数
                params = {}
                if category:
                    params['category'] = category
                if department:
                    params['department'] = department
                if keyword:
                    params['keyword'] = keyword
                
                # 调用 Java API（携带 JWT Token 进行认证）
                headers = {"Authorization": f"Bearer {MCP_JWT_TOKEN}"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/regulation-clause/mcp/query",
                        params=params,
                        headers=headers
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
                                    "message": f"查询成功，找到 {len(data)} 条条款"
                                }
                            else:
                                return {
                                    "success": False,
                                    "data": None,
                                    "message": "未找到匹配的条款"
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
                    "message": "无法连接到制度条款服务"
                }
            except Exception as e:
                logger.error(f"查询出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"查询出错：{str(e)}"
                }
    
    def _register_resources(self):
        """注册 MCP 资源"""
        
        @self.register_resource(
            uri="regulation://{regulation_number}",
            name="Regulation Clause Resource",
            description="通过制度编号获取制度条款",
            mime_type="application/json"
        )
        async def get_regulation_resource(uri: str) -> Dict[str, Any]:
            """获取制度资源"""
            try:
                # 从 URI 中提取制度编号
                regulation_number = uri.split("/")[-1]
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/regulation-clause/mcp/query",
                        params={"regulationNumber": regulation_number}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
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
                                    "message": f"制度 {regulation_number} 不存在"
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
        
        @self.register_resource(
            uri="regulation://categories",
            name="Regulation Categories Resource",
            description="获取所有制度类别列表",
            mime_type="application/json"
        )
        async def get_categories_resource(uri: str) -> Dict[str, Any]:
            """获取制度类别资源"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/regulation-clause/mcp/query"
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            # 提取所有类别
                            categories = list(set([
                                item.get('category') 
                                for item in data 
                                if item.get('category')
                            ]))
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
