"""
MCP Protocol 核心模块
实现标准的 Model Context Protocol 协议
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
import json


class MCPErrorCode(Enum):
    """MCP 错误码定义"""
    SUCCESS = 0
    PARAM_ERROR = 1001
    AUTH_ERROR = 1002
    PERMISSION_ERROR = 1003
    RESOURCE_NOT_FOUND = 1004
    INTERNAL_ERROR = 1005
    RATE_LIMIT_ERROR = 1006
    TOOL_NOT_FOUND = 1007
    RESOURCE_NOT_AVAILABLE = 1008


class MCPRequest(BaseModel):
    """MCP 请求模型"""
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP 响应模型"""
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPError(BaseModel):
    """MCP 错误模型"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class ResourceDefinition(BaseModel):
    """MCP 资源定义"""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None


class MCPServer:
    """
    MCP Server 基类
    实现标准的 MCP 协议
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, ToolDefinition] = {} #变量名: 类型提示 = 初始值
        self.resources: Dict[str, ResourceDefinition] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.resource_handlers: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any]):
        """注册 MCP 工具"""
        def decorator(func: Callable):
            self.tools[name] = ToolDefinition(
                name=name,
                description=description,
                inputSchema=input_schema
            )
            self.tool_handlers[name] = func
            return func
        return decorator
    
    def register_resource(self, uri: str, name: str, description: str, mime_type: str = "application/json"):
        """注册 MCP 资源"""
        def decorator(func: Callable):
            self.resources[uri] = ResourceDefinition(
                uri=uri,
                name=name,
                description=description,
                mimeType=mime_type
            )
            self.resource_handlers[uri] = func
            return func
        return decorator
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理 MCP 请求"""
        try:
            if request.method == "tools/list":
                # 列出所有工具
                return MCPResponse(
                    id=request.id,
                    result={
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in self.tools.values()
                        ]
                    }
                )
            
            elif request.method == "tools/call":
                # 调用工具
                tool_name = request.params.get("name")
                tool_args = request.params.get("arguments", {})
                
                if tool_name not in self.tool_handlers:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": MCPErrorCode.TOOL_NOT_FOUND.value,
                            "message": f"Tool not found: {tool_name}"
                        }
                    )
                
                # 调用工具处理函数
                result = await self.tool_handlers[tool_name](**tool_args)
                return MCPResponse(
                    id=request.id,
                    result=result
                )
            
            elif request.method == "resources/list":
                # 列出所有资源
                return MCPResponse(
                    id=request.id,
                    result={
                        "resources": [
                            {
                                "uri": res.uri,
                                "name": res.name,
                                "description": res.description,
                                "mimeType": res.mimeType
                            }
                            for res in self.resources.values()
                        ]
                    }
                )
            
            elif request.method == "resources/read":
                # 读取资源
                uri = request.params.get("uri")
                
                if uri not in self.resource_handlers:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": MCPErrorCode.RESOURCE_NOT_AVAILABLE.value,
                            "message": f"Resource not available: {uri}"
                        }
                    )
                
                # 调用资源处理函数
                result = await self.resource_handlers[uri](uri)
                return MCPResponse(
                    id=request.id,
                    result=result
                )
            
            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": MCPErrorCode.PARAM_ERROR.value,
                        "message": f"Unknown method: {request.method}"
                    }
                )
        
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={
                    "code": MCPErrorCode.INTERNAL_ERROR.value,
                    "message": str(e)
                }
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": self.version,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources)
        }
    
    def list_tools(self) -> list:
        """获取工具列表"""
        return list(self.tools.values())
    
    def list_resources(self) -> list:
        """获取资源列表"""
        return list(self.resources.values())
