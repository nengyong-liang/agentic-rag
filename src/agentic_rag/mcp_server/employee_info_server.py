"""
员工信息核验 MCP Server
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


class EmployeeInfoMCPServer(MCPServer):
    """员工信息核验 MCP Server"""
    
    def __init__(self):
        super().__init__(
            name="auto-employee-mcp",
            version="1.0.0"
        )
        
        # 注册工具
        self._register_tools()
        
        # 注册资源
        self._register_resources()
    
    def _register_tools(self):
        """注册 MCP 工具"""
        
        @self.register_tool(
            name="employee.verify",
            description="核验员工身份信息，包括员工编号、姓名、部门、职位等",
            input_schema={
                "type": "object",
                "properties": {
                    "employee_number": {
                        "type": "string",
                        "description": "员工编号，如 EMP-001"
                    },
                    "name": {
                        "type": "string",
                        "description": "员工姓名（可选，用于二次核验）"
                    }
                },
                "required": ["employee_number"]
            }
        )
        async def verify_employee(
            employee_number: str,
            name: Optional[str] = None
        ) -> Dict[str, Any]:
            """核验员工信息"""
            try:
                # 构建查询参数
                params = {"employeeNumber": employee_number}
                if name:
                    params['name'] = name
                
                # 调用 Java API（携带 JWT Token 进行认证）
                headers = {"Authorization": f"Bearer {MCP_JWT_TOKEN}"}
                async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/employee-info/mcp/verify",
                        params=params,
                        headers=headers
                    )
                    
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": data.get("success", False),
                        "data": data.get("data"),
                        "message": data.get("message", "核验完成")
                    }
                elif response.status_code == 404:
                    return {
                        "success": False,
                        "data": None,
                        "message": "员工不存在"
                    }
                else:
                    return {
                        "success": False,
                        "data": None,
                        "message": f"核验失败，HTTP 状态码：{response.status_code}"
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
                    "message": "无法连接到员工信息服务，请检查服务是否运行"
                }
            except Exception as e:
                logger.error(f"核验出错：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"核验出错：{str(e)}"
                }
        
        @self.register_tool(
            name="employee.query",
            description="查询员工信息，支持按部门、职位等条件查询",
            input_schema={
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "部门名称，如 人力资源部、财务部、技术研发部"
                    },
                    "position": {
                        "type": "string",
                        "description": "职位名称，如 经理、会计、工程师"
                    },
                    "status": {
                        "type": "string",
                        "description": "员工状态：1-在职，0-离职"
                    }
                },
                "required": []
            }
        )
        async def query_employee(
            department: Optional[str] = None,
            position: Optional[str] = None,
            status: Optional[str] = None
        ) -> Dict[str, Any]:
            """查询员工信息"""
            try:
                # 构建查询参数
                params = {}
                if department:
                    params['department'] = department
                if position:
                    params['position'] = position
                if status:
                    params['status'] = status
                
                # 调用 Java API（携带 JWT Token 进行认证）
                headers = {"Authorization": f"Bearer {MCP_JWT_TOKEN}"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/employee-info/mcp/query",
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
                                    "message": f"查询成功，找到 {len(data)} 个员工"
                                }
                            else:
                                return {
                                    "success": False,
                                    "data": None,
                                    "message": "未找到匹配的员工"
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
                    "message": "无法连接到员工信息服务"
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
            uri="employee://{employee_number}",
            name="Employee Info Resource",
            description="通过员工编号获取员工信息",
            mime_type="application/json"
        )
        async def get_employee_resource(uri: str) -> Dict[str, Any]:
            """获取员工资源"""
            try:
                # 从 URI 中提取员工编号
                employee_number = uri.split("/")[-1]
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/employee-info/mcp/verify",
                        params={"employeeNumber": employee_number}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            return {
                                "contents": [
                                    {
                                        "uri": uri,
                                        "mimeType": "application/json",
                                        "text": str(data.get("data"))
                                    }
                                ]
                            }
                        else:
                            return {
                                "error": {
                                    "code": MCPErrorCode.RESOURCE_NOT_FOUND.value,
                                    "message": f"员工 {employee_number} 不存在"
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
            uri="employee://departments",
            name="Employee Departments Resource",
            description="获取所有部门列表",
            mime_type="application/json"
        )
        async def get_departments_resource(uri: str) -> Dict[str, Any]:
            """获取部门资源"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/employee-info/mcp/query"
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            # 提取所有部门
                            departments = list(set([
                                item.get('department') 
                                for item in data 
                                if item.get('department')
                            ]))
                            return {
                                "contents": [
                                    {
                                        "uri": uri,
                                        "mimeType": "application/json",
                                        "text": str({"departments": departments, "total": len(departments)})
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
