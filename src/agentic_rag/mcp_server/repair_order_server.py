"""
维修工单生成 MCP Server
提供标准的 MCP 协议接口，支持维修工单的创建和查询
"""

import requests
import httpx
from typing import Dict, Any, Optional
from loguru import logger
from agentic_rag.mcp_server.protocol import MCPServer, MCPErrorCode
from agentic_rag.mcp_server.config import settings

# 配置（与 MCP Server 共用仓库根 .env）
JAVA_API_BASE_URL = settings.JAVA_API_URL


class RepairOrderMCPServer(MCPServer):
    """维修工单生成 MCP Server"""
    
    def __init__(self):
        super().__init__(name="auto-repair-mcp", version="1.0.0")
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        @self.register_tool(
            name="repair.create",
            description="创建汽车维修工单，包括客户信息、车辆信息、维修类型、故障描述等",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "客户姓名"},
                    "customer_phone": {"type": "string", "description": "客户电话"},
                    "vehicle_model": {"type": "string", "description": "车型"},
                    "vehicle_vin": {"type": "string", "description": "车架号（可选）"},
                    "repair_type": {"type": "string", "description": "维修类型（保养/维修/索赔）", "enum": ["保养", "维修", "索赔"]},
                    "problem_description": {"type": "string", "description": "故障描述"},
                    "technician": {"type": "string", "description": "维修技师（可选）"}
                },
                "required": ["customer_name", "customer_phone", "vehicle_model", "repair_type", "problem_description"]
            }
        )
        async def create_repair_order(
            customer_name: str,
            customer_phone: str,
            vehicle_model: str,
            repair_type: str,
            problem_description: str,
            vehicle_vin: Optional[str] = None,
            technician: Optional[str] = None
        ) -> Dict[str, Any]:
            """创建维修工单"""
            try:
                # 构建工单数据
                order_data = {
                    "customerName": customer_name,
                    "customerPhone": customer_phone,
                    "vehicleModel": vehicle_model,
                    "vehicleVin": vehicle_vin,
                    "repairType": repair_type,
                    "problemDescription": problem_description,
                    "technician": technician,
                    "status": "pending"
                }
                
                # 调用 Java API
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"{JAVA_API_BASE_URL}/repair-order/mcp/create",
                        json=order_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "data": data,
                            "message": f"工单创建成功，工单编号：{data.get('orderNumber')}"
                        }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"工单创建失败：{response.text}"
                        }
            except Exception as e:
                logger.error(f"创建工单失败：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"创建失败：{str(e)}"
                }
        
        @self.register_tool(
            name="repair.query",
            description="查询维修工单信息，支持按工单编号、客户姓名、状态等条件查询",
            input_schema={
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "工单编号"},
                    "customer_name": {"type": "string", "description": "客户姓名"},
                    "status": {"type": "string", "description": "工单状态（pending/processing/completed/cancelled）", "enum": ["pending", "processing", "completed", "cancelled"]},
                    "repair_type": {"type": "string", "description": "维修类型", "enum": ["保养", "维修", "索赔"]}
                },
                "required": []
            }
        )
        async def query_repair_order(
            order_number: Optional[str] = None,
            customer_name: Optional[str] = None,
            status: Optional[str] = None,
            repair_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """查询维修工单"""
            try:
                # 构建查询参数
                params = {}
                if status:
                    params['status'] = status
                if customer_name:
                    params['customerName'] = customer_name
                if repair_type:
                    params['repairType'] = repair_type
                
                # 调用 Java API
                async with httpx.AsyncClient(timeout=5.0) as client:
                    if order_number:
                        # 按工单编号查询
                        response = await client.get(
                            f"{JAVA_API_BASE_URL}/repair-order/mcp/{order_number}"
                        )
                    else:
                        # 按条件查询列表
                        response = await client.get(
                            f"{JAVA_API_BASE_URL}/repair-order/mcp/query",
                            params=params
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            return {
                                "success": True,
                                "data": {
                                    "items": data,
                                    "total": len(data)
                                },
                                "message": f"查询成功，找到 {len(data)} 个工单"
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
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"查询失败：{response.text}"
                        }
            except Exception as e:
                logger.error(f"查询工单失败：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"查询失败：{str(e)}"
                }
        
        @self.register_tool(
            name="repair.update_status",
            description="更新维修工单状态",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "工单 ID"},
                    "status": {"type": "string", "description": "新状态", "enum": ["pending", "processing", "completed", "cancelled"]}
                },
                "required": ["order_id", "status"]
            }
        )
        async def update_repair_status(order_id: int, status: str) -> Dict[str, Any]:
            """更新工单状态"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.put(
                        f"{JAVA_API_BASE_URL}/repair-order/mcp/status",
                        params={"id": order_id, "status": status}
                    )
                    
                    if response.status_code == 200:
                        return {
                            "success": True,
                            "data": {"order_id": order_id, "status": status},
                            "message": "工单状态更新成功"
                        }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": f"状态更新失败：{response.text}"
                        }
            except Exception as e:
                logger.error(f"更新工单状态失败：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"更新失败：{str(e)}"
                }
    
    def _register_resources(self):
        @self.register_resource(
            uri="repair://orders/{order_number}",
            name="维修工单资源",
            description="通过工单编号访问维修工单详情"
        )
        async def get_order_resource(order_number: str) -> Dict[str, Any]:
            """获取工单资源"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/repair-order/mcp/query",
                        params={"order_number": order_number}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "data": data,
                            "message": "获取成功"
                        }
                    else:
                        return {
                            "success": False,
                            "data": None,
                            "message": "获取失败"
                        }
            except Exception as e:
                logger.error(f"获取工单资源失败：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"获取失败：{str(e)}"
                }
        
        @self.register_resource(
            uri="repair://statistics/daily",
            name="工单统计资源",
            description="获取每日工单统计信息"
        )
        async def get_daily_statistics() -> Dict[str, Any]:
            """获取每日工单统计"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # 查询所有工单
                    response = await client.get(
                        f"{JAVA_API_BASE_URL}/repair-order/mcp/query"
                    )
                    
                    if response.status_code == 200:
                        orders = response.json()
                        if isinstance(orders, list):
                            # 统计数据
                            stats = {
                                "total": len(orders),
                                "pending": sum(1 for o in orders if o.get('status') == 'pending'),
                                "processing": sum(1 for o in orders if o.get('status') == 'processing'),
                                "completed": sum(1 for o in orders if o.get('status') == 'completed'),
                                "cancelled": sum(1 for o in orders if o.get('status') == 'cancelled'),
                                "by_type": {}
                            }
                            
                            # 按维修类型统计
                            for order in orders:
                                repair_type = order.get('repairType', '未知')
                                if repair_type not in stats['by_type']:
                                    stats['by_type'][repair_type] = 0
                                stats['by_type'][repair_type] += 1
                            
                            return {
                                "success": True,
                                "data": stats,
                                "message": "统计成功"
                            }
                    
                    return {
                        "success": False,
                        "data": None,
                        "message": "获取统计失败"
                    }
            except Exception as e:
                logger.error(f"获取工单统计失败：{e}")
                return {
                    "success": False,
                    "data": None,
                    "message": f"获取失败：{str(e)}"
                }
