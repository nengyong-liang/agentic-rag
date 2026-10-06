"""
MCP Bridge 模块
将标准 MCP Server 的工具桥接到 Agent 的 Function Calling

职责：
- 提供与 mcp_server/ 同一批工具（10 个，4 个子 Server）的 Function Calling schema
- 执行侧直连业务后端的工具接口（`JAVA_API_URL` + `/xxx/mcp/...`），不经过 MCP HTTP 端点
- 调用前置：沙箱授权（参数白名单 / 配额）+ HITL 风险审批
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List

import httpx

from agentic_rag.config import settings

logger = logging.getLogger(__name__)

JAVA_API_BASE_URL = settings.JAVA_API_URL


class MCPToolBridge:
    """MCP 工具桥接器：将 MCP Server 工具桥接到 Agent Function Calling

    安全集成：
    - 沙箱授权：ToolAuthorizer 校验参数白名单 + 配额
    - HITL 审批：ApprovalNode 判断风险等级，低风险自动通过，中高风险需人工审批
    """

    def __init__(self):
        self._tools_schema_cache = None
        self._tool_authorizer = None    # 延迟加载，避免循环依赖
        self._approval_node = None      # 延迟加载，避免循环依赖
        self._tool_map = {
            "inventory_query": {
                "name": "inventory_query",
                "description": "查询汽车配件库存信息，包括库存数量、仓库位置、价格、供应商等。可通过配件编号、车型或类别查询",
                "parameters": {
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
                },
                "mcp_tool": "inventory.query",
                "handler": self._handle_inventory_query
            },
            "inventory_detail": {
                "name": "inventory_detail",
                "description": "通过配件 ID 获取详细信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "配件 ID"
                        }
                    },
                    "required": ["id"]
                },
                "mcp_tool": "inventory.detail",
                "handler": self._handle_inventory_detail
            },
            "inventory_low_stock": {
                "name": "inventory_low_stock",
                "description": "查询库存预警，返回库存数量低于最低库存的配件列表",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "mcp_tool": "inventory.low_stock_alert",
                "handler": self._handle_inventory_low_stock
            },
            "repair_order_create": {
                "name": "repair_order_create",
                "description": "创建维修工单",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vehicle_model": {
                            "type": "string",
                            "description": "车辆型号"
                        },
                        "repair_type": {
                            "type": "string",
                            "description": "维修类型，如 保养、事故维修、故障维修"
                        },
                        "description": {
                            "type": "string",
                            "description": "维修描述"
                        }
                    },
                    "required": ["vehicle_model", "repair_type", "description"]
                },
                "mcp_tool": "repair_order.create",
                "handler": self._handle_repair_order_create
            },
            "repair_order_query": {
                "name": "repair_order_query",
                "description": "查询维修工单",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "工单 ID"
                        },
                        "status": {
                            "type": "string",
                            "description": "工单状态，如 待处理、进行中、已完成"
                        }
                    },
                    "required": []
                },
                "mcp_tool": "repair_order.query",
                "handler": self._handle_repair_order_query
            },
            "repair_order_update": {
                "name": "repair_order_update",
                "description": "更新维修工单状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "工单 ID"
                        },
                        "status": {
                            "type": "string",
                            "description": "新状态，如 进行中、已完成"
                        }
                    },
                    "required": ["id", "status"]
                },
                "mcp_tool": "repair_order.update_status",
                "handler": self._handle_repair_order_update
            },
            "employee_verify": {
                "name": "employee_verify",
                "description": "核验员工信息，通过员工编号查询员工基本信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_number": {
                            "type": "string",
                            "description": "员工编号"
                        }
                    },
                    "required": ["employee_number"]
                },
                "mcp_tool": "employee.verify",
                "handler": self._handle_employee_verify
            },
            "employee_query": {
                "name": "employee_query",
                "description": "按条件查询员工信息，支持按部门、职位等筛选",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {
                            "type": "string",
                            "description": "部门名称"
                        },
                        "position": {
                            "type": "string",
                            "description": "职位"
                        }
                    },
                    "required": []
                },
                "mcp_tool": "employee.query",
                "handler": self._handle_employee_query
            },
            "regulation_search": {
                "name": "regulation_search",
                "description": "按关键词搜索制度条款",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["keyword"]
                },
                "mcp_tool": "regulation.search",
                "handler": self._handle_regulation_search
            },
            "regulation_query": {
                "name": "regulation_query",
                "description": "按类别查询制度条款",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "制度类别，如 人事管理、财务管理、行政管理"
                        }
                    },
                    "required": ["category"]
                },
                "mcp_tool": "regulation.query_by_category",
                "handler": self._handle_regulation_query
            }
        }

    def _get_tool_authorizer(self):
        """延迟加载 ToolAuthorizer，避免循环依赖"""
        if self._tool_authorizer is None:
            from agent_harness.sandbox.tool_authorization import ToolAuthorizer
            self._tool_authorizer = ToolAuthorizer()
        return self._tool_authorizer

    def _get_approval_node(self):
        """延迟加载 ApprovalNode，避免循环依赖"""
        if self._approval_node is None:
            from agent_harness.hitl.approval import approval_node
            self._approval_node = approval_node
        return self._approval_node

    def _pre_check_tool(self, tool_name: str, tool_args: Dict, session_id: str = "default") -> Dict:
        """执行工具前的安全检查：沙箱授权 + HITL 审批

        策略：
        - 沙箱授权（ToolAuthorizer）：参数白名单 + 配额检查
        - HITL 审批（ApprovalNode）：风险等级判断
          - LOW 风险（只读查询）→ 自动通过，直接执行
          - MEDIUM/HIGH 风险（写操作）→ 创建审批请求，需人工决策

        Returns:
            Dict 含 approved/final_args/message 字段。
            approved=True 时可直接执行工具。
        """
        # 1. 沙箱授权
        authorizer = self._get_tool_authorizer()
        auth_result = authorizer.authorize(tool_name, tool_args, session_id)
        if not auth_result.allowed:
            logger.warning(
                f"[MCP Bridge] 沙箱拒绝工具调用：{tool_name}，"
                f"reason={auth_result.reason}"
            )
            return {
                "approved": False,
                "final_args": tool_args,
                "message": f"沙箱拒绝：{auth_result.reason}",
                "requires_approval": False,
            }

        # 2. HITL 审批风险判断
        approval_node = self._get_approval_node()
        need_approval, reason = approval_node.should_require_approval(tool_name, tool_args)

        if not need_approval:
            # LOW 风险工具：自动通过，直接执行
            return {
                "approved": True,
                "final_args": tool_args,
                "message": reason,
                "requires_approval": False,
            }

        # MEDIUM/HIGH 风险工具：创建审批请求
        request = approval_node.request_approval(
            tool_name=tool_name,
            tool_args=tool_args,
            session_id=session_id,
            requester="mcp_bridge",
            prompt_to_user=f"工具 {tool_name} 需要审批：{reason}",
        )

        # 尝试自动决策（LOW 自动过，MEDIUM/HIGH 不会自动决策）
        auto_decision = approval_node.auto_decide(request)
        if auto_decision.decision.value == "approve":
            # 自动通过（仅 LOW 风险）
            should_exec, final_args, _ = approval_node.apply_decision(
                auto_decision, tool_args
            )
            logger.info(
                f"[MCP Bridge] HITL 自动通过：{tool_name} "
                f"(request_id={request.request_id})"
            )
            return {
                "approved": should_exec,
                "final_args": final_args,
                "message": f"HITL 自动通过：{reason}",
                "requires_approval": False,
                "approval_request_id": request.request_id,
            }

        # MEDIUM/HIGH 风险 → 暂不执行，返回 pending 供人工审批
        logger.warning(
            f"[MCP Bridge] HITL 待审批：{tool_name} "
            f"(request_id={request.request_id})"
        )
        return {
            "approved": False,
            "final_args": tool_args,
            "message": f"工具 {tool_name} 需人工审批后才能执行",
            "requires_approval": True,
            "approval_request_id": request.request_id,
        }

    def get_tools_schema(self) -> List[Dict]:
        """获取 Function Calling 格式的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"]
                }
            }
            for tool_info in self._tool_map.values()
        ]

    def get_tools_dict(self) -> Dict:
        """获取工具注册表（兼容旧接口）"""
        return {
            name: {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self._tool_map.items()
        }

    def execute_tool(self, tool_name: str, tool_args: Dict, session_id: str = "default") -> Dict[str, Any]:
        """执行工具调用（含沙箱授权 + HITL 审批)

        低风险只读工具自动放行；中高风险写工具需人工审批。
        """
        if tool_name not in self._tool_map:
            return {
                "success": False,
                "data": None,
                "message": f"未知工具：{tool_name}"
            }

        tool_info = self._tool_map[tool_name]
        handler = tool_info["handler"]

        # 安全检查：沙箱授权 + HITL 审批
        check = self._pre_check_tool(tool_name, tool_args, session_id)
        if not check["approved"]:
            result = {
                "success": False,
                "data": None,
                "message": check["message"],
            }
            if check.get("requires_approval"):
                result["requires_approval"] = True
                result["approval_request_id"] = check.get("approval_request_id")
            return result

        tool_args = check["final_args"] or tool_args

        try:
            logger.info(f"[MCP Bridge] 执行工具：{tool_name}，参数：{tool_args}")
            result = handler(**tool_args)
            logger.info(f"[MCP Bridge] 工具执行完成：{tool_name}")
            return result
        except Exception as e:
            logger.error(f"[MCP Bridge] 工具执行失败：{tool_name}，错误：{e}")
            return {
                "success": False,
                "data": None,
                "message": f"工具执行失败：{str(e)}"
            }

    def _call_java_api_sync(self, method: str, path: str, params: Dict = None, json_data: Dict = None) -> Dict:
        """同步调用 Java API（通过 MCP 免认证接口）"""
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"{JAVA_API_BASE_URL}{path}"
                if method == "GET":
                    response = client.get(url, params=params)
                elif method == "POST":
                    response = client.post(url, json=json_data)
                elif method == "PUT":
                    response = client.put(url, json=json_data)
                else:
                    return {"success": False, "data": None, "message": f"不支持的 HTTP 方法：{method}"}

                if response.status_code == 200:
                    data = response.json()
                    return {"success": True, "data": data, "message": "调用成功"}
                elif response.status_code == 404:
                    return {"success": False, "data": None, "message": "资源不存在"}
                else:
                    return {"success": False, "data": None, "message": f"HTTP {response.status_code}"}
        except httpx.TimeoutException:
            return {"success": False, "data": None, "message": "请求超时"}
        except httpx.ConnectError:
            return {"success": False, "data": None, "message": "无法连接到后端服务"}
        except Exception as e:
            return {"success": False, "data": None, "message": str(e)}

    async def _call_java_api(self, method: str, path: str, params: Dict = None, json_data: Dict = None) -> Dict:
        """异步调用 Java API（通过 MCP 免认证接口），避免阻塞事件循环"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{JAVA_API_BASE_URL}{path}"
                if method == "GET":
                    response = await client.get(url, params=params)
                elif method == "POST":
                    response = await client.post(url, json=json_data)
                elif method == "PUT":
                    response = await client.put(url, json=json_data)
                else:
                    return {"success": False, "data": None, "message": f"不支持的 HTTP 方法：{method}"}

                if response.status_code == 200:
                    data = response.json()
                    return {"success": True, "data": data, "message": "调用成功"}
                elif response.status_code == 404:
                    return {"success": False, "data": None, "message": "资源不存在"}
                else:
                    return {"success": False, "data": None, "message": f"HTTP {response.status_code}"}
        except httpx.TimeoutException:
            return {"success": False, "data": None, "message": "请求超时"}
        except httpx.ConnectError:
            return {"success": False, "data": None, "message": "无法连接到后端服务"}
        except Exception as e:
            return {"success": False, "data": None, "message": str(e)}

    def _handle_inventory_query(self, part_number: str = None, vehicle_model: str = None, category: str = None) -> Dict:
        params = {}
        if part_number:
            params['partNumber'] = part_number
        if vehicle_model:
            params['vehicleModel'] = vehicle_model
        if category:
            params['category'] = category
        return self._call_java_api_sync("GET", "/part-inventory/mcp/query", params=params)

    def _handle_inventory_detail(self, id: int) -> Dict:
        return self._call_java_api_sync("GET", f"/part-inventory/mcp/{id}")

    def _handle_inventory_low_stock(self) -> Dict:
        return self._call_java_api_sync("GET", "/part-inventory/mcp/alert/low-stock")

    def _handle_repair_order_create(self, vehicle_model: str, repair_type: str, description: str) -> Dict:
        return self._call_java_api_sync("POST", "/repair-order/mcp/create", json_data={
            "vehicleModel": vehicle_model,
            "repairType": repair_type,
            "problemDescription": description
        })

    def _handle_repair_order_query(self, id: int = None, status: str = None) -> Dict:
        if id:
            return self._call_java_api_sync("GET", f"/repair-order/mcp/{id}")
        params = {}
        if status:
            params['status'] = status
        return self._call_java_api_sync("GET", "/repair-order/mcp/query", params=params)

    def _handle_repair_order_update(self, id: int, status: str) -> Dict:
        return self._call_java_api_sync("PUT", "/repair-order/mcp/status", json_data={
            "id": id,
            "status": status
        })

    def _handle_employee_verify(self, employee_number: str) -> Dict:
        return self._call_java_api_sync("GET", "/employee-info/mcp/verify", params={
            "employeeNumber": employee_number
        })

    def _handle_employee_query(self, department: str = None, position: str = None) -> Dict:
        params = {}
        if department:
            params['department'] = department
        if position:
            params['position'] = position
        return self._call_java_api_sync("GET", "/employee-info/mcp/query", params=params)

    def _handle_regulation_search(self, keyword: str) -> Dict:
        return self._call_java_api_sync("GET", "/regulation-clause/mcp/search", params={
            "keyword": keyword
        })

    def _handle_regulation_query(self, category: str) -> Dict:
        return self._call_java_api_sync("GET", "/regulation-clause/mcp/query", params={
            "category": category
        })


    async def execute_tool_async(self, tool_name: str, tool_args: Dict, session_id: str = "default") -> Dict[str, Any]:
        """异步执行工具调用（适用于 FastAPI 异步端点，避免阻塞事件循环）

        含沙箱授权 + HITL 审批（与 execute_tool 相同策略）。
        """
        if tool_name not in self._tool_map:
            return {
                "success": False,
                "data": None,
                "message": f"未知工具：{tool_name}"
            }

        tool_info = self._tool_map[tool_name]

        # 安全检查：沙箱授权 + HITL 审批
        check = self._pre_check_tool(tool_name, tool_args, session_id)
        if not check["approved"]:
            result = {
                "success": False,
                "data": None,
                "message": check["message"],
            }
            if check.get("requires_approval"):
                result["requires_approval"] = True
                result["approval_request_id"] = check.get("approval_request_id")
            return result

        tool_args = check["final_args"] or tool_args

        try:
            logger.info(f"[MCP Bridge] 异步执行工具：{tool_name}，参数：{tool_args}")
            # 异步调用 Java API
            mcp_tool = tool_info["mcp_tool"]
            result = await self._call_java_api_by_tool(mcp_tool, tool_args)
            logger.info(f"[MCP Bridge] 异步工具执行完成：{tool_name}")
            return result
        except Exception as e:
            logger.error(f"[MCP Bridge] 异步工具执行失败：{tool_name}，错误：{e}")
            return {
                "success": False,
                "data": None,
                "message": f"工具执行失败：{str(e)}"
            }

    async def _call_java_api_by_tool(self, mcp_tool: str, tool_args: Dict) -> Dict:
        """根据 MCP 工具名异步调用对应的 Java API"""
        # 配件库存相关
        if mcp_tool == "inventory.query":
            params = {}
            if tool_args.get('part_number'): params['partNumber'] = tool_args['part_number']
            if tool_args.get('vehicle_model'): params['vehicleModel'] = tool_args['vehicle_model']
            if tool_args.get('category'): params['category'] = tool_args['category']
            return await self._call_java_api("GET", "/part-inventory/mcp/query", params=params)
        elif mcp_tool == "inventory.detail":
            return await self._call_java_api("GET", f"/part-inventory/mcp/{tool_args['id']}")
        elif mcp_tool == "inventory.low_stock_alert":
            return await self._call_java_api("GET", "/part-inventory/mcp/alert/low-stock")
        # 维修工单相关
        elif mcp_tool == "repair_order.create":
            return await self._call_java_api("POST", "/repair-order/mcp/create", json_data={
                "vehicleModel": tool_args.get('vehicle_model', ''),
                "repairType": tool_args.get('repair_type', ''),
                "problemDescription": tool_args.get('description', '')
            })
        elif mcp_tool == "repair_order.query":
            if tool_args.get('id'):
                return await self._call_java_api("GET", f"/repair-order/mcp/{tool_args['id']}")
            params = {}
            if tool_args.get('status'): params['status'] = tool_args['status']
            return await self._call_java_api("GET", "/repair-order/mcp/query", params=params)
        elif mcp_tool == "repair_order.update_status":
            return await self._call_java_api("PUT", "/repair-order/mcp/status", json_data={
                "id": tool_args.get('id', 0),
                "status": tool_args.get('status', '')
            })
        # 员工信息相关
        elif mcp_tool == "employee.verify":
            return await self._call_java_api("GET", "/employee-info/mcp/verify", params={
                "employeeNumber": tool_args.get('employee_number', '')
            })
        elif mcp_tool == "employee.query":
            params = {}
            if tool_args.get('department'): params['department'] = tool_args['department']
            if tool_args.get('position'): params['position'] = tool_args['position']
            return await self._call_java_api("GET", "/employee-info/mcp/query", params=params)
        # 制度条款相关
        elif mcp_tool == "regulation.search":
            return await self._call_java_api("GET", "/regulation-clause/mcp/search", params={
                "keyword": tool_args.get('keyword', '')
            })
        elif mcp_tool == "regulation.query_by_category":
            return await self._call_java_api("GET", "/regulation-clause/mcp/query", params={
                "category": tool_args.get('category', '')
            })
        else:
            return {"success": False, "data": None, "message": f"未知的 MCP 工具：{mcp_tool}"}


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    mcp_tool_bridge = MCPToolBridge()
else:
    mcp_tool_bridge = None
