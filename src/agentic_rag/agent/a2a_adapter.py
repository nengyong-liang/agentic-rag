"""
Google A2A (Agent-to-Agent) 协议适配器
实现 A2A 协议的 Server/Client，支持 Agent 间协作通信
与 MCP 互补：Agent 内部使用 MCP 调用工具，外部使用 A2A 进行 Agent 间协作
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging
import json
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import httpx
    _has_httpx = True
except ImportError:
    _has_httpx = False
    logger.warning("httpx 未安装，A2AClient 功能将不可用")

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    _has_fastapi = True
except ImportError:
    _has_fastapi = False
    logger.warning("FastAPI 未安装，setup_a2a_routes 功能将不可用")


@dataclass
class AgentCard:
    """Agent 能力卡片，描述 Agent 的基本信息和技能"""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: Dict[str, bool] = field(default_factory=lambda: {"streaming": True, "pushNotifications": False})
    authentication: Dict[str, Any] = field(default_factory=lambda: {"schemes": ["bearer"]})
    skills: List[Dict[str, Any]] = field(default_factory=list)


class A2ATaskState(Enum):
    """A2A 任务状态枚举"""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2ATask:
    """A2A 任务实体"""
    id: str
    status: A2ATaskState
    message: Optional[Dict] = None
    artifacts: List[Dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "status": self.status.value,
            "message": self.message,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class A2AMessage:
    """A2A 消息实体"""
    role: str
    parts: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "role": self.role,
            "parts": self.parts,
        }


class A2AServer:
    """A2A 协议服务端，处理来自其他 Agent 的请求"""

    def __init__(self, agent_card: AgentCard, handler: Optional[Callable[[str], str]] = None):
        self.agent_card = agent_card
        self.handler = handler
        self._tasks: Dict[str, A2ATask] = {}

    def handle_request(self, request: Dict) -> Dict:
        """处理传入的 A2A JSON-RPC 请求"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        method_map = {
            "tasks/send": self._handle_tasks_send,
            "tasks/get": self._handle_tasks_get,
            "tasks/cancel": self._handle_tasks_cancel,
        }

        handler_func = method_map.get(method)
        if handler_func is None:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"未知方法: {method}"},
                "id": request_id,
            }

        try:
            task = handler_func(params)
            return {
                "jsonrpc": "2.0",
                "result": task.to_dict(),
                "id": request_id,
            }
        except Exception as e:
            logger.error(f"A2A 请求处理失败: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request_id,
            }

    def _handle_tasks_send(self, params: Dict) -> A2ATask:
        """处理 tasks/send 方法，创建或更新任务"""
        task_id = params.get("id", str(uuid.uuid4()))
        message = params.get("message", {})
        user_text = ""
        parts = message.get("parts", [])
        for part in parts:
            if part.get("type") == "text":
                user_text += part.get("text", "")

        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = A2ATaskState.WORKING
            task.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            task = A2ATask(
                id=task_id,
                status=A2ATaskState.WORKING,
                message=message,
            )
            self._tasks[task_id] = task

        result_text = self._route_to_handler(user_text)

        task.artifacts.append({
            "parts": [{"type": "text", "text": result_text}],
        })
        task.status = A2ATaskState.COMPLETED
        task.updated_at = datetime.now(timezone.utc).isoformat()

        return task

    def _handle_tasks_get(self, params: Dict) -> A2ATask:
        """处理 tasks/get 方法，获取任务状态"""
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"任务不存在: {task_id}")
        return task

    def _handle_tasks_cancel(self, params: Dict) -> A2ATask:
        """处理 tasks/cancel 方法，取消任务"""
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"任务不存在: {task_id}")
        task.status = A2ATaskState.CANCELED
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    def get_agent_card(self) -> Dict:
        """返回 Agent 卡片信息"""
        return asdict(self.agent_card)

    def _route_to_handler(self, user_text: str) -> str:
        """将请求路由到处理函数"""
        if self.handler:
            return self.handler(user_text)

        try:
            from agentic_rag.agent.agent import rag_agent
            route = rag_agent.route_query(user_text)
            if route == "agent":
                result = rag_agent.run_agent(user_text)
                if result.get("success"):
                    data = result.get("data", {})
                    return data.get("response", json.dumps(data, ensure_ascii=False))
                return result.get("message", "Agent 处理失败")
            else:
                return "此请求需要 RAG 检索，请通过 RAG 通道处理"
        except Exception as e:
            logger.error(f"路由处理失败: {e}")
            return f"处理失败: {str(e)}"


class A2AClient:
    """A2A 协议客户端，向其他 Agent 发送请求"""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url

    def discover_agent(self, agent_url: str) -> AgentCard:
        """从远端 Agent 获取 Agent Card"""
        if not _has_httpx:
            raise RuntimeError("httpx 未安装，无法使用 A2AClient")

        well_known_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(well_known_url)
            response.raise_for_status()
            data = response.json()
            return AgentCard(
                name=data.get("name", ""),
                description=data.get("description", ""),
                url=data.get("url", agent_url),
                version=data.get("version", "1.0.0"),
                capabilities=data.get("capabilities", {"streaming": True, "pushNotifications": False}),
                authentication=data.get("authentication", {"schemes": ["bearer"]}),
                skills=data.get("skills", []),
            )

    def send_task(self, agent_url: str, task_text: str) -> A2ATask:
        """向远端 Agent 发送任务"""
        if not _has_httpx:
            raise RuntimeError("httpx 未安装，无法使用 A2AClient")

        a2a_url = f"{agent_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task_text}],
                },
            },
            "id": str(uuid.uuid4()),
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(a2a_url, json=payload)
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            raise RuntimeError(f"A2A 错误: {result['error'].get('message', '未知错误')}")

        task_data = result.get("result", {})
        return A2ATask(
            id=task_data.get("id", ""),
            status=A2ATaskState(task_data.get("status", "submitted")),
            message=task_data.get("message"),
            artifacts=task_data.get("artifacts", []),
            created_at=task_data.get("created_at", ""),
            updated_at=task_data.get("updated_at", ""),
        )

    def get_task(self, agent_url: str, task_id: str) -> A2ATask:
        """获取远端任务状态"""
        if not _has_httpx:
            raise RuntimeError("httpx 未安装，无法使用 A2AClient")

        a2a_url = f"{agent_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(a2a_url, json=payload)
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            raise RuntimeError(f"A2A 错误: {result['error'].get('message', '未知错误')}")

        task_data = result.get("result", {})
        return A2ATask(
            id=task_data.get("id", ""),
            status=A2ATaskState(task_data.get("status", "submitted")),
            message=task_data.get("message"),
            artifacts=task_data.get("artifacts", []),
            created_at=task_data.get("created_at", ""),
            updated_at=task_data.get("updated_at", ""),
        )

    def cancel_task(self, agent_url: str, task_id: str) -> A2ATask:
        """取消远端任务"""
        if not _has_httpx:
            raise RuntimeError("httpx 未安装，无法使用 A2AClient")

        a2a_url = f"{agent_url.rstrip('/')}/a2a"
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(a2a_url, json=payload)
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            raise RuntimeError(f"A2A 错误: {result['error'].get('message', '未知错误')}")

        task_data = result.get("result", {})
        return A2ATask(
            id=task_data.get("id", ""),
            status=A2ATaskState(task_data.get("status", "canceled")),
            message=task_data.get("message"),
            artifacts=task_data.get("artifacts", []),
            created_at=task_data.get("created_at", ""),
            updated_at=task_data.get("updated_at", ""),
        )


class MCPvsA2AComparison:
    """MCP 与 A2A 协议对比分析"""

    @staticmethod
    def compare() -> Dict[str, Any]:
        """返回 MCP 与 A2A 的对比表"""
        return {
            "dimensions": [
                {
                    "dimension": "提出者",
                    "mcp": "Anthropic 2024.11",
                    "a2a": "Google 2025.4",
                },
                {
                    "dimension": "核心定位",
                    "mcp": "LLM-工具连接",
                    "a2a": "Agent-Agent 协作",
                },
                {
                    "dimension": "通信方向",
                    "mcp": "单向（LLM→工具）",
                    "a2a": "双向（Agent↔Agent）",
                },
                {
                    "dimension": "抽象层级",
                    "mcp": "工具层（Tool Layer）",
                    "a2a": "任务层（Task Layer）",
                },
                {
                    "dimension": "发现机制",
                    "mcp": "配置文件（config-based）",
                    "a2a": "动态发现（Agent Card）",
                },
                {
                    "dimension": "状态管理",
                    "mcp": "无状态（Stateless）",
                    "a2a": "有状态（Task 生命周期）",
                },
                {
                    "dimension": "典型用例",
                    "mcp": "工具调用（Tool Calling）",
                    "a2a": "多 Agent 协作",
                },
                {
                    "dimension": "协议格式",
                    "mcp": "JSON-RPC / stdio / SSE",
                    "a2a": "JSON-RPC / HTTP / SSE",
                },
                {
                    "dimension": "生态成熟度",
                    "mcp": "成熟（Mature）",
                    "a2a": "新兴（Emerging）",
                },
            ],
            "insight": "A2A 与 MCP 是互补关系而非竞争关系：Agent 内部使用 MCP 调用工具，外部使用 A2A 进行 Agent 间协作",
        }


def _get_default_skills() -> List[Dict]:
    """从 MCP Bridge 动态生成 Agent Card 技能列表"""
    try:
        from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
        tools_dict = mcp_tool_bridge.get_tools_dict()
        return [
            {"name": info["name"], "description": info["description"]}
            for info in tools_dict.values()
        ]
    except Exception:
        return [
            {"name": "knowledge_qa", "description": "知识库问答"},
            {"name": "inventory_query", "description": "配件库存查询"},
            {"name": "repair_order", "description": "维修工单管理"},
            {"name": "employee_info", "description": "员工信息核验"},
        ]

DEFAULT_AGENT_CARD = AgentCard(
    name="agentic-rag RAG Agent",
    description="企业级智能知识库问答系统 Agent",
    url="http://localhost:8001/a2a",
    version="1.0.0",
    skills=_get_default_skills(),
    capabilities={"streaming": True, "pushNotifications": False},
    authentication={"schemes": ["bearer"]},
)


def setup_a2a_routes(app: FastAPI, a2a_server: A2AServer):
    """将 A2A 端点注册到 FastAPI 应用"""
    if not _has_fastapi:
        logger.error("FastAPI 未安装，无法注册 A2A 路由")
        return

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        """返回 Agent Card"""
        return a2a_server.get_agent_card()

    @app.post("/a2a")
    async def handle_a2a_request(request: Request):
        """处理 A2A JSON-RPC 请求"""
        body = await request.json()
        result = a2a_server.handle_request(body)
        return JSONResponse(content=result)
