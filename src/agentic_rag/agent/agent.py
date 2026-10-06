"""
RAG Agent 核心类
基于 MCP Bridge（Function Calling 工具单点）实现；工具问答经 Harness 统一执行层。

升级要点：
1. 使用 MCP Bridge 统一调用所有 10 个工具（替代旧 tools.py 的 2 个工具）
2. 扩展 Agent 关键词覆盖员工信息/制度条款
3. 工具执行收敛到 Harness（优先）/ 手动循环（安全回退）双路径，
   均以豆包 Function Calling 为核心调用方式
"""

from typing import Dict, Any, List, Optional
from agent_harness.providers.doubao import hybrid_llm, DoubaoLLMService
from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
import json
import logging
import os

logger = logging.getLogger(__name__)


class SimpleConversationMemory:
    """简单的对话记忆实现（替代 langchain.memory）"""

    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.chat_history.append({"role": "user", "content": message})

    def add_ai_message(self, message: str):
        self.chat_history.append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        return self.chat_history

    def clear(self):
        self.chat_history = []


has_langchain = False  # langchain.memory 已被弃用，使用 SimpleConversationMemory


class RAGAgent:
    """RAG Agent 核心类（支持 Function Calling + MCP Bridge）

    记忆系统：
    - 优先复用全局 MemoryManager（state.memory_manager，三层记忆）
    - 无 MemoryManager 时降级为 SimpleConversationMemory（内存 List）
    - run_agent 内部自动检索相关记忆注入 system prompt，结束后记录经验
    """

    def __init__(self):
        self.llm = DoubaoLLMService() # 初始化豆包 LLM 服务
        self.mcp_bridge = mcp_tool_bridge # 初始化 MCP Bridge
        self.memory = SimpleConversationMemory() # 兜底：无 MemoryManager 时使用
        logger.info("Agent 初始化完成（MCP Bridge 模式，10 个工具）")

    def _get_memory_manager(self):
        """延迟获取全局 MemoryManager（避免循环依赖与初始化顺序问题）

        Returns:
            MemoryManager 或 None（不可用时返回 None，调用方降级处理）
        """
        try:
            from agentic_rag.api.state import state
            return getattr(state, "memory_manager", None)
        except Exception:
            return None

    def _build_tools_schema(self) -> List[Dict]:
        """构建 Function Calling 工具定义（通过 MCP Bridge 获取所有工具）"""
        return self.mcp_bridge.get_tools_schema()

    def _execute_tool(self, tool_name: str, tool_args: Dict, session_id: str = "default") -> Dict[str, Any]:
        """执行工具调用（通过 MCP Bridge，内部含沙箱授权 + HITL 审批）

        真实执行链为：agent → mcp_bridge.execute_tool → 沙箱授权(ToolAuthorizer)
        + HITL 审批(ApprovalNode)。高风险写工具会在 bridge 层创建人工审批请求
        （request_approval），返回 pending 等待批准，避免高风险调用绕过批准。
        """
        return self.mcp_bridge.execute_tool(tool_name, tool_args, session_id)

    def route_query(self, query: str) -> str:
        """路由判断：纯知识→RAG，需操作→Agent

        这是三层路由中的 Layer 1（关键词匹配），仅在 Agent 内部
        （A2A 协议 / run_agent）被调用。Java 端的 RouteServiceImpl
        是真正的外部路由入口。
        """
        # 输入：用户查询
        # 输出：路由结果（RAG 或 Agent）
        # 延迟导入避免循环依赖
        from agentic_rag.agent.agentic_rag import SupervisorAgent #SupervisorAgent用于判断是否需要调用工具调用
        agent_keywords = SupervisorAgent.ROUTE_KEYWORDS["react"]
        has_agent_signal = any(kw in query for kw in agent_keywords)
        if has_agent_signal:
            return "agent"
        return "rag"

    def _get_harness(self):
        """获取 Harness 主循环：优先复用全局 state.harness_loop，否则自建。

        自建仅用于 Harness 尚未被 main.py 装配的 standalone 上下文
        （如 run.py 直接调 run_agent），保证主链路仍统一走 Harness。
        """
        try:
            from agentic_rag.api.state import state
            loop = getattr(state, "harness_loop", None)
            if loop is not None:
                # 确保 TaskGoal 自动推导（D1-D6）已启用，复用既有审计器。
                # 用安全 getattr 链，避免 loop 无 .config（如测试桩）时异常。
                try:
                    if not getattr(getattr(loop, "config", None),
                                   "auto_compile_goal", False):
                        from agent_harness.core.task_compiler import build_production_compiler
                        loop.config.auto_compile_goal = True
                        loop.config.task_compiler = build_production_compiler(
                            auditor=getattr(loop, "_auditor", None))
                    # 启用 HITL 审批闭环（安全网关 requires_approval → ApprovalNode）
                    loop.config.enable_hitl_approval = True
                except Exception as e:
                    logger.warning(f"[RAGAgent] TaskCompiler 接入失败，跳过: {e}")
                return loop
        except Exception:
            pass
        try:
            from agent_harness import (
                HarnessLoop, HarnessConfig, get_default_provider,
                get_default_projector, get_default_breaker, get_default_validator,
                get_default_governor, get_default_gateway, get_default_auditor,
            )
            from agent_harness.core.task_compiler import build_production_compiler
            loop = HarnessLoop(
                config=HarnessConfig(
                    max_iterations=12,
                    enable_safety_gateway=True,
                    auto_compile_goal=True,
                    task_compiler=build_production_compiler(
                        auditor=get_default_auditor()),
                    enable_hitl_approval=True,
                    domain="general",
                    query_type="simple",
                ),
                model_provider=get_default_provider("doubao"),
            )
            loop.set_tool_executor(self.mcp_bridge.execute_tool)
            return loop
        except Exception as e:
            logger.warning(f"[RAGAgent] Harness 初始化失败，将回退手动循环: {e}")
            return None

    def _build_system_prompt(self, tools_schema: List[Dict]) -> str:
        """构建 Agent 系统提示（含工具描述；不掺入 LangGraph 表述——本实现基于
        Function Calling + MCP Bridge，D10 修正）"""
        tool_descriptions = "\n".join([
            f"{i + 1}. {t['function']['name']} - {t['function']['description']}"
            for i, t in enumerate(tools_schema)
        ])
        return f"""你是一个智能企业助手，可以调用工具来帮助用户查询信息或执行任务。

        你有以下工具可用：
        {tool_descriptions}

        当用户需要查询库存、生成工单、核验员工、搜索制度时，请调用相应的工具。
        如果不需要调用工具，直接回答用户的问题。
        回答时要专业、准确，并基于工具返回的数据进行总结。"""

    def _load_memory_context(self, query: str, memory_manager) -> str:
        """检索相关记忆并组装上下文（失败不阻塞）"""
        memory_context = ""
        if memory_manager is not None:
            try:
                memory_context = memory_manager.get_context_for_prompt(query, max_tokens=500) or ""
            except Exception as e:
                logger.warning(f"[RAGAgent] 记忆上下文生成失败，降级为无记忆：{e}")
                memory_context = ""
        return memory_context

    def _run_via_harness(self, harness, query: str, system_prompt: str,
                         tools_schema: List[Dict], memory_context: str,
                         chat_history, domain: str = "general",
                         query_type: str = "simple") -> Optional[Dict[str, Any]]:
        """委托 Harness 统一执行层，并把 HarnessResult 映射回兼容 dict。

        返回 None 表示 Harness 产出不可用（空答 / 模型不可用 / 异常），
        由上层安全回退到 proven 的手动循环，避免工具问答回归。
        """
        if memory_context:
            system_prompt = f"{system_prompt}\n\n【相关记忆】\n{memory_context}"
        try:
            from agent_harness import ContextPhase
            result = harness.run(
                session_id=f"agent-{query[:20]}",
                system_prompt=system_prompt,
                query=query,
                conversation_history=chat_history or [],
                tools_schema=tools_schema,
                phase=ContextPhase.EXECUTING,
                domain=domain,
                query_type=query_type,
            )
        except Exception as e:
            logger.error(f"[RAGAgent] Harness 执行异常: {e}")
            return None

        answer = (result.answer or "").strip()
        unusable = (
            not answer
            or answer == "系统繁忙，请稍后重试"
            or result.finish_reason == "model_unavailable"
        )
        if unusable:
            return None

        # 检查工具结果是否命中 HITL pending（高风险写工具需人工审批）
        requires_approval = False
        approval_request_id = None
        tool_used = None
        for tr in getattr(result, "tool_results", []) or []:
            if tr.get("requires_approval") or tr.get("status") in (
                "pending_approval", "approval_rejected", "approval_responded"):
                requires_approval = True
                approval_request_id = tr.get("approval_request_id") or approval_request_id
            tool_used = tr.get("tool_name") or tool_used
        # 兜底：HarnessResult 直接暴露的审批态
        if getattr(result, "requires_approval", False):
            requires_approval = True
            approval_request_id = getattr(result, "approval_request_id", None) or approval_request_id

        if requires_approval:
            return {
                "success": False,
                "data": {
                    "response": f"工具 {tool_used} 需要人工审批后才能执行，"
                                f"审批请求 ID：{approval_request_id or '未知'}",
                    "type": "agent",
                    "requires_approval": True,
                    "approval_request_id": approval_request_id,
                    "tool_used": tool_used,
                },
                "message": "需人工审批",
            }

        return {
            "success": True,
            "data": {
                "response": answer,
                "type": "agent",
                "tool_used": tool_used,
                "tool_result": (getattr(result, "tool_results", []) or [None])[0],
                "harness": {
                    "finish_reason": result.finish_reason,
                    "iteration_count": result.iteration_count,
                    "tool_call_count": result.tool_call_count,
                    "feedback_count": getattr(result, "feedback_count", 0),
                    "events_count": result.events_count,
                },
            },
            "message": "Agent 处理成功（经由 Harness 统一执行层）",
        }

    def run_agent(self, query: str, chat_history: List[Dict[str, str]] = None,
                  domain: str = "general", query_type: str = "simple") -> Dict[str, Any]:
        """运行 Agent 处理需要工具调用的查询。

        统一执行层（Harness）优先：Harness 可用且产出可用答案时走 Harness
        （工具执行仍收敛到 mcp_bridge 单点，含沙箱授权 + HITL 审批）。
        若 Harness 不可用或产出不可用（空答 / 模型不可用），回退到 proven 的
        手动循环，保证既有工具问答不回归。

        domain / query_type 透传给 Harness，供 TaskCompiler 兜底/复用判据使用。
        """
        tools_schema = self._build_tools_schema()
        memory_manager = self._get_memory_manager()
        system_prompt = self._build_system_prompt(tools_schema)
        memory_context = self._load_memory_context(query, memory_manager)

        harness = self._get_harness()
        if harness is not None:
            hresult = self._run_via_harness(
                harness, query, system_prompt, tools_schema, memory_context,
                chat_history, domain=domain, query_type=query_type)
            if hresult is not None:
                self._record_observation(query, hresult.get("data", {}).get("response", ""), memory_manager)
                logger.info(f"[RAGAgent] 完成（Harness 执行层），query={query[:40]}")
                return hresult
            logger.warning("[RAGAgent] Harness 产出不可用，回退手动循环")

        return self._run_manual(query, chat_history, tools_schema, system_prompt, memory_context, memory_manager)

    def _run_manual(self, query: str, chat_history, tools_schema: List[Dict],
                    system_prompt: str, memory_context: str, memory_manager) -> Dict[str, Any]:
        """手动 Function Calling 循环（仅作为 Harness 不可用时的安全回退）。"""
        try:
            if memory_context:
                system_prompt = f"{system_prompt}\n\n【相关记忆】\n{memory_context}"

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            if chat_history:
                messages.extend(chat_history)

            messages.append({
                "role": "user",
                "content": query
            })

            logger.info(f"Agent 开始处理查询：{query}（可用工具：{len(tools_schema)} 个）")
            response = self.llm.call_with_tools(messages, tools_schema, tool_choice="auto")

            tool_calls = response.get("tool_calls")

            if tool_calls and len(tool_calls) > 0:
                logger.info(f"LLM 决定调用工具：{tool_calls[0].get('function', {}).get('name')}")

                tool_call = tool_calls[0]
                tool_name = tool_call.get('function', {}).get('name')
                tool_args_str = tool_call.get('function', {}).get('arguments', '{}')

                try:
                    tool_args = json.loads(tool_args_str)
                except Exception:
                    tool_args = {}

                tool_result = self._execute_tool(tool_name, tool_args)

                # HITL 审批：高风险写工具（如 repair_order_create/update）在 bridge
                # 层经 approval_node.should_require_approval 判断 + request_approval 创建
                # 审批请求。若返回 pending，run_agent 将结果反馈给用户，等待人工决策。
                if isinstance(tool_result, dict) and tool_result.get("requires_approval"):
                    return {
                        "success": False,
                        "data": {
                            "response": f"工具 {tool_name} 需要人工审批后才能执行，"
                                        f"审批请求 ID：{tool_result.get('approval_request_id', '未知')}",
                            "type": "agent",
                            "requires_approval": True,
                            "approval_request_id": tool_result.get("approval_request_id"),
                        },
                        "message": tool_result.get("message", "需人工审批"),
                    }

                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_calls", "tool_calls": tool_calls}
                    ]
                })

                messages.append({
                    "role": "tool",
                    "content": [
                        {"type": "tool_result", "tool_result": json.dumps(tool_result, ensure_ascii=False)}
                    ],
                    "tool_call_id": tool_call.get('id', 'call_1')
                })

                final_response = self.llm._call_llm(messages)
                final_answer = final_response.get("text", "")

                # 记忆系统：记录本次对话经验
                self._record_observation(query, final_answer, memory_manager)

                return {
                    "success": True,
                    "data": {
                        "response": final_answer,
                        "type": "agent",
                        "tool_used": tool_name,
                        "tool_result": tool_result
                    },
                    "message": f"Agent 处理成功（使用工具：{tool_name}）"
                }
            else:
                logger.info("LLM 直接回答，未调用工具")
                answer = response.get("text", "")

                # 记忆系统：记录本次对话经验
                self._record_observation(query, answer, memory_manager)

                return {
                    "success": True,
                    "data": {
                        "response": answer,
                        "type": "agent"
                    },
                    "message": "Agent 处理成功"
                }

        except Exception as e:
            import traceback
            logger.error(f"Agent 处理失败：{e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "data": None,
                "message": f"Agent 处理失败：{str(e)}"
            }

    def _record_observation(self, query: str, answer: str, memory_manager) -> None:
        """记录对话经验到记忆系统

        优先使用 MemoryManager（三层记忆），降级使用 SimpleConversationMemory。
        失败不阻塞主流程。
        """
        try:
            if memory_manager is not None:
                memory_manager.add_observation("user", query)
                memory_manager.add_observation("assistant", answer)
            else:
                # 降级：仅记录到内存 List
                self.memory.add_user_message(query)
                self.memory.add_ai_message(answer)
        except Exception as e:
            logger.warning(f"[RAGAgent] 记忆记录失败，忽略：{e}")

    def get_available_tools(self) -> Dict[str, Any]:
        """获取可用工具列表"""
        return {
            "success": True,
            "data": {
                "tools": self.mcp_bridge.get_tools_dict(),
                "total": len(self.mcp_bridge.get_tools_dict())
            },
            "message": "获取工具列表成功"
        }


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    rag_agent = RAGAgent()
else:
    rag_agent = None


def get_agent() -> RAGAgent:
    """获取 Agent 实例"""
    return rag_agent
