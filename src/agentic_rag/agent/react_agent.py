import logging
import json
import re
import os
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

_mcp_bridge = None


def _get_mcp_bridge():
    global _mcp_bridge
    if _mcp_bridge is None:
        try:
            from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
            _mcp_bridge = mcp_tool_bridge
        except ImportError:
            logger.warning("[ReActAgent] MCP Bridge 不可用")
            _mcp_bridge = None
    return _mcp_bridge


class ReActStepType(Enum):
    """ReAct 推理步骤类型"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


class ReActStep(TypedDict, total=False):
    """ReAct 单步推理记录"""
    step_type: ReActStepType
    content: str
    tool_name: Optional[str]
    tool_args: Optional[Dict]
    iteration: int


class ReActState(TypedDict, total=False):
    """ReAct Agent 状态"""
    query: str
    steps: List[ReActStep]
    current_thought: Optional[str]
    current_action: Optional[Dict]
    current_observation: Optional[str]
    final_answer: Optional[str]
    iteration: int
    max_iterations: int
    scratchpad: str
    tool_results: List[Dict]


@dataclass
class ReActConfig:
    """ReAct Agent 配置"""
    max_iterations: int = 6
    enable_reflection: bool = True
    repetition_threshold: int = 2
    force_summary_on_limit: bool = True


_DEFAULT_TOOLS = []  # 已迁移到 mcp_bridge.py 统一管理，此处仅作fallback


class ReActAgent:
    """ReAct 模式智能体

    遵循 Thought→Action→Observation 循环进行逐步推理和工具调用，
    集成 MCP Bridge 执行工具，支持重复检测和强制摘要。
    """

    def __init__(self, max_iterations: int = 6, tools: List[Dict] = None):
        self.config = ReActConfig(max_iterations=max_iterations)
        self.tools = tools if tools is not None else self._load_tools()

    def _load_tools(self) -> List[Dict]:
        bridge = _get_mcp_bridge()
        if bridge is not None:
            try:
                tools_dict = bridge.get_tools_dict()
                return [
                    {
                        "name": info["name"],
                        "description": info["description"],
                        "parameters": info.get("parameters", {}).get("properties", {})
                    }
                    for info in tools_dict.values()
                ]
            except Exception as e:
                logger.warning(f"[ReActAgent] 从 MCP Bridge 加载工具失败: {e}")
        # fallback: 返回空列表，ReAct Agent 将无法调用工具
        logger.warning("[ReActAgent] MCP Bridge 不可用，ReAct Agent 将无法调用工具")
        return []

    def run(self, query: str, state: Optional[Dict] = None) -> Dict:
        """执行 ReAct 推理+工具问答。

        统一执行层（Harness）优先：Harness 可用且产出可用答案时走 Harness；
        否则回退到 provable 的手写 ReAct 循环（作为安全回退，保证不回归）。
        """
        harness = self._get_harness()
        if harness is not None:
            result = self._run_via_harness(harness, query, state)
            if result is not None:
                logger.info(f"[ReActAgent] 完成（Harness 执行层），query={query[:40]}")
                return result
            logger.warning("[ReActAgent] Harness 产出不可用，回退手写 ReAct 循环")
        return self._run_react_manual(query, state)

    def _get_harness(self):
        """获取 Harness 主循环（优先复用全局 state.harness_loop）。"""
        try:
            from agentic_rag.api.state import state
            loop = getattr(state, "harness_loop", None)
            if loop is not None:
                return loop
        except Exception:
            pass
        try:
            from agent_harness import HarnessLoop, HarnessConfig, get_default_provider
            bridge = _get_mcp_bridge()
            loop = HarnessLoop(
                config=HarnessConfig(max_iterations=self.config.max_iterations,
                                     enable_safety_gateway=True),
                model_provider=get_default_provider("doubao"),
            )
            if bridge is not None:
                loop.set_tool_executor(bridge.execute_tool)
            return loop
        except Exception as e:
            logger.warning(f"[ReActAgent] Harness 初始化失败: {e}")
            return None

    def _build_schema(self) -> List[Dict]:
        """把 ReAct 工具列表转换为 OpenAI Function Calling schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": {"type": "object", "properties": t.get("parameters", {}) or {}},
                }
            }
            for t in self.tools
        ]

    def _run_via_harness(self, harness, query: str, state: Optional[Dict]) -> Optional[Dict]:
        """委托 Harness 统一执行层，映射 HarnessResult → ReAct 兼容 dict。"""
        try:
            from agent_harness import ContextPhase
            tool_descriptions = self._format_tool_descriptions()
            system_prompt = (
                f"你是一个遵循 ReAct 模式的智能助手。\n"
                f"可用工具：\n{tool_descriptions}\n"
                f"规则：如需调用工具请发起工具调用请求；若信息足够则直接给出最终答案。"
            )
            result = harness.run(
                session_id=f"react-{query[:20]}",
                system_prompt=system_prompt,
                query=query,
                tools_schema=self._build_schema(),
                phase=ContextPhase.EXECUTING,
            )
        except Exception as e:
            logger.error(f"[ReActAgent] Harness 执行异常: {e}")
            return None

        answer = (result.answer or "").strip()
        if not answer or answer == "系统繁忙，请稍后重试" or result.finish_reason == "model_unavailable":
            return None

        tool_results = getattr(result, "tool_results", []) or []
        steps = [
            {
                "step_type": ReActStepType.OBSERVATION,
                "content": str(tr.get("result", ""))[:500],
                "tool_name": tr.get("tool_name"),
                "tool_args": tr.get("args"),
                "iteration": i,
            }
            for i, tr in enumerate(tool_results)
        ]
        tool_calls = [
            {"tool": tr.get("tool_name"), "args": tr.get("args"),
             "result": tr.get("result"), "iteration": i}
            for i, tr in enumerate(tool_results)
        ]

        result_dict = {
            "answer": answer,
            "steps": steps,
            "iteration_count": result.iteration_count,
            "tool_calls": tool_calls,
            "scratchpad": f"[ReAct via Harness] tools={len(tool_calls)}, "
                          f"finish={result.finish_reason}",
            "tool_results": [
                {"tool": tr.get("tool_name"), "params": tr.get("args"),
                 "result": tr.get("result"),
                 "success": tr.get("status") != "error"}
                for tr in tool_results
            ],
        }
        if isinstance(state, dict):
            state["tool_results"] = state.get("tool_results", []) + result_dict["tool_results"]
        return result_dict

    def _run_react_manual(self, query: str, state: Optional[Dict]) -> Dict:
        """手写 ReAct 循环（仅作为 Harness 不可用时的安全回退）。"""
        react_state: ReActState = {
            "query": query,
            "steps": [],
            "current_thought": None,
            "current_action": None,
            "current_observation": None,
            "final_answer": None,
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "scratchpad": "",
            "tool_results": state.get("tool_results", []) if state else []
        }

        logger.info(f"[ReActAgent] 开始处理查询: {query}")

        while not self._should_continue(react_state):
            react_state = self._thought(react_state)

            if react_state.get("final_answer") is not None:
                break

            if react_state.get("current_action") is not None:
                react_state = self._action(react_state)
                react_state = self._observation(react_state)

            react_state["iteration"] = react_state.get("iteration", 0) + 1

        if react_state.get("final_answer") is None and self.config.force_summary_on_limit:
            react_state = self._force_summary(react_state)

        tool_calls = [
            {
                "tool": step.get("tool_name"),
                "args": step.get("tool_args"),
                "iteration": step.get("iteration", 0)
            }
            for step in react_state.get("steps", [])
            if step.get("step_type") == ReActStepType.ACTION
        ]

        result = {
            "answer": react_state.get("final_answer", ""),
            "steps": react_state.get("steps", []),
            "iteration_count": react_state.get("iteration", 0),
            "tool_calls": tool_calls,
            "scratchpad": react_state.get("scratchpad", ""),
            "tool_results": react_state.get("tool_results", []),
        }

        logger.info(f"[ReActAgent] 处理完成: iterations={result['iteration_count']}, "
                    f"tool_calls={len(tool_calls)}")

        return result

    def _thought(self, state: ReActState) -> ReActState:
        """LLM 推理步骤：分析当前情况，决定下一步行动或给出最终答案"""
        query = state.get("query", "")
        scratchpad = state.get("scratchpad", "")
        tool_descriptions = self._format_tool_descriptions()

        prompt = (
            f"你是一个遵循 ReAct 模式的智能助手。\n\n"
            f"对于每一步，你必须：\n"
            f"1. Thought: 分析当前情况，决定下一步行动\n"
            f"2. Action: 选择一个工具并传入参数\n"
            f"3. Observation: 接收工具返回的结果\n\n"
            f"可用工具：\n{tool_descriptions}\n\n"
            f"重要规则：\n"
            f"- 每次只调用一个工具\n"
            f"- 如果已有足够信息，直接给出最终答案\n"
            f"- 最多进行 {self.config.max_iterations} 轮推理\n"
            f"- 必须基于 Observation 得出结论，不可凭空推断\n"
            f"- 不可重复调用相同的工具和参数\n\n"
            f"当前推理记录：\n{scratchpad}\n\n"
            f"用户问题：{query}\n\n"
            f"请按以下格式回复：\n"
            f"Thought: <你的推理过程>\n"
            f"Action: <工具名称>(<JSON参数>)\n\n"
            f"或者如果你已经有足够信息回答问题：\n"
            f"Thought: <你的推理过程>\n"
            f"Answer: <最终答案>"
        )

        llm_response = self._call_llm(prompt)

        action_parsed = self._parse_action(llm_response)

        if action_parsed is not None:
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", llm_response, re.DOTALL)
            thought_content = thought_match.group(1).strip() if thought_match else llm_response.strip()

            state["current_thought"] = thought_content
            state["current_action"] = action_parsed

            state["steps"].append({
                "step_type": ReActStepType.THOUGHT,
                "content": thought_content,
                "tool_name": None,
                "tool_args": None,
                "iteration": state.get("iteration", 0)
            })
        else:
            answer_match = re.search(r"Answer:\s*(.+)", llm_response, re.DOTALL)
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nAnswer:|$)", llm_response, re.DOTALL)

            if answer_match:
                thought_content = thought_match.group(1).strip() if thought_match else ""
                answer_content = answer_match.group(1).strip()

                state["current_thought"] = thought_content
                state["final_answer"] = answer_content

                state["steps"].append({
                    "step_type": ReActStepType.THOUGHT,
                    "content": thought_content,
                    "tool_name": None,
                    "tool_args": None,
                    "iteration": state.get("iteration", 0)
                })
                state["steps"].append({
                    "step_type": ReActStepType.ANSWER,
                    "content": answer_content,
                    "tool_name": None,
                    "tool_args": None,
                    "iteration": state.get("iteration", 0)
                })
            else:
                state["current_thought"] = llm_response.strip()
                state["final_answer"] = llm_response.strip()

                state["steps"].append({
                    "step_type": ReActStepType.THOUGHT,
                    "content": llm_response.strip(),
                    "tool_name": None,
                    "tool_args": None,
                    "iteration": state.get("iteration", 0)
                })
                state["steps"].append({
                    "step_type": ReActStepType.ANSWER,
                    "content": llm_response.strip(),
                    "tool_name": None,
                    "tool_args": None,
                    "iteration": state.get("iteration", 0)
                })

        return state

    def _action(self, state: ReActState) -> ReActState:
        """执行工具调用"""
        action = state.get("current_action", {})
        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})

        state["steps"].append({
            "step_type": ReActStepType.ACTION,
            "content": f"调用工具 {tool_name}，参数: {json.dumps(tool_args, ensure_ascii=False)}",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "iteration": state.get("iteration", 0)
        })

        logger.info(f"[ReActAgent] 执行工具: {tool_name}, 参数: {tool_args}")

        return state

    def _observation(self, state: ReActState) -> ReActState:
        """处理工具结果，更新推理记录"""
        action = state.get("current_action", {})
        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})

        observation_content = self._execute_tool(tool_name, tool_args)

        state["current_observation"] = observation_content

        tool_results = state.get("tool_results", [])
        tool_results.append({
            "tool": tool_name,
            "params": tool_args,
            "result": observation_content,
            "success": not observation_content.startswith("工具执行失败")
        })
        state["tool_results"] = tool_results

        state["steps"].append({
            "step_type": ReActStepType.OBSERVATION,
            "content": observation_content[:500],
            "tool_name": tool_name,
            "tool_args": None,
            "iteration": state.get("iteration", 0)
        })

        state["scratchpad"] = self._build_scratchpad(state.get("steps", []))
        state["current_action"] = None
        state["current_observation"] = None

        return state

    def _should_continue(self, state: ReActState) -> bool:
        """检查循环是否应终止"""
        if state.get("final_answer") is not None:
            return True

        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self.config.max_iterations)
        if iteration >= max_iterations:
            return True

        steps = state.get("steps", [])
        if self._detect_repetition(steps):
            logger.warning("[ReActAgent] 检测到重复动作，终止循环")
            return True

        return False

    def _detect_repetition(self, steps: List[ReActStep]) -> bool:
        """检测智能体是否陷入重复循环"""
        action_steps = [s for s in steps if s.get("step_type") == ReActStepType.ACTION]

        if len(action_steps) < self.config.repetition_threshold:
            return False

        last_action = action_steps[-1]
        repeat_count = 0

        for step in reversed(action_steps):
            if (step.get("tool_name") == last_action.get("tool_name") and
                    step.get("tool_args") == last_action.get("tool_args")):
                repeat_count += 1
            else:
                break

        return repeat_count >= self.config.repetition_threshold

    def _force_summary(self, state: ReActState) -> ReActState:
        """达到最大迭代次数时强制生成摘要"""
        query = state.get("query", "")
        scratchpad = state.get("scratchpad", "")
        tool_results = state.get("tool_results", [])

        results_summary = "\n".join([
            f"- {tr.get('tool', '未知')}: {str(tr.get('result', ''))[:200]}"
            for tr in tool_results
        ])

        summary_prompt = (
            f"基于以下推理过程和工具调用结果，请给出最终答案。\n\n"
            f"用户问题：{query}\n\n"
            f"推理记录：\n{scratchpad}\n\n"
            f"工具调用结果：\n{results_summary}\n\n"
            f"请给出简洁准确的最终答案："
        )

        summary = self._call_llm(summary_prompt)

        state["final_answer"] = summary.strip()

        state["steps"].append({
            "step_type": ReActStepType.ANSWER,
            "content": f"[强制摘要] {summary.strip()}",
            "tool_name": None,
            "tool_args": None,
            "iteration": state.get("iteration", 0)
        })

        logger.info("[ReActAgent] 强制摘要生成完成")

        return state

    def _build_scratchpad(self, steps: List[ReActStep]) -> str:
        """构建累积推理记录"""
        parts = []
        for step in steps:
            step_type = step.get("step_type")
            content = step.get("content", "")
            iteration = step.get("iteration", 0)

            if step_type == ReActStepType.THOUGHT:
                parts.append(f"[轮次{iteration}] Thought: {content}")
            elif step_type == ReActStepType.ACTION:
                tool_name = step.get("tool_name", "")
                tool_args = step.get("tool_args", {})
                parts.append(f"[轮次{iteration}] Action: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
            elif step_type == ReActStepType.OBSERVATION:
                parts.append(f"[轮次{iteration}] Observation: {content}")
            elif step_type == ReActStepType.ANSWER:
                parts.append(f"[轮次{iteration}] Answer: {content}")

        return "\n".join(parts)

    def _format_tool_descriptions(self) -> str:
        """格式化可用工具描述"""
        descriptions = []
        for i, tool in enumerate(self.tools, 1):
            name = tool.get("name", "未知")
            desc = tool.get("description", "无描述")
            params = tool.get("parameters", {})

            param_str = ""
            if params:
                param_parts = [f"{k}: {v}" for k, v in params.items()]
                param_str = f"（参数: {', '.join(param_parts)}）"

            descriptions.append(f"{i}. {name} - {desc}{param_str}")

        return "\n".join(descriptions)

    def _parse_action(self, llm_response: str) -> Optional[Dict]:
        """解析 LLM 响应，提取工具名称和参数"""
        action_match = re.search(r"Action:\s*(\w+)\s*\((.+?)\)", llm_response, re.DOTALL)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()
        args_str = action_match.group(2).strip()

        try:
            tool_args = json.loads(args_str)
        except json.JSONDecodeError:
            tool_args = {}
            pairs = re.findall(r'(\w+)\s*[:=]\s*["\']?([^"\',}]+)["\']?', args_str)
            for key, value in pairs:
                tool_args[key.strip()] = value.strip()

        available_tools = {t["name"] for t in self.tools}
        if tool_name not in available_tools:
            logger.warning(f"[ReActAgent] LLM 选择了不存在的工具: {tool_name}")
            return None

        return {"tool_name": tool_name, "tool_args": tool_args}

    def _execute_tool(self, tool_name: str, tool_args: Dict) -> str:
        """执行单个工具调用"""
        bridge = _get_mcp_bridge()

        if bridge is not None:
            try:
                result = bridge.execute_tool(tool_name, tool_args)
                if isinstance(result, dict):
                    if result.get("success", True):
                        data = result.get("data", result)
                        return json.dumps(data, ensure_ascii=False)[:1000]
                    else:
                        return f"工具执行失败: {result.get('message', '未知错误')}"
                return json.dumps(result, ensure_ascii=False)[:1000]
            except Exception as e:
                logger.error(f"[ReActAgent] 工具执行异常: {tool_name}, error: {e}")
                return f"工具执行失败: {str(e)}"
        else:
            return self._mock_tool_execute(tool_name, tool_args)

    def _mock_tool_execute(self, tool_name: str, tool_args: Dict) -> str:
        """MCP Bridge 不可用时的模拟工具执行"""
        mock_results = {
            "inventory_query": {"items": [{"name": "火花塞", "part_number": "ENG-001", "stock": 150, "price": 25.0}], "total": 1},
            "inventory_low_stock": {"items": [{"name": "刹车片", "part_number": "BRK-001", "stock": 3, "min_stock": 10}], "total": 1},
            "repair_order_create": {"order_number": "WO-2024-001", "status": "待处理", "message": "工单创建成功"},
            "repair_order_query": {"items": [{"order_number": "WO-2024-001", "status": "进行中", "vehicle_model": "长安CS75"}], "total": 1},
            "employee_verify": {"name": "张三", "department": "技术部", "position": "工程师", "status": "在职"},
            "employee_query": {"items": [{"name": "张三", "department": "技术部", "position": "工程师"}], "total": 1},
            "regulation_search": {"items": [{"title": "考勤管理制度", "category": "人事管理", "content": "工作时间为9:00-18:00"}], "total": 1},
            "regulation_query": {"items": [{"title": "财务报销制度", "category": "财务管理", "content": "报销需提供发票原件"}], "total": 1},
        }

        if tool_name in mock_results:
            return json.dumps(mock_results[tool_name], ensure_ascii=False)

        return f"工具 {tool_name} 暂不可用（MCP Bridge 未连接）"

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成响应"""
        try:
            from agent_harness.providers.doubao import hybrid_llm
            response = hybrid_llm.generate_answer(
                query=prompt,
                context="",
                history=None,
                domain="general",
                use_finetuned=False
            )
            return response.get("answer", response.get("text", ""))
        except ImportError:
            pass
        except Exception:
            pass

        try:
            from agent_harness.providers.doubao import DoubaoLLMService
            llm = DoubaoLLMService()
            messages = [{"role": "user", "content": prompt}]
            response = llm._call_llm(messages)
            return response.get("text", "")
        except ImportError:
            pass
        except Exception:
            pass

        logger.warning("[ReActAgent] LLM 服务不可用，使用规则回退")
        return self._rule_based_thought(prompt)

    def _rule_based_thought(self, prompt: str) -> str:
        """LLM 不可用时的规则回退推理"""
        query = ""
        query_match = re.search(r"用户问题：(.+?)(?=\n\n|\n请按)", prompt, re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()

        tool_keywords = {
            "inventory_query": ["库存", "配件", "缺货"],
            "inventory_low_stock": ["低库存", "库存预警", "缺货预警"],
            "repair_order_create": ["创建工单", "生成工单", "新建工单"],
            "repair_order_query": ["查询工单", "工单状态", "维修工单"],
            "employee_verify": ["核验员工", "员工编号"],
            "employee_query": ["查询员工", "员工信息"],
            "regulation_search": ["搜索制度", "制度条款"],
            "regulation_query": ["制度类别", "规章制度"],
        }

        for tool_name, keywords in tool_keywords.items():
            for kw in keywords:
                if kw in query:
                    return f"Thought: 用户问题涉及{kw}，需要调用工具查询\nAction: {tool_name}({{}})"

        return f"Thought: 根据现有信息直接回答\nAnswer: 抱歉，我暂时无法处理该问题，请稍后重试。"


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    react_agent = ReActAgent()
else:
    react_agent = None
