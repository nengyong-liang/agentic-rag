import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from datetime import datetime
from agentic_rag.agent.memory import MemoryManager, MemoryConfig
from agentic_rag.agent.sufficiency_judge import (
    Conflict, EvidenceState, SufficiencyJudge, SufficiencyVerdict,
)
from agent_harness.observability.tracing import get_current_span, record_attributes, traced

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig: # 配置类，用于 Agentic RAG 2.0 智能体的参数设置
    """Agentic RAG 2.0 配置"""
    enabled: bool = True  # 是否启用 Agentic RAG 2.0 智能体
    max_iterations: int = 3  # 最大迭代次数
    evidence_threshold: float = 0.7  # 证据阈值，用于判断是否需要调用工具
    enable_reflection: bool = True  # 是否启用自省机制
    enable_tool_agent: bool = True  # 是否启用工具智能体
    enable_knowledge_fusion: bool = True  # 是否启用知识融合机制


# 共享状态类，用于 Agentic RAG 2.0 多智能体之间的信息传递
class AgenticRAGState(TypedDict, total=False): 
    """Agentic RAG 2.0 多智能体共享状态
    query: 用户原始查询
    original_query: 用户原始查询，未进行任何修改
    rewritten_queries: 重写后的查询列表，用于后续迭代
    task_plan: 任务分解，包含任务类型、任务描述、任务依赖等

    """
    query: str
    original_query: str
    rewritten_queries: List[str]
    task_plan: Dict[str, Any]
    retrieval_results: List[Dict]  # 检索结果，包含文档ID、内容、元数据等
    evidence_chunks: List[Dict]  # 证据片段，用于支持最终答案
    tool_results: List[Dict]  # 工具调用结果，包含工具名称、参数、返回值等
    validation_result: Dict[str, Any]  # 验证结果，包含是否通过、原因等
    fused_knowledge: Dict[str, Any]  # 融合后的知识，包含多个来源的信息整合
    draft_answer: str  # 草稿回答，用于迭代更新
    final_answer: str  # 最终回答，用于输出
    citations: List[Dict]  # 引用来源，包含文档ID、段落ID、文本片段等
    iteration_count: int  # 迭代次数，记录智能体运行次数
    agent_trace: List[Dict]  # 智能体运行轨迹，包含每个智能体的输入、输出、中间状态等
    route_decision: str  # 路由决策，记录智能体选择的路由路径（如 standard_rag / agentic_rag / tool_call）
    confidence_score: float  # 置信度分数，用于评估回答质量（0-1之间）
    metadata: Dict[str, Any]  # 元数据，包含额外信息（如查询时间、用户ID、会话ID等）


# 合并状态更新，用于 Agentic RAG 2.0 多智能体之间的状态同步
def _merge_state(left: Dict, right: Dict) -> Dict:
    """合并状态更新"""
    merged = {**left, **right}
    for key in set(left.keys()) & set(right.keys()):
        if isinstance(left[key], list) and isinstance(right[key], list):
            merged[key] = left[key] + right[key]
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            merged[key] = {**left[key], **right[key]}
    return merged


class SupervisorAgent:
    """
    元控制器（Supervisor Agent）
    
    Agentic RAG 2.0 的核心编排器，串联 Planner / Retriever / Validator / ToolAgent / Synthesizer，
    在 LangGraph 中作为路由节点运行，控制整个多智能体协同流程的执行与终止。

    职责：
    1. 接收用户请求，分析意图（关键词匹配 + 历史记忆辅助）
    2. 路由决策：plan / react / retrieve / tool_call / synthesize
    3. 任务分解与分配（委派给各子 Agent）
    4. 迭代控制与终止判断（校验通过或达最大次数则终止）
    5. 智能体执行轨迹记录（agent_trace）

    路由优先级（逐级判断）：
        iteration >= max → synthesize
        iteration == 0  → _initial_route(query) 初次路由
        validation.passed → synthesize
        validation.abstain → synthesize（弃答降级，不再补检）
        validation.needs_conflict_resolution → retrieve（冲突消解定向补检）
        validation.needs_retrieval → retrieve
        validation.needs_tool → tool_call
        兜底 → synthesize
    """
    
    # Agent 内部路由关键词（与 agent.py 对齐）
    # 注意：此为 Agent 内部子流程路由（react/plan/simple_tool），
    # 与 Java 端 Layer 1 关键词路由无关。
    # Java 端关键词来自 llm_router.py（GET /route/config 拉取）。
    ROUTE_KEYWORDS = {
        # "复杂操作类"关键词 — 命中时走 react（规划 + 检索 + 工具混合流程）
        "react": ["查询库存", "工单", "员工信息", "配件", "维修", "低库存", "缺货",
                  "创建工单", "核验", "制度条款", "查询", "搜索", "生成", "创建", "批量", "下单"],
        # "复杂分析类"关键词 — 命中时走 plan（多跳推理 / 对比分析）
        "complex": ["对比", "分析", "区别", "关系", "影响", "为什么", "如何", "流程", "步骤", "多个", "综合"],
        # "简单工具类"关键词 — 命中时直接走 tool_call（不经过检索）
        "simple_tool": ["查询库存", "工单", "员工信息", "制度条款", "配件", "维修", "低库存", "缺货", "创建工单", "核验", "下单"],
    }
    
    def __init__(self, config: Optional[AgentConfig] = None, memory: Optional[MemoryManager] = None):
        """
        初始化 Supervisor，创建所有子 Agent 实例。

        Args:
            config: 全局配置，控制迭代次数、开关等（见 AgentConfig）
            memory: 可选的长短期记忆管理器，用于记忆辅助路由决策
        """
        self.config = config or AgentConfig()         # 全局配置
        self.planner = PlannerAgent()                  # 规划智能体：任务分解为子查询
        self.retriever = RetrieverAgent()              # 检索智能体：RAG 文档检索
        self.validator = ValidatorAgent(config=self.config)  # 校验智能体：三态证据判定 + 幻觉检测
        self.tool_agent = ToolAgent()                  # 工具智能体：调用外部工具（MCP）
        self.synthesizer = SynthesizerAgent()          # 综合智能体：知识融合与答案生成
        self.memory = memory                           # 可选记忆管理器，用于辅助路由

    def route(self, state: AgenticRAGState) -> str:
        """
        路由决策：根据当前迭代次数与校验结果，判断下一步走哪个分支。

        决策优先级：
            1. 超限终止 → synthesize（达到 max_iterations）
            2. 首次路由 → _initial_route(query) 关键词匹配
            3. 校验通过 → synthesize
            4. 判定弃答 → synthesize（降级表述，不硬答）
            5. 判定冲突 → retrieve（冲突消解定向补检）
            6. 需补充检索 → retrieve
            7. 需工具调用 → tool_call
            8. 兜底 → synthesize

        Args:
            state: 当前多智能体共享状态

        Returns:
            "plan"      - 进入规划流程（复杂问题多跳推理）
            "react"     - 进入反应流程（规划 + 检索 + 工具混合）
            "retrieve"  - 直接检索（简单知识问答）
            "tool_call" - 工具调用（操作型查询）
            "synthesize"- 直接综合生成（已有足够信息或迭代终止）
        """


        # state即为当前多智能体共享状态，包含 query、迭代次数、校验结果等信息

        query = state.get("query", "")
        iteration = state.get("iteration_count", 0) # 当前迭代次数
        validation = state.get("validation_result", {}) # 校验结果

        # 1. 达到最大迭代次数 → 强制进入综合生成
        if iteration >= self.config.max_iterations:
            logger.info(f"[Supervisor] 达到最大迭代次数 {self.config.max_iterations}，进入综合")
            return "synthesize"

        # 2. 首次运行 → 根据关键词 + 历史记忆作初始路由
        if iteration == 0:
            route_decision = self._initial_route(query) # _initial_route 返回路由决策，比如 "react" / "tool_call" / "plan" / "retrieve"
            state["route_decision"] = route_decision # 记录初始路由决策
            # _add_trace 用于记录初始路由决策，方便后续分析
            self._add_trace(state, "supervisor", f"初始路由决策: {route_decision}")
            return route_decision

        # 3. 校验通过 → 已有充足证据，不再迭代
        if validation.get("passed", False):
            logger.info(f"[Supervisor] 校验通过，进入综合")
            return "synthesize"

        # 4. 三态判定：证据不足且已到上限 → 显式弃答/澄清，不再补检
        if validation.get("abstain", False):
            logger.info("[Supervisor] 判定为不可作答（abstain），进入综合走降级表述")
            self._add_trace(state, "supervisor",
                            f"弃答降级: 缺失 {validation.get('missing_info', [])}")
            return "synthesize"

        # 5. 三态判定：证据矛盾 → 冲突消解（按冲突事实定向补检权威/最新版本）
        if validation.get("evidence_state") == EvidenceState.CONFLICTING.value:
            conflicts = validation.get("conflicts", [])
            logger.info(f"[Supervisor] 检测到 {len(conflicts)} 处证据冲突，触发冲突消解补检")
            self._add_trace(state, "supervisor",
                            f"冲突消解: {[c.get('fact') for c in conflicts]}")
            return "retrieve"

        # 6. 校验认为证据不足 → 补充检索
        if validation.get("needs_retrieval", False):
            logger.info(f"[Supervisor] 校验未通过，需要补充检索")
            return "retrieve"

        # 7. 校验认为需要工具辅助 → 触发工具调用
        if validation.get("needs_tool", False):
            logger.info(f"[Supervisor] 需要工具调用")
            return "tool_call"

        # 8. 兜底
        return "synthesize"

    def _initial_route(self, query: str) -> str:
        """
        初始路由判断：基于关键词匹配 + 历史记忆辅助。

        路由优先级：
            1. Memory 辅助：用户历史  多次涉及工具类查询  且   非复杂问题 → tool_call
            2. 复杂 + 操作关键词 → react（混合流程）
            3. 简单工具关键词 → tool_call（直接调工具）
            4. 复杂分析关键词 → plan（多跳推理规划）
            5. 仅有操作关键词 → react
            6. 兜底 → retrieve（标准 RAG 检索）

        Args:
            query: 用户原始查询字符串

        Returns:
            "react" / "tool_call" / "plan" / "retrieve"
        """
        react_keywords = self.ROUTE_KEYWORDS["react"] # 操作关键词，比如 "查询"、"添加"、"删除" 等
        complex_indicators = self.ROUTE_KEYWORDS["complex"] # 复杂分析关键词，比如 "复杂"、"多跳" 等
        simple_tool_keywords = self.ROUTE_KEYWORDS["simple_tool"]

        has_react_signal = any(kw in query for kw in react_keywords) # 是否包含操作关键词
        has_complex = any(indicator in query for indicator in complex_indicators) # 是否包含复杂分析关键词
        has_simple_tool = any(kw in query for kw in simple_tool_keywords) # 是否包含简单工具关键词

        # 优先检查历史记忆：用户若多次使用工具类查询，倾向走 tool_call
        if self.memory is not None:
            try:
                memories = self.memory.retrieve_relevant(query, top_k=3) # 从记忆中检索相关记忆
                # 过滤出涉及工具类查询的记忆
                tool_related_memories = [
                    m for m in memories
                    if any(kw in m.content for kw in ["库存", "工单", "配件", "维修", "查询"])
                ]

                if len(tool_related_memories) >= 2:
                    # 用户历史交互中多次涉及工具类查询，倾向走 tool_call
                    if not has_complex:
                        return "tool_call" # 非复杂问题 → 工具调用
            except Exception as e:
                logger.debug(f"[Supervisor] Memory 辅助路由失败: {e}")

        # 关键词优先级路由
        if has_complex and has_react_signal:
            return "react"           # 复杂操作 → 反应流程（规划 + 检索 + 工具混合）
        if has_simple_tool:
            return "tool_call"       # 简单工具 → 直接调工具
        if has_complex:
            return "plan"            # 复杂分析 → 任务规划与多跳推理
        if has_react_signal and not has_simple_tool:
            return "react"           # 操作意图 → 反应流程

        return "retrieve"            # 兜底 → 标准 RAG 检索

    def should_continue(self, state: AgenticRAGState) -> str:
        """
        判断是否继续迭代（在每次子 Agent 执行完毕后调用）。

        Args:
            state: 当前多智能体共享状态

        Returns:
            "validate"  - 进入校验环节，由 Validator 判断是否继续
            "synthesize"- 达到最大迭代次数，直接进入综合生成
        """
        iteration = state.get("iteration_count", 0)

        # 达到最大迭代次数则终止迭代，否则交由 Validator 判断
        if iteration >= self.config.max_iterations:
            return "synthesize"

        return "validate"

    def _add_trace(self, state: AgenticRAGState, agent_name: str, action: str):
        """
        记录智能体执行轨迹到 agent_trace，用于调试与可解释性。

        Args:
            state: 当前多智能体共享状态（会被原地修改）
            agent_name: 执行该动作的智能体名称（如 supervisor / planner / retriever）
            action: 执行的动作描述（如 "初始路由决策: tool_call"）
        """
        trace = state.get("agent_trace", [])
        trace.append({
            "agent": agent_name,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        state["agent_trace"] = trace


class PlannerAgent:
    """规划智能体（Planner Agent）
    
    负责任务分解与规划：
    1. 分析复杂问题，拆解为子任务
    2. 识别多跳推理路径
    3. 生成查询变体
    4. 确定子任务依赖关系
    """
    
    def plan(self, state: AgenticRAGState) -> AgenticRAGState:
        """执行任务规划（复用 query_rewriter 的 KG+LLM 分解器）

        工业级升级：
        - 废弃原 _decompose_query() 的纯字符串 split
        - 复用 rag.query_rewriter.decompose()：KG多跳 + LLM 分解 + HyDE
        - 分解结果全部参与检索，不再只取第一个
        """
        query = state.get("query", "")

        # 复用 query_rewriter.stepback_expander.decompose()：KG多跳 + LLM 分解
        from agentic_rag.rag.query_rewriter import query_rewriter
        sub_queries = query_rewriter.stepback_expander.decompose(query)

        task_plan = {
            "original_query": query,
            "sub_queries": sub_queries,
            "requires_multi_hop": len(sub_queries) > 1,
            "requires_tool": self._check_tool_need(query),
            "strategy": self._determine_strategy(query)
        }

        state["task_plan"] = task_plan
        state["rewritten_queries"] = sub_queries
        state["iteration_count"] = state.get("iteration_count", 0)

        logger.info(f"[Planner] 任务规划完成: {len(sub_queries)} 个子查询, "
                    f"策略={task_plan['strategy']}, 子查询={sub_queries}")

        return state

    def _check_tool_need(self, query: str) -> bool:
        """检查是否需要工具调用"""
        tool_keywords = ["库存", "工单", "员工", "配件", "维修", "制度条款"]
        return any(kw in query for kw in tool_keywords)
    
    def _determine_strategy(self, query: str) -> str:
        """确定检索策略"""
        if "最新" in query or "当前" in query:
            return "recency_weighted"
        if "对比" in query or "区别" in query:
            return "comparative"
        if "流程" in query or "步骤" in query:
            return "sequential"
        return "standard"


class RetrieverAgent:
    """检索智能体（Retriever Agent）—— 轻量级降级桩

    ⚠️ 降级桩说明：
    此类为轻量级实现，仅用于以下场景：
    1. AgenticRAGOrchestrator 的 use_crag=False 模式（单元测试/降级）
    2. CRAG / hybrid_retriever 不可用时的回退路径

    工业级流程（use_crag=True）直接复用 rag.crag.run_crag()，
    走完整 RETRIEVE_START 链路（Dense编码 → Milvus/ES → RRF → Rerank → LLM Judge），
    不经过本类的 _mock_retrieve。

    原有职责（降级模式下仍生效）：
    1. 执行多路混合检索（mock 数据）
    2. 对多个子查询并行检索
    3. 证据去重与质量过滤
    """
    
    def retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """执行检索"""
        queries = state.get("rewritten_queries", [state.get("query", "")])
        all_results = []
        
        for q in queries:
            results = self._mock_retrieve(q)
            all_results.extend(results)
        
        deduped = self._deduplicate(all_results)
        
        state["retrieval_results"] = deduped
        state["evidence_chunks"] = deduped[:10]
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        logger.info(f"[Retriever] 检索完成: {len(queries)} 个查询, "
                    f"原始={len(all_results)}, 去重后={len(deduped)}")
        
        return state
    
    def _mock_retrieve(self, query: str) -> List[Dict]:
        """降级桩：模拟检索（仅用于 use_crag=False 或单元测试）

        工业级流程走 rag.crag.run_crag() → hybrid_retriever.retrieve()，不经过此方法。
        """
        return [
            {
                "content": f"关于「{query}」的相关知识内容片段1",
                "score": 0.92,
                "metadata": {
                    "document_id": 1,
                    "document_name": "企业知识库",
                    "page_number": 10,
                    "section": "相关章节",
                    "security_level": 0,
                    "is_latest": True
                }
            },
            {
                "content": f"关于「{query}」的相关知识内容片段2",
                "score": 0.85,
                "metadata": {
                    "document_id": 2,
                    "document_name": "操作手册",
                    "page_number": 25,
                    "section": "操作规范",
                    "security_level": 0,
                    "is_latest": True
                }
            }
        ]
    
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重检索结果"""
        seen = set()
        deduped = []
        for r in results:
            content = r.get("content", "")
            if content not in seen:
                seen.add(content)
                deduped.append(r)
        deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
        return deduped





class ValidatorAgent:
    """校验智能体（Validator Agent）—— 轻量级降级桩

    ⚠️ 降级桩说明：
    此类用字符 Jaccard 相似度替代 LLM Judge，仅用于：
    1. AgenticRAGOrchestrator 的 use_crag=False 模式（单元测试/降级）
    2. CRAG / LLM Judge 不可用时的回退路径

    工业级流程（use_crag=True）复用 rag.crag.grade_documents()，
    走 DeepSeek + Qwen 双 Judge + MoA 投票，不经过本类的字符相似度校验。

    原有职责（降级模式下仍生效）：
    1. 评估检索结果与查询的相关性（字符重叠率）
    2. 证据充分性三态判定（SUFFICIENT / INSUFFICIENT / CONFLICTING）
    3. 依据判定结论产出定向补检查询（missing_info / 冲突事实）
    4. 幻觉风险预警
    """

    def __init__(self, config: Optional[AgentConfig] = None,
                 judge: Optional[SufficiencyJudge] = None):
        self.config = config or AgentConfig()
        self.judge = judge or SufficiencyJudge(
            min_evidence=2, relevance_threshold=self._get_threshold())

    def validate(self, state: AgenticRAGState) -> AgenticRAGState:
        """执行校验（规则优先的三态证据判定）"""
        query = state.get("query", "")
        evidence = state.get("evidence_chunks", [])
        iteration = state.get("iteration_count", 0)

        relevance_score = self._compute_relevance(query, evidence)
        verdict = self.judge.judge(
            query=query,
            evidence=evidence,
            required_info=self._required_info(state),
            iteration=iteration,
            max_iterations=self.config.max_iterations,
        )
        sufficient = verdict.is_sufficient
        needs_retrieval = (not sufficient) and not verdict.abstain
        needs_tool = self._check_tool_requirement(query, state.get("tool_results", []))

        validation = {
            "passed": sufficient and relevance_score >= self._get_threshold(),
            "relevance_score": relevance_score,
            "sufficiency": sufficient,
            "needs_retrieval": needs_retrieval,
            "needs_tool": needs_tool,
            "evidence_count": len(evidence),
            "hallucination_risk": "low" if relevance_score > 0.7 else "medium" if relevance_score > 0.4 else "high",
            # —— 三态判定结构化产出（可审计、可标注对齐）——
            "evidence_state": verdict.state.value,
            "supported_by": verdict.supported_by,
            "missing_info": verdict.missing_info,
            "conflicts": [c.to_dict() for c in verdict.conflicts],
            "needs_conflict_resolution": verdict.state is EvidenceState.CONFLICTING,
            "targeted_queries": self.build_targeted_queries(query, verdict),
            "abstain": verdict.abstain,
            "needs_clarification": verdict.needs_clarification,
            "judge_confidence": round(verdict.confidence, 4),
            "decided_by": verdict.decided_by,
        }

        state["validation_result"] = validation
        state["confidence_score"] = relevance_score

        logger.info(f"[Validator] 校验结果: passed={validation['passed']}, "
                    f"evidence_state={validation['evidence_state']}, "
                    f"relevance={relevance_score:.2f}, "
                    f"missing={validation['missing_info']}, "
                    f"hallucination_risk={validation['hallucination_risk']}")

        return state

    @staticmethod
    def build_targeted_queries(query: str, verdict: SufficiencyVerdict) -> List[str]:
        """把判定缺口转成下一轮检索的定向查询（而不是原样重检一次）"""
        targeted: List[str] = []
        for conflict in verdict.conflicts[:2]:
            fact = conflict.fact or query
            targeted.append(f"{fact} 最新 规定 权威")
        for missing in verdict.missing_info[:2]:
            text = str(missing).strip()
            if text and text != query:
                targeted.append(f"{query} {text}"[:80])
        seen: List[str] = []
        for q in targeted:
            if q not in seen:
                seen.append(q)
        return seen[:3]

    @staticmethod
    def _required_info(state: AgenticRAGState) -> Optional[List[str]]:
        """Planner 若给出了需覆盖的关键信息，则用于覆盖率判定"""
        plan = state.get("task_plan") or {}
        required = plan.get("required_info")
        return list(required) if required else None

    def _compute_relevance(self, query: str, evidence: List[Dict]) -> float:
        """计算相关性分数"""
        if not evidence:
            return 0.0
        
        scores = [e.get("score", 0) for e in evidence]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        query_chars = set(query)
        overlap_scores = []
        for e in evidence:
            content = e.get("content", "")
            content_chars = set(content)
            if query_chars and content_chars:
                overlap = len(query_chars & content_chars) / len(query_chars)
                overlap_scores.append(overlap)
        
        char_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        
        return 0.7 * avg_score + 0.3 * char_overlap
    
    def _check_tool_requirement(self, query: str, tool_results: List[Dict]) -> bool:
        """检查是否需要工具调用"""
        tool_keywords = ["库存", "工单", "员工", "配件"]
        needs_tool = any(kw in query for kw in tool_keywords)
        has_tool_result = len(tool_results) > 0
        return needs_tool and not has_tool_result
    
    def _get_threshold(self) -> float:
        """获取校验阈值"""
        return 0.6

    @staticmethod
    def tristate_from_crag(query: str, grade: str, score: float,
                           contradictions: List[Any]) -> Dict[str, Any]:
        """把 CRAG 的 LLM Judge 评级与矛盾检测映射到统一三态词汇

        工业级链路（use_crag=True）的证据评估由 CRAG 完成，本方法只做口径归一，
        使 Supervisor 与 Synthesizer 面对的是同一套 evidence_state 契约。
        """
        conflicts: List[Conflict] = []
        for c in contradictions or []:
            if isinstance(c, dict):
                conflicts.append(Conflict(
                    fact=str(c.get("fact") or c.get("statement") or c.get("topic") or "知识矛盾"),
                    values=[str(v) for v in (c.get("values") or [str(c.get("detail", c))])],
                    evidence_ids=[str(x) for x in (c.get("evidence_ids") or [])],
                ))
            elif c:
                conflicts.append(Conflict(fact=str(c)[:40], values=[str(c)]))

        if conflicts:
            verdict = SufficiencyVerdict(
                state=EvidenceState.CONFLICTING, conflicts=conflicts,
                confidence=0.4, decided_by="crag",
                reasoning=f"CRAG 检出 {len(conflicts)} 处矛盾")
        elif grade == "relevant" and score >= 0.6:
            verdict = SufficiencyVerdict(
                state=EvidenceState.SUFFICIENT, confidence=round(score, 4),
                decided_by="crag", reasoning="CRAG 评级 relevant 且分数达标")
        else:
            verdict = SufficiencyVerdict(
                state=EvidenceState.INSUFFICIENT, missing_info=[query[:60]],
                confidence=round(score, 4), decided_by="crag",
                reasoning=f"CRAG 评级 {grade}")

        out = verdict.to_dict()
        out["sufficiency"] = verdict.is_sufficient
        out["targeted_queries"] = ValidatorAgent.build_targeted_queries(query, verdict)
        out["needs_conflict_resolution"] = verdict.state is EvidenceState.CONFLICTING
        return out


class ToolAgent:
    """工具智能体（Tool Agent）
    
    负责MCP工具调用：
    1. 判断是否需要工具调用
    2. 选择合适的工具
    3. 执行工具并收集结果
    4. 将工具结果融入知识
    """
    
    def __init__(self):
        self._mcp_bridge = None
    
    def _get_bridge(self):
        """延迟加载 MCP Bridge"""
        if self._mcp_bridge is None:
            try:
                from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
                self._mcp_bridge = mcp_tool_bridge
            except ImportError:
                logger.warning("[ToolAgent] MCP Bridge 不可用")
                self._mcp_bridge = None
        return self._mcp_bridge
    
    def execute(self, state: AgenticRAGState) -> AgenticRAGState:
        """执行工具调用"""
        query = state.get("query", "")
        tool_results = state.get("tool_results", [])
        
        bridge = self._get_bridge()
        if bridge is None:
            logger.warning("[ToolAgent] MCP Bridge 不可用，跳过工具调用")
            return state
        
        tool_calls = self._plan_tool_calls(query)
        
        for tc in tool_calls:
            try:
                result = bridge.execute_tool(tc["tool"], tc["params"])
                tool_results.append({
                    "tool": tc["tool"],
                    "params": tc["params"],
                    "result": result,
                    "success": result.get("success", True) if isinstance(result, dict) else True
                })
                logger.info(f"[ToolAgent] 工具调用成功: {tc['tool']}")
            except Exception as e:
                logger.error(f"[ToolAgent] 工具调用失败: {tc['tool']}, error: {e}")
                tool_results.append({
                    "tool": tc["tool"],
                    "params": tc["params"],
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
        
        state["tool_results"] = tool_results
        return state
    
    def _plan_tool_calls(self, query: str) -> List[Dict]:
        """根据查询规划工具调用"""
        calls = []
        
        if any(kw in query for kw in ["库存", "配件", "缺货", "低库存"]):
            calls.append({
                "tool": "inventory_query",
                "params": {"query": query}
            })
        
        if any(kw in query for kw in ["工单", "维修"]):
            if "创建" in query or "生成" in query:
                calls.append({
                    "tool": "repair_order_create",
                    "params": {"description": query}
                })
            else:
                calls.append({
                    "tool": "repair_order_query",
                    "params": {"query": query}
                })
        
        if any(kw in query for kw in ["员工", "人员"]):
            calls.append({
                "tool": "employee_query",
                "params": {"query": query}
            })
        
        if any(kw in query for kw in ["制度", "条款", "规章"]):
            calls.append({
                "tool": "regulation_search",
                "params": {"query": query}
            })
        
        return calls[:3]

# 综合智能体（Synthesizer Agent）
class SynthesizerAgent:
    """综合智能体（Synthesizer Agent）
    
    负责答案生成与知识融合：
    1. 融合多源证据（检索结果 + 工具结果）
    2. 动态知识融合（冲突检测与解决）
    3. 生成带引用溯源的最终答案
    4. 幻觉抑制与一致性校验
    """
    

    """
    输入：AgenticRAGState 状态
    输出：AgenticRAGState 状态
    - 读取状态 — 从 AgenticRAGState 中取出 query、证据块、工具调用结果、校验结果
    - 知识融合 — 调用 _fuse_knowledge() 将检索结果和工具结果合并成统一结构，存入 fused_knowledge
    - 答案生成 — 调用 _generate_answer() 基于 query + 融合知识 + 校验结果生成最终回答，存入 final_answer
    - 引用提取 — 调用 _extract_citations() 从证据和工具结果中提取引用来源（文档名、章节等），存入 citations
    """
    def synthesize(self, state: AgenticRAGState) -> AgenticRAGState:
        """执行综合生成"""
        query = state.get("query", "")
        evidence = state.get("evidence_chunks", [])
        tool_results = state.get("tool_results", [])
        validation = state.get("validation_result", {})
        
        fused = self._fuse_knowledge(evidence, tool_results)
        state["fused_knowledge"] = fused
        
        answer = self._generate_answer(query, fused, validation)
        state["final_answer"] = answer
        
        citations = self._extract_citations(evidence, tool_results)
        state["citations"] = citations
        
        logger.info(f"[Synthesizer] 综合完成: 答案长度={len(answer)}, "
                    f"引用数={len(citations)}, 融合源={len(fused.get('sources', []))}")
        
        return state
    
    
    def _fuse_knowledge(self, evidence: List[Dict], tool_results: List[Dict]) -> Dict:
        """
        动态知识融合

        将检索证据和工具调用结果统一为结构化源列表（sources），
        并检测不同证据之间的矛盾（conflicts），
        最终确定融合策略（authority_weighted / simple_merge）。

        Args:
            evidence: RAG 检索得到的文档块列表（最多取前 5 条）
            tool_results: Agent 工具调用结果列表（仅保留成功的调用）

        Returns:
            Dict: {
                "sources": 结构化融合源列表,
                "conflicts": 冲突检测结果,
                "fusion_strategy": "simple_merge" 或 "authority_weighted"
            }
        """
        sources = []

        # 1. 融合检索结果 —— 提取内容、打分、元数据（文档名 / 页码 / 章节）
        for e in evidence[:5]:
            sources.append({
                "type": "retrieval",
                "content": e.get("content", ""),
                "score": e.get("score", 0),
                "document_id": e.get("metadata", {}).get("document_id"),
                "document_name": e.get("metadata", {}).get("document_name", ""),
                "page_number": e.get("metadata", {}).get("page_number"),
                "section": e.get("metadata", {}).get("section", "")
            })

        # 2. 融合工具调用结果 —— 提取工具名、内容摘要（限 500 字符），固定打分 0.9
        for tr in tool_results:
            if tr.get("success"):
                sources.append({
                    "type": "tool",
                    "tool_name": tr.get("tool", ""),
                    "content": json.dumps(tr.get("result", {}), ensure_ascii=False)[:500],
                    "score": 0.9
                })

        # 3. 检测不同来源间的语义冲突
        conflicts = self._detect_conflicts(sources)

        # 4. 返回融合结果：存在冲突时采用加权策略，否则简单合并
        return {
            "sources": sources,
            "conflicts": conflicts,
            "fusion_strategy": "authority_weighted" if conflicts else "simple_merge"
        }
    
    def _detect_conflicts(self, sources: List[Dict]) -> List[Dict]:
        """检测知识冲突"""
        conflicts = []
        retrieval_sources = [s for s in sources if s["type"] == "retrieval"]
        
        for i, s1 in enumerate(retrieval_sources):
            for s2 in retrieval_sources[i+1:]:
                content1 = s1.get("content", "")
                content2 = s2.get("content", "")
                if content1 and content2 and content1 != content2:
                    overlap = len(set(content1) & set(content2))
                    total = len(set(content1) | set(content2))
                    similarity = overlap / total if total > 0 else 0
                    if 0.3 < similarity < 0.8:
                        conflicts.append({
                            "source_a": s1.get("document_name", "未知"),
                            "source_b": s2.get("document_name", "未知"),
                            "similarity": round(similarity, 2)
                        })
        
        return conflicts
    
    def _generate_answer(self, query: str, fused: Dict, validation: Dict) -> str:
        """生成最终答案"""
        sources = fused.get("sources", [])
        conflicts = fused.get("conflicts", [])

        # 三态判定为弃答（证据不足且已到轮次上限）→ 不硬答
        if validation.get("abstain"):
            missing = validation.get("missing_info", [])
            hint = f"（仍缺：{'、'.join(str(m) for m in missing[:3])}）" if missing else ""
            return (f"抱歉，关于「{query}」现有知识库中的证据不足以支撑可靠回答{hint}，"
                    f"已停止作答以避免编造。请补充相关文档或缩小问题范围。")

        if not sources:
            return "抱歉，未找到与您问题相关的知识，请尝试换一种方式提问。"
        
        answer_parts = []
        answer_parts.append(f"关于「{query}」，根据知识库检索结果：\n")
        
        for i, source in enumerate(sources[:5], 1):
            if source["type"] == "retrieval":
                doc_name = source.get("document_name", "未知文档")
                page = source.get("page_number", "")
                section = source.get("section", "")
                answer_parts.append(f"{i}. {source['content'][:200]} "
                                   f"[来源: {doc_name}" + 
                                   (f", 第{page}页" if page else "") + 
                                   (f", {section}" if section else "") + "]")
            elif source["type"] == "tool":
                answer_parts.append(f"{i}. [工具调用: {source.get('tool_name', '')}] "
                                   f"{source['content'][:200]}")
        
        if conflicts:
            answer_parts.append(f"\n⚠️ 检测到 {len(conflicts)} 处知识冲突，请以最新版本为准。")

        if validation.get("evidence_state") == EvidenceState.CONFLICTING.value:
            facts = [str(c.get("fact", "")) for c in validation.get("conflicts", [])
                     if isinstance(c, dict) and c.get("fact")]
            scope = f"（{'、'.join(facts[:3])}）" if facts else ""
            answer_parts.append(f"\n⚠️ 证据{scope}互相矛盾，补检后仍未消解，"
                                f"请勿直接采用单一来源数值，建议核实最新文档。")
        
        hallucination_risk = validation.get("hallucination_risk", "low")
        if hallucination_risk == "high":
            answer_parts.append("\n⚠️ 检索结果相关性较低，以上回答仅供参考，建议核实原始文档。")
        
        return "\n".join(answer_parts)
    
    def _extract_citations(self, evidence: List[Dict], tool_results: List[Dict]) -> List[Dict]:
        """提取引用溯源"""
        citations = []
        
        for e in evidence[:5]:
            meta = e.get("metadata", {})
            citations.append({
                "document_id": meta.get("document_id"),
                "document_name": meta.get("document_name", ""),
                "page_number": meta.get("page_number"),
                "section": meta.get("section", ""),
                "score": e.get("score", 0),
                "type": "retrieval"
            })
        
        for tr in tool_results:
            if tr.get("success"):
                citations.append({
                    "tool_name": tr.get("tool", ""),
                    "type": "tool",
                    "score": 0.9
                })
        
        return citations


class AgenticRAGOrchestrator:
    """Agentic RAG 2.0 编排器（双模式：工业级 CRAG 复用 / 轻量降级桩）

    基于 Supervisor-Worker 模式的多智能体协作编排：
    - Supervisor: 路由决策 + 迭代控制
    - Planner: 任务分解 + 查询规划
    - Retriever/Validator: 轻量降级桩（use_crag=False 时使用）
    - Tool: MCP 工具调用（Agent 特有，CRAG 不处理）
    - Synthesizer: 工具结果 + 检索结果融合（Agent 特有）

    双模式架构（高内聚低耦合）：
    - use_crag=True（默认，工业级）：retrieve/plan/react 路由复用 rag.crag.run_crag()
      走完整 RETRIEVE_START 链路（Dense编码 → Milvus/ES → RRF → Rerank → LLM Judge + MoA）
    - use_crag=False（降级/测试）：走 RetrieverAgent._mock_retrieve + ValidatorAgent 字符相似度
    - tool_call 路由始终走独立 ToolAgent + Synthesizer（CRAG 不处理工具调用）

    用户可通过 config.enabled 开关选择是否启用多智能体模式。
    """

    def __init__(self, config: Optional[AgentConfig] = None, memory_manager: Optional[MemoryManager] = None):
        self.config = config or AgentConfig()
        self.memory = memory_manager if memory_manager is not None else MemoryManager(MemoryConfig())
        self.supervisor = SupervisorAgent(self.config, self.memory)

    @traced("agent.run", attributes_fn=lambda self, query, **kw: {"agent.query": str(query)[:60]})
    async def run(self, query: str, history: Optional[List[Dict]] = None,
                  use_crag: bool = True, route: Optional[str] = None,
                  domain: str = "general", use_finetuned: bool = False) -> Dict[str, Any]:
        """运行 Agentic RAG 2.0 多智能体协作流程

        Args:
            query: 用户查询
            history: 对话历史
            use_crag: 是否复用 CRAG 完整链路
                     - True（默认，工业级）：retrieve/plan/react 走 run_crag() 完整 RETRIEVE_START
                     - False（降级/测试）：走 RetrieverAgent + ValidatorAgent 轻量桩
            route: 路由覆盖（react/plan/retrieve/tool_call）。传入时优先采用，
                   否则由 SupervisorAgent 内部路由决定（调度网关可贯穿 Java route）。
            domain: 领域标识（贯穿到 run_crag，修复 D6 固定 "general" 断点）
            use_finetuned: 是否使用微调模型（贯穿到 run_crag）

        Returns:
            包含 answer, citations, agent_trace, confidence 等的结果字典
        """
        if not self.config.enabled:
            return {
                "answer": "",
                "citations": [],
                "agent_trace": [],
                "confidence": 0.0,
                "mode": "disabled",
                "message": "Agentic RAG 2.0 未启用"
            }

        logger.info(f"[AgenticRAG 2.0] 开始处理查询: {query} (use_crag={use_crag})")

        # 检索相关记忆
        relevant_memories = self.memory.retrieve_relevant(query, top_k=3)
        memory_context = self.memory.get_context_for_prompt(query, max_tokens=500)

        state: AgenticRAGState = {
            "query": query,
            "original_query": query,
            "rewritten_queries": [query],
            "retrieval_results": [],
            "evidence_chunks": [],
            "tool_results": [],
            "validation_result": {},
            "fused_knowledge": {},
            "iteration_count": 0,
            "agent_trace": [],
            "route_decision": "",
            "confidence_score": 0.0,
            "metadata": {"history": history or [], "memory_context": memory_context,
                         "domain": domain, "use_finetuned": use_finetuned}
        }

        # 路由决策：支持调度网关贯穿 route（react/plan/retrieve/tool_call），否则由 Supervisor 决定
        internal_routes = ("react", "plan", "retrieve", "tool_call")
        if route in internal_routes:
            route_decision = route
            state["route_decision"] = route
        else:
            route_decision = self.supervisor.route(state)  # 路由决策
        route = route_decision

        # plan 路由：先任务分解，再对每个子查询走 CRAG
        sub_queries = [query]  # 默认只查原查询
        if route == "plan":
            state = self.supervisor.planner.plan(state)
            sub_queries = state.get("rewritten_queries", [query])
            route = "retrieve"  # 落入 retrieve

        # retrieve 路由：复用 CRAG 完整链路 或 走轻量降级桩
        if route == "retrieve":
            if use_crag:
                state = await self._run_multi_crag(state, sub_queries, history, domain, use_finetuned)
            else:
                state = await self._lightweight_retrieve(state)

        # tool_call 路由：独立 ToolAgent + Synthesizer（CRAG 不处理工具）
        elif route == "tool_call":
            state = self.supervisor.tool_agent.execute(state)
            state = self.supervisor.synthesizer.synthesize(state)

        # react 路由：ReAct 工具调用 + CRAG 检索 + Synthesizer 融合
        elif route == "react":
            # 1. ReAct 工具调用（Agent 独有）
            try:
                from agentic_rag.agent.react_agent import ReActAgent
                react_agent = ReActAgent()
                react_result = react_agent.run(query, state)
                state["final_answer"] = react_result.get("answer", "")
                state["tool_results"] = react_result.get("tool_results", state.get("tool_results", []))
                state["iteration_count"] = react_result.get("iteration_count", 0)
                trace = state.get("agent_trace", [])
                for step in react_result.get("steps", []):
                    trace.append({
                        "agent": "react",
                        "step_type": step.get("step_type", ""),
                        "content": step.get("content", ""),
                        "tool_name": step.get("tool_name"),
                        "iteration": step.get("iteration", 0)
                    })
                state["agent_trace"] = trace
            except ImportError:
                logger.warning("[AgenticRAG 2.0] ReAct Agent 不可用，降级为 tool_call")
                state = self.supervisor.tool_agent.execute(state)

            # 2. 复用 CRAG 检索+生成（高内聚：复用 RAG 能力）
            if use_crag:
                try:
                    crag_result = await self._run_crag_with_retry(query, history, domain, use_finetuned)
                    state = self._merge_crag_result(state, crag_result)
                except Exception as e:
                    logger.warning(f"[AgenticRAG 2.0] react CRAG 失败，降级到轻量检索: {e}")
                    state = await self._lightweight_retrieve(state)
            else:
                state = await self._lightweight_retrieve(state)

            # 3. Synthesizer 融合工具结果 + 检索结果（Agent 独有）
            state = self.supervisor.synthesizer.synthesize(state)

        # 兜底：直接综合
        if "final_answer" not in state or not state.get("final_answer"):
            state = self.supervisor.synthesizer.synthesize(state)

        result = {
            "answer": state.get("final_answer", ""),
            "citations": state.get("citations", []),
            "agent_trace": state.get("agent_trace", []),
            "confidence": state.get("confidence_score", 0.0),
            "validation": state.get("validation_result", {}),
            "fused_knowledge": state.get("fused_knowledge", {}),
            "iteration_count": state.get("iteration_count", 0),
            "route_decision": state.get("route_decision", ""),
            "mode": "agentic_rag_2.0"
        }

        # 记录本次交互到记忆
        self.memory.add_observation("user", query)
        self.memory.add_observation("assistant", result.get("answer", ""))

        record_attributes(get_current_span(), {
            "agent.route": result["route_decision"],
            "agent.iterations": result["iteration_count"],
            "agent.confidence": result["confidence"],
            "agent.evidence_state": (result.get("validation") or {}).get("evidence_state", ""),
            "agent.citation_count": len(result["citations"]),
        })

        logger.info(f"[AgenticRAG 2.0] 处理完成: route={result['route_decision']}, "
                    f"iterations={result['iteration_count']}, "
                    f"confidence={result['confidence']:.2f}")

        return result

    async def _run_crag_with_retry(self, query: str, history: Optional[List[Dict]],
                                   domain: str = "general", use_finetuned: bool = False) -> Dict[str, Any]:
        """复用 CRAG 完整流程（高内聚：检索+评估+改写+生成）

        封装 run_crag 调用，统一异常处理。domain/use_finetuned 贯穿到 run_crag，
        修复 D6：编排器不再固定 "general"/False，而是透传调度网关的下发参数。
        """
        from agentic_rag.rag.crag import run_crag
        return await run_crag(
            query=query,
            domain=domain,
            use_finetuned=use_finetuned,
            history=history,
        )

    def _merge_crag_result(self, state: AgenticRAGState, crag_result: Dict[str, Any]) -> AgenticRAGState:
        """将 CRAG 结果适配到 AgenticRAGState

        CRAG 返回字段 → AgenticRAGState 字段映射：
        - answer → final_answer
        - citations → citations
        - relevance_grade → validation_result.passed (grade=="relevant")
        - relevance_score → validation_result.relevance_score / confidence_score
        - chain_log → 合并到 agent_trace
        - contradictions → validation_result.contradictions
        """
        # 答案与引用
        state["final_answer"] = crag_result.get("answer", "")
        state["citations"] = crag_result.get("citations", [])

        # 校验结果（CRAG 的 LLM Judge + MoA 投票）
        grade = crag_result.get("relevance_grade", "partial")
        score = crag_result.get("relevance_score", 0.0)
        contradictions = crag_result.get("contradictions", [])

        state["validation_result"] = {
            "passed": grade == "relevant",
            "relevance_score": score,
            "relevance_grade": grade,
            "needs_retrieval": grade in ("partial", "irrelevant"),
            "needs_tool": False,
            "evidence_count": len(state.get("evidence_chunks", [])),
            "hallucination_risk": "low" if score > 0.7 else "medium" if score > 0.4 else "high",
            "contradictions": contradictions,
            **ValidatorAgent.tristate_from_crag(state.get("query", ""), grade, score, contradictions),
        }
        state["confidence_score"] = score

        # 合并 CRAG chain_log 到 agent_trace
        trace = state.get("agent_trace", [])
        for log_entry in crag_result.get("chain_log", []):
            trace.append({
                "agent": "crag",
                "step": log_entry.get("step", ""),
                "data": log_entry.get("data", {}),
                "timestamp": log_entry.get("timestamp", ""),
            })
        state["agent_trace"] = trace

        # 迭代计数（CRAG 改写次数）
        rewrite_count = crag_result.get("rewrite_count", 0)
        state["iteration_count"] = state.get("iteration_count", 0) + rewrite_count

        logger.info(f"[AgenticRAG 2.0] CRAG 结果合并完成: grade={grade}, "
                    f"score={score:.2f}, rewrite={rewrite_count}, "
                    f"contradictions={len(contradictions)}")

        return state

    async def _run_multi_crag(self, state: AgenticRAGState, sub_queries: List[str],
                              history: Optional[List[Dict]], domain: str = "general",
                              use_finetuned: bool = False) -> AgenticRAGState:
        """执行 CRAG（支持多子查询并行 + 结果合并 + 降级策略）

        - 单查询：直接走 _run_crag_with_retry + _merge_crag_result
        - 多查询：asyncio.gather 并行 → _merge_multi_crag_results 聚合
        - 降级链：多查询全失败 → 单查询首条 → 轻量检索
        """
        if len(sub_queries) <= 1:
            # 单查询：直接走 CRAG
            try:
                crag_result = await self._run_crag_with_retry(sub_queries[0], history, domain, use_finetuned)
                return self._merge_crag_result(state, crag_result)
            except Exception as e:
                logger.warning(f"[AgenticRAG 2.0] CRAG 失败，降级到轻量检索: {e}")
                return await self._lightweight_retrieve(state)

        # 多子查询并行
        try:
            tasks = [self._run_crag_with_retry(sq, history, domain, use_finetuned) for sq in sub_queries]
            crag_results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = [r for r in crag_results if not isinstance(r, Exception)]
            if not valid_results:
                raise RuntimeError("所有子查询 CRAG 均失败")

            if len(valid_results) == 1:
                logger.info(f"[AgenticRAG 2.0] 多子查询仅1个成功，走单结果合并")
                return self._merge_crag_result(state, valid_results[0])

            logger.info(f"[AgenticRAG 2.0] {len(valid_results)}/{len(sub_queries)} 个子查询 CRAG 成功，合并结果")
            return self._merge_multi_crag_results(state, valid_results)
        except Exception as e:
            logger.warning(f"[AgenticRAG 2.0] 多子查询 CRAG 失败，降级到首查询: {e}")
            try:
                crag_result = await self._run_crag_with_retry(sub_queries[0], history, domain, use_finetuned)
                return self._merge_crag_result(state, crag_result)
            except Exception as e2:
                logger.warning(f"[AgenticRAG 2.0] 降级也失败，走轻量检索: {e2}")
                return await self._lightweight_retrieve(state)

    def _merge_multi_crag_results(self, state: AgenticRAGState,
                                   crag_results: List[Dict[str, Any]]) -> AgenticRAGState:
        """合并多个子查询的 CRAG 结果（工业级聚合）

        聚合策略：
        - citations: 按 chunk id 去重合并
        - answer: 按子查询顺序拼接，strategy 感知前缀标注
        - confidence: 取最大值
        - validation: 取最优评级
        - chain_log: 全部合并到 agent_trace
        """
        all_citations = []
        seen_chunk_ids = set()
        answer_parts = []
        max_score = 0.0
        best_grade = "irrelevant"
        all_contradictions = []
        all_chain_logs = []

        for crag_result in crag_results:
            # 合并 citations（按 id 去重）
            for cite in crag_result.get("citations", []):
                chunk_id = cite.get("id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_citations.append(cite)

            # 收集答案
            answer = crag_result.get("answer", "")
            if answer:
                answer_parts.append(answer)

            # 取最高置信度
            score = crag_result.get("relevance_score", 0.0)
            if score > max_score:
                max_score = score

            # 取最佳评级
            grade = crag_result.get("relevance_grade", "irrelevant")
            if grade == "relevant":
                best_grade = "relevant"
            elif grade == "partial" and best_grade != "relevant":
                best_grade = "partial"

            # 合并矛盾
            all_contradictions.extend(crag_result.get("contradictions", []))

            # 合并 chain_log
            all_chain_logs.extend(crag_result.get("chain_log", []))

        # 拼接最终答案（strategy 感知）
        strategy = state.get("task_plan", {}).get("strategy", "standard")
        if len(answer_parts) > 1:
            if strategy == "comparative":
                prefix = "【对比分析】\n"
            elif strategy == "sequential":
                prefix = "【流程说明】\n"
            elif strategy == "recency_weighted":
                prefix = "【时效分析】\n"
            else:
                prefix = ""
            final_answer = prefix + "\n\n".join(answer_parts)
        elif answer_parts:
            final_answer = answer_parts[0]
        else:
            final_answer = ""

        state["final_answer"] = final_answer
        state["citations"] = all_citations
        state["confidence_score"] = max_score
        state["validation_result"] = {
            "passed": best_grade == "relevant",
            "relevance_score": max_score,
            "relevance_grade": best_grade,
            "needs_retrieval": best_grade in ("partial", "irrelevant"),
            "needs_tool": False,
            "evidence_count": len(all_citations),
            "hallucination_risk": "low" if max_score > 0.7 else "medium" if max_score > 0.4 else "high",
            "contradictions": all_contradictions,
            **ValidatorAgent.tristate_from_crag(state.get("query", ""), best_grade,
                                                max_score, all_contradictions),
        }

        # 合并 chain_log 到 agent_trace
        trace = state.get("agent_trace", [])
        for log_entry in all_chain_logs:
            trace.append({
                "agent": "crag",
                "step": log_entry.get("step", ""),
                "data": log_entry.get("data", {}),
                "timestamp": log_entry.get("timestamp", ""),
            })
        state["agent_trace"] = trace

        logger.info(f"[AgenticRAG 2.0] 多子查询合并完成: "
                    f"{len(crag_results)} 子查询, "
                    f"{len(all_citations)} 引用(去重), "
                    f"confidence={max_score:.2f}, "
                    f"strategy={strategy}")

        return state

    async def _lightweight_retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """轻量级降级检索：RetrieverAgent + ValidatorAgent（use_crag=False 时使用）

        走 mock 检索 + 三态证据判定，不依赖 Milvus/ES/LLM。
        适用于单元测试和低配环境降级。
        """
        state = self.supervisor.retriever.retrieve(state)
        state = self.supervisor.validator.validate(state)

        # 迭代校验（轻量级，最多 max_iterations 次）
        while self.supervisor.should_continue(state) == "validate":
            validation = state["validation_result"]
            if validation.get("passed") or validation.get("abstain"):
                break
            if validation.get("needs_retrieval"):
                # 缺口驱动：用判定产出的定向查询补检，而不是原样重检
                state = self._apply_targeted_queries(state, validation)
                state = self.supervisor.retriever.retrieve(state)
            state = self.supervisor.validator.validate(state)

        return state

    @staticmethod
    def _apply_targeted_queries(state: AgenticRAGState,
                                validation: Dict[str, Any]) -> AgenticRAGState:
        """将三态判定的定向查询并入 rewritten_queries（去重、限量）"""
        targeted = validation.get("targeted_queries") or []
        if not targeted:
            return state
        existing = list(state.get("rewritten_queries", []))
        for q in targeted:
            if q not in existing:
                existing.append(q)
        state["rewritten_queries"] = existing[:6]
        logger.info(f"[AgenticRAG 2.0] 定向补检查询: {targeted}")
        return state
