"""
本地调试入口：逐步骤模拟用户查询经过整条 AI 服务链路

用法：
  python scripts/debug_query_pipeline.py [query] [--auto]

  不传参数则交互选择查询
  --auto 自动模式，跳过所有暂停

例：python scripts/debug_query_pipeline.py "查询配件 ENG-001 的库存"
    python scripts/debug_query_pipeline.py "公司年假政策是什么" --auto

流程：查询输入 → 安全过滤 → 路由决策 → 按路由分流

注意：低配环境友好，Milvus/ES/Redis 不可用时自动跳过并提示
"""

import sys
import os
import json
import logging
import time
import asyncio
import traceback

# src 布局：未安装包时也能直接跑（把 <repo>/src 放进导入路径）
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 屏蔽噪音日志
for _name in ("httpx", "urllib3", "elastic_transport", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger("run")


# ========================================================================
# 工具函数
# ========================================================================

def step(title: str):
    print()
    print("=" * 70)
    print(f"  【{title}】")
    print("=" * 70)


def step_detail(label: str, content):
    print(f"  {label}: ", end="")
    if isinstance(content, str):
        print(content)
    else:
        print(json.dumps(content, ensure_ascii=False, indent=2))
    print(f"  {'─' * 50}")


def pause(hint: str = "按 Enter 继续下一步，输入 q 退出..."):
    if "--auto" in sys.argv:
        print(f"  ⏸ {hint} (自动跳过)")
        return
    try:
        inp = input(f"  ⏸ {hint}")
        if inp.strip().lower() == "q":
            print("  ⛔ 用户中止")
            sys.exit(0)
    except EOFError:
        pass


def safe_step(label: str, fn, fallback=None):
    """安全执行一步，失败时跳过并返回 fallback"""
    step_detail(label, "进行中...")
    try:
        result = fn()
        return result
    except Exception as e:
        # 只取第一行错误，避免大段 traceback
        err_msg = str(e).split("\n")[0][:120]
        step_detail(label, f"⚠️ 跳过（环境不可用）: {err_msg}")
        return fallback


# ========================================================================
# 第 0 步：初始化（模拟 main.py 的初始化流程）
# ========================================================================

step("0. 系统初始化（模拟 main.py）")

# 0a. 全局状态
step_detail("0a. 初始化全局状态", "进行中...")
from agentic_rag.api.state import state
state.DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
os.makedirs(state.DATA_DIR, exist_ok=True)
step_detail("0a. 全局状态", f"DATA_DIR = {state.DATA_DIR}")

# 0b. Agent
state.agent = safe_step("0b. Agent", lambda: __import__("agent.agent", fromlist=["get_agent"]).get_agent())

# 0c. MCP Bridge
state.mcp_bridge = safe_step("0c. MCP Bridge", lambda: __import__("agent.mcp_bridge", fromlist=["mcp_tool_bridge"]).mcp_tool_bridge)

# 0d. Hybrid LLM
state.hybrid_llm = safe_step("0d. Hybrid LLM", lambda: __import__("utils.llm", fromlist=["hybrid_llm"]).hybrid_llm)

# 0e. Redis 缓存（可选）
state.redis_cache = safe_step("0e. Redis 缓存（可选）", lambda: __import__("utils.redis_cache", fromlist=["get_redis_cache"]).get_redis_cache())

# 0f. Agentic RAG 2.0（可选）
def _init_agentic_rag():
    from agentic_rag.agent.agentic_rag import AgenticRAGOrchestrator, AgentConfig
    return AgenticRAGOrchestrator(AgentConfig(enabled=True))
state.agentic_rag_orchestrator = safe_step("0f. Agentic RAG 2.0（可选）", _init_agentic_rag)

pause() # 等待用户确认


# ========================================================================
# 第 1 步：查询输入
# ========================================================================

step("1. 查询输入")

DEFAULT_QUERIES = [
    ("简单知识", "公司年假政策是什么"),
    ("Agent工具", "查询配件 ENG-001 的库存"),
    ("复杂知识", "刚入职五险一金怎么交"),
    ("模糊查询", "那个系统怎么用"),
    ("复合查询", "查库存并且看库存制度"),
    ("敏感词", "操你妈的"),
    ("含PII", "13800138000 的维修记录"),
]

print("  可选查询：")
for i, (label, q) in enumerate(DEFAULT_QUERIES, 1):
    print(f"    {i}. [{label}] {q}")

# 解析参数：去掉 --auto
args = [a for a in sys.argv[1:] if not a.startswith("--")]
query = " ".join(args) if args else ""
if not query:
    try:
        choice = input(f"\n  选择 (1-{len(DEFAULT_QUERIES)}, 默认 1): ").strip()
        idx = int(choice) - 1 if choice else 0
        query = DEFAULT_QUERIES[idx][1]
    except (ValueError, IndexError, EOFError):
        query = DEFAULT_QUERIES[0][1]

step_detail("用户输入", query)
pause() # 等待用户确认


# ========================================================================
# 第 2 步：安全过滤（必经）
# ========================================================================

step("2. 安全过滤")

from agentic_rag.utils.sensitive_filter import sensitive_filter
from agentic_rag.utils.prompt_injection import PromptInjectionDefense, InjectionRiskLevel

# 2a. 敏感词检查
step_detail("2a. 敏感词检查", "进行中...")
sensitive_result = sensitive_filter.check_sensitive(query)
if sensitive_result["is_sensitive"]:
    step_detail("2a. 敏感词检查 → 拦截",
                f"命中敏感词: {sensitive_result['sensitive_words']}")
    print("  ⛔ 流程终止：包含敏感内容")
    sys.exit(0)
step_detail("2a. 敏感词检查", "✅ 通过")

# 2b. 注入检测（工业级多层防御）
step_detail("2b. 注入检测（多层防御）", "进行中...")
prompt_defense = PromptInjectionDefense(enable_ml_classifier=False)

# 第1层：输入过滤
input_result = prompt_defense.detect_input(query)
if not input_result.is_safe:
    step_detail("2b. 注入检测 → 拦截",
                f"第1层拦截 | 风险等级: {input_result.risk_level.value} | "
                f"评分: {input_result.score:.2f} | "
                f"检测模式: {len(input_result.detected_patterns)} 个")
    print("  ⛔ 流程终止：检测到提示注入攻击")
    print(f"  💡 建议: {input_result.suggestion}")
    sys.exit(0)
step_detail("2b. 注入检测（第1层-输入过滤）", 
            f"✅ 通过 (风险: {input_result.risk_level.value}, 评分: {input_result.score:.2f})")

# 2c. PII 掩码
step_detail("2c. PII 掩码", "进行中...")
masked_query, mask_result = sensitive_filter.mask_pii_with_mapping(query)
if len(mask_result.pii_items) > 0:
    step_detail("2c. PII 掩码",
                f"检测到 {len(mask_result.pii_items)} 个PII → 已掩码为 {masked_query}")
else:
    step_detail("2c. PII 掩码", "✅ 无PII")
    mask_result = None

pause()


# ========================================================================
# 第 3 步：路由决策
# ========================================================================

step("3. 路由决策")

from agentic_rag.rag.llm_router import llm_router
from agentic_rag.utils.model_router import select_model, select_model_or_moa

# 3a. 三层路由（含多意图检测 — route_query 内部已处理）
route = safe_step("3a. 三层路由（含多意图检测）",
    lambda: llm_router.route_query(masked_query), fallback="rag")
if route == "multi":
    step_detail("3a. 三层路由 → 复合查询", "route=multi, 建议并行走 RAG + Agent")
step_detail("3a. 三层路由 → 结果", f"route={route}")

# 3c. 模型路由（含 MoA 策略）
model_info = safe_step("3c. 模型路由（含 MoA）",
    lambda: select_model_or_moa(route, "default", "general"),
    fallback={"mode": "single", "model": "unknown", "reason": "模型路由跳过"})
if model_info.get("mode") == "moa":
    config = model_info["config"]
    step_detail("3c. 模型路由 → MoA",
                f"proposers={config.proposers}, aggregator={config.aggregator}, "
                f"strategy={config.strategy} ({model_info['reason']})")
else:
    step_detail("3c. 模型路由 → 单模型",
                f"model={model_info.get('model', 'unknown')} ({model_info.get('reason', '')})")

pause()


# ========================================================================
# 第 4 步：按路由结果分流
# ========================================================================


async def run_rag_pipeline(query_text: str):
    """RAG 路径完整流程"""
    print()
    step("4. RAG 路径")

    # 4a. 查询分类
    qtype = safe_step("4a. 查询分类",
        lambda: __import__("rag.query_classifier", fromlist=["query_classifier"]).query_classifier.classify(query_text),
        fallback="unknown")
    step_detail("4a. 查询分类 → 结果", f"type={qtype}")
    pause()

    # 4b. 查询扩展
    def _rewrite():
        from agentic_rag.rag.query_rewriter import query_rewriter
        expanded = query_rewriter.rewrite(query_text)
        return expanded
    expanded = safe_step("4b. 查询扩展", _rewrite, fallback=None)
    if expanded and expanded != query_text:
        step_detail("4b. 查询扩展 → 结果", f"扩展: {query_text} → {expanded}")
    else:
        step_detail("4b. 查询扩展", "✅ 不扩展 (simple 或跳过)")
    pause()

    # 4c. 混合检索（依赖 Milvus + ES，低配环境大概率不可用）
    docs = []
    def _retrieve():
        from agentic_rag.rag.retriever import hybrid_retriever
        return hybrid_retriever.retrieve(query_text)
    retrieval_start = time.time()
    docs = safe_step("4c. 混合检索", _retrieve, fallback=[])
    elapsed = time.time() - retrieval_start
    if docs:
        step_detail("4c. 混合检索 → 结果",
                    f"检索到 {len(docs)} 条文档 (耗时 {elapsed:.2f}s)")
        for i, d in enumerate(docs[:3], 1):
            content_preview = d.get("content", "")[:80]
            source = d.get("metadata", {}).get("source", "unknown")
            step_detail(f"  doc[{i}]", f"[{source}] {content_preview}...")
        if len(docs) > 3:
            step_detail(f"  ...", f"还有 {len(docs)-3} 条")
    else:
        step_detail("4c. 混合检索", "⚠️ 无检索结果（Milvus/ES 未启动，使用空上下文）")
    pause()

    # 4c-2. 第2层防御：检索结果过滤（防御间接注入）
    if docs:
        step_detail("4c-2. 检索结果过滤（第2层防御）", "进行中...")
        doc_contents = [d.get("content", "") for d in docs]
        safe_docs, context_result = prompt_defense.filter_safe_documents(doc_contents)
        if not context_result.is_safe:
            step_detail("4c-2. 检索过滤 → 拦截",
                        f"拦截 {len(context_result.blocked_documents)} 个可疑文档 | "
                        f"风险: {context_result.risk_level.value}")
            # 更新 docs，移除被污染的文档
            docs = [d for idx, d in enumerate(docs) if idx not in context_result.blocked_documents]
            step_detail("  安全文档", f"剩余 {len(docs)} 条可用文档")
        else:
            step_detail("4c-2. 检索结果过滤", 
                        f"✅ 通过 (检测 {len(doc_contents)} 个文档，全部安全)")
    pause()

    # 4d. CRAG（依赖 LLM API + 检索结果）
    answer = ""
    crag_result = {}
    async def _run_crag():
        from agentic_rag.rag.crag import run_crag
        return await run_crag(
            query=query_text,
            domain="general",
            use_finetuned=False,
            history=None,
        )
    try:
        step_detail("4d. CRAG 评估", "进行中...")
        crag_result = await _run_crag()
        grade = crag_result.get("relevance_grade", "unknown")
        rewrite_count = crag_result.get("rewrite_count", 0)
        answer = crag_result.get("answer", "")
        moa_judges = crag_result.get("chain_log", [])
        # 检查是否有 MoA 投票信息
        grade_logs = [l for l in moa_judges if l.get("step") == "grade"]
        if grade_logs and grade_logs[0].get("details", {}).get("moa_judges"):
            step_detail("4d. CRAG → MoA 评估",
                        f"grade={grade}, MoA Judges={grade_logs[0]['details']['moa_judges']}")
        else:
            step_detail("4d. CRAG → 结果", f"grade={grade}, 改写次数={rewrite_count}")
        step_detail("  answer 预览", answer[:200])
        
        # 4d-2. 第3层防御：输出净化（防止敏感信息泄露）
        if answer:
            step_detail("4d-2. 输出净化（第3层防御）", "进行中...")
            sanitized_answer, output_result = prompt_defense.sanitize_output(answer)
            if not output_result.is_safe:
                step_detail("4d-2. 输出净化 → 检测到敏感信息",
                            f"风险: {output_result.risk_level.value} | "
                            f"检测模式: {len(output_result.detected_patterns)} 个")
                # 使用净化后的回答
                answer = sanitized_answer
                crag_result["answer"] = sanitized_answer
                step_detail("  净化后预览", sanitized_answer[:200])
            else:
                step_detail("4d-2. 输出净化", "✅ 通过（无敏感信息泄露）")
    except Exception as e:
        err_msg = str(e).split("\n")[0][:120]
        step_detail("4d. CRAG", f"⚠️ 跳过（LLM/检索不可用）: {err_msg}")
    pause()

    # 4d-3. MoA 独立测试（可选，仅当查询类型为 complex 时）
    if qtype == "complex":
        step_detail("4d-3. MoA 独立生成测试", "进行中...")
        try:
            from agent_harness.providers.moa import MixtureOfAgents, MoAConfig
            from agent_harness.providers.doubao import hybrid_llm
            moa_config = MoAConfig(
                proposers=["deepseek", "qwen"],
                aggregator="deepseek",
                strategy="aggregate",
            )
            moa = MixtureOfAgents(hybrid_llm)
            context_text = "（低配环境无检索结果，使用空上下文测试 MoA 流程）"
            if docs:
                context_text = "\n".join([d.get("content", "")[:200] for d in docs[:3]])
            moa_result = await moa.generate(query_text, context_text, moa_config)
            step_detail("4d-3. MoA → 结果",
                        f"strategy={moa_result.strategy}, "
                        f"proposers={len(moa_result.proposer_answers)}, "
                        f"耗时={moa_result.elapsed_seconds}s")
            for i, pa in enumerate(moa_result.proposer_answers, 1):
                preview = pa.get("answer", "")[:80] or "(空)"
                step_detail(f"  proposer[{i}] ({pa.get('model', '?')})",
                            preview)
            step_detail("  aggregator → 最终回答",
                        moa_result.final_answer[:150])
        except Exception as e:
            err_msg = str(e).split("\n")[0][:120]
            step_detail("4d-3. MoA", f"⚠️ 跳过: {err_msg}")
        pause()

    # 4e. NLI 校验
    if answer and docs:
        def _nli():
            from agentic_rag.utils.nli_checker import nli_checker
            contexts = [d.get("content", "") for d in docs[:3]]
            return nli_checker.check_consistency(answer, contexts)
        nli_result = safe_step("4e. NLI 校验", _nli, fallback=None)
        if nli_result:
            if nli_result.get("is_consistent", True):
                step_detail("4e. NLI 校验", "✅ 通过")
            else:
                step_detail("4e. NLI 校验 → 失败",
                            f"冲突概率: {nli_result.get('contradiction_prob', 0):.2f}")
    else:
        step_detail("4e. NLI 校验", "⚠️ 跳过（无 answer 或无检索结果）")
    pause()

    # 4f. PII 还原
    if mask_result and len(mask_result.pii_items) > 0 and answer:
        def _restore():
            return sensitive_filter.restore_pii(answer, mask_result.request_id)
        restored = safe_step("4f. PII 还原", _restore, fallback=None)
        if restored:
            step_detail("4f. PII 还原", f"✅ 已还原: {restored[:100]}...")
        else:
            step_detail("4f. PII 还原", "✅ 无需还原 (无PII占位符)")
    else:
        step_detail("4f. PII 还原", "✅ 无需还原 (原始无PII 或无 answer)")

    print()
    print("=" * 70)
    print("  ✅ RAG 路径完成")
    print(f"  最终回答: {answer or '(无，环境不可用)'}")
    print("=" * 70)
    return answer


async def run_agent_pipeline(query_text: str):
    """Agent 路径完整流程（含 Agentic RAG 2.0 多智能体协作）"""
    print()
    step("4. Agent 路径")

    # 4a. SupervisorAgent 内部路由
    def _supervisor_route():
        from agentic_rag.agent.agentic_rag import SupervisorAgent
        supervisor = SupervisorAgent()
        return supervisor._initial_route(query_text)
    route_decision = safe_step("4a. SupervisorAgent 内部路由", _supervisor_route, fallback="unknown")
    
    step_detail("4a. 内部路由 → 结果", f"route={route_decision}")
    pause()

    # 4b. 根据路由分流
    #   retrieve/plan/react → Agentic RAG 2.0 多智能体协作（复用 CRAG 完整链路）
    #   tool_call → Function Calling 模式（RAGAgent + MCP Bridge）
    result = {"success": False, "message": "未执行"}

    if route_decision in ("retrieve", "plan", "react"):
        
        step_detail("4b. Agentic RAG 2.0 执行", "进行中...")# Agentic RAG 2.0 多智能体协作模式

        if state.agentic_rag_orchestrator is None:
            step_detail("4b. Agentic RAG 2.0", "⚠️ orchestrator 未初始化，跳过")
        else:
            try:
                # agentic_result 
                # 调用 AgenticRAGOrchestrator
                agentic_result = await state.agentic_rag_orchestrator.run(
                    query_text, use_crag=True
                )

                # 返回 路由 迭代次数 置信度
                step_detail("4b. Agentic RAG 2.0 → 结果",
                            f"route={agentic_result.get('route_decision', '')}, "
                            f"iterations={agentic_result.get('iteration_count', 0)}, "
                            f"confidence={agentic_result.get('confidence', 0):.2f}")
                
                # 返回 答案 检索引用 智能体轨迹
                answer = agentic_result.get("answer", "")
                step_detail("  answer 预览", answer[:200])
                step_detail("  citations 数", len(agentic_result.get("citations", [])))
                step_detail("  agent_trace 数", len(agentic_result.get("agent_trace", [])))
                
                result = {
                    "success": True,
                    "data": {"response": answer, "type": "agentic_rag_2.0"},
                    "message": f"Agentic RAG 2.0 处理成功（route={agentic_result.get('route_decision', '')}）"
                }
            except Exception as e:
                err_msg = str(e).split("\n")[0][:120]
                step_detail("4b. Agentic RAG 2.0", f"⚠️ 跳过: {err_msg}")

    else:
        # tool_call → Function Calling 模式
        step_detail("4b. Function Calling 执行", "进行中...")
        def _run_agent():
            from agentic_rag.agent.agent import rag_agent
            return rag_agent.run_agent(query_text)
        try:
            result = _run_agent()
            if result.get("success"):
                response = result["data"]["response"]
                step_detail("4b. Function Calling → 结果", response[:200])
            else:
                step_detail("4b. Function Calling → 失败", result.get("message", "未知错误"))
        except Exception as e:
            err_msg = str(e).split("\n")[0][:120]
            step_detail("4b. Function Calling", f"⚠️ 跳过（LLM 不可用）: {err_msg}")

    print()
    print("=" * 70)
    print("  ✅ Agent 路径完成")
    resp = result.get("data", {}).get("response", "(无，环境不可用)")
    print(f"  最终回答: {resp[:200]}")
    print("=" * 70)
    return result


async def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  agentic-rag 调试运行器".center(60) + "║")
    print("║" + f"  查询: {query}".center(60) + "║")
    print("╚" + "═" * 68 + "╝")

    if route == "rag": # 运行 RAG 流程
        await run_rag_pipeline(masked_query)
    elif route == "agent":
        await run_agent_pipeline(masked_query)
    elif route == "multi":
        print()
        print("  📌 多意图复合查询，分别运行 RAG 和 Agent")
        await run_rag_pipeline(masked_query)
        await run_agent_pipeline(masked_query)
    else:
        print(f"  未知路由: {route}")


if __name__ == "__main__":
    asyncio.run(main())
    # RAG 流程里有 await run_crag() ，所以外层main函数必须是 async 
    # 最终需要 asyncio.run() 来启动。

# python.exe run.py "公司年假政策是什么" --auto 2>&1