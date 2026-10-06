"""
CRAG (Corrective RAG) 模块 - 工业级升级版
基于 LangGraph StateGraph 实现检索-评估-重试循环

⚠️ 低配版说明：
   架构图 handle_no_info 节点标注为 MCP 协议联网搜索。
   实际使用 web_search.py 通过 SerpAPI / Bing Search REST API
   实现联网搜索，未接入 MCP 协议层。在接入 MCP Server 之前，
   直接用 REST API 是更低成本的原型方案。

升级要点（v2.0）：
1. 四级决策：relevant(知识提炼) / partial(改写重试) / irrelevant(改写重试) / contradictory(标记冲突)
2. 矛盾文档处理：冲突事实分别罗列，不强行统一，标注来源
3. 知识提炼：relevant时剥离噪声、重组上下文，提升生成质量
4. 动态阈值：根据领域和置信度调整评估阈值
5. 全链路日志：记录 query→扩展→检索→评估→生成 完整链路
6. 降级支持：LLM Judge不可用时回退规则打分
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from agentic_rag.rag.retriever import hybrid_retriever
from agentic_rag.rag.rewrite_governor import rewrite_governor
from agent_harness.observability.tracing import get_current_span, record_attributes, traced
from agentic_rag.config import settings
from agent_harness.providers.doubao import DoubaoLLMService, hybrid_llm
from agent_harness.providers.deepseek import DeepSeekLLMService
from agentic_rag.utils.sensitive_filter import sensitive_filter
from agentic_rag.utils.web_search import web_search_service  # ⚠️ 低配版：架构图标 MCP，实际 REST API
from agent_harness.observability.monitor import rag_monitor
from agentic_rag.utils.tenant_filter import tenant_permission_filter, TenantContext

try:
    from agent_harness.context import LayeredContextManager, ContextConfig, ContextPhase
    _has_context_engine = True
except ImportError:
    _has_context_engine = False
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("context_engine 模块不可用，CRAG 将使用原始上下文构建")

logger = logging.getLogger(__name__)

MAX_REWRITES = 2

DOMAIN_THRESHOLDS = {
    "hr": {"relevant": 0.7, "partial": 0.4},
    "finance": {"relevant": 0.8, "partial": 0.5},
    "it": {"relevant": 0.6, "partial": 0.3},
    "general": {"relevant": 0.6, "partial": 0.3},
}


class CragState(TypedDict):
    query: str
    original_query: str
    documents: List[Dict]
    relevance_grade: str
    relevance_score: float
    rewrite_count: int
    answer: str
    citations: List[Dict]
    domain: str
    use_finetuned: bool
    history: Optional[List[Dict]]
    web_search_results: Optional[Dict]
    chain_log: List[Dict]
    start_time: float
    contradictions: List[Dict]
    request_id: str


def _log_chain(chain_log: List[Dict], step: str, data: Dict) -> List[Dict]:
    # 记录链路日志
    entry = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    chain_log.append(entry)
    return chain_log


@traced("crag.retrieve", attributes_fn=lambda state: {"crag.query": str(state.get("query", ""))[:60]})
def retrieve(state: CragState) -> Dict:
    # 接收一个类型为 `CragState` 的参数 state，标注该函数应当返回字典类型
    query = state["query"]
    chain_log = state.get("chain_log", []) # 从状态中获取链路日志，若不存在则创建空列表
    _log_chain(chain_log, "retrieve_start", {"query": query}) # 记录检索开始时间

    try:
        # 构建租户过滤上下文
        tenant_ctx = TenantContext(
            tenant_id="default",
            user_id="system",
            roles=["admin"]
        )
        # 从 state 中尝试读取 tenant 信息
        if "tenant_context" in state:
            tc = state["tenant_context"]
            tenant_ctx = TenantContext(
                tenant_id=tc.get("tenant_id", "default"),
                user_id=tc.get("user_id", "system"),
                roles=tc.get("roles", ["admin"]),
                security_level=tc.get("security_level", "public"),
                allowed_document_ids=tc.get("allowed_document_ids")
            )
        elif "user_id" in state:
            tenant_ctx = TenantContext(
                tenant_id="default",
                user_id=state["user_id"],
                roles=["admin"]
            )
        filter_expr = tenant_permission_filter.build_milvus_filter(tenant_ctx)
        results = hybrid_retriever.retrieve(query=query, filter_dict=filter_expr) # 调用检索器
        logger.info(f"[CRAG] 检索到 {len(results)} 条结果（租户过滤：{tenant_ctx.tenant_id}）")
        _log_chain(chain_log, "retrieve_end", {"result_count": len(results)}) # 记录检索结束时间
    except Exception as e:
        logger.error(f"[CRAG] 检索失败：{e}")
        results = []
        _log_chain(chain_log, "retrieve_error", {"error": str(e)}) # 记录检索错误

    return {"documents": results, "chain_log": chain_log}


@traced("crag.grade_documents", attributes_fn=lambda state: {"crag.doc_count": len(state.get("documents", []))})
def grade_documents(state: CragState) -> Dict:
    """
    评估节点：LLM Judge 四级相关性判定

    四级决策（面试核心）：
    - relevant：完全相关 → 知识提炼
    - partial：部分相关 → 改写Query重走检索（最大2次）
    - irrelevant：不相关 → 改写Query重走检索
    - contradictory：矛盾 → 标记冲突，传递至生成层特殊处理
    """
    query = state["query"]
    documents = state["documents"]
    domain = state.get("domain", "general")
    chain_log = state.get("chain_log", [])

    if not documents:
        logger.info("[CRAG] 无检索结果，判定为 no_info")
        _log_chain(chain_log, "grade", {"grade": "no_info", "score": 0.0})
        return {"relevance_grade": "no_info", "relevance_score": 0.0, "chain_log": chain_log}

    logger.info(f"[CRAG] 评估节点：评估 {len(documents)} 条文档与查询的相关性")

    llm = DeepSeekLLMService()

    doc_summaries = []
    for i, doc in enumerate(documents[:10], 1):
        content = doc.get("content", "")[:200]
        source = doc.get("metadata", {}).get("source", f"文档{doc.get('document_id', i)}")
        doc_summaries.append(f"[{i}] 来源:{source} | {content}...")

    docs_text = "\n".join(doc_summaries)

    grade_prompt = f"""你是一个文档相关性评估专家。请判断以下检索到的文档片段是否与用户问题相关，并检测是否存在矛盾信息。

用户问题：{query}

检索到的文档片段：
{docs_text}

请按以下格式输出（严格JSON）：
{{
  "score": 0.0-1.0的评分,
  "grade": "relevant/partial/irrelevant/contradictory",
  "contradictions": [],
  "reason": "简短说明"
}}

评分标准：
- 1.0=完全相关，0.7+=相关(relevant)
- 0.4-0.7=部分相关(partial)
- <0.4=不相关(irrelevant)
- contradictory=文档间存在事实矛盾

contradictions格式（仅当grade=contradictory时填写）：
[{{"fact": "矛盾的事实点", "source_a": "来源A描述", "source_b": "来源B描述"}}]

只输出JSON，不要其他内容。"""

    messages = [
        {"role": "system", "content": "你是文档相关性评估专家，输出严格JSON格式的评估结果。"},
        {"role": "user", "content": grade_prompt}
    ]

    try:
        # ⚠️ 低配版 MoA：2 个 Judge 投票（DeepSeek + Qwen），工业级通常 3-5 个
        judge_grades = []
        judge_models = [llm]  # 第 1 个 Judge：DeepSeek
        try:
            from agent_harness.providers.qwen import QwenLLMService
            judge_models.append(QwenLLMService())
        except Exception:
            pass  # Qwen 不可用时只用 DeepSeek

        for judge_llm in judge_models:
            try:
                response = judge_llm._call_llm(messages)
                grade_text = response.get("text", "").strip()

                import re
                import json

                json_match = re.search(r'\{[\s\S]*\}', grade_text)
                if json_match:
                    grade_result = json.loads(json_match.group())
                    score = float(grade_result.get("score", 0.5))
                    grade = grade_result.get("grade", "partial")
                    contradictions = grade_result.get("contradictions", [])
                    reason = grade_result.get("reason", "")
                else:
                    numbers = re.findall(r'[0-9]*\.?[0-9]+', grade_text)
                    score = float(numbers[0]) if numbers else 0.5
                    grade = "partial"
                    contradictions = []
                    reason = ""

                score = max(0.0, min(1.0, score))
                if grade not in ["relevant", "partial", "irrelevant", "contradictory"]:
                    thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["general"])
                    if score >= thresholds["relevant"]:
                        grade = "relevant"
                    elif score >= thresholds["partial"]:
                        grade = "partial"
                    else:
                        grade = "irrelevant"

                judge_grades.append({
                    "grade": grade, "score": score,
                    "contradictions": contradictions, "reason": reason,
                })
            except Exception as e:
                logger.warning(f"[CRAG] Judge 评估失败: {e}")

        # 多 Judge 投票共识
        if len(judge_grades) > 1:
            from agent_harness.providers.moa import vote_on_grades
            final = vote_on_grades(judge_grades)
            grade = final["grade"]
            score = final["score"]
            contradictions = final.get("contradictions", [])
            reason = f"MoA 投票: {final.get('vote_details', {})}"
            logger.info(f"[CRAG] MoA 多 Judge 投票: {final.get('vote_details', {})} → {grade} (avg_score={score:.2f})")
        elif judge_grades:
            grade = judge_grades[0]["grade"]
            score = judge_grades[0]["score"]
            contradictions = judge_grades[0]["contradictions"]
            reason = judge_grades[0].get("reason", "")
        else:
            # 所有 Judge 均失败，回退规则打分
            raise Exception("所有 Judge 评估失败")

        logger.info(f"[CRAG] 评估结果：{grade} (评分={score:.2f}, 矛盾={len(contradictions)}, 原因={reason})")
        _log_chain(chain_log, "grade", {
            "grade": grade, "score": score,
            "contradictions": len(contradictions),
            "reason": reason,
            "moa_judges": len(judge_grades),
        })

        return {
            "relevance_grade": grade,
            "relevance_score": score,
            "contradictions": contradictions,
            "chain_log": chain_log
        }

    except Exception as e:
        logger.error(f"[CRAG] 评估失败：{e}，回退规则打分")
        grade, score = _rule_based_grade(query, documents, domain)
        _log_chain(chain_log, "grade_fallback", {"grade": grade, "score": score, "error": str(e)})

        return {
            "relevance_grade": grade,
            "relevance_score": score,
            "contradictions": [],
            "chain_log": chain_log
        }


def _rule_based_grade(query: str, documents: List[Dict], domain: str) -> tuple:
    """降级规则打分：LLM Judge不可用时回退"""
    if not documents:
        return "irrelevant", 0.0

    query_chars = set(query)
    scores = []
    for doc in documents[:5]:
        content = doc.get("content", "")
        content_chars = set(content)
        overlap = len(query_chars & content_chars) / max(len(query_chars), 1)
        rerank_score = float(doc.get("rerank_score", doc.get("combined_score", 0)))
        combined = overlap * 0.3 + rerank_score * 0.7
        scores.append(combined)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["general"])

    if avg_score >= thresholds["relevant"]:
        return "relevant", avg_score
    elif avg_score >= thresholds["partial"]:
        return "partial", avg_score
    else:
        return "irrelevant", avg_score


def detect_contradictions(documents: List[Dict], query: str) -> List[Dict]:
    """检测文档间矛盾（轻量规则版，不依赖LLM）"""
    contradictions = []
    if len(documents) < 2:
        return contradictions

    contradiction_keywords = [
        ("是", "不是"), ("有", "没有"), ("可以", "不可以"),
        ("允许", "禁止"), ("需要", "不需要"), ("包含", "不包含"),
        ("支持", "不支持"), ("必须", "不必"),
    ]

    contents = [doc.get("content", "") for doc in documents[:5]]

    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            for affirm, negate in contradiction_keywords:
                if affirm in contents[i] and negate in contents[j]:
                    if any(kw in contents[i] and kw in contents[j] for kw in query):
                        contradictions.append({
                            "fact": f"关于'{query}'的矛盾",
                            "source_a": contents[i][:100],
                            "source_b": contents[j][:100],
                            "type": "negation_conflict"
                        })
                        break

    return contradictions[:3]


@traced("crag.refine_knowledge")
def refine_knowledge(state: CragState) -> Dict:
    query = state["original_query"]
    documents = state["documents"]
    chain_log = state.get("chain_log", [])

    logger.info(f"[CRAG] 知识提炼：对 {len(documents)} 条文档进行噪声过滤和上下文重组")

    refined_docs = []
    for doc in documents:
        content = doc.get("content", "")
        score = doc.get("rerank_score", doc.get("combined_score", 0))

        if score and float(score) < 0.3:
            continue

        if len(content) > 500:
            content = _extract_relevant_sentences(query, content)

        refined_doc = dict(doc)
        refined_doc["content"] = content
        refined_docs.append(refined_doc)

    if not refined_docs:
        refined_docs = documents[:3]

    logger.info(f"[CRAG] 知识提炼完成：{len(documents)} → {len(refined_docs)} 条")
    _log_chain(chain_log, "refine", {
        "original_count": len(documents),
        "refined_count": len(refined_docs)
    })

    return {"documents": refined_docs, "chain_log": chain_log}


def _extract_relevant_sentences(query: str, content: str, max_length: int = 400) -> str:
    sentences = content.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]

    query_chars = set(query)
    scored_sentences = []
    for sent in sentences:
        overlap = len(set(sent) & query_chars)
        scored_sentences.append((sent, overlap))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    result = []
    total_len = 0
    for sent, _ in scored_sentences:
        if total_len + len(sent) > max_length:
            break
        result.append(sent)
        total_len += len(sent)

    return "".join(result) if result else content[:max_length]


@traced("crag.rewrite_query", attributes_fn=lambda state: {"crag.rewrite_count": state.get("rewrite_count", 0)})
def rewrite_query(state: CragState) -> Dict:
    query = state["query"]
    original_query = state["original_query"]
    rewrite_count = state.get("rewrite_count", 0)
    chain_log = state.get("chain_log", [])

    logger.info(f"[CRAG] 改写节点：第 {rewrite_count + 1} 次改写，原查询={query}")

    llm = DeepSeekLLMService()

    rewrite_prompt = f"""用户提出了一个问题，但检索系统没有找到相关的文档。请改写这个问题，使其更容易被检索系统理解。

原始问题：{original_query}
当前查询：{query}

改写要求：
1. 换一种表述方式，使用同义词或更具体的描述
2. 补充可能缺失的上下文信息
3. 只输出改写后的问题，不要解释
4. 改写后的问题不超过50字"""

    messages = [
        {"role": "system", "content": "你是一个查询改写专家，擅长将问题转化为更易检索的形式。"},
        {"role": "user", "content": rewrite_prompt}
    ]

    try:
        response = llm._call_llm(messages)
        rewritten = response.get("text", "").strip()

        if not rewritten:
            rewritten = original_query

        # 改写护栏：意图偏移过大 / 改写原地打转 → 回退原查询并终止改写循环
        final_query, guard = rewrite_governor.guard_rewrite(
            original_query, query, rewritten)
        _log_chain(chain_log, "rewrite", {
            "from": query, "to": final_query, "count": rewrite_count + 1,
            "action": guard["action"], "intent_drift": guard.get("intent_drift"),
            "reason": guard.get("reason", "")
        })
        if guard["action"] == "fallback_original":
            _log_chain(chain_log, "rewrite_fallback", {
                "to": original_query, "reason": guard.get("reason", "")
            })
    except Exception as e:
        logger.error(f"[CRAG] 查询改写失败：{e}，使用原查询")
        final_query = original_query
        _log_chain(chain_log, "rewrite_error", {"error": str(e)})

    return {"query": final_query, "rewrite_count": rewrite_count + 1, "chain_log": chain_log}


def generate_with_contradictions(state: CragState) -> Dict:
    """
    矛盾文档特殊生成：冲突事实分别罗列，不强行统一

    面试核心：
    1. 冲突事实分别罗列，不强行统一
    2. 做可信度加权/来源优先级（官方>行业>自媒体，时间新>旧）
    3. 触发质疑式回答，不给出确定结论
    """
    query = state["original_query"]
    documents = state["documents"]
    contradictions = state.get("contradictions", [])
    domain = state.get("domain", "general")
    use_finetuned = state.get("use_finetuned", False)
    history = state.get("history")
    chain_log = state.get("chain_log", [])

    logger.info(f"[CRAG] 矛盾文档生成：{len(contradictions)} 处冲突")

    context_parts = []
    citations = []

    for idx, result in enumerate(documents, 1):
        content = result.get("content", "")
        context_parts.append(f"[{idx}] {content}")

        metadata = result.get('metadata', {})
        citations.append({
            "document_id": result.get('document_id', 0),
            "chunk_index": result.get('chunk_index', 0),
            "section": metadata.get('section', ''),
            "page_number": metadata.get('page_number', ''),
            "score": result.get('rerank_score', result.get('combined_score', 0)),
            "rank": idx
        })

    context = "\n\n".join(context_parts)

    contradiction_text = ""
    if contradictions:
        contradiction_text = "\n\n⚠️ 检测到以下信息冲突：\n"
        for i, c in enumerate(contradictions, 1):
            contradiction_text += f"冲突{i}：{c.get('fact', '')}\n"
            contradiction_text += f"  来源A：{c.get('source_a', '')}\n"
            contradiction_text += f"  来源B：{c.get('source_b', '')}\n"
        contradiction_text += "\n请按以下原则处理冲突：\n"
        contradiction_text += "1. 冲突事实分别罗列，不强行统一\n"
        contradiction_text += "2. 标注各来源，让用户自行判断\n"
        contradiction_text += "3. 禁止直接输出'是/否'，改为'现有资料存在矛盾，不同来源观点如下'\n"

    contradiction_prompt = f"""基于以下检索到的文档回答用户问题。

{context}

{contradiction_text}

用户问题：{query}

回答要求：
1. 如果存在矛盾信息，必须标注冲突点和来源
2. 不对矛盾信息强行统一结论
3. 优先采信官方文档、时间更新的来源
4. 格式示例：
   现有检索信息存在冲突：
   来源1：xxx
   来源2：xxx
   暂无统一结论，供参考。"""

    try:
        llm_response = hybrid_llm.generate_answer(
            query=query,
            context=contradiction_prompt,
            history=history,
            domain=domain,
            use_finetuned=use_finetuned
        )

        answer = llm_response.get('answer', '')
        filtered_answer = sensitive_filter.filter_sensitive(answer)

        _log_chain(chain_log, "generate_contradictory", {
            "model": llm_response.get('model', 'unknown'),
            "context_docs": len(documents),
            "contradictions": len(contradictions),
            "answer_length": len(filtered_answer)
        })
    except Exception as e:
        logger.error(f"[CRAG] 矛盾文档生成失败：{e}")
        filtered_answer = "系统繁忙，请稍后重试"
        citations = []
        _log_chain(chain_log, "generate_error", {"error": str(e)})

    return {"answer": filtered_answer, "citations": citations, "chain_log": chain_log}


def _build_context_with_ce(query: str, documents: List[Dict], history: Optional[List[Dict]],
                            domain: str, chain_log: List[Dict]) -> str:
    """使用 Context Engineering 构建分层上下文，降级回退原始拼接"""
    if not _has_context_engine:
        context_parts = []
        for idx, result in enumerate(documents, 1):
            content = result.get("content", "")
            context_parts.append(f"[{idx}] {content}")
        return "\n\n".join(context_parts)

    try:
        ctx_manager = LayeredContextManager(ContextConfig())

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                ctx_manager.update_conversation(role, content)

        system_prompt = f"你是{domain}领域的专业问答助手。基于检索到的文档准确回答用户问题，标注来源。"

        context = ctx_manager.build_context(
            query=query,
            system_prompt=system_prompt,
            retrieved_docs=documents,
            tool_outputs=[],
            phase=ContextPhase.ANSWERING,
        )

        stats = ctx_manager.get_stats()
        _log_chain(chain_log, "context_engineering", {
            "total_tokens": stats.get("total_tokens", 0),
            "phase": "answering",
            "layers": stats.get("layers", {}),
        })

        return context
    except Exception as e:
        logger.warning(f"[CRAG] Context Engineering 构建失败，回退原始拼接: {e}")
        context_parts = []
        for idx, result in enumerate(documents, 1):
            content = result.get("content", "")
            context_parts.append(f"[{idx}] {content}")
        return "\n\n".join(context_parts)


@traced("crag.generate", attributes_fn=lambda state: {"crag.doc_count": len(state.get("documents", []))})
def generate(state: CragState) -> Dict:
    query = state["original_query"]
    documents = state["documents"]
    domain = state.get("domain", "general")
    use_finetuned = state.get("use_finetuned", False)
    history = state.get("history")
    chain_log = state.get("chain_log", [])

    logger.info(f"[CRAG] 生成节点：基于 {len(documents)} 条文档生成回答")

    citations = []
    for idx, result in enumerate(documents, 1):
        metadata = result.get('metadata', {})
        citations.append({
            "document_id": result.get('document_id', 0),
            "chunk_index": result.get('chunk_index', 0),
            "section": metadata.get('section', ''),
            "page_number": metadata.get('page_number', ''),
            "score": result.get('rerank_score', result.get('combined_score', 0)),
            "rank": idx
        })

    context = _build_context_with_ce(query, documents, history, domain, chain_log)

    try:
        # 根据查询类型决定是否使用 MoA（多模型协作）
        # ⚠️ MoA 仅当配置 ENABLE_MOA=True 且查询为 complex 时才启用，避免默认依赖多模型/外网
        query_type = state.get("query_type", "simple")
        enable_moa = bool(getattr(settings, "ENABLE_MOA", False))
        if query_type == "complex" and enable_moa:
            from agentic_rag.utils.model_router import select_model_or_moa
            moa_result = select_model_or_moa("rag", query_type, domain)
            if moa_result.get("mode") == "moa":
                logger.info(f"[CRAG] MoA 模式生成：{moa_result['reason']}")
                llm_response = hybrid_llm.generate_with_moa(
                    query=query,
                    context=context,
                    history=history,
                    domain=domain,
                    moa_config=moa_result["config"],
                )
            else:
                llm_response = hybrid_llm.generate_answer(
                    query=query, context=context, history=history,
                    domain=domain, use_finetuned=use_finetuned,
                )
        else:
            llm_response = hybrid_llm.generate_answer(
                query=query,
                context=context,
                history=history,
                domain=domain,
                use_finetuned=use_finetuned
            )

        answer = llm_response.get('answer', '')
        filtered_answer = sensitive_filter.filter_sensitive(answer)

        _log_chain(chain_log, "generate", {
            "model": llm_response.get('model', 'unknown'),
            "context_docs": len(documents),
            "answer_length": len(filtered_answer),
            "context_engineering": _has_context_engine
        })
    except Exception as e:
        logger.error(f"[CRAG] 生成失败：{e}")
        filtered_answer = "系统繁忙，请稍后重试"
        citations = []
        _log_chain(chain_log, "generate_error", {"error": str(e)})

    return {"answer": filtered_answer, "citations": citations, "chain_log": chain_log}


def decide_to_generate(state: CragState) -> str:
    """
    条件边：四级决策（面试核心）

    - relevant → 知识提炼后生成
    - partial → 改写Query重走检索（最大2次）
    - irrelevant → 改写Query重走检索
    - contradictory → 标记冲突，传递至生成层特殊处理
    - no_info → 联网搜索fallback
    """
    grade = state.get("relevance_grade", "partial")
    rewrite_count = state.get("rewrite_count", 0)

    if grade == "relevant":
        logger.info("[CRAG] 决策：文档相关 → 知识提炼后生成")
        return "refine"

    if grade == "contradictory":
        logger.info("[CRAG] 决策：文档矛盾 → 标记冲突，传递至生成层特殊处理")
        return "contradictory"

    if rewrite_count > 0 and state.get("query", "") == state.get("original_query", ""):
        # 改写护栏已回退原查询（意图偏移过大 / 改写原地打转），继续改写无收益
        logger.info("[CRAG] 决策：改写已回退原查询，终止改写重试")
        return "generate" if grade == "partial" else "no_info"

    if grade == "no_info":
        if rewrite_count < MAX_REWRITES:
            logger.info(f"[CRAG] 决策：无检索结果，尝试改写查询（{rewrite_count}/{MAX_REWRITES}）")
            return "rewrite"
        else:
            logger.info(f"[CRAG] 决策：无检索结果且已达最大改写次数({MAX_REWRITES})")
            return "no_info"

    if grade == "partial" and rewrite_count < MAX_REWRITES:
        logger.info(f"[CRAG] 决策：文档部分相关，改写查询重试（{rewrite_count}/{MAX_REWRITES}）")
        return "rewrite"

    if grade == "irrelevant" and rewrite_count < MAX_REWRITES:
        logger.info(f"[CRAG] 决策：文档不相关，改写查询重试（{rewrite_count}/{MAX_REWRITES}）")
        return "rewrite"

    if rewrite_count >= MAX_REWRITES:
        if grade == "partial":
            logger.info(f"[CRAG] 决策：已达最大改写次数，部分相关文档直接生成")
            return "generate"
        else:
            logger.info(f"[CRAG] 决策：已达最大改写次数，尝试联网搜索")
            return "no_info"

    logger.info("[CRAG] 决策：默认改写查询重试")
    return "rewrite"


async def handle_no_info(state: CragState) -> Dict:
    query = state.get("original_query", "")
    chain_log = state.get("chain_log", [])

    _log_chain(chain_log, "no_info_start", {"query": query})

    if web_search_service.is_available():
        logger.info(f"[CRAG] 知识库无结果，尝试联网搜索：{query}")
        try:
            search_results = await web_search_service.search(query, num_results=5)

            if "error" not in search_results and search_results.get("results"):
                logger.info(f"[CRAG] 联网搜索成功，获取 {len(search_results['results'])} 条结果")

                web_docs = []
                for item in search_results.get("results", []):
                    web_docs.append({
                        "content": f"{item.get('title', '')}：{item.get('snippet', '')}",
                        "document_id": 0,
                        "chunk_index": 0,
                        "metadata": {"source": "web_search", "url": item.get("link", "")},
                        "rerank_score": 0.5
                    })

                context_parts = [doc["content"] for doc in web_docs]
                context = "\n\n".join(context_parts)

                try:
                    llm_response = hybrid_llm.generate_answer(
                        query=query,
                        context=context,
                        history=state.get("history"),
                        domain=state.get("domain", "general"),
                        use_finetuned=state.get("use_finetuned", False)
                    )

                    answer = llm_response.get('answer', '')
                    filtered_answer = sensitive_filter.filter_sensitive(answer)

                    citations = []
                    for idx, item in enumerate(search_results.get("results", []), 1):
                        citations.append({
                            "document_id": 0,
                            "chunk_index": idx,
                            "section": item.get("title", ""),
                            "page_number": "",
                            "score": 0.5,
                            "rank": idx,
                            "source": "web_search",
                            "url": item.get("link", "")
                        })

                    _log_chain(chain_log, "web_search_success", {
                        "result_count": len(search_results.get("results", []))
                    })

                    return {
                        "answer": filtered_answer,
                        "citations": citations,
                        "web_search_results": search_results,
                        "chain_log": chain_log
                    }
                except Exception as e:
                    logger.error(f"[CRAG] 联网搜索后生成失败：{e}")
            else:
                logger.info(f"[CRAG] 联网搜索无结果")
        except Exception as e:
            logger.error(f"[CRAG] 联网搜索失败：{e}")

    _log_chain(chain_log, "no_info_end", {"fallback": True})

    return {
        "answer": "根据现有文档，无法回答该问题。请尝试换一种方式提问，或联系管理员补充相关文档。",
        "citations": [],
        "web_search_results": None,
        "chain_log": chain_log
    }


def build_crag_graph() -> StateGraph: # 返回值类型注解，标注函数执行完毕一定会返回 `StateGraph` 类型实例
    """构建 CRAG LangGraph 工作流（四级决策 + 矛盾处理）"""

    # 
    workflow = StateGraph(CragState)
    # StateGraph是LangGraph 提供的图构造类
    # 相当于定义一个实例，后续这个实例添加节点，格式都必须是 CragState

    workflow.add_node("retrieve", retrieve) # 检索节点
    workflow.add_node("grade_documents", grade_documents) # 评估文档节点
    workflow.add_node("refine_knowledge", refine_knowledge)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_with_contradictions", generate_with_contradictions)
    workflow.add_node("handle_no_info", handle_no_info)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "refine": "refine_knowledge",
            "generate": "generate",
            "rewrite": "rewrite_query",
            "no_info": "handle_no_info",
            "contradictory": "generate_with_contradictions"
        }
    )

    workflow.add_edge("refine_knowledge", "generate")
    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_edge("generate", END)
    workflow.add_edge("generate_with_contradictions", END)
    workflow.add_edge("handle_no_info", END)

    graph = workflow.compile()
    logger.info("[CRAG] LangGraph CRAG 工作流构建完成（四级决策+矛盾处理+降级规则打分）")
    return graph


crag_graph = build_crag_graph()


@traced("crag.run", attributes_fn=lambda query, **kw: {"crag.query": str(query)[:60],
                                                        "crag.domain": kw.get("domain", "general")})
async def run_crag(
    query: str,
    domain: str = "general", # 领域标识
    use_finetuned: bool = False, # 是否使用微调模型
    history: List[Dict] = None, # 对话历史
    request_id: str = None # 请求ID（用于PII映射绑定）
) -> Dict[str, Any]:
    """
    运行 CRAG 流程

    Args:
        query: 用户查询
        domain: 领域标识
        use_finetuned: 是否使用微调模型
        history: 对话历史
        request_id: 请求ID（用于PII映射绑定）

    Returns:
        包含 answer、citations、chain_log、contradictions 的字典
    """
    start_time = time.time() # 记录开始时间

    with rag_monitor.trace(query=query, domain=domain) as trace_record: # 记录CRAG流程
        # 计算查询类型（simple/fuzzy/complex），使 MoA 可配置触发（D4 修复）
        try:
            from agentic_rag.rag.query_rewriter import query_rewriter
            initial_query_type = (query_rewriter.get_expansion_info(query) or {}).get("query_type", "simple")
        except Exception:
            initial_query_type = "simple"

        initial_state = {
            "query": query,
            "original_query": query,
            "documents": [],
            "relevance_grade": "",
            "relevance_score": 0.0,
            "rewrite_count": 0,
            "answer": "",
            "citations": [],
            "domain": domain,
            "use_finetuned": use_finetuned,
            "history": history,
            "web_search_results": None,
            "chain_log": [],
            "start_time": start_time,
            "contradictions": [],
            "request_id": request_id or "",
            "query_type": initial_query_type,
        }

        try:
            result = await crag_graph.ainvoke(initial_state) # 调用CRAG工作流
            rewrite_count = result.get('rewrite_count', 0)
            elapsed = time.time() - start_time

            logger.info(f"[CRAG] 流程完成，改写次数：{rewrite_count}，耗时：{elapsed:.2f}s")

            chain_log = result.get("chain_log", [])
            _log_chain(chain_log, "complete", {
                "elapsed_seconds": round(elapsed, 2),
                "rewrite_count": rewrite_count,
                "grade": result.get("relevance_grade", ""),
                "score": result.get("relevance_score", 0),
                "contradictions": len(result.get("contradictions", []))
            })

            trace_record.rewrite_count = rewrite_count
            trace_record.retrieval_count = len(result.get("documents", []))
            trace_record.web_search_used = result.get("web_search_results") is not None
            trace_record.answer = result.get("answer", "")[:200]

            return {
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "rewrite_count": rewrite_count,
                "relevance_grade": result.get("relevance_grade", ""),
                "relevance_score": result.get("relevance_score", 0),
                "contradictions": result.get("contradictions", []),
                "chain_log": chain_log,
                "context_snippets": [d.get("content", "") for d in result.get("documents", [])[:3]],
                "query_type": initial_query_type,
                "elapsed_seconds": round(elapsed, 2)
            }
        except Exception as e:
            logger.error(f"[CRAG] 流程异常：{e}")
            return {
                "answer": "系统繁忙，请稍后重试",
                "citations": [],
                "rewrite_count": 0,
                "relevance_grade": "error",
                "relevance_score": 0.0,
                "contradictions": [],
                "chain_log": [],
                "context_snippets": [],
                "query_type": initial_query_type,
                "elapsed_seconds": round(time.time() - start_time, 2)
            }
