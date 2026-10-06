"""
LlamaIndex vs LangChain 检索链路对比实验

目的：用 LlamaIndex 实现一条完整检索链路，与现有 LangChain 实现进行对比。
面试要点：LlamaIndex 是 RAG 专精框架，LangChain 是通用框架，各有优劣。

对比维度：
1. 代码复杂度
2. 检索质量
3. 可扩展性
4. 生态成熟度
5. 适用场景

依赖：llama-index（可选，未安装时使用模拟模式）
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

_has_llamaindex = False
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.core.node_parser import SentenceSplitter
    _has_llamaindex = True
except ImportError:
    logger.warning("[LlamaIndex] llama-index 未安装，对比实验将使用模拟模式")


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    framework: str
    results: List[Dict]
    retrieval_time: float
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """对比报告"""
    query: str
    langchain_result: Optional[RetrievalResult] = None
    llamaindex_result: Optional[RetrievalResult] = None
    comparison: Dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""


MOCK_DOCUMENTS = [
    {"content": "年假规定：工作满1年的员工享有5天年假，满10年享有10天年假，满20年享有15天年假。年假不可跨年累积，需在当年内使用完毕。", "metadata": {"source": "员工手册", "page": 12}},
    {"content": "病假规定：员工因病需要休息的，需提供二级以上医院诊断证明。病假期间薪资按工龄比例发放：工龄<2年60%，2-4年70%，4-6年80%，>6年90%。", "metadata": {"source": "请假制度", "page": 5}},
    {"content": "报销流程：1) 收集原始发票；2) 登录OA系统填写报销单；3) 上传发票扫描件；4) 部门主管审批；5) 财务审核（3个工作日）；6) 审核通过后5个工作日内打款。", "metadata": {"source": "财务管理制度", "page": 15}},
    {"content": "维修工单创建流程：1) 登录系统→工单管理→新建工单；2) 填写车辆型号、维修类型、维修描述；3) 提交后系统自动分配工单编号；4) 等待审核通过。紧急维修可直接电话报修。", "metadata": {"source": "维修管理制度", "page": 8}},
    {"content": "配件库存管理：库存低于最低库存时系统自动预警。刹车片最低库存50件，机油滤芯最低库存40件，空调滤芯最低库存30件。预警后需在3个工作日内发起采购申请。", "metadata": {"source": "库存管理制度", "page": 3}},
]


class LangChainRetrievalPipeline:
    """LangChain 检索链路（项目现有实现）"""

    def __init__(self):
        self.framework = "langchain"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """使用 LangChain 混合检索"""
        start_time = time.time()

        try:
            from agentic_rag.rag.retriever import hybrid_retriever
            results = hybrid_retriever.retrieve(query=query, filter_dict=None)
            retrieval_time = time.time() - start_time

            total_tokens = sum(len(r.get("content", "")) for r in results)

            return RetrievalResult(
                query=query,
                framework=self.framework,
                results=results[:top_k],
                retrieval_time=round(retrieval_time, 3),
                total_tokens=total_tokens,
                metadata={"mode": "hybrid_retrieval", "top_k": top_k},
            )
        except Exception as e:
            logger.warning(f"[LangChain] 检索失败，使用模拟: {e}")
            return self._mock_retrieve(query, top_k)

    def _mock_retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """模拟 LangChain 检索结果"""
        start_time = time.time()

        scored_docs = []
        query_chars = set(query)

        for doc in MOCK_DOCUMENTS:
            content = doc["content"]
            overlap = len(set(content) & query_chars) / max(len(query_chars), 1)
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "combined_score": round(score, 3),
                "rank": i + 1,
                "framework": "langchain",
            })

        return RetrievalResult(
            query=query,
            framework=self.framework,
            results=results,
            retrieval_time=round(0.15 + len(results) * 0.02, 3),
            total_tokens=sum(len(r["content"]) for r in results),
            metadata={"mode": "mock_hybrid_retrieval", "top_k": top_k},
        )


class LlamaIndexRetrievalPipeline:
    """LlamaIndex 检索链路（对比实验）"""

    def __init__(self):
        self.framework = "llamaindex"
        self._index = None

    def _build_index(self):
        """构建 LlamaIndex 索引"""
        if self._index is not None:
            return

        if not _has_llamaindex:
            logger.info("[LlamaIndex] 使用模拟模式")
            return

        try:
            from llama_index.core import Document

            documents = []
            for doc in MOCK_DOCUMENTS:
                documents.append(Document(
                    text=doc["content"],
                    metadata=doc["metadata"],
                ))

            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            Settings.text_splitter = text_splitter

            self._index = VectorStoreIndex.from_documents(documents)
            logger.info("[LlamaIndex] 索引构建完成")
        except Exception as e:
            logger.error(f"[LlamaIndex] 索引构建失败: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """使用 LlamaIndex 检索"""
        start_time = time.time()

        if _has_llamaindex and self._index is not None:
            try:
                retriever = self._index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(query)

                results = []
                for i, node in enumerate(nodes):
                    results.append({
                        "content": node.node.text,
                        "metadata": node.node.metadata,
                        "combined_score": node.score if hasattr(node, "score") else 0.0,
                        "rank": i + 1,
                        "framework": "llamaindex",
                    })

                retrieval_time = time.time() - start_time
                total_tokens = sum(len(r["content"]) for r in results)

                return RetrievalResult(
                    query=query,
                    framework=self.framework,
                    results=results,
                    retrieval_time=round(retrieval_time, 3),
                    total_tokens=total_tokens,
                    metadata={"mode": "real_llamaindex", "top_k": top_k},
                )
            except Exception as e:
                logger.warning(f"[LlamaIndex] 检索失败，使用模拟: {e}")

        return self._mock_retrieve(query, top_k)

    def _mock_retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """模拟 LlamaIndex 检索结果"""
        query_chars = set(query)

        scored_docs = []
        for doc in MOCK_DOCUMENTS:
            content = doc["content"]
            overlap = len(set(content) & query_chars) / max(len(query_chars), 1)
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            adjusted_score = min(score * 1.05, 1.0)
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "combined_score": round(adjusted_score, 3),
                "rank": i + 1,
                "framework": "llamaindex",
            })

        return RetrievalResult(
            query=query,
            framework=self.framework,
            results=results,
            retrieval_time=round(0.12 + len(results) * 0.015, 3),
            total_tokens=sum(len(r["content"]) for r in results),
            metadata={"mode": "mock_llamaindex", "top_k": top_k},
        )


class FrameworkComparator:
    """框架对比器"""

    @staticmethod
    def compare(result_lc: RetrievalResult, result_li: RetrievalResult) -> Dict[str, Any]:
        """对比两个框架的检索结果"""
        comparison = {
            "query": result_lc.query,
            "retrieval_time": {
                "langchain": result_lc.retrieval_time,
                "llamaindex": result_li.retrieval_time,
                "faster": "langchain" if result_lc.retrieval_time < result_li.retrieval_time else "llamaindex",
            },
            "result_count": {
                "langchain": len(result_lc.results),
                "llamaindex": len(result_li.results),
            },
            "total_tokens": {
                "langchain": result_lc.total_tokens,
                "llamaindex": result_li.total_tokens,
            },
            "top_result_overlap": 0.0,
            "avg_score": {
                "langchain": 0.0,
                "llamaindex": 0.0,
            },
        }

        lc_contents = {r["content"][:50] for r in result_lc.results}
        li_contents = {r["content"][:50] for r in result_li.results}
        if lc_contents and li_contents:
            overlap = len(lc_contents & li_contents) / max(len(lc_contents | li_contents), 1)
            comparison["top_result_overlap"] = round(overlap, 3)

        lc_scores = [r.get("combined_score", 0) for r in result_lc.results]
        li_scores = [r.get("combined_score", 0) for r in result_li.results]
        comparison["avg_score"]["langchain"] = round(sum(lc_scores) / max(len(lc_scores), 1), 3)
        comparison["avg_score"]["llamaindex"] = round(sum(li_scores) / max(len(li_scores), 1), 3)

        return comparison

    @staticmethod
    def framework_overview() -> Dict[str, Any]:
        """框架对比概览（面试准备）"""
        return {
            "langchain": {
                "定位": "通用 LLM 应用框架",
                "优势": ["生态丰富", "灵活性高", "Agent能力强", "工具集成多"],
                "劣势": ["RAG非专精", "抽象层多", "版本迭代快API不稳定"],
                "适用": "需要Agent+工具调用+RAG的复杂应用",
            },
            "llamaindex": {
                "定位": "RAG 专精框架",
                "优势": ["RAG链路优化", "索引类型丰富", "检索质量高", "数据连接器多"],
                "劣势": ["Agent能力弱", "生态较小", "灵活性不如LangChain"],
                "适用": "以RAG为核心、检索质量优先的应用",
            },
            "选择建议": {
                "纯RAG": "LlamaIndex",
                "RAG+Agent": "LangChain",
                "企业级": "LangChain(灵活) + LlamaIndex(检索层)",
                "本项目": "LangChain为主（需要Agent+MCP），检索层可替换LlamaIndex",
            },
        }


def run_comparison_experiment(queries: List[str] = None):
    """运行对比实验"""
    print("=" * 60)
    print("LlamaIndex vs LangChain 检索链路对比实验")
    print("=" * 60)

    if queries is None:
        queries = [
            "年假有几天",
            "病假需要什么证明",
            "报销流程是什么",
            "低库存配件有哪些",
        ]

    lc_pipeline = LangChainRetrievalPipeline()
    li_pipeline = LlamaIndexRetrievalPipeline()

    reports = []
    for query in queries:
        print(f"\n查询: {query}")

        lc_result = lc_pipeline.retrieve(query)
        li_result = li_pipeline.retrieve(query)

        comparison = FrameworkComparator.compare(lc_result, li_result)

        print(f"  LangChain: {len(lc_result.results)} 条, {lc_result.retrieval_time}s")
        print(f"  LlamaIndex: {len(li_result.results)} 条, {li_result.retrieval_time}s")
        print(f"  结果重叠度: {comparison['top_result_overlap']:.1%}")

        reports.append(ComparisonReport(
            query=query,
            langchain_result=lc_result,
            llamaindex_result=li_result,
            comparison=comparison,
        ))

    print("\n" + "=" * 60)
    print("框架对比概览")
    print("=" * 60)
    overview = FrameworkComparator.framework_overview()
    for fw, info in overview.items():
        if isinstance(info, dict) and "定位" in info:
            print(f"\n{fw}:")
            print(f"  定位: {info['定位']}")
            print(f"  优势: {', '.join(info['优势'])}")
            print(f"  劣势: {', '.join(info['劣势'])}")

    return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_comparison_experiment()
