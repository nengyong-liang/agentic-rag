# 已知问题与口径纪律（KNOWN ISSUES）

> 这份文档记录本仓**真实存在过的问题**、**它们现在的状态**，以及引用本仓数字时的口径。
> 数字一律带分母与时点，全部指向已提交产物（`docs/EVIDENCE.md` 给复算命令）。
> 本文不承诺"已全部修复"——每条都标状态。

状态标记与 `agent-harness` 仓保持一致：✅ 已修复 / 🟡 部分修复 / ⚪ 记录在案 / 🔴 已作废。

编号沿用逐条 finding 的序号，**跳号的编号在本仓没有对应物**
（属于未随本仓公开的早期记录），本文不主张其内容。

---

## §1 本仓在回答什么问题

业务型 RAG 的失败方式不是"检索不到"，而是**检索到了还硬答**。
本仓把"要不要答"做成三层判定，并把"证据本身脏不脏"也做成一层：

| 层 | 模块 | 判据 | 失败时的行为 |
|---|---|---|---|
| 检索质量 | `rag/retriever.py` `rag/crag.py` | CRAG 四级判定（`relevant`/`partial`/`irrelevant`/`contradictory`）+ DeepSeek/Qwen 双 Judge 复核 | 触发改写重试 / 提炼 / 标记冲突 |
| 证据充分性 | `agent/sufficiency_judge.py` | `SUFFICIENT / INSUFFICIENT / CONFLICTING` 三态 | 澄清或显式弃答，不硬答 |
| 改写收益 | `rag/rewrite_governor.py` | 试跑择优 + 意图偏移度量 | 无收益**回退原 Query** |
| 语料保鲜 | `rag/corpus_governance.py` | 增量 diff + 过期/冲突/近重复/缺陷发现 | 只标记、只出建议，**不删数据** |

这四层的共同设计约束是 **fail-closed**：LLM 只允许**加严**判定，
不允许把"不充分"翻成"充分"（§12.3 缺口 9）。

---

## §2 问题登记册

### §2.26 检索链路审计：多样性重排是死代码（RAG-01）

- **§2.26-E**：`rag/retriever.py::retrieve()` 的两条分支（跳过 Rerank / 正常 Rerank）
  产出的列表长度**恒等于** `dynamic_rerank_top_k`，
  而 `_diversity_rerank()` 的第一句判断是 `if len(results) <= top_k: return results`
  → **恒真早退**，去重/多样性过滤逻辑**从未执行过**。
  这是典型的"机制在、链路不可达"（同 `agent-harness` 的 §2.20-A②）。
- 修法：启用多样性重排时，重排/截断到**候选池** `dynamic_top_k`（5/10/20），
  再由 `_diversity_rerank` 收敛到 `dynamic_rerank_top_k`（3/5/10）。
  落点：`rag/retriever.py:359` 附近，见代码内注释。
- 守住用例（`tests/test_retriever_diversity_reachability.py`，零 API）：
  ① `test_diversity_filter_is_reachable`（**新行为**，改前必红）：
  冗余候选下最终结果必须包含候选池第 6 位之后（旧代码视野之外）的文档；
  ② `test_diversity_disabled_keeps_legacy_slice`（**非回归守卫**，两态皆绿）：
  关闭开关时输出恒等于候选前 `dynamic_rerank_top_k` 条。
- 效应层复算（E5，见 `docs/EVIDENCE.md §2`）：
  修复**之前**的 `plus_dedup` 档与基线 `adaptive` 档
  nDCG@10 / Recall@10 / MRR@10 / hit_rate@10 / precision@5 **五项逐位相同**
  → 这就是"死代码"在指标上的签名：**加了一层等于没加**。
  状态：✅ 已修复（可达性有用例守住）；🟡 修复后**实测为负向**，见下条。

### §2.27 修复之后：可达了，但在公开数据上是**负向**的

- **§2.27-A**：RAG-01 修好之后，多样性重排真的参与排序了，
  于是在 WixQA（200 queries / 6,221 docs）上第一次得到它的**真实效应量**：

  | 档 | nDCG@10 | Recall@10 | 相对 `adaptive` 基线 |
  |---|---|---|---|
  | `adaptive`（基线，无去重） | 0.5025 | 0.7092 | — |
  | `plus_dedup_pool`（候选池 20） | 0.4790 | 0.6492 | **−0.0235 / −0.0600** |
  | `plus_dedup_prod`（生产接线，池 10） | 0.4577 | 0.5842 | **−0.0448 / −0.1250** |

  结论要按字面读：**这个机制在本数据集上是减分的**，
  而且越接近生产配置（池更小）减得越多。
  本仓**不把它包装成优化**——它的价值是"暴露了一个从未生效、且生效后有害的默认开关"。
- 🔴 **未收口的配置决策**：`HybridRetriever.__init__`（`rag/retriever.py:61`）里
  `enable_diversity_rerank` 的默认值**仍是 `True`**，与上面的实测结论相反。
  改默认值会改变所有调用方的行为，因此本文只把它登记为待决项，而不是宣布已处理。
  在此之前，任何"多样性重排提升了效果"的表述在本仓**不成立**。
- 边界：单数据集、单份 qrels、节点级指标，未做区间估计
  （分母 = 200 条 query；`+dedup` 与基线逐位相同本身就是"该档无观测差异"的读法）。
  要翻案需要第二个数据集或按查询分层的对照，不是重跑同一条链。

---

## §3 与 `agent-harness` 的依赖真相（不装、也不低估）

`agent_harness` 在本仓是 **import 期硬依赖**，不是"可选增强"。
真实导入点（可自行 `grep`）：

| 用到的 harness 能力 | 本仓导入点 |
|---|---|
| Provider 抽象（doubao / deepseek / qwen / moa / registry） | `agent/agent.py` `agent/react_agent.py` `rag/crag.py` `rag/llm_router.py` `rag/query_classifier.py` `rag/query_rewriter.py` `utils/model_router.py` `api/main.py` |
| 上下文投影与 Token 计数（`context` / `ContextPhase` / `LayeredContextManager`） | `agent/agent.py` `agent/react_agent.py` `agent/memory.py` `rag/crag.py` |
| 可观测（`observability.tracing` / `monitor`） | `agent/agentic_rag.py` `rag/retriever.py` `rag/crag.py` `api/routes/monitor.py` `api/main.py` |
| 主循环与目标编译（`HarnessLoop` / `build_production_compiler`） | `agent/agent.py` `agent/react_agent.py` `api/main.py` |
| 安全网关侧的工具授权与 HITL 审批 | `agent/mcp_bridge.py` `api/routes/agent.py` `api/main.py` |
| 评测基准（`evaluation.benchmarks`） | `api/routes/monitor.py` |

**只有执行循环是运行时可选的**：`agent/agent.py::_get_harness()` 拿不到 `HarnessLoop`
时回退到内置手动循环 `_run_manual()`，业务链路不因缺少运行时层而中断。
装配与回退都可实跑验证（零 API）：

```bash
python examples/03_harness_seam.py
```

它会打印两件事：装配成功时主循环 / Provider / 工具执行器 / 安全网关 / HITL / 目标编译的来源，
以及**把 `agent_harness` 从可导入集合里摘掉之后** `_get_harness()` 返回 `None`。

🔴 表述纪律：
- 可以写"应用层与运行时层分离，运行时缺失时业务链路可降级"；
- 不可以写"本仓零依赖、可独立运行"（`pip install` 本仓后不装 harness **导入就会失败**）。

---

## §4 口径纪律（引用本仓数字时）

与 `agent-harness/docs/KNOWN-ISSUES.md §6` 同源，这里只列本仓最常踩的四条：

1. **节点级 ≠ 链路级**。nDCG@10 / Recall@10 的分母是 **200 条 query**（检索质量）；
   Pass^k / Mean@k 的分母是**任务数**（端到端可靠性）。
   两把尺**不得同表比较，更不得一起排名**。
2. **"没差异" ≠ "没作用"**。§2.26 的 `+dedup` 与基线逐位相同，
   读法是"该层不可达"，不是"该层被验证为中性"。
3. **桩读数只支撑管控类主张**。本仓的 mcp_server 工具依赖业务后端，
   后端未随本仓开源 → 无后端时工具返回连接错误（`mcp_server/README.md` 已声明）。
   不支撑"真实数据一致性"。
4. **未跑的配置写"未跑"**。`+CRAG` 档截至本文撰写**没有产物**，
   不得写成"效果接近"（`docs/EVIDENCE.md §3`）。

---

## §12.3 尚未闭合的能力缺口（10 项）

与 `agent-harness` 的同一张表（编号一致，两仓各自填自己的现状）：

| # | 缺口 | 本仓现状 | 已落到的代码 | 还差什么 |
|---|---|---|---|---|
| 1 | OTel 可观测投影 | 🟡 | `api/routes/monitor.py` 读 harness 事件流 | 无 collector 端闭环；本仓自身未建 span |
| 2 | RPM/TPM 双桶与背压 | 🟡 | 复用 `agent-harness` 的 `governance/rate_limiter.py`；本仓 `rag/llm_router.py` 接线 | 未做跨进程共享配额 |
| 3 | 确定性回放 | 🟡 | 复用 harness `ToolResultFreezer`；本仓未单独建夹具 | 工具结果冻结只在跑批侧验证过 |
| 4 | 三类运行时指标 | 🟡 | `rag/retriever.py` 的 `latency_breakdown`（分段耗时）已入产物 | 未形成稳定阈值/回归 |
| 5 | 换模型回归基线 | 🟡 | 复用 harness `model_regression.py` | 本仓生成侧未建自己的基线批次 |
| 6 | 公开数据集与 Benchmark 四要素 | 🔴 | `experiments/retrieval/` 只有 WixQA 一份产物 | **BEIR 未下载未接入**；不可写"公开数据集 SOTA" |
| 7 | **Corpus 保鲜治理** | 🟡 | `rag/corpus_governance.py`：增量 diff（`plan_incremental_update`）、5 类发现（`expired`/`superseded`/`conflict`/`duplicate`/`defect`）、4 类处置（`review`/`suppress`/`drop_duplicate`/`reindex`）、`filter_fresh` 检索侧剔除；`tests/test_corpus_governance.py` 覆盖 | 冲突裁决仍靠人工；未在真实半年期语料上演练过 |
| 8 | 裁判妥协的三层防御 | 🟡 | `rag/crag.py` CRAG 四级判定 + DeepSeek/Qwen **双 Judge**；改写护栏（§12.3 缺口 10） | 双 Judge 不一致时的仲裁策略未定档 |
| 9 | **证据充分性三态判定** | ✅ | `agent/sufficiency_judge.py`：`SUFFICIENT/INSUFFICIENT/CONFLICTING` + 结构化依据；**规则优先、LLM 只可加严（fail-closed）**；引用不可核查即作废；`INSUFFICIENT` 到轮次上限 → 显式弃答/澄清；`ValidatorAgent`/`SupervisorAgent`/`SynthesizerAgent` 全链路接入；CRAG 评级归一到同一 `evidence_state` 契约（`tests/test_sufficiency_judge.py`） | 只在自测数据上验证；无端到端"弃答正确率"的公开基准 |
| 10 | **Query 改写回退治理** | ✅ | `rag/rewrite_governor.py`：试跑择优、**无收益回退原 Query**、意图偏移度量与超限剪枝（HyDE 长文本不因篇幅误判）、按策略归因（计划数/检索数/命中率/采纳数/平均偏移 + 整体回退率）；`rag/crag.py` 侧空改写/原地打转/漂移过大 → 回退并终止改写循环（`tests/test_rewrite_governor.py`） | 归因统计未接线上流量；偏移度量是词面近似，非语义 |

🔴 **由本表直接导出的表述禁令**：
- 不可写"检索指标达到/超过 SOTA"（缺口 6）；
- 不可写"多样性去重提升了召回"（§2.27 实测**负向**）；
- 不可写"零依赖运行时"（§3 导入期硬依赖）；
- 不可写"全链路可观测/自动限流已闭环"（缺口 1/2/4 为 🟡）。

---

## §5 MCP Server 的边界（说明白，不装）

本仓自带一个**自研 JSON-RPC 2.0** MCP Server（`mcp_server/`，不用 MCP SDK）：
10 个工具 / 8 个资源 / 4 个子 Server。三条必须说清的边界：

1. **业务后端未随本仓开源** → 没有后端时工具调用返回连接错误，不是"能力缺失"的伪装；
2. `MCP_API_KEY` **留空即 fail-closed**（外部调用全部拒绝），启动只打 warning；
3. 需要 `MCP_JWT_TOKEN` 才能访问受保护资源。

详见 `mcp_server/README.md`。

---

## 相关文档

- `docs/EVIDENCE.md`：E5 八档检索消融的产物、复算命令与未收口项
- `README.md`：架构、快速开始与能力边界
- 运行时层仓库：`agent-harness`（主循环 / 终止校验 / 熔断 / 安全网关 / 事件流）
