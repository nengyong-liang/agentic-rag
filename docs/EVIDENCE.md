# 实验证据（EVIDENCE）

> 本文只放**指向已提交产物**的读数。每条数字都带分母、口径与产物文件名。
> 本文一律不写成本金额；付费实验只说明**是否执行**与**计量读数**。
>
> 引用前请先读 `docs/KNOWN-ISSUES.md §4 口径纪律`：
> 本文的 nDCG@10 / Recall@10 是**节点级**指标（分母 = 200 条 query），
> 与 `agent-harness` 的 Pass^k / Mean@k（分母 = 任务数）**不是同一把尺，不得同表比较**。

---

## §0 复算命令与一条必须说明的复算限制

```bash
# 从已提交产物打印八档对照表（零 API、零外呼、无需数据集）
python scripts/analysis/retrieval_ablation_report.py
```

产物：`experiments/retrieval/e5_*_report.json`（8 份档位报告）。

🔴 **复算限制（不装）**：这 8 份报告由一个**未随本仓发布**的离线跑批脚本产出。
因此本仓能复算的是**"从产物重新打印对照表与派生差值"**，
**不能**复算"从原始文档重新检索打分"这一层。
要独立重跑，需要自备 WixQA 语料 + `BAAI/bge-m3` 编码 + 本仓 `rag/` 检索链路。
本文所有结论都限定在这个边界内。

---

## §1 数据集与口径

| 项 | 值 | 来源字段 |
|---|---|---|
| 数据集 | WixQA（公开） | `dataset_scale` |
| 语料 | **6,221** docs | 同上 |
| 查询 | **200** queries | 同上 |
| 相关性标注 | 200 条 qrels | 同上 |
| 存储/打分 | **进程内 numpy cosine**（无向量库） | `storage` |
| 编码模型 | `BAAI/bge-m3` | 同上 |
| `max_seq_len` | **512** | `max_seq_len` |
| 指标 | nDCG@10 / Recall@10 / Recall@1 / MRR@10 / hit_rate@10 / precision@5 | `metrics` |

⚠️ 两点口径：
1. 分母 = **200 条 query**，没有做区间估计（单一数据集、单一 qrels）；
2. `storage` 是进程内 numpy cosine，**不是**本仓生产的 ES + 向量库混合检索链路
   → 本档测的是**检索策略层**（top-k / 去重 / 重排 / 改写融合），
   不含真实混合检索的召回上界。

---

## §2 八档对照（全部由 `scripts/analysis/retrieval_ablation_report.py` 打印）

| 档位 | nDCG@10 | Recall@10 | Recall@1 | MRR@10 | hit_rate@10 | precision@5 | 叠加的阶段 |
|---|---|---|---|---|---|---|---|
| `fixed` | 0.5043 | 0.7142 | 0.2733 | 0.4563 | 0.7750 | 0.1520 | dense |
| `adaptive`（基线） | 0.5025 | 0.7092 | 0.2733 | 0.4555 | 0.7700 | 0.1520 | dense + dynamic_topk |
| `plus_dedup` | 0.5025 | 0.7092 | 0.2733 | 0.4555 | 0.7700 | 0.1520 | + diversity_rerank |
| `plus_dedup_pool` | 0.4790 | 0.6492 | 0.2733 | 0.4431 | 0.7150 | 0.1430 | + diversity_rerank(pool=20) |
| `plus_dedup_prod` | 0.4577 | 0.5842 | 0.2733 | 0.4340 | 0.6400 | 0.1430 | + diversity_rerank(prod-fixed, 池 10) |
| `plus_reranker` | 0.5399 | 0.7092 | 0.3442 | 0.5151 | 0.7700 | 0.1490 | + cross_encoder_rerank |
| `plus_rewrite` | 0.5153 | **0.7225** | 0.2858 | 0.4751 | 0.7800 | 0.1520 | + llm_multi_query + RRF |
| `combo_prod` | 0.5399 | 0.7092 | 0.3442 | 0.5151 | 0.7700 | 0.1490 | rerank → diversity（生产顺序） |

档位专属配置（产物原文）：`plus_dedup`/`plus_dedup_pool` 的 `diversity_threshold = 0.85`；
`plus_reranker` 的 `rerank_model = BAAI/bge-reranker-base`、`rerank_pool = 10`；
`plus_rewrite` 的 `rewriter = QueryRewriter.plan_variants(use_llm=True)`、
`fusion = RRF 多路等权 k=60`。

---

## §3 四条能站住的读法

1. **`plus_dedup` 与 `adaptive` 六项指标逐位相同**（0.5025 / 0.7092 / 0.2733 / 0.4555 / 0.7700 / 0.1520）。
   这不是"去重被验证为中性"，而是 **RAG-01 死代码的指标签名**：
   该档的多样性重排**从未参与排序**（`KNOWN-ISSUES §2.26-E`）。
   产物 `e5_plus_dedup_pool_report.json` 的 `note` 字段原文即写明这一点
   （"修好接线版；生产版因 len==top_k 早退 → 见 RAG-01"）。
2. **接好线之后，去重是减分的**：`plus_dedup_pool` nDCG@10 −0.0235、Recall@10 −0.0600；
   越接近生产配置（池 10）减得越多：`plus_dedup_prod` −0.0448 / **−0.1250**，
   hit_rate@10 从 0.7700 掉到 0.6400。
   → 见 `KNOWN-ISSUES §2.27`，本仓**不**把这一层包装成优化。
3. **Cross-encoder 重排是唯一明确正向的一层**：相对 `adaptive`，
   nDCG@10 **+0.0374**、Recall@1 **+0.0709**、MRR@10 **+0.0596**，
   而 Recall@10 持平（0.7092）→ 它改变的是**排序质量**，不是召回集合覆盖。
4. **改写的收益在召回侧**：`plus_rewrite` Recall@10 **0.7225**（基线 0.7092，+0.0133）、
   hit_rate@10 0.7800、nDCG@10 0.5153；Recall@1 0.2858。
   计量读数（`e5_plus_rewrite_report.json` 的 `meter` 字段）：
   **200 条 query 触发 281 次调用**（`calls_per_query = 1.405`），
   prompt 54,014 tokens / completion **149,038** tokens（provider = deepseek）。
   ⚠️ completion 是 prompt 的 2.8 倍 → 改写阶段的输出体积远大于输入，
   这是本档最该被审视的成本项（本仓未做长度约束的对照实验）。

⚠️ **`combo_prod` 与 `plus_reranker` 读数逐位相同**（0.5399 / 0.7092 / …），
但两档的**阶段顺序相反**（前者 rerank→diversity，后者 diversity→rerank）。
→ 这组产物**无法区分两种顺序的差异**，
因此"生产顺序更优/更差"这类结论**在本仓不成立**（§6 规则：没跑就写没跑，
跑了但分不出来就写分不出来）。

---

## §4 🔴 未跑的档位

`+CRAG` 档**截至本文撰写没有产物**（本仓 `experiments/retrieval/` 里只有上面 8 份）。
CRAG 在本仓是**证据评级与改写重试的控制逻辑**（`rag/crag.py`，
四级判定 `relevant` / `partial` / `irrelevant` / `contradictory` +
DeepSeek/Qwen 双 Judge 复核），它有单测覆盖，但**没有检索指标**。

因此：
- ✅ 可写"实现了 CRAG 评级与双 Judge 复核，并有单测"；
- 🔴 不可写"CRAG 提升了检索指标"——**那一档没跑**。

---

## §5 零成本守卫（可天天跑）

| 检查 | 命令 | 当前读数 |
|---|---|---|
| 全量单测（零 API、零计费） | `pytest -q` | **185 passed**（无需设 `PYTHONPATH`，见 `conftest.py` 的兄弟目录发现） |
| 装配接缝实跑 | `python examples/03_harness_seam.py` | exit 0：装配成功 + 摘掉 harness 后回退 `_run_manual()` |
| 检索对照表复算 | `python scripts/analysis/retrieval_ablation_report.py` | 打印 §2 全表 |

本仓 185 条用例**全部不发起计费调用**：服务层（LLM / 检索器 / ES / 向量库）
在测试里用假对象注入。两道护栏（`conftest.py`）：
`PYTEST_COLLECT_ONLY` 在收集阶段置位，让模块级单例跳过联网与重型初始化；
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 默认打开，需要真实模型权重的用例各自带 skip 条件。
需要真实配对的端到端用例在 `agent-harness` 仓，由 `RUN_LIVE_TESTS=1` 门禁保护。

---

## §6 未收口项

| 项 | 状态 | 缺什么 |
|---|---|---|
| `+CRAG` 检索指标 | ⚪ 未跑 | §4 |
| E5 跑批脚本 | ⚪ 未发布 | §0 的复算限制；本文只发布产物 |
| `enable_diversity_rerank` 默认值 | 🔴 与实测结论相反 | `rag/retriever.py:61` 默认仍是 `True`（`KNOWN-ISSUES §2.27`） |
| BEIR 公开数据集 | 🔴 未接入 | 与 `agent-harness` 的 §12.3 缺口 6 同一条；不可写"公开数据集 SOTA" |
| 第二个数据集 / 分层对照 | ⚪ 未做 | §2.27 的负向结论要翻案需要它，不是重跑同一条链 |
| 生产混合检索链路（ES + 向量库） | ⚪ 未测 | §1：全部读数来自进程内 numpy cosine |
| 生成侧端到端（答案质量） | ⚪ 未测 | 本文只有检索侧节点级指标 |
| 真实服务栈端到端 | ⚪ 未做 | `mcp_server/` 依赖未开源的业务后端（`mcp_server/README.md`） |

---

## §7 引用本文时的禁令

1. 不与 `agent-harness` 的 Pass^k / Mean@k 同表比较或排名（不同 estimand）。
2. 不写"检索达到 SOTA"（单数据集、无区间、缺口 6 未闭合）。
3. 不写"多样性去重提升召回"（§3 第 2 条实测**负向**）。
4. 不写"CRAG 提升检索指标"（§4 未跑）。
5. 不写"两种重排顺序中生产顺序更好"（§3 末尾：读数逐位相同，分不出来）。
6. 不写"本仓可独立运行"（`KNOWN-ISSUES §3`：`agent_harness` 是导入期硬依赖）。
7. 报任何差值必须同时报分母（200 queries / 6,221 docs）与档位 stages。

---

## 相关文档

- `docs/KNOWN-ISSUES.md`：RAG-01 死代码、§2.27 负向结论、依赖真相、缺口表
- `README.md`：架构与快速开始
- 运行时层仓库：`agent-harness`（E1~E4b 端到端可靠性证据在该仓）
