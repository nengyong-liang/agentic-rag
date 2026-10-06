# Examples

三个示例覆盖应用层里最常被问的三段逻辑，全部**零 API**（不产生模型计费）：

| 文件 | 看什么 | 是否需要 agent-harness |
| --- | --- | --- |
| `01_corpus_governance.py` | 知识库保鲜：一次扫描给出「该重转 / 该下线 / 该人工复核」三类结论，并把结论落回检索链路 | 否 |
| `02_evidence_sufficiency.py` | 证据充分性三态判定：sufficient / insufficient / conflicting，以及「不够到上限就弃答，不编造」；含 LLM 只能加严（fail-closed）的对比 | 否 |
| `03_harness_seam.py` | 应用层与运行时层的装配接缝：装配成功时列出接管的主循环与治理开关；摘掉 `agent_harness` 后确认真的回退到内置手动循环 | 是 |

```bash
# 两仓并列放在同一个父目录时，示例会自己发现 ../agent-harness/src，无需安装
python examples/01_corpus_governance.py
python examples/02_evidence_sufficiency.py
python examples/03_harness_seam.py
```

`01`、`02` 只用应用层的纯规则模块，不需要 `agent_harness`，也不需要 Milvus / Elasticsearch /
Redis / 任何模型密钥：判定都是纯函数式的，时间基准显式传 `as_of`，所以输出可重跑、可写进断言。

真实链路的端到端示例（需要密钥与检索中间件）在 `scripts/debug_query_pipeline.py`。
