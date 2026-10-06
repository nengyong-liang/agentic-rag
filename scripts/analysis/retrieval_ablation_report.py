"""从 `experiments/retrieval/e5_*_report.json` 复现 E5 消融对照表（零成本、零外呼）。

读的是**已落盘的档位报告**：这些报告由一次性 E5 runner 在 WixQA（200 queries /
6,221 docs）上产出，指标口径 = `agent_harness.evaluation.retrieval_benchmark`
（节点级：nDCG@10 / Recall@10 / MRR@10 / HitRate@10 / Precision@5）。

生成侧的 runner 脚本**未随本仓提交**（需要 WixQA 全量下载 + 本地 bge-m3 权重），
所以这里能复核的是「表格与产物一致」，不是「从零重跑档位」。边界见
`docs/EVIDENCE.md` 与 `docs/KNOWN-ISSUES.md` §2.27。

用法：
    python scripts/analysis/retrieval_ablation_report.py
"""

from __future__ import annotations

import glob
import io
import json
import os

COLUMNS = ["ndcg@10", "recall@10", "mrr@10", "hit_rate@10", "precision@5"]


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = sorted(glob.glob(os.path.join(root, "experiments", "retrieval", "e5_*_report.json")))
    if not paths:
        print("未找到档位报告：experiments/retrieval/e5_*_report.json")
        return 1

    rows = []
    for p in paths:
        d = json.load(io.open(p, encoding="utf-8"))
        rows.append((d.get("tier", os.path.basename(p)), d.get("metrics", {}), d))

    scale = rows[0][2].get("dataset_scale", {})
    print(f"数据集规模：{scale}   存储层：{rows[0][2].get('storage')}")
    print(f"max_seq_len：{rows[0][2].get('max_seq_len')}   阶段：每档 stages 见产物\n")
    header = f"{'档位':<18}" + "".join(f"{c:>14}" for c in COLUMNS) + "   stages"
    print(header)
    print("-" * len(header))
    for tier, m, d in rows:
        line = f"{tier:<18}" + "".join(f"{m.get(c, float('nan')):>14.4f}" for c in COLUMNS)
        print(f"{line}   {'+'.join(d.get('stages', []))}")
    print("\n口径提醒：nDCG@10 / Recall@10 是**节点级**指标，分母 = 200 条 query；"
          "与链路级 Pass^k / Mean@k 属不同 estimand，不得同表比较或排名。")
    print("逐档附注（对照锚点 / 过滤占比 / 阈值）见各产物的 note、pool、p4_stats 字段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
