"""示例 1：知识库保鲜 —— 一次扫描同时回答「哪些切片该重转、该下线、该人工复核」。

零 API、零中间件：治理器是纯函数式的，所有时间判定都显式传 `as_of`，
所以结果可重跑、可写进断言。

三段演示：
  1. `scan()`     —— 过期 / 被取代 / 重复 / 脏切片 / 事实冲突；
  2. `plan_incremental_update()` —— 增量索引 diff（只重转真正变过的切片）；
  3. `filter_fresh()` —— 治理结论如何落到检索链路（下线清单直接剔除）。

运行：
    python examples/01_corpus_governance.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentic_rag.rag.corpus_governance import (  # noqa: E402
    CorpusGovernor, filter_fresh,
)

AS_OF = date(2026, 9, 30)          # 判定基准日：写死它，结论才可比


def chunk(chunk_id: str, content: str, **meta) -> dict:
    """最小切片形态：与索引导出的字段命名保持一致"""
    record = {"chunk_id": chunk_id, "content": content, "kb_id": "hr_policy"}
    if meta:
        record["metadata"] = meta
    return record


CORPUS = [
    # 现行版本：应保留
    chunk("annual_leave_v2", "公司年假政策：入职满 1 年 5 天，满 5 年 10 天，需提前 3 个工作日申请。",
          topic_key="annual_leave", version=2, updated_at="2026-01-15",
          effective_until="2027-12-31", is_latest=True),
    # 被 v2 取代的旧版：应下线，不再召回
    chunk("annual_leave_v1", "公司年假政策：入职满 1 年 3 天，满 5 年 8 天，需提前 3 个工作日申请。",
          topic_key="annual_leave", version=1, updated_at="2024-01-10", is_latest=False),
    # 明确过了生效截止日
    chunk("attendance_2025", "2025 年考勤制度：上班打卡时间为 9:00，弹性 15 分钟。",
          topic_key="attendance", updated_at="2025-03-01", effective_until="2026-06-30"),
    # 同一内容的重复副本
    chunk("duplicate_of_v2", "公司年假政策：入职满 1 年 5 天，满 5 年 10 天，需提前 3 个工作日申请。",
          topic_key="annual_leave", version=2, updated_at="2026-01-15"),
    # 脏切片：截断
    chunk("truncated", "加班申请需要提交审批单，并由直属主管审批后，",
          topic_key="overtime"),
]


def main() -> int:
    governor = CorpusGovernor(as_of=AS_OF, ttl_days=180, min_chars=8)

    print("—— 1. 治理扫描 ——")
    report = governor.scan(CORPUS)
    print(f"  切片数 {report.chunk_count}，结论 {len(report.findings)} 条")
    for f in report.findings:
        print(f"  [{f.rule:<20}] {f.chunk_id:<16} {f.action:<10} {f.detail[:52]}")
    print(f"  按处置动作汇总: {report.counts_by_action()}")

    print("\n—— 2. 增量索引 diff ——")
    indexed = CORPUS[:2]
    current = [chunk("annual_leave_v2",
                     "公司年假政策：入职满 1 年 5 天，满 5 年 10 天，满 10 年 15 天，需提前 3 个工作日申请。",
                     topic_key="annual_leave", version=3, updated_at="2026-09-01",
                     is_latest=True),
               CORPUS[1], CORPUS[2], CORPUS[4]]
    plan = governor.plan_incremental_update(indexed, current)
    print(f"  {plan.to_dict()}")
    print(f"  需要重转 {plan.changed_count} 个切片 —— 其余沿用原索引，不整库重建")

    print("\n—— 3. 结论落到检索链路 ——")
    retrieved = list(CORPUS)          # 假设检索层原样返回了全库切片
    kept, removed = filter_fresh(retrieved, report)
    print(f"  剔除 {len(removed)} 条：{sorted(removed)}")
    print(f"  保留 {len(kept)} 条：{[c['chunk_id'] for c in kept]}")
    print("  全部被剔除时如实返回空列表，交给证据充分性判定走 CRAG 补救，"
          "而不是把过期证据硬塞进上下文。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
