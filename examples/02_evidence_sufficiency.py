"""示例 2：证据充分性三态判定 —— 「够了 / 不够 / 打架」，以及不够时怎么收场。

零 API：判定器是「规则骨架 + 可选 LLM 语义增强」，本例只走规则骨架。
这条链路的价值在于它把「要不要继续检索 / 要不要弃答」从模型的主观感受，
变成了一个可审计的判定：

  SUFFICIENT   证据数量与关键信息覆盖都达标
  INSUFFICIENT 证据不足或关键信息有缺口 —— 到迭代上限即弃答，不编造
  CONFLICTING  同一事实出现不同取值 —— 分别罗列，不强行统一

运行：
    python examples/02_evidence_sufficiency.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentic_rag.agent.sufficiency_judge import SufficiencyJudge  # noqa: E402

QUERY = "这款发动机滤清器的库存是多少，谁能审批出库？"
REQUIRED = ["库存数量", "出库审批人"]

GOOD = [
    {"chunk_id": "e1", "score": 0.92,
     "content": "备件目录：这款发动机滤清器的库存数量 15 件，低于 20 件触发补货预警。"},
    {"chunk_id": "e2", "score": 0.85,
     "content": "出库审批人：备件出库由仓储主管审批，单笔超过 50 件需部门经理复核。"},
]
TOO_FEW = [GOOD[0], {"chunk_id": "e2b", "score": 0.31, "content": "无关的食堂菜单公告。"}]
CONFLICT = [
    {"chunk_id": "e1", "score": 0.92,
     "content": "备件目录：这款发动机滤清器的库存数量 15 件。"},
    {"chunk_id": "e3", "score": 0.88,
     "content": "盘点快照：这款发动机滤清器的库存数量 5 件。"},
    GOOD[1],
]


def report(tag: str, verdict) -> None:
    print(f"  {tag}")
    print(f"    state          : {verdict.state.value}")
    print(f"    supported_by   : {verdict.supported_by}")
    print(f"    missing_info   : {verdict.missing_info}")
    if verdict.conflicts:
        for c in verdict.conflicts:
            print(f"    conflict       : {c.fact} -> {c.values} (证据 {c.evidence_ids})")
    print(f"    abstain        : {verdict.abstain}  由 {verdict.decided_by} 决定"
          f"  confidence={verdict.confidence:.2f}")
    print(f"    reasoning      : {verdict.reasoning}\n")


def main() -> int:
    judge = SufficiencyJudge(min_evidence=2, relevance_threshold=0.6)

    print("—— 1. 证据充分 ——")
    report("两条高相关证据覆盖全部关键信息", judge.judge(QUERY, GOOD, REQUIRED))

    print("—— 2. 证据不足（第 2 条相关性低于阈值）——")
    report("首轮", judge.judge(QUERY, TOO_FEW, REQUIRED, iteration=0))
    report("已到迭代上限 → 弃答而不是编造",
           judge.judge(QUERY, TOO_FEW, REQUIRED, iteration=3, max_iterations=3))

    print("—— 3. 证据打架 ——")
    report("同一事实两个取值", judge.judge(QUERY, CONFLICT, REQUIRED))

    print("—— 4. LLM 只能加严，不能反向抬升（fail-closed）——")

    def optimistic_llm(prompt: str) -> str:
        # 假设模型乐观地宣称"证据已经够了"
        return '{"state": "SUFFICIENT", "supported_by": ["e1"], "reasoning": "看起来够了"}'

    strict = SufficiencyJudge(llm_callable=optimistic_llm,
                              min_evidence=2, relevance_threshold=0.6)
    verdict = strict.judge(QUERY, TOO_FEW, REQUIRED, iteration=0)
    print(f"    规则判 {verdict.state.value}，LLM 想升成 SUFFICIENT —— 结果仍是 "
          f"{verdict.state.value}（decided_by={verdict.decided_by}）")
    print("    说明：LLM 只在规则已判充分后被调用，且只允许把结论改严；"
          "它不能把「不够」说成「够」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
