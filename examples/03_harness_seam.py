"""示例 3：应用层与运行时层的装配接缝（A ⇄ B）。

`agentic-rag` 不自己实现主循环，而是在运行时向 `agent-harness` 要一个
`HarnessLoop`；拿不到就回退到内置的手动循环。这个文件把这条接缝显式跑一遍：

  1. 正常装配：主循环 / 上下文投影 / 终止校验 / 熔断 / 安全网关 / 审计 / 事件流
     全部来自 agent-harness，工具执行器指向本仓的 MCP bridge；
  2. 缺失装配：把 `agent_harness` 从可导入集合里摘掉，确认回退路径真的可用
     （不是写在 README 里的口头承诺）。

零 API：只装配、不发起模型调用。

运行：
    python examples/03_harness_seam.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

# Provider / 可观测 / 上下文计数来自 agent-harness：优先用已安装的包，
# 其次发现并列放置的 ../agent-harness/src。
try:
    import agent_harness  # noqa: F401
except ModuleNotFoundError:
    _sibling = os.path.abspath(os.path.join(HERE, "..", "..", "agent-harness", "src"))
    if not os.path.isdir(_sibling):
        sys.exit("缺少 agent-harness：pip install -e ../agent-harness，"
                 "或把两仓并列放在同一个父目录下")
    sys.path.insert(0, _sibling)

from agentic_rag.agent.agent import RAGAgent  # noqa: E402


def describe(loop) -> None:
    cfg = loop.config
    executor = loop._tool_executor
    executor_name = (getattr(executor, "__qualname__", None)
                     or type(executor).__name__) if executor else None
    print(f"  主循环          : {type(loop).__module__}.{type(loop).__name__}")
    print(f"  模型 Provider   : {loop._model_provider.get_model_name()}")
    print(f"  工具执行器      : {executor_name}（本仓 MCP bridge）")
    print(f"  max_iterations  : {cfg.max_iterations}")
    print(f"  安全网关        : {cfg.enable_safety_gateway}")
    print(f"  HITL 审批       : {cfg.enable_hitl_approval}")
    print(f"  目标自动编译    : {cfg.auto_compile_goal}"
          f"（compiler={type(cfg.task_compiler).__name__ if cfg.task_compiler else None}）")


def main() -> int:
    agent = RAGAgent()

    print("—— 1. 装配成功（已安装 agent-harness）——")
    loop = agent._get_harness()
    if loop is None:
        print("  未装配：agent-harness 不可用（pip install -e ../agent-harness）")
        return 1
    describe(loop)

    print("\n—— 2. 摘掉 agent_harness 后：回退手动循环 ——")
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "agent_harness" or name.startswith("agent_harness."):
            raise ImportError(f"simulated: no module named {name}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        fallback = agent._get_harness()
    finally:
        builtins.__import__ = real_import

    print(f"  _get_harness() -> {fallback}")
    print("  返回 None 时 run_agent() 走 _run_manual() 内置手动循环："
          "业务链路不会因缺少运行时层而中断。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
