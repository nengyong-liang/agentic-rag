"""pytest 根配置：收集期护栏。

`PYTEST_COLLECT_ONLY` 在收集阶段置位：应用侧大量模块级单例（LLM 服务、检索器、
MCP 桥、Skill 注册表）据此跳过初始化，保证 `pytest --collect-only` 不联网、
不加载模型、不产生任何 API 费用。收集结束后清除，测试内 import 时恢复正常。

`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 默认打开：本仓库的默认测试档**不需要**
模型权重；需要真实向量模型的用例各自带 skip 条件。

运行时层 `agent_harness`（Provider / 可观测 / 上下文计数）在本仓 import 时即需要：
推荐 `pip install -e ../agent-harness`；若两仓并列放在同一父目录且未安装，下面的
路径发现会让 `pytest` 直接可跑。
"""

import os
import sys
from pathlib import Path

os.environ["PYTEST_COLLECT_ONLY"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_harness_src = Path(__file__).resolve().parents[1] / "agent-harness" / "src"
if _harness_src.is_dir() and str(_harness_src) not in sys.path:
    sys.path.insert(0, str(_harness_src))


def pytest_collection_modifyitems(config, items):
    os.environ.pop("PYTEST_COLLECT_ONLY", None)
