"""
Skill 抽象层模块
将多个 Tool 组合为标准化 SOP（Skill），提供版本化与回滚机制

设计理念（对标工业级 Agent 平台的 Skill 体系）：
- Skill = Tool 组合 SOP + Prompt + 约束 + 版本化
- 声明式 JSON 定义，易维护（无需改代码即可新增/调整 Skill）
- 按 prompt_template 的 SOP 步骤顺序调用工具，遵守 constraints
  （max_tool_calls / timeout_seconds / retry_on_failure / max_retries）

与 mcp_bridge.py 的关系：
- SkillManager 不重复实现工具，而是编排 mcp_tool_bridge.execute_tool
- 延迟导入 mcp_bridge 避免循环依赖
"""

import json
import os
import time
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Skill 数据类：包含多个 Tool 的标准化操作流程定义"""

    name: str                                    # Skill 名称（唯一标识）
    description: str                             # Skill 描述
    tools: List[str]                             # 包含的 Tool 名称列表（SOP 顺序）
    prompt_template: str                         # Skill 级 Prompt 模板（SOP 步骤说明，供 Agent/LLM 遵循）
    constraints: Dict[str, Any]                  # 约束逻辑（max_tool_calls/timeout_seconds/retry_on_failure/max_retries）
    version: str = "1.0.0"                       # 版本号（语义化版本）
    rollback_on_failure: bool = False            # 失败是否回滚（中断后续步骤）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """从字典构造 Skill（兼容 JSON 反序列化）"""
        now = datetime.now().isoformat()
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tools=list(data.get("tools", [])),
            prompt_template=data.get("prompt_template", ""),
            constraints=dict(data.get("constraints", {})),
            version=data.get("version", "1.0.0"),
            rollback_on_failure=bool(data.get("rollback_on_failure", False)),
            created_at=data.get("created_at", now),
            updated_at=data.get("updated_at", now),
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)


class SkillManager:
    """Skill 管理器：加载/注册/版本化/回滚/执行 Skill"""

    def __init__(self, skill_registry_path: str = "agent/skills/"):
        self.skill_registry_path = skill_registry_path
        self._skills: Dict[str, Skill] = {}                 # 当前激活版本 name -> Skill
        self._version_history: Dict[str, List[Skill]] = {}  # 版本历史 name -> [Skill...]（用于回滚）
        self._mcp_bridge = None                              # 延迟加载，避免循环依赖
        self._load_skills()

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _resolve_registry_path(self) -> str:
        """解析技能注册目录路径，显式传入优先，其次包内 skills/ 目录"""
        # 候选路径：依次尝试原始路径 / 模块同级 skills / cwd 下 agent/skills
        candidates = [
            self.skill_registry_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills"),
            os.path.join("agent", "skills"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        # 全部未命中则返回原始路径（_load_skills 会处理不存在的情况）
        return self.skill_registry_path

    def _get_mcp_bridge(self):
        """延迟导入 mcp_tool_bridge，避免循环依赖"""
        if self._mcp_bridge is None:
            from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
            self._mcp_bridge = mcp_tool_bridge
        return self._mcp_bridge

    def _load_skills(self) -> None:
        """从 skill_registry_path 加载所有 JSON 技能定义"""
        registry_path = self._resolve_registry_path()
        if not os.path.isdir(registry_path):
            logger.warning(f"[SkillManager] 技能注册目录不存在：{registry_path}")
            return

        loaded = 0
        for filename in sorted(os.listdir(registry_path)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(registry_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                # 同名技能保留较新版本（按 version 字段去重）
                self._skills[skill.name] = skill
                self._version_history.setdefault(skill.name, []).append(skill)
                loaded += 1
                logger.info(f"[SkillManager] 加载技能：{skill.name} v{skill.version}")
            except Exception as e:
                logger.error(f"[SkillManager] 加载技能失败：{filename}，错误：{e}")

        logger.info(f"[SkillManager] 共加载 {loaded} 个技能（目录：{registry_path}）")

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get_skill(self, name: str) -> Optional[Skill]:
        """获取指定名称的当前激活版本 Skill"""
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        """列出所有 Skill（当前激活版本）"""
        return list(self._skills.values())

    # ------------------------------------------------------------------
    # 执行接口
    # ------------------------------------------------------------------
    def execute_skill(self, name: str, args: Dict, context: Dict) -> Dict[str, Any]:
        """
        按 SOP 执行技能：根据 prompt_template 步骤顺序调用工具，遵守 constraints

        执行契约：
        - 步骤来源：优先使用 args["tool_steps"]（显式指定调用序列），否则使用 skill.tools 全部
        - 工具参数：从 args[tool_name] 取该工具的参数字典，缺省为空 dict
        - 工具范围校验：tool_steps 中的工具必须在 skill.tools 允许列表内
        - 约束检查：
            * max_tool_calls: 限制最大工具调用次数（含显式 tool_steps）
            * timeout_seconds: 整体执行超时（超时则中断后续步骤）
            * retry_on_failure + max_retries: 单工具失败时重试次数

        Args:
            name: 技能名称
            args: 执行参数，可包含 tool_steps 与各工具参数（按工具名分组）
            context: 上下文（session_id 等，预留扩展，本方法不强制使用）

        Returns:
            Dict: 执行结果，含 success/skill/version/steps/violations/elapsed_seconds/message
        """
        skill = self._skills.get(name)
        if skill is None:
            return {
                "success": False,
                "data": None,
                "message": f"未找到技能：{name}"
            }

        args = args if isinstance(args, dict) else {}
        constraints = skill.constraints or {}
        max_tool_calls = int(constraints.get("max_tool_calls", len(skill.tools)))
        timeout_seconds = float(constraints.get("timeout_seconds", 30))
        retry_on_failure = bool(constraints.get("retry_on_failure", False))
        max_retries = int(constraints.get("max_retries", 1))

        # 决定执行步骤序列
        tool_steps = args.get("tool_steps")
        if not tool_steps:
            tool_steps = skill.tools

        # 约束：最大调用次数
        if len(tool_steps) > max_tool_calls:
            return {
                "success": False,
                "data": None,
                "skill": name,
                "version": skill.version,
                "message": f"工具调用次数超限：要求 {len(tool_steps)} 次，上限 {max_tool_calls} 次",
                "violations": ["max_tool_calls_exceeded"]
            }

        bridge = self._get_mcp_bridge()
        start_time = time.time()
        results: List[Dict[str, Any]] = []
        violations: List[str] = []

        # 安全集成：工具调用经沙箱授权（ToolAuthorizer）+ HITL 审批后执行。
        # 真实执行链由 MCPToolBridge._pre_check_tool 落实（FSWhitelist / ToolAuthorizer /
        # HITL ApprovalNode），此处对桥接器统一入口做前置检查；Mock 桥接器无该能力时跳过。
        _sandbox_check = getattr(bridge, "_pre_check_tool", None)

        for step_index, tool_name in enumerate(tool_steps):
            # 约束：整体超时
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                violations.append(f"timeout_exceeded_at_step_{step_index + 1}")
                logger.warning(
                    f"[SkillManager] 技能 {name} 在第 {step_index + 1} 步超时"
                    f"（{elapsed:.1f}s > {timeout_seconds}s）"
                )
                break

            # 约束：工具必须在 Skill 允许范围内
            if tool_name not in skill.tools:
                violations.append(f"tool_not_in_skill:{tool_name}")
                logger.warning(
                    f"[SkillManager] 工具 {tool_name} 不在技能 {name} 的允许列表中，跳过"
                )
                results.append({
                    "step": step_index + 1,
                    "tool": tool_name,
                    "success": False,
                    "result": None,
                    "error": f"工具 {tool_name} 不在技能 {name} 的允许列表中"
                })
                continue

            # 提取该工具的参数（按工具名分组）
            tool_args = args.get(tool_name, {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            # 带重试的工具调用
            max_attempt = (max_retries + 1) if retry_on_failure else 1
            attempt = 0
            tool_result: Optional[Dict[str, Any]] = None
            last_error: Optional[str] = None

            # 沙箱授权 + HITL 审批前置检查（真实桥接器生效；Mock 桥接器跳过以保证单测隔离）
            if _sandbox_check is not None:
                step_args = args.get(tool_name, {})
                if not isinstance(step_args, dict):
                    step_args = {}
                _check = _sandbox_check(
                    tool_name,
                    step_args,
                    context.get("session_id", "default") if isinstance(context, dict) else "default",
                )
                if not _check.get("approved"):
                    results.append({
                        "step": step_index + 1,
                        "tool": tool_name,
                        "success": False,
                        "result": None,
                        "error": _check.get("message", "沙箱/审批拒绝"),
                    })
                    violations.append(f"sandbox_rejected:{tool_name}")
                    logger.warning(
                        f"[SkillManager] 技能 {name} 第 {step_index + 1} 步被沙箱/审批拒绝："
                        f"{_check.get('message')}"
                    )
                    if skill.rollback_on_failure:
                        violations.append(f"rollback_triggered_at_step_{step_index + 1}")
                        break
                    continue

            while attempt < max_attempt:
                try:
                    logger.info(
                        f"[SkillManager] 技能 {name} 第 {step_index + 1} 步调用工具 "
                        f"{tool_name}（尝试 {attempt + 1}/{max_attempt}）"
                    )
                    tool_result = bridge.execute_tool(tool_name, tool_args)
                    if isinstance(tool_result, dict) and tool_result.get("success"):
                        break
                    # 工具返回失败
                    last_error = (
                        tool_result.get("message", "工具执行失败")
                        if isinstance(tool_result, dict) else "工具执行失败"
                    )
                    if retry_on_failure and attempt < max_attempt - 1:
                        logger.warning(f"[SkillManager] 工具 {tool_name} 执行失败，将重试：{last_error}")
                        attempt += 1
                        continue
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[SkillManager] 工具 {tool_name} 执行异常：{e}")
                    if retry_on_failure and attempt < max_attempt - 1:
                        attempt += 1
                        continue
                    break

            step_success = bool(
                tool_result and isinstance(tool_result, dict) and tool_result.get("success")
            )
            results.append({
                "step": step_index + 1,
                "tool": tool_name,
                "success": step_success,
                "result": tool_result,
                "error": None if step_success else last_error
            })

            # 约束：失败回滚（中断后续步骤）
            if not step_success and skill.rollback_on_failure:
                violations.append(f"rollback_triggered_at_step_{step_index + 1}")
                logger.warning(
                    f"[SkillManager] 技能 {name} 在第 {step_index + 1} 步失败，"
                    f"触发回滚（rollback_on_failure=True）"
                )
                break

        success = bool(results) and all(r["success"] for r in results)
        elapsed_total = time.time() - start_time

        return {
            "success": success,
            "data": results,
            "skill": name,
            "version": skill.version,
            "steps": results,
            "violations": violations,
            "elapsed_seconds": round(elapsed_total, 3),
            "message": "技能执行完成" if success else "技能执行失败（详见 violations/steps）"
        }

    # ------------------------------------------------------------------
    # 注册 / 版本化 / 回滚
    # ------------------------------------------------------------------
    def register_skill(self, skill: Skill) -> bool:
        """
        注册新技能或更新现有技能
        - 同名同版本：仅刷新 updated_at（视为更新元数据）
        - 同名异版本：归档当前版本，激活新版本
        - 新增技能：直接激活并记入版本历史
        """
        if not isinstance(skill, Skill):
            logger.error("[SkillManager] register_skill 入参非 Skill 实例")
            return False

        existing = self._skills.get(skill.name)
        if existing and existing.version == skill.version:
            # 同版本：保留原 created_at，刷新 updated_at
            skill.created_at = existing.created_at
            skill.updated_at = datetime.now().isoformat()
            self._skills[skill.name] = skill
            logger.info(f"[SkillManager] 技能 {skill.name} v{skill.version} 已刷新")
            return True

        # 新增或版本升级
        self._skills[skill.name] = skill
        self._version_history.setdefault(skill.name, []).append(skill)
        logger.info(f"[SkillManager] 注册技能 {skill.name} v{skill.version}")
        return True

    def version_skill(self, name: str, new_version: str) -> Optional[str]:
        """
        升级技能版本：在当前版本基础上创建新版本并激活（拷贝当前定义，仅改版本号）

        Args:
            name: 技能名称
            new_version: 新版本号（如 "1.1.0"）

        Returns:
            激活的版本号，失败返回 None
        """
        skill = self._skills.get(name)
        if skill is None:
            logger.error(f"[SkillManager] 版本化失败：技能 {name} 不存在")
            return None

        if new_version == skill.version:
            logger.warning(f"[SkillManager] 新版本号与当前版本相同：{new_version}")
            return skill.version

        new_skill = Skill(
            name=skill.name,
            description=skill.description,
            tools=list(skill.tools),
            prompt_template=skill.prompt_template,
            constraints=deepcopy(skill.constraints),
            version=new_version,
            rollback_on_failure=skill.rollback_on_failure,
            created_at=skill.created_at,
            updated_at=datetime.now().isoformat(),
        )
        self._skills[name] = new_skill
        self._version_history.setdefault(name, []).append(new_skill)
        logger.info(f"[SkillManager] 技能 {name} 从 v{skill.version} 升级到 v{new_version}")
        return new_version

    def rollback_skill(self, name: str, to_version: str) -> bool:
        """
        回滚技能到指定历史版本

        Args:
            name: 技能名称
            to_version: 目标版本号

        Returns:
            是否回滚成功
        """
        history = self._version_history.get(name, [])
        target = None
        for skill in history:
            if skill.version == to_version:
                target = skill
                break

        if target is None:
            logger.error(f"[SkillManager] 回滚失败：技能 {name} 无版本 {to_version}")
            return False

        # 创建回滚快照（更新 updated_at，便于追溯回滚动作）
        rolled_back = Skill(
            name=target.name,
            description=target.description,
            tools=list(target.tools),
            prompt_template=target.prompt_template,
            constraints=deepcopy(target.constraints),
            version=to_version,
            rollback_on_failure=target.rollback_on_failure,
            created_at=target.created_at,
            updated_at=datetime.now().isoformat(),
        )
        self._skills[name] = rolled_back
        # 回滚版本也记入历史，便于审计
        self._version_history[name].append(rolled_back)
        logger.info(f"[SkillManager] 技能 {name} 已回滚到 v{to_version}")
        return True


# 模块级单例：与 mcp_tool_bridge 保持一致的导出风格
if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    skill_manager = SkillManager()
else:
    skill_manager = None
