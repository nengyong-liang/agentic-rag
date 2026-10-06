"""
Agent 长程任务管理系统
支持断点恢复、步骤回滚、状态持久化

=== 面试要点 ===

1. 长程任务 vs 短程任务：
   - 短程任务：单次 LLM 调用即可完成（如简单问答）
   - 长程任务：Agent 需要执行多步骤任务（如：查询库存→生成工单→通知相关人员），
     不能假设一次完成，需要逐步推进并记录中间状态

2. 断点恢复（类似数据库 WAL）：
   - WAL（Write-Ahead Log）先写日志再执行，崩溃后可重放恢复
   - 本系统每完成一个步骤就持久化状态（checkpoint），进程崩溃后从最近 checkpoint 恢复
   - 与"从头重做"的区别：跳过已完成步骤，直接从断点继续，节省时间和资源

3. 步骤回滚（不是 undo）：
   - 不是撤销已执行的操作（如已创建的工单无法撤销）
   - 而是标记后续步骤为 pending，让它们重新执行
   - 适用场景：某步失败后，修复条件再重试后续步骤

4. 与 LangGraph checkpoint 的区别：
   - LangGraph 是图级 checkpoint：保存整个图的状态（节点+边），用于图执行的断点恢复
   - 本系统是任务级 checkpoint：保存任务步骤的执行进度，用于线性多步骤任务的断点恢复
   - LangGraph 侧重于复杂图的分支/循环，本系统侧重于简单线性任务的可靠性
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# 可选 Redis 依赖
has_redis = False
try:
    import redis
    has_redis = True
except ImportError:
    pass


# ============================================================
# 核心数据结构
# ============================================================

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    """任务步骤

    每个步骤包含输入/输出数据、执行状态、重试计数等信息，
    是断点恢复和步骤回滚的基本单位。
    """
    step_id: str
    step_name: str
    status: str  # "pending" / "running" / "completed" / "failed" / "skipped"
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class LongRunningTask:
    """长程任务

    包含多个步骤的线性任务，支持断点恢复和步骤回滚。
    checkpoint_data 保存断点恢复所需的关键数据。
    """
    task_id: str
    task_name: str
    description: str
    status: TaskStatus
    steps: List[TaskStep]
    current_step_index: int
    created_at: str
    updated_at: str
    checkpoint_data: Optional[Dict[str, Any]]  # 断点恢复数据
    metadata: Dict[str, Any]
    parent_task_id: Optional[str] = None  # 支持子任务


@dataclass
class TaskConfig:
    """任务管理器配置"""
    max_steps: int = 20
    step_timeout_seconds: int = 300
    max_retries_per_step: int = 2
    enable_checkpoint: bool = True
    checkpoint_interval: int = 1  # 每完成 N 步保存一次
    storage_backend: str = "memory"  # "memory" / "redis" / "file"
    file_storage_dir: str = ""  # file 后端的存储目录


# ============================================================
# 存储后端
# ============================================================

class MemoryStorage:
    """内存存储后端（默认）"""

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    def save_task(self, task_data: Dict[str, Any]) -> bool:
        task_id = task_data.get("task_id", "")
        self._tasks[task_id] = task_data
        return True

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_id)

    def delete_task(self, task_id: str) -> bool:
        self._tasks.pop(task_id, None)
        self._checkpoints.pop(task_id, None)
        return True

    def list_tasks(self) -> List[Dict[str, Any]]:
        return list(self._tasks.values())

    def save_checkpoint(self, task_id: str, checkpoint: Dict[str, Any]) -> bool:
        self._checkpoints[task_id] = checkpoint
        return True

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._checkpoints.get(task_id)

    def delete_checkpoint(self, task_id: str) -> bool:
        self._checkpoints.pop(task_id, None)
        return True


class RedisStorage:
    """Redis 存储后端（可选）

    使用项目已有的 Redis 服务，以 JSON 字符串存储任务和 checkpoint 数据。
    """

    TASK_PREFIX = "long_running_task:"
    CHECKPOINT_PREFIX = "long_running_task_checkpoint:"

    def __init__(self, redis_client=None):
        self._redis = redis_client
        if self._redis is None and has_redis:
            try:
                from agentic_rag.config import settings
                self._redis = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True
                )
                self._redis.ping()
                logger.info("[RedisStorage] Redis 连接成功")
            except Exception as e:
                logger.error(f"[RedisStorage] Redis 连接失败: {e}")
                self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def save_task(self, task_data: Dict[str, Any]) -> bool:
        if not self.available:
            return False
        try:
            key = f"{self.TASK_PREFIX}{task_data['task_id']}"
            self._redis.set(key, json.dumps(task_data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.error(f"[RedisStorage] 保存任务失败: {e}")
            return False

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            key = f"{self.TASK_PREFIX}{task_id}"
            data = self._redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"[RedisStorage] 加载任务失败: {e}")
            return None

    def delete_task(self, task_id: str) -> bool:
        if not self.available:
            return False
        try:
            self._redis.delete(f"{self.TASK_PREFIX}{task_id}")
            self._redis.delete(f"{self.CHECKPOINT_PREFIX}{task_id}")
            return True
        except Exception as e:
            logger.error(f"[RedisStorage] 删除任务失败: {e}")
            return False

    def list_tasks(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            keys = self._redis.keys(f"{self.TASK_PREFIX}*")
            result = []
            for key in keys:
                data = self._redis.get(key)
                if data:
                    result.append(json.loads(data))
            return result
        except Exception as e:
            logger.error(f"[RedisStorage] 列出任务失败: {e}")
            return []

    def save_checkpoint(self, task_id: str, checkpoint: Dict[str, Any]) -> bool:
        if not self.available:
            return False
        try:
            key = f"{self.CHECKPOINT_PREFIX}{task_id}"
            self._redis.set(key, json.dumps(checkpoint, ensure_ascii=False))
            return True
        except Exception as e:
            logger.error(f"[RedisStorage] 保存 checkpoint 失败: {e}")
            return False

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            key = f"{self.CHECKPOINT_PREFIX}{task_id}"
            data = self._redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"[RedisStorage] 加载 checkpoint 失败: {e}")
            return None

    def delete_checkpoint(self, task_id: str) -> bool:
        if not self.available:
            return False
        try:
            self._redis.delete(f"{self.CHECKPOINT_PREFIX}{task_id}")
            return True
        except Exception as e:
            logger.error(f"[RedisStorage] 删除 checkpoint 失败: {e}")
            return False


class FileStorage:
    """文件存储后端（JSON 文件持久化，可选）"""

    def __init__(self, storage_dir: str = ""):
        if storage_dir:
            self._dir = storage_dir
        else:
            self._dir = os.path.join(os.path.dirname(__file__), ".task_storage")
        os.makedirs(self._dir, exist_ok=True)

    def _task_path(self, task_id: str) -> str:
        return os.path.join(self._dir, f"task_{task_id}.json")

    def _checkpoint_path(self, task_id: str) -> str:
        return os.path.join(self._dir, f"checkpoint_{task_id}.json")

    def save_task(self, task_data: Dict[str, Any]) -> bool:
        try:
            path = self._task_path(task_data["task_id"])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"[FileStorage] 保存任务失败: {e}")
            return False

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            path = self._task_path(task_id)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[FileStorage] 加载任务失败: {e}")
            return None

    def delete_task(self, task_id: str) -> bool:
        try:
            task_path = self._task_path(task_id)
            cp_path = self._checkpoint_path(task_id)
            if os.path.exists(task_path):
                os.remove(task_path)
            if os.path.exists(cp_path):
                os.remove(cp_path)
            return True
        except Exception as e:
            logger.error(f"[FileStorage] 删除任务失败: {e}")
            return False

    def list_tasks(self) -> List[Dict[str, Any]]:
        try:
            result = []
            for filename in os.listdir(self._dir):
                if filename.startswith("task_") and filename.endswith(".json"):
                    path = os.path.join(self._dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        result.append(json.load(f))
            return result
        except Exception as e:
            logger.error(f"[FileStorage] 列出任务失败: {e}")
            return []

    def save_checkpoint(self, task_id: str, checkpoint: Dict[str, Any]) -> bool:
        try:
            path = self._checkpoint_path(task_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"[FileStorage] 保存 checkpoint 失败: {e}")
            return False

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            path = self._checkpoint_path(task_id)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[FileStorage] 加载 checkpoint 失败: {e}")
            return None

    def delete_checkpoint(self, task_id: str) -> bool:
        try:
            path = self._checkpoint_path(task_id)
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception as e:
            logger.error(f"[FileStorage] 删除 checkpoint 失败: {e}")
            return False


# ============================================================
# 序列化工具
# ============================================================

def _task_to_dict(task: LongRunningTask) -> Dict[str, Any]:
    """将 LongRunningTask 转换为可序列化的字典"""
    return {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "description": task.description,
        "status": task.status.value,
        "steps": [asdict(s) for s in task.steps],
        "current_step_index": task.current_step_index,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "checkpoint_data": task.checkpoint_data,
        "metadata": task.metadata,
        "parent_task_id": task.parent_task_id,
    }


def _dict_to_task(data: Dict[str, Any]) -> LongRunningTask:
    """将字典转换为 LongRunningTask"""
    steps = []
    for s in data.get("steps", []):
        steps.append(TaskStep(**s))
    return LongRunningTask(
        task_id=data["task_id"],
        task_name=data["task_name"],
        description=data["description"],
        status=TaskStatus(data["status"]),
        steps=steps,
        current_step_index=data["current_step_index"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        checkpoint_data=data.get("checkpoint_data"),
        metadata=data.get("metadata", {}),
        parent_task_id=data.get("parent_task_id"),
    )


# ============================================================
# TaskManager 核心类
# ============================================================

class TaskManager:
    """长程任务管理器

    支持断点恢复、步骤回滚、状态持久化的多步骤任务管理系统。

    用法示例::

        manager = TaskManager()

        # 创建任务
        steps_def = [
            {"step_name": "查询库存", "input_data": {"part_number": "ENG-001"}},
            {"step_name": "生成工单", "input_data": {"vehicle_model": "长安CS75"}},
        ]
        task_id = manager.create_task("库存查询与工单生成", "查询配件库存并生成维修工单", steps_def)

        # 定义步骤执行器
        async def executor(step: TaskStep, context: Dict) -> Dict:
            # 根据步骤名称执行不同逻辑
            if step.step_name == "查询库存":
                return {"stock": 150, "price": 25.0}
            elif step.step_name == "生成工单":
                return {"order_number": "WO-2024-001"}
            return {}

        # 执行任务
        result = await manager.execute_task(task_id, executor)

        # 暂停/恢复
        manager.pause_task(task_id)
        result = await manager.resume_task(task_id, executor)
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        """初始化任务管理器

        Args:
            config: 任务配置，为 None 时使用默认配置
        """
        self.config = config or TaskConfig()
        self._storage = self._create_storage()
        # 内存中的任务缓存，避免频繁反序列化
        self._task_cache: Dict[str, LongRunningTask] = {}
        # 运行中的任务事件循环引用，用于取消
        self._running_tasks: Dict[str, asyncio.Task] = {}
        logger.info(f"[TaskManager] 初始化完成，存储后端: {self.config.storage_backend}")

    def _create_storage(self):
        """根据配置创建存储后端"""
        backend = self.config.storage_backend
        if backend == "redis":
            storage = RedisStorage()
            if not storage.available:
                logger.warning("[TaskManager] Redis 不可用，回退到内存存储")
                return MemoryStorage()
            return storage
        elif backend == "file":
            return FileStorage(storage_dir=self.config.file_storage_dir)
        else:
            return MemoryStorage()

    def _now(self) -> str:
        """获取当前时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _persist_task(self, task: LongRunningTask) -> None:
        """持久化任务到存储后端"""
        task.updated_at = self._now()
        self._task_cache[task.task_id] = task
        self._storage.save_task(_task_to_dict(task))

    def _load_task(self, task_id: str) -> Optional[LongRunningTask]:
        """从缓存或存储后端加载任务"""
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        data = self._storage.load_task(task_id)
        if data is None:
            return None
        task = _dict_to_task(data)
        self._task_cache[task_id] = task
        return task

    # ----------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------

    def create_task(
        self,
        name: str,
        description: str,
        steps_def: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
    ) -> str:
        """创建长程任务

        Args:
            name: 任务名称
            description: 任务描述
            steps_def: 步骤定义列表，每项包含 step_name 和 input_data
            metadata: 任务元数据
            parent_task_id: 父任务 ID（支持子任务）

        Returns:
            任务 ID

        Raises:
            ValueError: 步骤数量超过限制
        """
        if len(steps_def) > self.config.max_steps:
            raise ValueError(
                f"步骤数量 {len(steps_def)} 超过最大限制 {self.config.max_steps}"
            )

        task_id = str(uuid.uuid4())
        now = self._now()

        steps = []
        for i, step_def in enumerate(steps_def):
            step = TaskStep(
                step_id=f"{task_id}_step_{i}",
                step_name=step_def.get("step_name", f"步骤{i + 1}"),
                status="pending",
                input_data=step_def.get("input_data", {}),
                output_data=None,
                error=None,
                started_at=None,
                completed_at=None,
                retry_count=0,
                max_retries=step_def.get("max_retries", self.config.max_retries_per_step),
            )
            steps.append(step)

        task = LongRunningTask(
            task_id=task_id,
            task_name=name,
            description=description,
            status=TaskStatus.PENDING,
            steps=steps,
            current_step_index=0,
            created_at=now,
            updated_at=now,
            checkpoint_data=None,
            metadata=metadata or {},
            parent_task_id=parent_task_id,
        )

        self._persist_task(task)
        logger.info(f"[TaskManager] 创建任务: {name}（ID={task_id}，{len(steps)} 步）")
        return task_id

    async def execute_task(
        self,
        task_id: str,
        executor_fn: Callable,
    ) -> Dict[str, Any]:
        """执行任务

        按顺序执行任务中的每个步骤，支持断点恢复和自动重试。
        executor_fn 签名：async def executor(step: TaskStep, context: Dict) -> Dict

        Args:
            task_id: 任务 ID
            executor_fn: 步骤执行函数

        Returns:
            包含执行结果的字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if task.status == TaskStatus.COMPLETED:
            return {"success": True, "message": "任务已完成", "task_id": task_id}
        if task.status == TaskStatus.CANCELLED:
            return {"success": False, "message": "任务已取消", "task_id": task_id}
        if task.status == TaskStatus.RUNNING:
            return {"success": False, "message": "任务正在运行中", "task_id": task_id}

        # 如果有 checkpoint，从断点恢复
        if task.checkpoint_data:
            start_index = task.checkpoint_data.get("current_step_index", 0)
            task.current_step_index = start_index
            logger.info(f"[TaskManager] 从断点恢复，起始步骤: {start_index}")

        task.status = TaskStatus.RUNNING
        self._persist_task(task)

        # 构建已完成步骤的输出上下文
        completed_outputs = task.checkpoint_data.get("completed_outputs", {}) if task.checkpoint_data else {}

        try:
            while task.current_step_index < len(task.steps):
                step = task.steps[task.current_step_index]

                # 跳过已完成的步骤（断点恢复场景）
                if step.status == "completed":
                    logger.info(f"[TaskManager] 跳过已完成步骤: {step.step_name}")
                    task.current_step_index += 1
                    continue

                # 跳过被标记为 skipped 的步骤
                if step.status == "skipped":
                    task.current_step_index += 1
                    continue

                # 执行步骤
                step_result = await self._execute_step(task, step, executor_fn, completed_outputs)

                if step_result["success"]:
                    # 步骤成功，记录输出
                    step.status = "completed"
                    step.output_data = step_result.get("output", {})
                    step.completed_at = self._now()
                    completed_outputs[step.step_id] = step.output_data
                    task.current_step_index += 1

                    # 按配置间隔保存 checkpoint
                    if self.config.enable_checkpoint and task.current_step_index % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(task_id)
                    else:
                        self._persist_task(task)
                else:
                    # 步骤失败
                    step.status = "failed"
                    step.error = step_result.get("error", "未知错误")
                    step.completed_at = self._now()

                    # 检查是否可重试
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        step.status = "pending"  # 重置为待执行，下次可重试
                        logger.warning(
                            f"[TaskManager] 步骤 {step.step_name} 失败，"
                            f"准备重试（{step.retry_count}/{step.max_retries}）"
                        )
                        self._persist_task(task)
                        continue
                    else:
                        # 重试次数耗尽，任务失败
                        task.status = TaskStatus.FAILED
                        self._persist_task(task)
                        logger.error(
                            f"[TaskManager] 任务失败: {task.task_name}，"
                            f"步骤 {step.step_name} 重试 {step.max_retries} 次后仍失败"
                        )
                        return {
                            "success": False,
                            "message": f"任务失败：步骤 {step.step_name} 执行失败",
                            "task_id": task_id,
                            "failed_step": step.step_name,
                            "error": step.error,
                        }

            # 所有步骤完成
            task.status = TaskStatus.COMPLETED
            self._persist_task(task)
            logger.info(f"[TaskManager] 任务完成: {task.task_name}（ID={task_id}）")
            return {
                "success": True,
                "message": "任务完成",
                "task_id": task_id,
                "completed_outputs": completed_outputs,
            }

        except asyncio.CancelledError:
            task.status = TaskStatus.PAUSED
            self._persist_task(task)
            logger.info(f"[TaskManager] 任务被取消/暂停: {task.task_name}")
            return {"success": False, "message": "任务被取消/暂停", "task_id": task_id}
        except Exception as e:
            task.status = TaskStatus.FAILED
            self._persist_task(task)
            logger.error(f"[TaskManager] 任务执行异常: {e}")
            return {"success": False, "message": f"任务执行异常: {str(e)}", "task_id": task_id}

    async def _execute_step(
        self,
        task: LongRunningTask,
        step: TaskStep,
        executor_fn: Callable,
        completed_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行单个步骤

        支持同步和异步执行器，带超时处理。

        Args:
            task: 任务对象
            step: 步骤对象
            executor_fn: 执行器函数
            completed_outputs: 已完成步骤的输出

        Returns:
            包含执行结果的字典
        """
        step.status = "running"
        step.started_at = self._now()
        self._persist_task(task)

        # 构建上下文：之前所有步骤的输出 + 任务元数据 + checkpoint 数据
        context = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_metadata": task.metadata,
            "completed_outputs": completed_outputs,
            "checkpoint_data": task.checkpoint_data,
            "current_step_index": task.current_step_index,
        }

        try:
            # 支持同步和异步执行器
            if asyncio.iscoroutinefunction(executor_fn):
                result = await asyncio.wait_for(
                    executor_fn(step, context),
                    timeout=self.config.step_timeout_seconds,
                )
            else:
                # 同步函数在线程池中执行，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: executor_fn(step, context)),
                    timeout=self.config.step_timeout_seconds,
                )

            return {"success": True, "output": result if isinstance(result, dict) else {"result": result}}

        except asyncio.TimeoutError:
            error_msg = f"步骤超时（{self.config.step_timeout_seconds}秒）"
            logger.warning(f"[TaskManager] {error_msg}: {step.step_name}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[TaskManager] 步骤执行异常: {step.step_name}, error: {e}")
            return {"success": False, "error": error_msg}

    def pause_task(self, task_id: str) -> Dict[str, Any]:
        """暂停任务

        暂停后可调用 resume_task 从断点恢复。

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if task.status != TaskStatus.RUNNING:
            return {"success": False, "message": f"任务状态为 {task.status.value}，无法暂停"}

        # 保存 checkpoint
        if self.config.enable_checkpoint:
            self.save_checkpoint(task_id)

        task.status = TaskStatus.PAUSED
        self._persist_task(task)

        # 取消运行中的 asyncio.Task
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]

        logger.info(f"[TaskManager] 任务已暂停: {task.task_name}（ID={task_id}）")
        return {"success": True, "message": "任务已暂停", "task_id": task_id}

    def mark_task_resumable(self, task_id: str) -> Dict[str, Any]:
        """将任务标记为可恢复状态（PENDING）

        与 resume_task 不同，此方法仅修改状态，不执行任务。
        需要外部调用 execute_task 并传入 executor_fn 才能真正恢复执行。

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if task.status not in (TaskStatus.PAUSED, TaskStatus.FAILED):
            return {"success": False, "message": f"任务状态为 {task.status.value}，无法恢复"}

        task.status = TaskStatus.PENDING
        self._persist_task(task)

        logger.info(f"[TaskManager] 任务已标记为可恢复: {task.task_name}（ID={task_id}）")
        return {"success": True, "message": "任务已标记为可恢复，等待执行器接入", "task_id": task_id}

    async def resume_task(
        self,
        task_id: str,
        executor_fn: Callable,
    ) -> Dict[str, Any]:
        """从断点恢复执行

        加载 checkpoint，跳过已完成步骤，从当前步骤继续执行。

        Args:
            task_id: 任务 ID
            executor_fn: 步骤执行函数

        Returns:
            执行结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if task.status not in (TaskStatus.PAUSED, TaskStatus.FAILED):
            return {"success": False, "message": f"任务状态为 {task.status.value}，无法恢复"}

        # 加载 checkpoint
        self.load_checkpoint(task_id)

        logger.info(f"[TaskManager] 恢复任务: {task.task_name}（ID={task_id}）")
        return await self.execute_task(task_id, executor_fn)

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消任务

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return {"success": False, "message": f"任务状态为 {task.status.value}，无法取消"}

        task.status = TaskStatus.CANCELLED
        self._persist_task(task)

        # 取消运行中的 asyncio.Task
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]

        logger.info(f"[TaskManager] 任务已取消: {task.task_name}（ID={task_id}）")
        return {"success": True, "message": "任务已取消", "task_id": task_id}

    def retry_step(self, task_id: str, step_index: int) -> Dict[str, Any]:
        """重试失败的步骤

        将指定步骤状态重置为 pending，重试计数归零。

        Args:
            task_id: 任务 ID
            step_index: 步骤索引

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if step_index < 0 or step_index >= len(task.steps):
            return {"success": False, "message": f"步骤索引越界: {step_index}"}

        step = task.steps[step_index]
        if step.status != "failed":
            return {"success": False, "message": f"步骤状态为 {step.status}，无需重试"}

        step.status = "pending"
        step.retry_count = 0
        step.error = None
        step.started_at = None
        step.completed_at = None
        task.current_step_index = step_index
        task.status = TaskStatus.PAUSED  # 设为暂停，等待手动 resume
        self._persist_task(task)

        logger.info(f"[TaskManager] 重试步骤: {step.step_name}（任务={task_id}）")
        return {"success": True, "message": f"步骤 {step.step_name} 已重置，可调用 resume_task 继续执行"}

    def rollback_step(self, task_id: str, step_index: int) -> Dict[str, Any]:
        """回滚到某一步

        不是 undo（已执行的操作无法撤销），而是将指定步骤及后续步骤标记为 pending，
        让它们重新执行。

        Args:
            task_id: 任务 ID
            step_index: 回滚到的步骤索引（该步骤也会被重新执行）

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        if step_index < 0 or step_index >= len(task.steps):
            return {"success": False, "message": f"步骤索引越界: {step_index}"}

        # 从 step_index 开始，将所有后续步骤重置为 pending
        reset_count = 0
        for i in range(step_index, len(task.steps)):
            step = task.steps[i]
            if step.status in ("completed", "failed", "running"):
                step.status = "pending"
                step.output_data = None
                step.error = None
                step.started_at = None
                step.completed_at = None
                step.retry_count = 0
                reset_count += 1

        task.current_step_index = step_index
        task.status = TaskStatus.PAUSED  # 设为暂停，等待手动 resume

        # 清除 checkpoint 中该步骤之后的输出
        if task.checkpoint_data and "completed_outputs" in task.checkpoint_data:
            for i in range(step_index, len(task.steps)):
                task.checkpoint_data["completed_outputs"].pop(task.steps[i].step_id, None)
            task.checkpoint_data["current_step_index"] = step_index

        self._persist_task(task)

        logger.info(
            f"[TaskManager] 回滚任务到步骤 {step_index}: "
            f"{task.steps[step_index].step_name}（重置 {reset_count} 步）"
        )
        return {
            "success": True,
            "message": f"已回滚到步骤 {task.steps[step_index].step_name}，{reset_count} 步将被重新执行",
            "reset_count": reset_count,
        }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务状态信息字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        completed = sum(1 for s in task.steps if s.status == "completed")
        failed = sum(1 for s in task.steps if s.status == "failed")
        pending = sum(1 for s in task.steps if s.status == "pending")

        return {
            "success": True,
            "data": {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "description": task.description,
                "status": task.status.value,
                "current_step_index": task.current_step_index,
                "total_steps": len(task.steps),
                "completed_steps": completed,
                "failed_steps": failed,
                "pending_steps": pending,
                "progress": f"{completed}/{len(task.steps)}",
                "progress_percent": round(completed / len(task.steps) * 100, 1) if task.steps else 0,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "step_name": s.step_name,
                        "status": s.status,
                        "retry_count": s.retry_count,
                        "error": s.error,
                    }
                    for s in task.steps
                ],
                "has_checkpoint": task.checkpoint_data is not None,
                "parent_task_id": task.parent_task_id,
            },
        }

    def list_tasks(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """列出所有任务

        Args:
            status_filter: 按状态筛选，如 "running"、"completed" 等

        Returns:
            任务列表字典
        """
        all_tasks_data = self._storage.list_tasks()

        tasks = []
        for data in all_tasks_data:
            task = _dict_to_task(data)
            self._task_cache[task.task_id] = task

            if status_filter and task.status.value != status_filter:
                continue

            completed = sum(1 for s in task.steps if s.status == "completed")
            tasks.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "status": task.status.value,
                "progress": f"{completed}/{len(task.steps)}",
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            })

        return {
            "success": True,
            "data": {
                "tasks": tasks,
                "total": len(tasks),
            },
        }

    def save_checkpoint(self, task_id: str) -> Dict[str, Any]:
        """保存断点

        保存当前步骤索引和所有已完成步骤的输出，用于断点恢复。
        checkpoint 数据结构：{task_id, current_step_index, completed_outputs, timestamp}

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        completed_outputs = {}
        for step in task.steps:
            if step.status == "completed" and step.output_data is not None:
                completed_outputs[step.step_id] = step.output_data

        checkpoint = {
            "task_id": task_id,
            "current_step_index": task.current_step_index,
            "completed_outputs": completed_outputs,
            "timestamp": self._now(),
        }

        task.checkpoint_data = checkpoint
        self._persist_task(task)
        self._storage.save_checkpoint(task_id, checkpoint)

        logger.info(
            f"[TaskManager] 保存 checkpoint: 任务={task.task_name}，"
            f"进度={task.current_step_index}/{len(task.steps)}"
        )
        return {"success": True, "message": "checkpoint 已保存", "task_id": task_id}

    def load_checkpoint(self, task_id: str) -> Dict[str, Any]:
        """加载断点

        从存储后端加载 checkpoint 数据，更新任务的当前步骤索引和已完成输出。

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        checkpoint = self._storage.load_checkpoint(task_id)
        if checkpoint is None:
            # 尝试从任务本身的 checkpoint_data 恢复
            if task.checkpoint_data:
                checkpoint = task.checkpoint_data
            else:
                return {"success": False, "message": "无可用 checkpoint"}

        # 恢复已完成步骤的输出
        completed_outputs = checkpoint.get("completed_outputs", {})
        for step in task.steps:
            if step.step_id in completed_outputs:
                step.output_data = completed_outputs[step.step_id]
                step.status = "completed"

        task.current_step_index = checkpoint.get("current_step_index", 0)
        task.checkpoint_data = checkpoint
        self._persist_task(task)

        logger.info(
            f"[TaskManager] 加载 checkpoint: 任务={task.task_name}，"
            f"起始步骤={task.current_step_index}"
        )
        return {"success": True, "message": "checkpoint 已加载", "task_id": task_id}

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """删除任务

        Args:
            task_id: 任务 ID

        Returns:
            操作结果字典
        """
        task = self._load_task(task_id)
        if task is None:
            return {"success": False, "message": f"任务不存在: {task_id}"}

        # 取消运行中的 asyncio.Task
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]

        self._task_cache.pop(task_id, None)
        self._storage.delete_task(task_id)

        logger.info(f"[TaskManager] 删除任务: {task.task_name}（ID={task_id}）")
        return {"success": True, "message": "任务已删除", "task_id": task_id}


# ============================================================
# 全局单例
# ============================================================

_task_manager: Optional[TaskManager] = None


def get_task_manager(config: Optional[TaskConfig] = None) -> TaskManager:
    """获取全局 TaskManager 实例（懒加载单例）

    Args:
        config: 首次调用时可传入配置，后续调用忽略

    Returns:
        TaskManager 实例
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(config)
    return _task_manager
