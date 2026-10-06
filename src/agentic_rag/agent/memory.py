"""
Agent 记忆管理系统 (Memory Management System)

三层记忆架构，为 Agent 提供跨 conversation/thread 的经验积累与上下文感知能力。

=== 核心概念 ===

1. 为什么 Agent 需要记忆？
   无状态 Agent 无法积累经验，每次对话从零开始，导致：
   - 重复提问相同信息（用户偏好、上下文）
   - 无法利用历史决策经验优化当前推理
   - 长对话中丢失关键信息

2. 三层记忆的区别（划分依据：作用域 scope + 生命周期 lifetime + 语义 semantics + 检索方式 retrieval，
   **不要用"是否落盘"或"是否跨聊天窗口"来定义**）：
   - 短期记忆 (ShortTermMemory)：当前推理直接需要的工作上下文（最近 N 轮消息、当前变量、
     tool outputs），生命周期与当前活动上下文绑定；检索=直接读取窗口。快速访问但容量有限。
   - 工作记忆 (WorkingMemory)：≈ 业界"中期记忆"。保存**超出当前上下文窗口生命周期、但仍与
     进行中任务相关的临时状态与中间知识**（任务进度、中间变量、临时知识缓存）；检索=按 key/任务 ID
     精确取。本质是"长时任务的状态持久化与恢复"，支持上下文卸载、会话中断恢复，必要时跨
     conversation/thread 续接。
   - 长期记忆 (LongTermMemory)：跨任务/跨线程持久化的用户偏好 / 历史经验 / 领域知识，
     向量化存储、语义检索，类似人类的长期经验。

   术语层级（避免混用）：request/turn（一次 LLM 调用）< session（一段连续交互）<
   conversation/thread（一个聊天窗口，可跨多次 session）< task（一个目标性任务，可跨多个线程）。

3. 记忆压缩的必要性：
   Token 预算有限，不能无限注入历史。短期记忆溢出时需归档到长期记忆，
   长期记忆需定期合并相似条目去重，保证检索效率和质量。

4. 与 RAG 的区别：
   RAG 检索的是文档知识（外部知识库），Memory 检索的是 Agent 自身经验
   （对话历史、用户偏好、任务状态）。两者互补，Memory 是 Agent 的"自我认知"。

=== 降级策略 ===
- Redis 不可用 → 内存 dict 实现短期记忆
- Milvus 不可用 → 内存 list + TF-IDF 相似度实现长期记忆
- LLM 不可用 → 抽取式摘要（extractive summarization）替代生成式摘要
- 不使用 tiktoken，使用项目已有的 TokenCounter（from utils.context_engine）
"""

import re
import math
import uuid
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter

from agent_harness.context import TokenCounter

logger = logging.getLogger(__name__)

# ============================================================
# 可选依赖：Redis / Milvus / 向量模型
# ============================================================

_redis_client = None
try:
    import redis
    _redis_client = redis
except ImportError:
    logger.info("[Memory] redis 未安装，短期记忆将使用内存实现")

_pymilvus = None
try:
    from pymilvus import connections as _milvus_connections
    from pymilvus import Collection as _milvus_collection_cls
    from pymilvus import utility as _milvus_utility
    from pymilvus import FieldSchema, DataType, CollectionSchema
    _pymilvus = True
except Exception:
    logger.info("[Memory] pymilvus 未安装或加载失败，长期记忆将使用内存 + TF-IDF 实现")


# ============================================================
# 模块级连接缓存：避免每次实例化 MemoryManager 都重连 Redis / Milvus
# ============================================================

_redis_clients: Dict[str, Any] = {}
_milvus_connected: bool = False


def _get_redis_client(redis_url: str):
    """获取（或复用）Redis 客户端，按 url 缓存连接"""
    if redis_url in _redis_clients:
        return _redis_clients[redis_url]
    client = _redis_client.from_url(redis_url)
    client.ping()
    _redis_clients[redis_url] = client
    return client


def _ensure_milvus(config: "MemoryConfig") -> bool:
    """模块级复用 Milvus 连接（同一进程仅连接一次）"""
    global _milvus_connected
    if _milvus_connected:
        return True
    if _pymilvus is None:
        return False
    try:
        _milvus_connections.connect(
            alias="memory",
            host=config.milvus_host,
            port=config.milvus_port,
            timeout=3,
        )
        _milvus_connected = True
        return True
    except Exception:
        return False


# ============================================================
# 数据结构
# ============================================================

@dataclass # 作用是自动给类生成样板代码，专门用来做数据存储类（只存数据、很少写方法的类）
class MemoryItem:
    """记忆条目（承载三层记忆的表结构字段）

    Attributes:
        id: 唯一标识（长期记忆对应 memory_id）
        content: 记忆内容（长期记忆对应 raw_text）
        memory_type: 记忆类型 ("short_term" / "long_term" / "working")
        timestamp: 创建时间（ISO 格式，对应 created_at）
        metadata: 元数据（source, role, task_id, importance_score 等）
        embedding: 向量表示（长期记忆对应 content_embedding，可选）

    分层专有字段（对应面试标准记忆表设计）：
        updated_at: 最后更新 / 合并 / 刷新权重时间（长期记忆）
        source_type: 来源类型 user_message / document / tool_result / system_summary / feedback
        agent_id: 归属 Agent（中期记忆）
        expires_at: 失效时间，中期记忆 TTL，到期自动清理
        token_count: Token 数（短期记忆，用于预算与压缩）
    """
    id: str
    content: str
    memory_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    # --- 记忆表设计字段 ---
    updated_at: Optional[str] = None
    source_type: str = ""
    agent_id: str = ""
    expires_at: Optional[str] = None
    token_count: int = 0


    # 调用 to_dict() 之前是一个 Python 数据类（dataclass）实例对象 ，不是字典
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "updated_at": self.updated_at,
            "source_type": self.source_type,
            "agent_id": self.agent_id,
            "expires_at": self.expires_at,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            memory_type=data.get("memory_type", "short_term"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            updated_at=data.get("updated_at"),
            source_type=data.get("source_type", ""),
            agent_id=data.get("agent_id", ""),
            expires_at=data.get("expires_at"),
            token_count=data.get("token_count", 0),
        )


@dataclass
class MemoryConfig:
    """记忆管理配置

    Attributes:
        short_term_window: 短期记忆保留最近 N 轮对话
        long_term_max_items: 长期记忆最大条目数
        working_memory_max_keys: 工作记忆最大 key 数，最大 key 数 = 最大条目数
        archive_threshold: 短期记忆相似度超过此值才归档（避免归档重复信息）
        importance_decay: 重要性衰减因子（每次检索后乘以此值）
        enable_auto_archive: 是否启用自动归档（短期→长期）
        redis_url: Redis 连接地址（为空则使用内存实现）
        redis_key_prefix: Redis key 前缀
        milvus_collection: Milvus 集合名称（为空则使用内存实现）
        milvus_host: Milvus 地址
        milvus_port: Milvus 端口
        working_memory_ttl_seconds: 中期记忆默认 TTL（秒），0 表示不过期
    """
    short_term_window: int = 10
    long_term_max_items: int = 1000
    working_memory_max_keys: int = 50
    working_memory_ttl_seconds: int = 86400   # 中期记忆默认 24h TTL
    archive_threshold: float = 0.7
    importance_decay: float = 0.95
    enable_auto_archive: bool = True
    redis_url: str = ""
    redis_key_prefix: str = "agentic_rag:memory:short"
    default_user_id: str = "default"   # 跨会话记忆的默认用户（无 user_id 时全局共享，兼容旧数据）
    milvus_collection: str = "agent_long_term_memory"
    milvus_host: str = "localhost"
    milvus_port: int = 19530


# ============================================================
# 短期记忆：对话上下文滑动窗口
# ============================================================

class ShortTermMemory:
    """短期记忆管理器

    存储最近 N 轮对话，基于 Redis（可选）或内存 dict。
    溢出时触发归档到长期记忆。

    存储结构：
    - Redis: List 类型，每条消息 JSON 序列化后追加
    - 内存: List[MemoryItem]，按时间顺序排列
    """

    def __init__(self, config: MemoryConfig, user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        self.config = config
        self.user_id = user_id or config.default_user_id
        self.session_id = session_id
        self._items: List[MemoryItem] = []
        self._redis = None

        # 尝试连接 Redis（模块级缓存，复用连接）
        if config.redis_url and _redis_client is not None:
            try:
                self._redis = _get_redis_client(config.redis_url)
                logger.info("[短期记忆] Redis 连接成功")
            except Exception as e:
                logger.warning(f"[短期记忆] Redis 连接失败，降级为内存实现: {e}")
                self._redis = None

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> MemoryItem:
        """添加一条对话消息

        Args:
            role: 角色 ("user" / "assistant" / "system")
            content: 消息内容
            metadata: 额外元数据

        Returns:
            创建的 MemoryItem
        """
        now = datetime.now().isoformat()
        role_to_source = {
            "user": "user_message",
            "assistant": "assistant_message",
            "system": "system_summary",
        }
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type="short_term",
            timestamp=now,
            updated_at=now,
            source_type=role_to_source.get(role, "message"),
            token_count=TokenCounter.count_tokens(content),
            metadata={
                "role": role,          # 对应表字段 message_role
                "importance_score": 1.0,
                "user_id": (metadata or {}).get("user_id", self.user_id),
                "session_id": (metadata or {}).get("session_id", self.session_id),
                **(metadata or {}),
            },
        )

        if self._redis is not None:
            self._redis_add(item)
        else:
            self._items.append(item)
            # 超出窗口时截断
            if len(self._items) > self.config.short_term_window:
                self._items = self._items[-self.config.short_term_window:]

        logger.debug(f"[短期记忆] 添加消息: role={role}, 长度={len(content)}")
        return item

    def get_recent(self, n: Optional[int] = None) -> List[MemoryItem]:
        """获取最近 N 轮对话

        Args:
            n: 获取条数，默认取配置的窗口大小

        Returns:
            最近的记忆条目列表（按时间升序）
        """
        count = n or self.config.short_term_window

        if self._redis is not None:
            return self._redis_get_recent(count)

        return self._items[-count:]

    def get_overflow(self) -> List[MemoryItem]:
        """获取溢出的旧消息（超出窗口的部分），用于归档

        Returns:
            溢出的记忆条目列表
        """
        if self._redis is not None:
            # Redis 模式下溢出由 LRU 自动管理，此处返回空
            return []

        overflow_count = len(self._items) - self.config.short_term_window
        if overflow_count <= 0:
            return []

        overflow = self._items[:overflow_count]
        self._items = self._items[overflow_count:]
        return overflow

    def clear(self):
        """清空短期记忆"""
        if self._redis is not None:
            try:
                self._redis.delete(self._redis_key())
            except Exception as e:
                logger.warning(f"[短期记忆] Redis 清空失败: {e}")
        else:
            self._items.clear()

        logger.info("[短期记忆] 已清空")

    def size(self) -> int:
        """当前条目数"""
        if self._redis is not None:
            try:
                return self._redis.llen(self._redis_key())
            except Exception:
                return 0
        return len(self._items)

    def _redis_key(self) -> str:
        """Redis key 按 user_id + session_id 分片：短期记忆会话隔离、用户间不串味"""
        sid = self.session_id or "nosession"
        return f"{self.config.redis_key_prefix}:{self.user_id}:{sid}"

    # --- Redis 内部方法 ---

    def _redis_add(self, item: MemoryItem):
        """Redis 模式添加消息"""
        import json
        key = self._redis_key()
        self._redis.rpush(key, json.dumps(item.to_dict(), ensure_ascii=False))
        # 保留最近 N 条
        self._redis.ltrim(key, -self.config.short_term_window, -1)

    def _redis_get_recent(self, n: int) -> List[MemoryItem]:
        """Redis 模式获取最近 N 条"""
        import json
        key = self._redis_key()
        raw_items = self._redis.lrange(key, -n, -1)
        items = []
        for raw in raw_items:
            try:
                data = json.loads(raw)
                items.append(MemoryItem.from_dict(data))
            except Exception:
                continue
        return items


# ============================================================
# 长期记忆：向量化存储 + 语义检索
# ============================================================

class LongTermMemory:
    """长期记忆管理器

    存储用户偏好、历史决策、知识积累。
    基于 Milvus（可选）或内存 list + TF-IDF 相似度。

    核心能力：
    1. 语义相似度检索（Milvus 向量检索 / TF-IDF 文本相似度）
    2. 重要性衰减（每次检索后 importance_score *= decay）
    3. 定期合并相似条目（去重）
    """

    def __init__(self, config: MemoryConfig, user_id: Optional[str] = None):
        self.config = config
        self.user_id = user_id or config.default_user_id
        self._items: List[MemoryItem] = []
        self._milvus_collection = None
        self._embedding_fn = None

        # 尝试初始化 Milvus（模块级复用连接）
        if _pymilvus is not None:
            try:
                if _ensure_milvus(config):
                    self._milvus_collection = config.milvus_collection
                    # 自动建表（如果集合不存在）
                    self._ensure_collection()
                    logger.info("[长期记忆] Milvus 连接成功")
                else:
                    self._milvus_collection = None
            except Exception as e:
                logger.warning(f"[长期记忆] Milvus 不可用，降级为内存 + TF-IDF: {e}")
                self._milvus_collection = None

        # 尝试加载向量模型
        try:
            from agentic_rag.utils.embedding import EmbeddingService
            service = EmbeddingService()
            self._embedding_fn = service.embed_text
            logger.info("[长期记忆] 向量模型加载成功（EmbeddingService）")
        except (ImportError, Exception) as e:
            logger.info(f"[长期记忆] 向量模型不可用，使用 TF-IDF 相似度: {e}")
            self._embedding_fn = None

    def add(self, content: str, metadata: Optional[Dict] = None) -> MemoryItem:
        """添加一条长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据（可包含 source, category, importance_score 等）

        Returns:
            创建的 MemoryItem
        """
        # 超出最大条目数时，先合并相似条目
        if len(self._items) >= self.config.long_term_max_items:
            self._merge_similar()

        now = datetime.now().isoformat()
        source_type = (metadata or {}).get("source_type") or (metadata or {}).get("source") or "unknown"
        item = MemoryItem(
            id=str(uuid.uuid4()),          # 对应表字段 memory_id
            content=content,               # 对应表字段 raw_text
            memory_type="long_term",
            timestamp=now,                 # 对应表字段 created_at
            updated_at=now,                # 对应表字段 updated_at
            source_type=source_type,       # 来源类型（判断可信度）
            metadata={
                "importance_score": 1.0,
                "access_count": 0,
                "user_id": (metadata or {}).get("user_id", self.user_id),
                **(metadata or {}),
            },
        )

        # 尝试生成向量
        if self._embedding_fn is not None:
            try:
                embedding = self._embedding_fn(content)
                if isinstance(embedding, list):
                    item.embedding = embedding
                elif hasattr(embedding, 'tolist'):
                    item.embedding = embedding.tolist()
            except Exception as e:
                logger.debug(f"[长期记忆] 向量生成失败: {e}")

        self._items.append(item)

        # 同步到 Milvus（如果可用）
        if self._milvus_collection is not None and item.embedding is not None:
            self._milvus_insert(item)

        logger.debug(f"[长期记忆] 添加记忆: 长度={len(content)}, "
                     f"有向量={item.embedding is not None}")
        return item

    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """语义相似度检索（按 user_id 隔离共享）

        Args:
            query: 查询文本
            top_k: 返回最相似的 K 条

        Returns:
            按相似度降序排列的记忆条目列表（仅包含当前 user_id 的记忆）
        """
        if not self._items:
            return []

        # 优先使用 Milvus 向量检索
        if self._milvus_collection is not None and self._embedding_fn is not None:
            # 多取一些再按 user_id 过滤（Milvus 元数据为字符串，后置过滤更稳妥）
            results = self._milvus_search(query, top_k * 4)
            results = self._filter_by_user(results)
            if results:
                return results[:top_k]

        # 降级：TF-IDF 文本相似度（同样按 user_id 过滤）
        return self._filter_by_user(self._tfidf_search(query, top_k))

    def _filter_by_user(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """按 user_id 过滤记忆（跨会话按用户隔离/共享）

        - 默认用户（"default"）不过滤，兼容历史无 user_id 的全局数据
        - 指定真实 user_id 时，仅返回该用户的记忆
        """
        if not self.user_id or self.user_id == self.config.default_user_id:
            return items
        return [it for it in items if it.metadata.get("user_id") == self.user_id]

    def decay_importance(self):
        """对所有长期记忆执行重要性衰减"""
        for item in self._items:
            current_score = item.metadata.get("importance_score", 1.0)
            item.metadata["importance_score"] = current_score * self.config.importance_decay

    def adjust_weight(self, memory_id: str, delta: float) -> bool:
        """反馈机制：根据用户反馈动态调整某条长期记忆的重要性权重

        Args:
            memory_id: 记忆 ID
            delta: 权重增量（正向=有用被采纳，负向=无用/错误）

        Returns:
            是否命中并更新
        """
        for item in self._items:
            if item.id == memory_id:
                score = item.metadata.get("importance_score", 1.0)
                item.metadata["importance_score"] = max(0.0, min(score + delta, 5.0))
                item.updated_at = datetime.now().isoformat()
                logger.debug(f"[长期记忆] 权重调整 {memory_id}: {score} -> {item.metadata['importance_score']}")
                return True
        return False

    def _merge_similar(self):
        """合并相似条目（去重）

        策略：计算所有条目间的文本相似度，超过阈值的合并为一条，
        保留重要性更高的那条，内容合并。
        """
        if len(self._items) < 2:
            return

        merged_indices: Set[int] = set()
        new_items: List[MemoryItem] = []

        for i, item_a in enumerate(self._items):
            if i in merged_indices:
                continue

            similar_group = [item_a]
            for j in range(i + 1, len(self._items)):
                if j in merged_indices:
                    continue
                item_b = self._items[j]
                similarity = self._compute_text_similarity(item_a.content, item_b.content)
                if similarity > 0.85:
                    similar_group.append(item_b)
                    merged_indices.add(j)

            if len(similar_group) > 1:
                # 保留重要性最高的条目，内容合并
                similar_group.sort(
                    key=lambda x: x.metadata.get("importance_score", 0), reverse=True
                )
                best = similar_group[0]
                merged_content = best.content
                for other in similar_group[1:]:
                    if other.content not in merged_content:
                        merged_content += f"；{other.content}"
                best.content = merged_content
                best.metadata["merged_count"] = len(similar_group)
                new_items.append(best)
            else:
                new_items.append(item_a)

        removed = len(self._items) - len(new_items)
        self._items = new_items

        if removed > 0:
            logger.info(f"[长期记忆] 合并相似条目: {len(self._items) + removed} → "
                        f"{len(self._items)}，去除 {removed} 条")

    def clear(self):
        """清空长期记忆"""
        self._items.clear()
        logger.info("[长期记忆] 已清空")

    def size(self) -> int:
        """当前条目数"""
        return len(self._items)

    # --- TF-IDF 相似度检索 ---

    def _tfidf_search(self, query: str, top_k: int) -> List[MemoryItem]:
        """基于 TF-IDF 的文本相似度检索

        使用词频 + 逆文档频率计算查询与记忆条目的相似度，
        结合重要性分数加权排序。
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return self._items[-top_k:]

        # 构建文档词频
        doc_freq: Dict[str, int] = Counter()
        all_docs_terms: List[List[str]] = []

        for item in self._items:
            terms = self._tokenize(item.content)
            all_docs_terms.append(terms)
            unique_terms = set(terms)
            for t in unique_terms:
                doc_freq[t] += 1

        total_docs = len(self._items)

        # 计算查询的 TF-IDF 向量
        query_tf = Counter(query_terms)
        query_vec: Dict[str, float] = {}
        for term, tf in query_tf.items():
            idf = math.log((total_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1
            query_vec[term] = tf * idf

        # 计算每条记忆的 TF-IDF 向量并求余弦相似度
        scored_items: List[tuple] = []
        for idx, (item, doc_terms) in enumerate(zip(self._items, all_docs_terms)):
            doc_tf = Counter(doc_terms)
            doc_vec: Dict[str, float] = {}
            for term, tf in doc_tf.items():
                idf = math.log((total_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1
                doc_vec[term] = tf * idf

            similarity = self._cosine_similarity(query_vec, doc_vec)
            importance = item.metadata.get("importance_score", 1.0)

            # 综合分数 = 语义相似度 × 0.7 + 重要性 × 0.3
            final_score = similarity * 0.7 + importance * 0.3
            scored_items.append((final_score, similarity, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        # 更新访问计数
        for _, _, item in scored_items[:top_k]:
            item.metadata["access_count"] = item.metadata.get("access_count", 0) + 1

        return [item for _, _, item in scored_items[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词（基于字符 + 英文单词的简单分词）"""
        # 提取中文词（2-4字组合）和英文单词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'[a-zA-Z]+', text.lower())

        # 中文 bigram
        bigrams = []
        for i in range(len(chinese_chars) - 1):
            bigrams.append(chinese_chars[i] + chinese_chars[i + 1])

        return bigrams + english_words

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """计算两个稀疏向量的余弦相似度"""
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    @staticmethod
    def _compute_text_similarity(text_a: str, text_b: str) -> float:
        """计算两段文本的 Jaccard 相似度（用于去重合并）"""
        set_a = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text_a.lower()))
        set_b = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text_b.lower()))

        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # --- Milvus 内部方法 ---

    def _ensure_collection(self):
        """确保 Milvus 集合存在，不存在则创建

        Schema 与 _milvus_insert 的字段保持一致：
        - id: VARCHAR(64) 主键
        - content: VARCHAR(8192) 记忆内容
        - embedding: FLOAT_VECTOR(1024) 向量（BGE-M3 默认维度）
        - metadata: VARCHAR(8192) 元数据 JSON 字符串

        失败不阻塞，降级为内存模式（_milvus_collection 置 None）。
        """
        try:
            if _milvus_utility is None or _milvus_collection_cls is None:
                return
            if _milvus_utility.has_collection(self._milvus_collection, using="memory"):
                logger.debug(f"[长期记忆] Milvus 集合已存在：{self._milvus_collection}")
                return

            # 创建集合 schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
            ]
            schema = CollectionSchema(fields, description="Agent 长期记忆存储")
            collection = _milvus_collection_cls(self._milvus_collection, schema=schema, using="memory")

            # 创建向量索引（IVF_FLAT + COSINE）
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
            )
            logger.info(f"[长期记忆] Milvus 集合已创建：{self._milvus_collection}（dim=1024, IVF_FLAT/COSINE）")
        except Exception as e:
            logger.warning(f"[长期记忆] Milvus 集合创建失败，降级为内存模式：{e}")
            self._milvus_collection = None

    def _milvus_insert(self, item: MemoryItem):
        """Milvus 插入向量"""
        try:
            collection = _milvus_collection_cls(self._milvus_collection)
            collection.insert([{
                "id": item.id,
                "content": item.content,
                "embedding": item.embedding,
                "metadata": str(item.metadata),
            }])
        except Exception as e:
            logger.debug(f"[长期记忆] Milvus 插入失败: {e}")

    def _milvus_search(self, query: str, top_k: int) -> List[MemoryItem]:
        """Milvus 向量检索"""
        try:
            if self._embedding_fn is None:
                return []

            query_embedding = self._embedding_fn(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            collection = _milvus_collection_cls(self._milvus_collection)
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["content", "metadata"],
            )

            items = []
            for hit in results[0]:
                item = MemoryItem(
                    id=hit.id,
                    content=hit.entity.get("content", ""),
                    memory_type="long_term",
                    timestamp=datetime.now().isoformat(),
                    metadata=eval(hit.entity.get("metadata", "{}")),
                    embedding=None,
                )
                items.append(item)
            return items
        except Exception as e:
            logger.debug(f"[长期记忆] Milvus 检索失败，降级 TF-IDF: {e}")
            return []


# ============================================================
# 工作记忆：当前任务进度 / 中间结果
# ============================================================

class WorkingMemory:
    """工作记忆管理器（≈ 业界"中期记忆"）

    结构化存储（dict），支持 CRUD 操作。
    用于保存**超出当前上下文窗口生命周期、但仍与进行中任务相关的临时状态与中间知识**
    （任务进度、中间变量、临时知识缓存），即"长时任务的状态持久化与恢复"，
    支持上下文卸载（context offloading）与会话中断恢复，必要时跨 conversation/thread 续接。

    注意：当前实现为进程内内存 Dict，进程重启即丢失；若要真正"中断后重启仍可续接"
    （cross-session state persistence），需落到 Redis/DB。
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._store: Dict[str, MemoryItem] = {} # 工作记忆存储

    def set(self, key: str, value: Any, metadata: Optional[Dict] = None,
            agent_id: str = "", ttl_seconds: Optional[int] = None) -> MemoryItem:
        """设置工作记忆（中期记忆：事件驱动写入 + TTL 自动过期）

        Args:
            key: 键名（对应表字段 state_key，如 current_step / selected_tool）
            value: 值（对应表字段 structured_data，建议 dict/list，结构化保存）
            metadata: 额外元数据
            agent_id: 归属 Agent（对应表字段 agent_id）
            ttl_seconds: 存活秒数（对应表字段 expires_at）；默认取 config.working_memory_ttl_seconds

        Returns:
            创建或更新的 MemoryItem
        """
        self._purge_expired()

        # 超出最大 key 数时，删除最旧的
        if key not in self._store and len(self._store) >= self.config.working_memory_max_keys:
            oldest_key = min(
                self._store.keys(),
                key=lambda k: self._store[k].timestamp,
            )
            del self._store[oldest_key]
            logger.debug(f"[工作记忆] 淘汰最旧 key: {oldest_key}")

        now = datetime.now()
        ttl = ttl_seconds if ttl_seconds is not None else self.config.working_memory_ttl_seconds
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl and ttl > 0 else None
        content = str(value) if not isinstance(value, str) else value
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type="working",
            timestamp=now.isoformat(),     # 对应表字段 created_at
            updated_at=now.isoformat(),
            agent_id=agent_id,
            expires_at=expires_at,         # 对应表字段 expires_at
            metadata={
                "key": key,
                "state_key": key,          # 对应表字段 state_key
                "value_type": type(value).__name__,
                "structured_data": value if isinstance(value, (dict, list)) else content,
                **(metadata or {}),
            },
        )
        self._store[key] = item
        logger.debug(f"[工作记忆] 设置: {key}={content[:50]} (expires_at={expires_at})")
        return item

    def _purge_expired(self):
        """清理已过期的中期记忆（TTL 到期自动清理）"""
        if not self._store:
            return
        now = datetime.now()
        expired = []
        for k, item in self._store.items():
            if item.expires_at:
                try:
                    if datetime.fromisoformat(item.expires_at) <= now:
                        expired.append(k)
                except Exception:
                    continue
        for k in expired:
            del self._store[k]
        if expired:
            logger.info(f"[工作记忆] TTL 清理过期条目 {len(expired)} 个: {expired}")

    def get(self, key: str) -> Optional[MemoryItem]:
        """获取工作记忆

        Args:
            key: 键名

        Returns:
            对应的 MemoryItem，不存在返回 None（已过期的视为不存在）
        """
        self._purge_expired()
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """删除工作记忆

        Args:
            key: 键名

        Returns:
            是否删除成功
        """
        if key in self._store:
            del self._store[key]
            logger.debug(f"[工作记忆] 删除: {key}")
            return True
        return False

    def keys(self) -> List[str]:
        """获取所有 key"""
        self._purge_expired()
        return list(self._store.keys())

    def items(self) -> List[MemoryItem]:
        """获取所有工作记忆条目"""
        self._purge_expired()
        return list(self._store.values())

    def clear(self):
        """清空工作记忆"""
        self._store.clear()
        logger.info("[工作记忆] 已清空")

    def size(self) -> int:
        """当前条目数"""
        return len(self._store)


# ============================================================
# 记忆压缩与归档
# ============================================================

class MemoryCompressor:
    """记忆压缩器

    提供两种压缩策略：
    1. 抽取式摘要（extractive summarization）：无需 LLM，基于关键词频率和位置评分
    2. 归档：短期记忆溢出时，旧消息摘要后归档到长期记忆
    """

    @staticmethod
    def extractive_summarize(text: str, target_tokens: int) -> str:
        """抽取式摘要

        按句子评分（关键词频率 × 位置权重 × 长度权重），
        选取 Top-K 句子组成摘要，无需 LLM 依赖。

        Args:
            text: 原始文本
            target_tokens: 目标 Token 数

        Returns:
            压缩后的文本
        """
        if not text or target_tokens <= 0:
            return ""

        current_tokens = TokenCounter.count_tokens(text)
        if current_tokens <= target_tokens:
            return text

        # 按句子切分
        sentence_ends = re.compile(r'(?<=[。！？.!?])\s*')
        sentences = [s.strip() for s in sentence_ends.split(text) if s.strip()]

        if not sentences:
            return TokenCounter.truncate_to_tokens(text, target_tokens)

        if len(sentences) == 1:
            return TokenCounter.truncate_to_tokens(sentences[0], target_tokens)

        # 计算词频
        all_words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
        word_freq: Dict[str, int] = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1

        # 句子评分
        total = len(sentences)
        scored: List[tuple] = []
        for i, s in enumerate(sentences):
            # 位置权重：首尾句子更重要
            position_score = 1.0
            if i < 2:
                position_score = 1.5
            elif i >= total - 2:
                position_score = 1.3

            # 关键词权重
            s_words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', s.lower())
            keyword_score = sum(
                word_freq.get(w, 0) / max_freq for w in s_words
            ) / max(len(s_words), 1)

            # 长度权重：过短的句子信息量少
            length_score = min(len(s) / 50.0, 1.0)

            final_score = position_score * 0.4 + keyword_score * 0.4 + length_score * 0.2
            scored.append((i, s, final_score))

        # 按分数降序排列，选取不超过目标 Token 的句子
        scored.sort(key=lambda x: x[2], reverse=True)
        selected: List[tuple] = []
        current_tokens = 0
        for idx, s, score in scored:
            s_tokens = TokenCounter.count_tokens(s)
            if current_tokens + s_tokens > target_tokens:
                continue
            selected.append((idx, s))
            current_tokens += s_tokens

        # 按原文顺序排列
        selected.sort(key=lambda x: x[0])
        return "".join(s for _, s in selected)

    @staticmethod
    def summarize_conversation(messages: List[MemoryItem], target_tokens: int = 200) -> str:
        """将多条对话消息摘要为一条

        Args:
            messages: 对话消息列表
            target_tokens: 摘要目标 Token 数

        Returns:
            摘要文本
        """
        if not messages:
            return ""

        # 拼接所有消息
        parts = []
        for msg in messages:
            role = msg.metadata.get("role", "unknown")
            role_label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(role, role)
            parts.append(f"{role_label}：{msg.content}")

        full_text = "\n".join(parts)
        return MemoryCompressor.extractive_summarize(full_text, target_tokens)


# ============================================================
# MemoryManager 统一管理器
# ============================================================

class MemoryManager:
    """Agent 记忆统一管理器

    整合三层记忆，提供统一的记忆操作接口：
    - 添加观察 / 用户偏好 / 任务状态
    - 检索相关记忆
    - 归档短期记忆到长期
    - 生成可注入 Prompt 的记忆上下文

    使用示例::

        manager = MemoryManager()

        # 添加对话
        manager.add_observation("user", "我想查询火花塞的库存")
        manager.add_observation("assistant", "火花塞(ENG-001)当前库存150个...")

        # 记录用户偏好
        manager.add_user_preference("语言风格", "简洁专业")

        # 记录任务状态
        manager.add_task_state("task_001", {"step": "retrieving", "progress": 0.6})

        # 检索相关记忆
        memories = manager.retrieve_relevant("火花塞库存", top_k=5)

        # 生成 Prompt 上下文
        context = manager.get_context_for_prompt("火花塞还有货吗？", max_tokens=2000)
    """

    def __init__(self, config: Optional[MemoryConfig] = None,
                 user_id: Optional[str] = None, session_id: Optional[str] = None):
        self.config = config or MemoryConfig()
        self.user_id = user_id or self.config.default_user_id
        self.session_id = session_id or str(uuid.uuid4())

        # 短期记忆按 user+session 隔离；长期记忆按 user 共享（跨会话互通）
        self.short_term = ShortTermMemory(self.config, user_id=self.user_id, session_id=self.session_id)
        self.long_term = LongTermMemory(self.config, user_id=self.user_id)
        self.working = WorkingMemory(self.config)
        self.compressor = MemoryCompressor()

        logger.info(
            f"[MemoryManager] 初始化完成 user_id={self.user_id}, "
            f"短期窗口={self.config.short_term_window}, "
            f"长期上限={self.config.long_term_max_items}, "
            f"工作记忆上限={self.config.working_memory_max_keys}"
        )

    def add_observation(self, role: str, content: str, metadata: Optional[Dict] = None,
                        session_id: Optional[str] = None) -> MemoryItem:
        """添加观察（对话消息）

        Args:
            role: 角色 ("user" / "assistant" / "system")
            content: 消息内容
            metadata: 额外元数据（可包含 user_id、session_id 等，缺省时自动用管理器级别）
            session_id: 会话 ID（不传则使用管理器初始化时的 session_id）

        Returns:
            创建的 MemoryItem
        """
        obs_meta = {
            "user_id": self.user_id,
            "session_id": session_id or self.session_id,
            **(metadata or {}),
        }
        item = self.short_term.add(role, content, obs_meta)

        # 自动归档检查
        if self.config.enable_auto_archive:
            self._auto_archive()

        return item

    def add_user_preference(self, key: str, value: str) -> MemoryItem:
        """记录用户偏好到长期记忆（自动归属当前 user_id，跨会话共享）

        Args:
            key: 偏好键名（如 "语言风格"、"输出格式"）
            value: 偏好值

        Returns:
            创建的 MemoryItem
        """
        return self.long_term.add(
            content=f"用户偏好 - {key}：{value}",
            metadata={
                "source": "user_preference",
                "category": "preference",
                "preference_key": key,
                "user_id": self.user_id,
                "importance_score": 1.5,  # 用户偏好重要性更高
            },
        )

    def add_task_state(self, task_id: str, state: Dict[str, Any],
                       agent_id: str = "", ttl_seconds: Optional[int] = None) -> MemoryItem:
        """记录任务状态到工作记忆（≈ 中期记忆；事件驱动写入 + TTL 自动过期）

        Args:
            task_id: 任务 ID（对应表字段 task_id）
            state: 任务状态字典（对应表字段 structured_data）
            agent_id: 归属 Agent（对应表字段 agent_id）
            ttl_seconds: 存活秒数（对应表字段 expires_at）；默认取 config

        Returns:
            创建的 MemoryItem
        """
        return self.working.set(
            key=f"task:{task_id}",
            value=state,
            agent_id=agent_id,
            ttl_seconds=ttl_seconds,
            metadata={"task_id": task_id, "source": "task_state", "source_type": "task_state",
                      "user_id": self.user_id},
        )

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """检索相关记忆

        从三层记忆中分别检索，合并后按相关性排序返回。

        Args:
            query: 查询文本
            top_k: 返回最多 K 条

        Returns:
            按相关性排序的记忆条目列表
        """
        results: List[MemoryItem] = []

        # 短期记忆：最近 N 轮直接返回
        recent = self.short_term.get_recent()
        for item in recent:
            # 简单关键词匹配评分
            score = self._compute_relevance(query, item.content)
            item.metadata["_relevance_score"] = score
            results.append(item)

        # 长期记忆：语义相似度检索
        long_term_results = self.long_term.search(query, top_k=top_k)
        for item in long_term_results:
            score = item.metadata.get("_relevance_score", 0.5)
            item.metadata["_relevance_score"] = score
            results.append(item)

        # 工作记忆：按 key 精确匹配
        for item in self.working.items():
            key = item.metadata.get("key", "")
            if key and (key in query or query in key):
                item.metadata["_relevance_score"] = 1.0
                results.append(item)

        # 去重（按 id）
        seen_ids: Set[str] = set()
        unique_results: List[MemoryItem] = []
        for item in results:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_results.append(item)

        # 按相关性降序排序
        unique_results.sort(
            key=lambda x: x.metadata.get("_relevance_score", 0), reverse=True
        )

        # 清理临时评分
        for item in unique_results:
            item.metadata.pop("_relevance_score", None)

        logger.debug(
            f"[MemoryManager] 检索完成: query='{query[:30]}...', "
            f"短期={len(recent)}, 长期={len(long_term_results)}, "
            f"合并后={len(unique_results[:top_k])}"
        )

        return unique_results[:top_k]

    def archive_short_term(self) -> int:
        """手动归档短期记忆到长期

        将短期记忆中溢出的旧消息摘要后归档到长期记忆。

        Returns:
            归档的条目数
        """
        overflow = self.short_term.get_overflow()
        if not overflow:
            return 0

        # 摘要后归档
        summary = self.compressor.summarize_conversation(overflow, target_tokens=200)
        if summary:
            self.long_term.add(
                content=summary,
                metadata={
                    "source": "short_term_archive",
                    "category": "conversation_summary",
                    "original_count": len(overflow),
                    "user_id": self.user_id,
                    "importance_score": 0.8,
                },
            )

        logger.info(f"[MemoryManager] 归档短期记忆: {len(overflow)} 条 → 摘要归档")
        return len(overflow)

    def get_context_for_prompt(self, query: str, max_tokens: int = 2000,
                               user_id: Optional[str] = None,
                               session_id: Optional[str] = None) -> str:
        """生成可直接注入 Prompt 的记忆上下文

        按优先级组装三层记忆，控制在 max_tokens 预算内：
        1. 工作记忆（当前任务状态，最高优先级）
        2. 短期记忆（最近对话，可按 user_id/session_id 过滤）
        3. 长期记忆（相关经验）

        Args:
            query: 当前查询
            max_tokens: 最大 Token 预算
            user_id: 用户 ID（可选，指定后只检索该用户的记忆）
            session_id: 会话 ID（可选，指定后只检索该会话的记忆）

        Returns:
            格式化的记忆上下文字符串
        """
        parts: List[str] = []
        current_tokens = 0

        # 1. 工作记忆（占预算 20%）
        working_budget = max(int(max_tokens * 0.2), 100)
        working_items = self.working.items()
        if working_items:
            working_text = self._format_working_memory(working_items, working_budget)
            if working_text:
                parts.append(f"[当前任务状态]\n{working_text}")
                current_tokens += TokenCounter.count_tokens(working_text)

        # 2. 短期记忆（占预算 40%）
        short_budget = max(int(max_tokens * 0.4), 200)
        recent_items = self.short_term.get_recent()
        if recent_items:
            # 如果指定了 user_id 或 session_id，过滤短期记忆
            if user_id or session_id:
                filtered_items = []
                for item in recent_items:
                    meta = item.metadata
                    if not self._match_user(meta, user_id):
                        continue
                    if session_id and meta.get("session_id") != session_id:
                        continue
                    filtered_items.append(item)
                recent_items = filtered_items
            short_text = self._format_short_term_memory(recent_items, short_budget)
            if short_text:
                parts.append(f"[近期对话]\n{short_text}")
                current_tokens += TokenCounter.count_tokens(short_text)

        # 3. 长期记忆（剩余预算）
        remaining_budget = max(max_tokens - current_tokens, 200)
        relevant_items = self.long_term.search(query, top_k=5)
        if relevant_items:
            # 如果指定了 user_id 或 session_id，过滤长期记忆
            if user_id or session_id:
                filtered_items = []
                for item in relevant_items:
                    meta = item.metadata
                    if not self._match_user(meta, user_id):
                        continue
                    if session_id and meta.get("session_id") != session_id:
                        continue
                    filtered_items.append(item)
                relevant_items = filtered_items
            long_text = self._format_long_term_memory(relevant_items, remaining_budget)
            if long_text:
                parts.append(f"[相关经验]\n{long_text}")

        context = "\n\n".join(parts)

        # 最终 Token 截断保护
        if TokenCounter.count_tokens(context) > max_tokens:
            context = TokenCounter.truncate_to_tokens(context, max_tokens)

        logger.debug(
            f"[MemoryManager] 生成 Prompt 上下文: "
            f"{TokenCounter.count_tokens(context)}/{max_tokens} tokens"
        )

        return context

    def clear(self):
        """清空所有记忆"""
        self.short_term.clear()
        self.long_term.clear()
        self.working.clear()
        logger.info("[MemoryManager] 所有记忆已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "short_term": {
                "size": self.short_term.size(),
                "max_window": self.config.short_term_window,
            },
            "long_term": {
                "size": self.long_term.size(),
                "max_items": self.config.long_term_max_items,
            },
            "working": {
                "size": self.working.size(),
                "max_keys": self.config.working_memory_max_keys,
                "keys": self.working.keys(),
            },
        }

    # --- 内部方法 ---

    def _auto_archive(self):
        """自动归档短期记忆溢出部分"""
        overflow = self.short_term.get_overflow()
        if not overflow:
            return

        # 检查溢出消息是否值得归档（相似度低于阈值才归档，避免重复）
        summary = self.compressor.summarize_conversation(overflow, target_tokens=200)
        if not summary:
            return

        # 检查与已有长期记忆的相似度
        existing = self.long_term.search(summary, top_k=1)
        if existing:
            similarity = self._compute_relevance(summary, existing[0].content)
            if similarity > self.config.archive_threshold:
                logger.debug("[MemoryManager] 溢出消息与长期记忆高度相似，跳过归档")
                return

        self.long_term.add(
            content=summary,
            metadata={
                "source": "auto_archive",
                "category": "conversation_summary",
                "original_count": len(overflow),
                "user_id": self.user_id,
                "importance_score": 0.8,
            },
        )
        logger.info(f"[MemoryManager] 自动归档: {len(overflow)} 条短期记忆 → 长期记忆")

    @staticmethod
    def _match_user(meta: Dict, user_id: Optional[str]) -> bool:
        """判断记忆条目是否属于指定用户（兼容无 user_id 的旧数据）

        - user_id 为空：匹配全部
        - 条目无 user_id：仅当当前为默认用户（"default"）时匹配（兼容历史全局数据）
        - 否则要求 user_id 精确相等
        """
        if not user_id:
            return True
        item_user = meta.get("user_id")
        if item_user is None:
            return user_id == "default"
        return item_user == user_id

    @staticmethod
    def _compute_relevance(query: str, content: str) -> float:
        """计算查询与内容的简单相关性（Jaccard 系数）"""

        # 提取查询和内容的词项（考虑中文和英文）
        query_terms = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', query.lower()))
        
        # 提取内容的词项（考虑中文和英文）
        content_terms = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', content.lower()))
        
        # 
        if not query_terms or not content_terms:
            return 0.0

        # 计算交集和并集的词项数
        # 交集：查询和内容中都存在的词项
        # 并集：查询和内容中至少存在一个词项
        intersection = len(query_terms & content_terms)
        union = len(query_terms | content_terms)

        # 计算 Jaccard 系数
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _format_working_memory(items: List[MemoryItem], budget: int) -> str:
        """格式化工作记忆"""
        parts = []
        current_tokens = 0

        for item in items:
            key = item.metadata.get("key", "unknown")
            entry = f"- {key}：{item.content}"
            entry_tokens = TokenCounter.count_tokens(entry)

            if current_tokens + entry_tokens > budget:
                remaining = budget - current_tokens
                if remaining > 20:
                    truncated = TokenCounter.truncate_to_tokens(
                        item.content, remaining - TokenCounter.count_tokens(f"- {key}：") - 3
                    )
                    parts.append(f"- {key}：{truncated}...")
                break

            parts.append(entry)
            current_tokens += entry_tokens

        return "\n".join(parts)

    @staticmethod
    def _format_short_term_memory(items: List[MemoryItem], budget: int) -> str:
        """格式化短期记忆"""
        parts = []
        current_tokens = 0

        for item in items:
            role = item.metadata.get("role", "unknown")
            role_label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(role, role)
            entry = f"{role_label}：{item.content}"
            entry_tokens = TokenCounter.count_tokens(entry)

            if current_tokens + entry_tokens > budget:
                remaining = budget - current_tokens
                if remaining > 20:
                    truncated = TokenCounter.truncate_to_tokens(
                        item.content, remaining - TokenCounter.count_tokens(f"{role_label}：") - 3
                    )
                    parts.append(f"{role_label}：{truncated}...")
                break

            parts.append(entry)
            current_tokens += entry_tokens

        return "\n".join(parts)

    @staticmethod
    def _format_long_term_memory(items: List[MemoryItem], budget: int) -> str:
        """格式化长期记忆"""
        parts = []
        current_tokens = 0

        for item in items:
            source = item.metadata.get("source", "经验")
            category = item.metadata.get("category", "")
            entry = f"- [{source}] {item.content}"
            entry_tokens = TokenCounter.count_tokens(entry)

            if current_tokens + entry_tokens > budget:
                remaining = budget - current_tokens
                if remaining > 20:
                    truncated = TokenCounter.truncate_to_tokens(
                        item.content, remaining - TokenCounter.count_tokens(f"- [{source}] ") - 3
                    )
                    parts.append(f"- [{source}] {truncated}...")
                break

            parts.append(entry)
            current_tokens += entry_tokens

        return "\n".join(parts)
