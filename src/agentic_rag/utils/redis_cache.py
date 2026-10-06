import json
import os
import logging
import time
import fnmatch
from typing import Optional, Dict, Any, List
from agentic_rag.config import settings
import hashlib

has_redis = False
try:
    import redis
    has_redis = True
    logging.info("redis 已加载")
except ImportError:
    logging.warning("redis 未安装，缓存功能将不可用")

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis 缓存服务（工业级：扩展结果缓存 + TTL随知识库更新 + 命中率统计）

    ⚠️ 低配版说明：
        架构图设计 TTL=7 天（604800s），为高频问答实现尽可能高的缓存命中率。
        实际 TTL 较短（问答 1 小时 / 检索结果 2 小时 / 查询扩展 1 天），
        因为知识库数据量不大且更新频率低，短 TTL 可以减少"回答过时"的风险。
        可根据实际使用场景调大 TTL（如企业内部固定知识库可设为 7d）。
    """

    def __init__(self):
        self.redis_client = None
        self.default_ttl = 3600
        self.retrieval_ttl = 7200
        self.expansion_ttl = 86400
        self._hit_count = 0
        self._miss_count = 0

        if has_redis:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis 连接成功")
            except Exception as e:
                logger.error(f"Redis 连接失败: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis 未安装，使用内存缓存")

        self.memory_cache = {}
        self.memory_expiry = {}

    def get_cache_key(self, prefix: str, data: str, tenant_id: Optional[str] = None) -> str:
        data_hash = hashlib.md5(data.encode()).hexdigest()
        if tenant_id:
            return f"{prefix}:{tenant_id}:{data_hash}"
        return f"{prefix}:{data_hash}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            ttl = ttl or self.default_ttl

            if self.redis_client:
                value_str = json.dumps(value, ensure_ascii=False)
                self.redis_client.setex(key, ttl, value_str)
                return True
            else:
                self.memory_cache[key] = value
                self.memory_expiry[key] = time.time() + ttl
                return True
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        try:
            if self.redis_client:
                value_str = self.redis_client.get(key)
                if value_str:
                    return json.loads(value_str)
            else:
                if key in self.memory_cache:
                    if time.time() < self.memory_expiry.get(key, 0):
                        return self.memory_cache[key]
                    else:
                        del self.memory_cache[key]
                        del self.memory_expiry[key]
            return None
        except Exception as e:
            logger.error(f"缓存获取失败: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.memory_expiry:
                    del self.memory_expiry[key]
            return True
        except Exception as e:
            logger.error(f"缓存删除失败: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
            else:
                count = 0
                keys_to_delete = []
                for key in self.memory_cache:
                    if fnmatch.fnmatch(key, pattern):
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    del self.memory_cache[key]
                    del self.memory_expiry[key]
                    count += 1

                return count
            return 0
        except Exception as e:
            logger.error(f"缓存清除失败: {e}")
            return 0

    def cache_query(self, query: str, result: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        key = self.get_cache_key("query", query, tenant_id)
        return self.set(key, result)

    def get_cached_query(self, query: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = self.get_cache_key("query", query, tenant_id)
        return self.get(key)

    def cache_retrieval(self, query: str, results: List[Dict[str, Any]], tenant_id: Optional[str] = None) -> bool:
        key = self.get_cache_key("retrieval", query, tenant_id)
        return self.set(key, results, ttl=self.retrieval_ttl)

    def get_cached_retrieval(self, query: str, tenant_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        key = self.get_cache_key("retrieval", query, tenant_id) # 生成缓存键
        result = self.get(key) # 从缓存获取检索结果
        if result:
            self._hit_count += 1 # 命中缓存，增加命中计数
        else:
            self._miss_count += 1 # 未命中缓存，增加未命中计数
        return result

    def cache_expansion(self, query: str, expanded_queries: List[str]) -> bool:
        """缓存查询扩展结果（TTL=24h，随知识库更新）"""
        key = self.get_cache_key("expansion", query)
        return self.set(key, expanded_queries, ttl=self.expansion_ttl)

    def get_cached_expansion(self, query: str) -> Optional[List[str]]:
        """获取缓存的查询扩展结果"""
        key = self.get_cache_key("expansion", query)
        return self.get(key)

    def invalidate_on_knowledge_update(self) -> int:
        """知识库更新时清除相关缓存"""
        count = 0
        count += self.clear_pattern("retrieval:*")
        count += self.clear_pattern("expansion:*")
        count += self.clear_pattern("query:*")
        logger.info(f"[缓存] 知识库更新，清除 {count} 条缓存")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total
        }


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    redis_cache = RedisCache()
else:
    redis_cache = None


def get_redis_cache() -> RedisCache:
    return redis_cache
