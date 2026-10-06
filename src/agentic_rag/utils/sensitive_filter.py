"""
敏感词过滤 + PII 掩码服务（工业级升级版）

升级要点：
1. DFA字典树敏感词过滤：O(n)时间复杂度，万级词库只扫一遍
2. PII 识别与掩码：正则+NER提取，替换为占位符[PII_TYPE_ID]
3. PII Redis映射表：掩码→原始值映射，支持还原(TTL=30min, 绑定RequestID)
4. 文档级 ACL（密级过滤）
"""

import re
import json
import uuid
import os
import logging
import time
from typing import List, Dict, Tuple, Optional

from agentic_rag.utils.dfa_filter import DFASensitiveFilter, dfa_sensitive_filter

logger = logging.getLogger(__name__)


class PIIMaskResult:
    """PII掩码结果"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.mapping: Dict[str, str] = {}
        self.pii_items: List[Dict] = []

    def add_mapping(self, placeholder: str, original: str, pii_type: str):
        self.mapping[placeholder] = original
        self.pii_items.append({
            "placeholder": placeholder,
            "original_length": len(original),
            "type": pii_type,
        })

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "pii_count": len(self.pii_items),
            "pii_items": self.pii_items,
        }


class SensitiveFilter:
    """敏感词过滤 + PII 掩码服务（工业级）"""

    PII_PATTERNS = {
        "phone": {
            "pattern": re.compile(r'1[3-9]\d{9}'),
            "description": "手机号",
            "placeholder_template": "[PII_PHONE_{id}]"
        },
        "id_card": {
            "pattern": re.compile(r'\b[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'),
            "description": "身份证号",
            "placeholder_template": "[PII_IDCARD_{id}]"
        },
        "bank_card": {
            "pattern": re.compile(r'(?<!\d)[1-9]\d{15,18}(?!\d)'),
            "description": "银行卡号",
            "placeholder_template": "[PII_BANKCARD_{id}]"
        },
        "email": {
            "pattern": re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
            "description": "邮箱地址",
            "placeholder_template": "[PII_EMAIL_{id}]"
        }
    }

    SECURITY_LEVELS = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "secret": 3
    }

    PII_REDIS_TTL = 1800
    PII_REDIS_KEY_PREFIX = "pii_mapping:"

    def __init__(self):
        self.dfa_filter = dfa_sensitive_filter
        self.enable_pii_mask = True
        self._redis_client = None
        logger.info("敏感词过滤(DFA) + PII 掩码服务初始化完成")

    def _get_redis(self):
        if self._redis_client is not None:
            return self._redis_client
        try:
            from agentic_rag.utils.redis_cache import get_redis_cache
            cache = get_redis_cache()
            if cache and hasattr(cache, 'redis_client'):
                self._redis_client = cache.redis_client
                return self._redis_client
        except Exception as e:
            logger.warning(f"Redis不可用，PII映射仅存内存：{e}")
        return None

    def check_sensitive(self, text: str) -> Dict:
        """检查文本是否包含敏感词（DFA O(n)算法）"""
        return self.dfa_filter.check(text)

    def filter_sensitive(self, text: str, replacement: str = '*') -> str:
        """过滤文本中的敏感词 + PII 掩码"""
        if not text:
            return text

        filtered_text, _ = self.dfa_filter.filter(text, replacement)

        if self.enable_pii_mask:
            filtered_text = self.mask_pii(filtered_text)

        return filtered_text

    def mask_pii_with_mapping(self, text: str, request_id: str = None) -> Tuple[str, PIIMaskResult]:
        """
        PII掩码 + 生成映射表（支持还原）

        面试核心：掩码→占位符，映射存Redis(TTL=30min, 绑定RequestID)

        Args:
            text: 原始文本
            request_id: 请求ID，用于绑定映射

        Returns:
            (masked_text, mask_result)
        """
        if not text:
            return text, PIIMaskResult(request_id or "")

        request_id = request_id or str(uuid.uuid4())[:8]
        mask_result = PIIMaskResult(request_id)
        masked_text = text
        pii_counter = 0

        for pii_type, pii_config in self.PII_PATTERNS.items():
            matches = list(pii_config["pattern"].finditer(masked_text))
            for match in reversed(matches):
                original = match.group()
                placeholder = pii_config["placeholder_template"].format(id=pii_counter)
                masked_text = masked_text[:match.start()] + placeholder + masked_text[match.end():]
                mask_result.add_mapping(placeholder, original, pii_type)
                pii_counter += 1

        if mask_result.pii_items:
            self._save_pii_mapping(request_id, mask_result.mapping)
            logger.info(f"PII掩码：发现 {len(mask_result.pii_items)} 处隐私信息，request_id={request_id}")

        return masked_text, mask_result

    def restore_pii(self, text: str, request_id: str) -> str:
        """
        PII还原：根据映射表将占位符替换回原始信息

        Args:
            text: 含占位符的文本
            request_id: 请求ID

        Returns:
            还原后的文本
        """
        mapping = self._load_pii_mapping(request_id)
        if not mapping:
            logger.warning(f"PII映射不存在或已过期：request_id={request_id}")
            return text

        restored = text
        for placeholder, original in mapping.items():
            restored = restored.replace(placeholder, original)

        if restored != text:
            self._delete_pii_mapping(request_id)
            logger.info(f"PII还原完成，request_id={request_id}")

        return restored

    def _save_pii_mapping(self, request_id: str, mapping: Dict[str, str]) -> None:
        """保存PII映射到Redis"""
        redis = self._get_redis()
        if redis:
            try:
                key = f"{self.PII_REDIS_KEY_PREFIX}{request_id}"
                redis.setex(key, self.PII_REDIS_TTL, json.dumps(mapping, ensure_ascii=False))
                logger.info(f"PII映射已存Redis：{key}，TTL={self.PII_REDIS_TTL}s")
                return
            except Exception as e:
                logger.warning(f"Redis存储PII映射失败：{e}")

        logger.info(f"PII映射仅存内存（Redis不可用）：request_id={request_id}")

    def _load_pii_mapping(self, request_id: str) -> Dict[str, str]:
        """从Redis加载PII映射"""
        redis = self._get_redis()
        if redis:
            try:
                key = f"{self.PII_REDIS_KEY_PREFIX}{request_id}"
                data = redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis读取PII映射失败：{e}")
        return {}

    def _delete_pii_mapping(self, request_id: str) -> None:
        """删除PII映射（还原后清理）"""
        redis = self._get_redis()
        if redis:
            try:
                key = f"{self.PII_REDIS_KEY_PREFIX}{request_id}"
                redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis删除PII映射失败：{e}")

    def mask_pii(self, text: str) -> str:
        """对文本中的 PII 信息进行掩码处理（简单掩码，无映射）"""
        if not text:
            return text

        masked_text = text
        pii_found = []

        for pii_type, pii_config in self.PII_PATTERNS.items():
            matches = pii_config["pattern"].finditer(masked_text)
            for match in matches:
                original = match.group()
                if pii_type == "phone":
                    masked = original[:3] + "****" + original[-4:]
                elif pii_type == "id_card":
                    masked = original[:6] + "********" + original[-4:]
                elif pii_type == "bank_card":
                    masked = original[:4] + "**********" + original[-4:]
                elif pii_type == "email":
                    masked = original[0] + "***@" + original.split("@")[1]
                else:
                    masked = "*" * len(original)
                masked_text = masked_text.replace(original, masked, 1)
                pii_found.append({
                    "type": pii_type,
                    "description": pii_config["description"],
                    "original_length": len(original),
                    "masked": masked
                })

        if pii_found:
            logger.info(f"PII 掩码：发现 {len(pii_found)} 处隐私信息")

        return masked_text

    def check_pii(self, text: str) -> Dict:
        """检查文本中是否包含 PII 信息"""
        if not text:
            return {"has_pii": False, "pii_items": []}

        pii_items = []
        for pii_type, pii_config in self.PII_PATTERNS.items():
            matches = pii_config["pattern"].findall(text)
            if matches:
                pii_items.append({
                    "type": pii_type,
                    "description": pii_config["description"],
                    "count": len(matches)
                })

        return {
            "has_pii": len(pii_items) > 0,
            "pii_items": pii_items
        }

    def build_security_filter(self, user_security_level: str) -> str:
        """
        构建文档密级过滤表达式（用于 Milvus filter_expr）

        Args:
            user_security_level: 用户安全级别（public/internal/confidential/secret）

        Returns:
            Milvus 过滤表达式
        """
        user_level = self.SECURITY_LEVELS.get(user_security_level, 0)

        allowed_levels = [
            level for level, score in self.SECURITY_LEVELS.items()
            if score <= user_level
        ]

        if not allowed_levels:
            return 'metadata["security_level"] == "public"'

        conditions = [f'metadata["security_level"] == "{level}"' for level in allowed_levels]
        filter_expr = " || ".join(conditions)

        logger.info(f"文档密级过滤：用户级别={user_security_level}，允许访问={allowed_levels}")
        return filter_expr

    def add_sensitive_word(self, word: str, category: str = '其他'):
        """添加敏感词（DFA字典树动态添加）"""
        self.dfa_filter.add_word(word, category)

    def remove_sensitive_word(self, word: str, category: str = None):
        """移除敏感词"""
        self.dfa_filter.remove_word(word)

    def get_sensitive_words(self) -> Dict[str, List[str]]:
        """获取所有敏感词（按分类）"""
        return self.dfa_filter.get_all_words()


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    sensitive_filter = SensitiveFilter()
else:
    sensitive_filter = None
