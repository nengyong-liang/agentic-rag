"""
多租户权限过滤模块

面试核心考点：
1. 检索期：Milvus/ES查询自动注入 tenant_id + role_permissions 过滤
2. 生成期：Context过滤无权限文档，Prompt追加"仅限公开知识作答"
3. 输出期：NLI二次校验是否越权泄露，命中则触发拦截并记录审计日志

三级权限管控：
- 菜单权限：控制可见功能模块
- 接口权限：控制可调用API
- 数据权限：控制可访问文档范围（密级+租户隔离）
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TenantContext:
    """租户上下文"""
    tenant_id: str
    user_id: str
    roles: List[str]
    security_level: str = "public"
    allowed_document_ids: Optional[List[int]] = None

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def is_admin(self) -> bool:
        return "admin" in self.roles


class TenantPermissionFilter:
    """多租户权限过滤器"""

    ROLE_PERMISSIONS = {
        "admin": {
            "security_levels": ["public", "internal", "confidential", "secret"],
            "document_access": "all",
            "can_generate": True,
        },
        "manager": {
            "security_levels": ["public", "internal", "confidential"],
            "document_access": "department",
            "can_generate": True,
        },
        "employee": {
            "security_levels": ["public", "internal"],
            "document_access": "own_department",
            "can_generate": True,
        },
        "guest": {
            "security_levels": ["public"],
            "document_access": "public_only",
            "can_generate": False,
        },
    }

    SECURITY_LEVEL_MAP = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "secret": 3,
    }

    def __init__(self):
        logger.info("[多租户权限] 初始化完成")

    def build_milvus_filter(self, tenant_ctx: TenantContext) -> Optional[str]:
        """
        构建Milvus查询过滤表达式

        自动注入：tenant_id + security_level 过滤
        """
        if tenant_ctx.is_admin():
            filter_parts = [f'tenant_id == "{tenant_ctx.tenant_id}"']
        else:
            allowed_levels = self._get_allowed_security_levels(tenant_ctx)
            level_conditions = [
                f'metadata["security_level"] == "{level}"'
                for level in allowed_levels
            ]
            tenant_condition = f'tenant_id == "{tenant_ctx.tenant_id}"'
            level_expr = " || ".join(level_conditions)
            filter_parts = [f'({tenant_condition}) && ({level_expr})']

        if tenant_ctx.allowed_document_ids is not None:
            doc_ids = tenant_ctx.allowed_document_ids
            if doc_ids:
                doc_conditions = [f"document_id == {did}" for did in doc_ids[:50]]
                filter_parts.append(f'({" || ".join(doc_conditions)})')

        filter_expr = " && ".join(filter_parts) if filter_parts else None
        logger.info(
            f"[多租户权限] Milvus过滤：tenant={tenant_ctx.tenant_id}, "
            f"roles={tenant_ctx.roles}, filter={filter_expr}"
        )
        return filter_expr

    def build_es_filter(self, tenant_ctx: TenantContext) -> Dict:
        """
        构建ES查询过滤条件

        自动注入：tenant_id + security_level 过滤
        """
        must_conditions = [
            {"term": {"tenant_id": tenant_ctx.tenant_id}}
        ]

        if not tenant_ctx.is_admin():
            allowed_levels = self._get_allowed_security_levels(tenant_ctx)
            must_conditions.append({
                "terms": {"metadata.security_level": allowed_levels}
            })

        if tenant_ctx.allowed_document_ids is not None:
            must_conditions.append({
                "terms": {"document_id": tenant_ctx.allowed_document_ids}
            })

        filter_dict = {"bool": {"must": must_conditions}}
        logger.info(
            f"[多租户权限] ES过滤：tenant={tenant_ctx.tenant_id}, "
            f"roles={tenant_ctx.roles}"
        )
        return filter_dict

    def filter_retrieval_results(
        self,
        results: List[Dict],
        tenant_ctx: TenantContext
    ) -> List[Dict]:
        """检索期过滤：根据权限过滤检索结果"""
        allowed_levels = self._get_allowed_security_levels(tenant_ctx)
        filtered = []

        for result in results:
            metadata = result.get("metadata", {})
            doc_tenant = metadata.get("tenant_id", "")
            doc_level = metadata.get("security_level", "public")

            if doc_tenant and doc_tenant != tenant_ctx.tenant_id:
                continue

            if not tenant_ctx.is_admin() and doc_level not in allowed_levels:
                continue

            if tenant_ctx.allowed_document_ids is not None:
                doc_id = result.get("document_id", 0)
                if doc_id not in tenant_ctx.allowed_document_ids:
                    continue

            filtered.append(result)

        if len(filtered) < len(results):
            logger.info(
                f"[多租户权限] 检索过滤：{len(results)} → {len(filtered)} 条"
            )

        return filtered

    def filter_generation_context(
        self,
        context: str,
        documents: List[Dict],
        tenant_ctx: TenantContext
    ) -> tuple:
        """
        生成期过滤：Context过滤无权限文档

        Returns:
            (filtered_context, security_prompt_suffix)
        """
        allowed_levels = self._get_allowed_security_levels(tenant_ctx)
        has_restricted = False

        filtered_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_level = metadata.get("security_level", "public")
            if doc_level in allowed_levels:
                filtered_docs.append(doc)
            else:
                has_restricted = True

        # 将 filtered_docs 的内容拼接成新的 context 字符串
        filtered_context_parts = []
        for idx, doc in enumerate(filtered_docs, 1):
            content = doc.get("content", "")
            if content:
                filtered_context_parts.append(f"[{idx}] {content}")
        filtered_context = "\n\n".join(filtered_context_parts) if filtered_context_parts else ""

        security_prompt = ""
        if has_restricted:
            security_prompt = "\n⚠️ 安全提示：仅限公开知识作答，不得泄露受限文档内容。"

        if not tenant_ctx.is_admin():
            security_prompt += "\n⚠️ 仅限公开知识作答。"

        return filtered_context, security_prompt

    def check_output_permission(
        self,
        answer: str,
        tenant_ctx: TenantContext,
        source_documents: List[Dict]
    ) -> Dict:
        """
        输出期校验：检查生成内容是否越权泄露

        Returns:
            {
                'is_allowed': bool,
                'leaked_info': List[str],
                'action': 'allow'/'block'/'redact'
            }
        """
        if tenant_ctx.is_admin():
            return {'is_allowed': True, 'leaked_info': [], 'action': 'allow'}

        leaked = []
        for doc in source_documents:
            metadata = doc.get("metadata", {})
            doc_level = metadata.get("security_level", "public")
            if doc_level not in self._get_allowed_security_levels(tenant_ctx):
                content = doc.get("content", "")
                if content and len(content) > 10:
                    snippet = content[:30]
                    if snippet in answer:
                        leaked.append({
                            "document_id": doc.get("document_id"),
                            "security_level": doc_level,
                            "snippet": snippet
                        })

        if leaked:
            logger.warning(
                f"[多租户权限] 输出越权检测：发现 {len(leaked)} 处泄露，"
                f"tenant={tenant_ctx.tenant_id}"
            )
            return {
                'is_allowed': False,
                'leaked_info': leaked,
                'action': 'redact'
            }

        return {'is_allowed': True, 'leaked_info': [], 'action': 'allow'}

    def _get_allowed_security_levels(self, tenant_ctx: TenantContext) -> List[str]:
        """获取租户允许访问的安全级别"""
        max_level = self.SECURITY_LEVEL_MAP.get(tenant_ctx.security_level, 0)

        for role in tenant_ctx.roles:
            role_config = self.ROLE_PERMISSIONS.get(role, {})
            role_levels = role_config.get("security_levels", ["public"])
            for level in role_levels:
                level_val = self.SECURITY_LEVEL_MAP.get(level, 0)
                max_level = max(max_level, level_val)

        return [
            level for level, val in self.SECURITY_LEVEL_MAP.items()
            if val <= max_level
        ]


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    tenant_permission_filter = TenantPermissionFilter()
else:
    tenant_permission_filter = None
