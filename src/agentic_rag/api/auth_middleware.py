"""
API 认证中间件（简化版）

支持两种认证方式：
1. Authorization: Bearer <token>
2. X-API-Key: <token>

配置文件中设置 API_KEY 环境变量，默认值为空（空时不做认证，兼容开发模式）。
"""

import os
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# 从环境变量读取 API_KEY，默认空字符串（开发模式不认证）
API_KEY = os.getenv("API_KEY", "")

# 免认证路径
SKIP_PATHS = ["/health", "/docs", "/openapi.json", "/redoc"]


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件"""

    async def dispatch(self, request: Request, call_next):
        # 开发模式：API_KEY 为空时跳过认证
        if not API_KEY:
            return await call_next(request)

        # 免认证路径放行
        path = request.url.path
        if any(path.startswith(skip) for skip in SKIP_PATHS):
            return await call_next(request)

        # 提取 token
        token = None

        # 从 Authorization header 提取
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        # 从 X-API-Key header 提取
        if not token:
            token = request.headers.get("X-API-Key", "")

        # 验证 token
        if not token or token != API_KEY:
            logger.warning(f"认证失败: path={path}, client={request.client.host if request.client else 'unknown'}")
            raise HTTPException(status_code=401, detail="认证失败，请提供有效的 API Key")

        return await call_next(request)