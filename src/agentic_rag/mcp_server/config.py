"""
MCP Server 配置管理
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Optional
from loguru import logger

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class Settings:
    """MCP Server 配置类"""
    
    # 服务配置
    HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("MCP_PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # 身份认证
    API_KEY: str = os.getenv("MCP_API_KEY", "")
    
    # Java API 配置
    JAVA_API_URL: str = os.getenv("JAVA_API_URL", "http://localhost:8080/api")
    JAVA_API_TIMEOUT: int = int(os.getenv("JAVA_API_TIMEOUT", "30"))
    # MCP Server 调用 Java 后端使用的 JWT Token（从环境变量读取，不可硬编码）
    MCP_JWT_TOKEN: str = os.getenv("MCP_JWT_TOKEN", "")
    
    # 限流配置
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "100/minute")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/mcp_server.log")
    
    
    # Redis 配置（可选，用于缓存）
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # 监控配置
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_INTERVAL: int = int(os.getenv("METRICS_INTERVAL", "60"))
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.API_KEY:
            logger.warning("未设置 MCP_API_KEY，外部调用将全部被拒绝（fail-closed）")
        
        if cls.DEBUG:
            logger.warning("调试模式已启用，生产环境请关闭！")
        
        logger.info("配置验证通过")
        logger.info(f"服务端口：{cls.PORT}")
        logger.info(f"限流配置：{cls.RATE_LIMIT}")
        logger.info(f"日志级别：{cls.LOG_LEVEL}")
        logger.info(f"Java API: {cls.JAVA_API_URL}")
        
        return True


# 全局配置实例
settings = Settings()
