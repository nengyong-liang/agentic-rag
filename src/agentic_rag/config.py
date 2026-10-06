"""应用配置：检索链路、中间件、服务与模型凭据。

真实密钥放在仓库根 `.env`（已 gitignore）；这里在导入时把它载入进程环境，
使可选依赖 `agent_harness` 的 Provider 也能读到同一份凭据。
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class RagSettings(BaseSettings):
    """AI 服务配置"""

    # 服务配置
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8001

    # 豆包大模型配置
    DOUBAO_API_URL: str = "https://ark.cn-beijing.volces.com/api/v3/responses"
    DOUBAO_API_KEY: str = ""
    DOUBAO_MODEL: str = "doubao-seed-2-0-pro-260215"

    # 阿里云 Qwen 大模型配置
    # ⚠️ 低配版说明：
    #    架构图设计为 Qwen2.5-72B（本地部署，需 ~144GB 显存）。
    #    实际使用 qwen-plus API 模型（云端 API 调用），免本地 GPU 部署。
    #    16GB 无 GPU 机器上只能使用云端 API 方案。
    QWEN_API_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    QWEN_API_KEY: str = ""
    ALIBABA_API_KEY: str = ""
    QWEN_MODEL: str = "qwen-plus"

    # DeepSeek 大模型配置
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"

    # Milvus 配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_chunks"

    # Elasticsearch 配置
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_INDEX: str = "document_chunks"

    # 业务后端：工具执行侧转发的目标（与 mcp_server 共用同一份 .env）
    JAVA_API_URL: str = "http://localhost:8080/api"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # 向量模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # 切片配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # 检索配置
    # ⚠️ 低配版说明：
    #    架构图设计为 Milvus 检索 Top-50 / Reranker Top-10，追求更高召回率。
    #    实际默认 TOP_K=10、RERANK_TOP_K=5，在保证核心精度的前提下
    #    减少 LLM 上下文长度和 Token 消耗，适合原型验证阶段。
    #    最高支持动态扩大至 Top-20（复杂查询场景）。
    TOP_K: int = 10
    RERANK_TOP_K: int = 5

    # 联网搜索配置
    SERPAPI_KEY: str = ""
    BING_API_KEY: str = ""
    SEARCH_ENGINE: str = "google"

    # 多模态视觉配置
    DASHSCOPE_API_KEY: str = ""
    VISION_MODEL: str = "qwen-vl-plus"

    # 调度网关开关（统一调度网关 execution/dispatcher.py 使用）
    # ENABLE_NLI：生成后 NLI 一致性校验。默认 False；即使开启也走规则式，
    # 不强制加载 DeBERTa/外网，保证全量测试与弱网环境稳定。
    ENABLE_NLI: bool = False
    # ENABLE_MOA：复杂查询是否启用 MoA 多模型协作。默认 False（仅 simple 单模型），
    # 置 True 后 complex 查询才会聚合多模型（依赖外部模型可用性）。
    ENABLE_MOA: bool = False

    # CORS 配置
    CORS_ORIGINS: str = "http://localhost:5173"  # 逗号分隔的允许域名

    # 监控配置
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "agentic-rag"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    MONITOR_MAX_TRACES: str = "1000"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )


settings = RagSettings()
