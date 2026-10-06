import os
import sys
import logging


# 尝试导入 FastAPI 相关模块
has_fastapi = False
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    has_fastapi = True
    logging.info("FastAPI 已加载")

    # 检查 python-multipart 是否安装
    try:
        import python_multipart
        from agentic_rag.api.state import state
        state.has_multipart = True
        logging.info("python-multipart 已加载")
    except ImportError:
        from agentic_rag.api.state import state
        state.has_multipart = False
        logging.warning("缺少 python-multipart，文件上传功能将不可用")
except ImportError as e:
    logging.warning(f"FastAPI 未安装: {e}")

# 尝试导入其他模块
try:
    from agentic_rag.config import settings
    from agent_harness.providers.doubao import hybrid_llm
    from agentic_rag.agent.agent import get_agent
    from agentic_rag.agent.mcp_bridge import mcp_tool_bridge
    from agentic_rag.utils.redis_cache import get_redis_cache
    from agentic_rag.agent.agentic_rag import AgenticRAGOrchestrator, AgentConfig
    from agentic_rag.agent.a2a_adapter import A2AServer, DEFAULT_AGENT_CARD, setup_a2a_routes
    from agentic_rag.agent.memory import MemoryManager, MemoryConfig
    from agentic_rag.agent.long_running_task import TaskManager, TaskConfig
    has_dependencies = True
    logging.info("核心依赖已加载")
except ImportError as e:
    logging.warning(f"部分依赖未安装: {e}")
    has_dependencies = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入全局状态容器
from agentic_rag.api.state import state

# 初始化应用
app = None

if has_fastapi and has_dependencies:
    try:
        app = FastAPI(title="agentic-rag AI Service", version="1.0.0")

        # OpenTelemetry 链路追踪：未配置目的地时 init_tracing 自行保持 no-op
        try:
            from agent_harness.observability.tracing import init_tracing
            init_tracing(
                service_name=os.environ.get("OTEL_SERVICE_NAME", "agentic-rag"),
                console_export=os.environ.get(
                    "OTEL_CONSOLE_EXPORT", "").lower() in ("1", "true", "yes"),
            )
        except Exception as e:
            logger.warning(f"OpenTelemetry 初始化失败（不影响业务）: {e}")

        cors_origins = getattr(settings, 'CORS_ORIGINS', 'http://localhost:5173')
        allowed_origins = [origin.strip() for origin in cors_origins.split(',') if origin.strip()]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册 API Key 认证中间件
        try:
            from agentic_rag.api.auth_middleware import AuthMiddleware
            app.add_middleware(AuthMiddleware)
            logger.info("API Key 认证中间件已注册")
        except Exception as e:
            logger.warning(f"API Key 认证中间件注册失败: {e}")

        # 设置 DATA_DIR
        state.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
        os.makedirs(state.DATA_DIR, exist_ok=True)

        # 初始化 Agent
        try:
            state.agent = get_agent()
        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}")
            state.agent = None

        # 初始化 MCP Bridge
        try:
            state.mcp_bridge = mcp_tool_bridge
            logger.info("MCP Bridge 初始化成功")
        except Exception as e:
            logger.error(f"MCP Bridge 初始化失败: {e}")
            state.mcp_bridge = None

        # 初始化 Redis 缓存
        try:
            state.redis_cache = get_redis_cache()
        except Exception as e:
            logger.error(f"Redis 缓存初始化失败：{e}")
            state.redis_cache = None

        # 注册 A2A 协议端点
        try:
            a2a_server = A2AServer(DEFAULT_AGENT_CARD)
            setup_a2a_routes(app, a2a_server)
            logger.info("A2A 协议端点已注册")
        except Exception as e:
            logger.error(f"A2A 协议端点注册失败：{e}")

        # 初始化 Agent Memory
        try:
            state.memory_manager = MemoryManager(MemoryConfig())
            logger.info("Agent Memory 初始化成功")
        except Exception as e:
            logger.error(f"Agent Memory 初始化失败：{e}")
            state.memory_manager = None

        # 初始化长程任务管理器
        try:
            state.task_manager = TaskManager(TaskConfig())
            logger.info("长程任务管理器初始化成功")
        except Exception as e:
            logger.error(f"长程任务管理器初始化失败：{e}")
            state.task_manager = None

        # 初始化 Agentic RAG 2.0（使用全局 memory_manager）
        try:
            agentic_config = AgentConfig(enabled=True)
            state.agentic_rag_orchestrator = AgenticRAGOrchestrator(agentic_config, memory_manager=state.memory_manager)
            logger.info("Agentic RAG 2.0 初始化成功（使用全局 Memory）")
        except Exception as e:
            logger.error(f"Agentic RAG 2.0 初始化失败：{e}")

        # 初始化 hybrid_llm
        state.hybrid_llm = hybrid_llm

        # ============================================================
        # 初始化 Harness 统一运行层
        # ============================================================
        try:
            from agent_harness import (
                EventStore, get_event_store,
                get_default_provider,
                get_default_projector,
                get_default_breaker, get_default_validator,
                get_default_governor,
                get_default_gateway,
                get_default_auditor,
                HarnessLoop, HarnessConfig,
            )
            from agentic_rag.utils.redis_cache import get_redis_cache

            # 初始化各子系统
            redis_client = state.redis_cache

            # Event Store（带 Redis 后端）
            state.harness_event_store = get_event_store(redis_client=redis_client)

            # 模型 Provider（使用现有 llm 服务）
            try:
                from agent_harness.providers.doubao import DoubaoLLMService
                from agent_harness.providers.registry import DoubaoModelProvider
                doubao_llm = DoubaoLLMService()
                state.harness_provider = DoubaoModelProvider(llm_service=doubao_llm)
            except Exception as e:
                logger.warning(f"Harness ModelProvider 初始化失败: {e}")
                state.harness_provider = get_default_provider("doubao")

            # 其他子系统
            state.harness_projector = get_default_projector()
            state.harness_breaker = get_default_breaker()
            state.harness_validator = get_default_validator()
            state.harness_governor = get_default_governor()
            state.harness_gateway = get_default_gateway()
            state.harness_auditor = get_default_auditor()

            # 安全网关：关联 ToolAuthorizer
            try:
                from agent_harness.sandbox.tool_authorization import ToolAuthorizer
                state.harness_gateway = get_default_gateway()
                authorizer = ToolAuthorizer()
                # 通过内部属性注入 authorizer
                state.harness_gateway._authorizer = authorizer
            except Exception as e:
                logger.warning(f"Harness 安全网关授权器初始化失败: {e}")

            # Harness 主循环
            harness_config = HarnessConfig(
                max_iterations=25,
                max_tool_calls=50,
                max_wall_clock_seconds=300,
                enable_safety_gateway=True,
                enable_audit=True,
                enable_event_store=True,
            )
            state.harness_loop = HarnessLoop(
                config=harness_config,
                event_store=state.harness_event_store,
                model_provider=state.harness_provider,
                projector=state.harness_projector,
                breaker=state.harness_breaker,
                validator=state.harness_validator,
                governor=state.harness_governor,
                gateway=state.harness_gateway,
                auditor=state.harness_auditor,
            )

            # 关联工具执行器（MCP Bridge）
            if state.mcp_bridge is not None:
                state.harness_loop.set_tool_executor(state.mcp_bridge.execute_tool)
                logger.info("Harness 工具执行器已关联 MCP Bridge")

            logger.info("Harness 统一运行层初始化成功")
        except Exception as e:
            logger.error(f"Harness 统一运行层初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # ============================================================
        # 初始化数据合成管线
        try:
            from agentic_rag.data_synthesis.synthesizer import DataSynthesisPipeline, PipelineConfig
            state.synthesis_pipeline = DataSynthesisPipeline(PipelineConfig())
            state._synthesis_available = True
        except ImportError as e:
            logger.warning(f"数据合成模块不可用: {e}")
            state.synthesis_pipeline = None
            state._synthesis_available = False

        # ============================================================
        # 注册路由
        # ============================================================
        from agentic_rag.api.routes.chat import router as chat_router
        from agentic_rag.api.routes.document import router as document_router
        from agentic_rag.api.routes.agent import router as agent_router
        from agentic_rag.api.routes.knowledge import router as knowledge_router
        from agentic_rag.api.routes.monitor import router as monitor_router
        from agentic_rag.api.routes.security import router as security_router
        from agentic_rag.api.routes.task import router as task_router
        from agentic_rag.api.routes.inference import router as inference_router
        from agentic_rag.api.routes.memory import router as memory_router

        app.include_router(chat_router)
        app.include_router(document_router)
        app.include_router(agent_router)
        app.include_router(knowledge_router)
        app.include_router(monitor_router)
        app.include_router(security_router)
        app.include_router(task_router)
        app.include_router(inference_router)
        app.include_router(memory_router)

        # 健康检查端点保留在 main.py
        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}

    except Exception as e:
        logger.error(f"应用初始化失败: {e}")
        app = None


if __name__ == "__main__":
    if app and hasattr(settings, 'SERVICE_HOST') and hasattr(settings, 'SERVICE_PORT'):
        import uvicorn
        uvicorn.run(app, host=settings.SERVICE_HOST, port=settings.SERVICE_PORT)
    else:
        logger.error("应用未初始化或配置不完整，无法启动服务")
