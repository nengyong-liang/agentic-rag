"""
MCP Server 主入口
提供标准 MCP 协议接口，支持内部 Python Agent 和外部第三方系统调用
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
import time
import os
from dotenv import load_dotenv

from agentic_rag.mcp_server.protocol import MCPServer, MCPRequest, MCPResponse
from agentic_rag.mcp_server.inventory_server import InventoryMCPServer
from agentic_rag.mcp_server.repair_order_server import RepairOrderMCPServer
from agentic_rag.mcp_server.employee_info_server import EmployeeInfoMCPServer
from agentic_rag.mcp_server.regulation_clause_server import RegulationClauseMCPServer

# 加载环境变量
load_dotenv()

# ============== 配置管理 ==============
class MCPConfig:
    """MCP Server 配置"""
    HOST = os.getenv("MCP_HOST", "0.0.0.0")
    PORT = int(os.getenv("MCP_PORT", "8001"))
    API_KEY = os.getenv("MCP_API_KEY", "")
    RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============== FastAPI 应用 ==============
app = FastAPI(
    title="agentic-rag MCP Server",
    description="标准 MCP 协议服务器，提供配件库存查询、维修工单生成等工具接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件（支持外部调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== MCP Server 实例 ==============
inventory_server = InventoryMCPServer()
repair_server = RepairOrderMCPServer()
employee_server = EmployeeInfoMCPServer()
regulation_server = RegulationClauseMCPServer()

# 合并所有工具
class CombinedMCPServer(MCPServer):
    """组合 MCP Server"""
    def __init__(self, *servers):
        super().__init__(name="combined-mcp", version="1.0.0")
        for server in servers:
            # 合并工具
            for name, tool_def in server.tools.items():
                self.tools[name] = tool_def
                self.tool_handlers[name] = server.tool_handlers[name]
            # 合并资源
            for uri, resource_def in server.resources.items():
                self.resources[uri] = resource_def
                self.resource_handlers[uri] = server.resource_handlers[uri]

combined_server = CombinedMCPServer(inventory_server, repair_server, employee_server, regulation_server)

# ============== 身份认证 ==============
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """验证 API Key"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key"
        )
    if api_key != MCPConfig.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无效的 API Key"
        )
    return api_key

# ============== 日志中间件 ==============
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求日志"""
    start_time = time.time()
    
    # 获取请求信息
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    
    logger.info(f"请求开始 | {method} {path} | IP: {client_ip}")
    
    # 处理请求
    response = await call_next(request)
    
    # 计算耗时
    process_time = time.time() - start_time
    
    logger.info(
        f"请求结束 | {method} {path} | "
        f"状态：{response.status_code} | "
        f"耗时：{process_time:.3f}s | "
        f"IP: {client_ip}"
    )
    
    return response

# ============== MCP 接口 ==============
@app.post("/mcp", response_model=MCPResponse)
async def mcp_endpoint(
    request: MCPRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    MCP 标准接口
    
    支持以下方法：
    - tools/list: 获取可用工具列表
    - tools/call: 调用指定工具
    - resources/list: 获取可用资源列表
    - resources/read: 读取指定资源
    
    请求格式遵循 JSON-RPC 2.0 规范
    """
    try:
        logger.debug(f"处理 MCP 请求：{request.method}")
        response = await combined_server.handle_request(request)
        return response
    except Exception as e:
        logger.error(f"MCP 请求处理失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ============== 健康检查接口 ==============
@app.get("/health")
async def health_check():
    """健康检查接口（无需认证）"""
    return {
        "status": "healthy",
        "service": MCPConfig.HOST,
        "version": "1.0.0",
        "timestamp": time.time()
    }

# ============== 工具列表接口（便捷访问） ==============
@app.get("/tools")
async def list_tools(api_key: str = Depends(verify_api_key)):
    """获取可用工具列表（便捷接口）"""
    tools = combined_server.list_tools()
    return {
        "jsonrpc": "2.0",
        "result": tools,
        "id": "tools-list"
    }

# ============== 资源列表接口（便捷访问） ==============
@app.get("/resources")
async def list_resources(api_key: str = Depends(verify_api_key)):
    """获取可用资源列表（便捷接口）"""
    resources = combined_server.list_resources()
    return {
        "jsonrpc": "2.0",
        "result": resources,
        "id": "resources-list"
    }

# ============== 启动入口 ==============
def main():
    """启动 MCP Server"""
    logger.info(f"启动 MCP Server...")
    logger.info(f"服务地址：http://{MCPConfig.HOST}:{MCPConfig.PORT}")
    logger.info(f"API 文档：http://{MCPConfig.HOST}:{MCPConfig.PORT}/docs")
    logger.info(f"限流配置：{MCPConfig.RATE_LIMIT}")
    logger.info(f"日志级别：{MCPConfig.LOG_LEVEL}")
    
    uvicorn.run(
        app,
        host=MCPConfig.HOST,
        port=MCPConfig.PORT,
        log_level=MCPConfig.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
