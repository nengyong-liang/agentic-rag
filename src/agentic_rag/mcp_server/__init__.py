"""
MCP Server 包
提供标准 MCP 协议接口，支持内部和外部调用
"""

from .protocol import MCPServer, MCPRequest, MCPResponse, ToolDefinition, ResourceDefinition
from .inventory_server import InventoryMCPServer
from .config import settings

__version__ = "1.0.0"
__all__ = [
    "MCPServer",
    "MCPRequest",
    "MCPResponse",
    "ToolDefinition",
    "ResourceDefinition",
    "InventoryMCPServer",
    "settings"
]
