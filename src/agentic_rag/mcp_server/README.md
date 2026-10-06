# mcp_server：自研 JSON-RPC 2.0 工具服务

汽车售后业务的工具侧：4 个子 Server 聚合为一个 `combined-mcp` 服务，对外暴露统一的
`POST /mcp` 端点，共 **10 个工具 + 8 个资源**。协议层不依赖现成 MCP SDK，请求/响应模型
与路由实现见 `protocol.py`。

| 子 Server | 领域 | 代表工具 |
|-----------|------|----------|
| `inventory_server.py` | 配件库存 | `inventory.query` / `inventory.detail` / `inventory.low_stock_alert` |
| `employee_info_server.py` | 员工信息 | `employee.verify` / `employee.query` |
| `repair_order_server.py` | 维修工单 | `repair.query` / `repair.create` / `repair.update_status` |
| `regulation_clause_server.py` | 制度条款 | `regulation.query` / `regulation.search` |

支持方法：`initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`；
另有便捷端点 `GET /tools`、`GET /resources`（同受 API Key 保护）。

## 两种调用路径

- **内部**：Agent 经 `agent/mcp_bridge.py` 把工具清单转成 Function Calling schema，
  调用前先过 Harness 的安全网关（参数白名单 / 配额 / 风险分级 / HITL 审批）。
- **外部**：第三方系统直接 POST JSON-RPC，Header 携带 `X-API-Key`。

请求示例：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": { "name": "inventory.query", "arguments": { "part_number": "ENG-001" } },
  "id": "1"
}
```

## 启动

```bash
cp .env.example .env            # 至少填 MCP_API_KEY，可选 JAVA_API_URL / MCP_JWT_TOKEN
pip install -e .                # 或 export PYTHONPATH=src
python -m agentic_rag.mcp_server.main
```

配置由 `config.py` 从**仓库根 `.env`** 读取（不再依赖当前工作目录）。默认端口 8001 与 AI
服务相同，同机并跑时改 `MCP_PORT`。

## 边界（说明白，不装）

- 工具执行底层是 HTTP 转发到业务后端（`JAVA_API_URL`）。**业务后端与数据未随本仓开源**，
  因此没有后端时工具调用返回连接错误；本仓可复现的是协议层契约、聚合路由与 Agent 侧桥接。
- `MCP_API_KEY` 留空即 fail-closed：缺少 Key 的请求 401，Key 不匹配 403。
- 调用受保护的业务接口需要 `MCP_JWT_TOKEN`，缺失会收到 403。
