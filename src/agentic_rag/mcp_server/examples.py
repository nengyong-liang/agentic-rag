"""
MCP 客户端使用示例
演示如何使用 MCP 客户端 SDK 调用配件库存查询服务
"""

import asyncio
import os
from agentic_rag.mcp_server.client import MCPClient, InventoryClient


async def example_1_basic_usage():
    """示例 1：基本用法"""
    print("=" * 60)
    print("示例 1：基本用法")
    print("=" * 60)
    
    # 初始化客户端
    client = MCPClient(
        base_url="http://localhost:8001",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    
    # 1. 健康检查
    print("\n1. 健康检查")
    health = await client.health_check()
    print(f"   服务状态：{health['status']}")
    print(f"   服务版本：{health['version']}")
    
    # 2. 获取工具列表
    print("\n2. 获取工具列表")
    tools = await client.list_tools()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    # 3. 获取资源列表
    print("\n3. 获取资源列表")
    resources = await client.list_resources()
    for resource in resources:
        print(f"   - {resource['name']}: {resource['uri']}")


async def example_2_call_tool():
    """示例 2：调用工具"""
    print("\n" + "=" * 60)
    print("示例 2：调用工具")
    print("=" * 60)
    
    client = MCPClient(
        base_url="http://localhost:8001",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    
    # 1. 查询配件库存（按编号）
    print("\n1. 查询配件库存（按编号）")
    result = await client.call_tool(
        "inventory.query",
        {"part_number": "PN001"}
    )
    if result.get("success"):
        data = result.get("data", [])
        print(f"   查询成功，共 {result.get('total', 0)} 条记录")
        if data:
            item = data[0]
            print(f"   配件编号：{item.get('partNumber')}")
            print(f"   配件名称：{item.get('partName')}")
            print(f"   库存数量：{item.get('stockQuantity')}")
    else:
        print(f"   查询失败：{result.get('message')}")
    
    # 2. 查询配件库存（按车型）
    print("\n2. 查询配件库存（按车型）")
    result = await client.call_tool(
        "inventory.query",
        {"vehicle_model": "宝马 3 系"}
    )
    if result.get("success"):
        print(f"   查询成功，共 {result.get('total', 0)} 条记录")
    
    # 3. 获取配件详情
    print("\n3. 获取配件详情")
    result = await client.call_tool(
        "inventory.detail",
        {"part_number": "PN001"}
    )
    if result.get("success"):
        data = result.get("data", {})
        print(f"   配件名称：{data.get('partName')}")
        print(f"   配件类别：{data.get('category')}")
        print(f"   适用车型：{data.get('vehicleModel')}")
        print(f"   库存数量：{data.get('stockQuantity')}")
        print(f"   仓库位置：{data.get('warehouseLocation')}")
        print(f"   单价：¥{data.get('unitPrice')}")
        print(f"   供应商：{data.get('supplier')}")
    
    # 4. 获取低库存预警
    print("\n4. 获取低库存预警")
    result = await client.call_tool(
        "inventory.low_stock_alert",
        {"threshold": 20}
    )
    if result.get("success"):
        data = result.get("data", [])
        print(f"   预警查询成功，共 {len(data)} 个配件库存不足")
        for item in data[:3]:  # 只显示前 3 个
            print(f"   - {item.get('partNumber')}: {item.get('stockQuantity')}件")


async def example_3_inventory_client():
    """示例 3：使用高级客户端"""
    print("\n" + "=" * 60)
    print("示例 3：使用高级客户端（InventoryClient）")
    print("=" * 60)
    
    # 初始化基础客户端
    mcp_client = MCPClient(
        base_url="http://localhost:8001",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    
    # 初始化高级客户端
    inventory_client = InventoryClient(mcp_client)
    
    # 1. 查询库存
    print("\n1. 查询库存（配件编号 PN001）")
    result = await inventory_client.query(part_number="PN001")
    if result.get("success"):
        for item in result.get("data", [])[:2]:
            print(f"   - {item.get('partName')}: {item.get('stockQuantity')}件")
    
    # 2. 查询某车型的所有配件
    print("\n2. 查询某车型的所有配件（奥迪 A6L）")
    result = await inventory_client.query(vehicle_model="奥迪 A6L")
    if result.get("success"):
        print(f"   共查询到 {result.get('total', 0)} 个配件")
    
    # 3. 获取配件详情
    print("\n3. 获取配件详情（PN002）")
    result = await inventory_client.get_detail("PN002")
    if result.get("success"):
        data = result.get("data", {})
        print(f"   配件：{data.get('partName')}")
        print(f"   库存：{data.get('stockQuantity')}件")
        print(f"   位置：{data.get('warehouseLocation')}")
    
    # 4. 低库存预警
    print("\n4. 低库存预警（阈值 20）")
    result = await inventory_client.get_low_stock_alert(threshold=20)
    if result.get("success"):
        data = result.get("data", [])
        print(f"   共 {len(data)} 个配件需要补货")


async def example_4_error_handling():
    """示例 4：错误处理"""
    print("\n" + "=" * 60)
    print("示例 4：错误处理")
    print("=" * 60)
    
    # 1. 使用错误的 API Key
    print("\n1. 使用错误的 API Key")
    client = MCPClient(
        base_url="http://localhost:8001",
        api_key="wrong-key"
    )
    try:
        await client.list_tools()
    except Exception as e:
        print(f"   认证失败：{e}")
    
    # 2. 调用不存在的工具
    print("\n2. 调用不存在的工具")
    client = MCPClient(
        base_url="http://localhost:8001",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    try:
        await client.call_tool("nonexistent.tool", {})
    except Exception as e:
        print(f"   工具不存在：{e}")
    
    # 3. 服务不可用
    print("\n3. 服务不可用")
    client = MCPClient(
        base_url="http://localhost:9999",
        api_key=os.getenv("MCP_API_KEY", "")
    )
    try:
        await client.health_check()
    except Exception as e:
        print(f"   服务不可用：{e}")


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("MCP 客户端使用示例")
    print("=" * 60)
    
    try:
        # 运行示例
        await example_1_basic_usage()
        await example_2_call_tool()
        await example_3_inventory_client()
        await example_4_error_handling()
    except Exception as e:
        print(f"\n运行示例时出错：{e}")
        print("请确保 MCP Server 已启动：python -m mcp_server")
    
    print("\n" + "=" * 60)
    print("示例运行结束")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
