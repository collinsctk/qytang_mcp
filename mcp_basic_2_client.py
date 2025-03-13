import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

# 将 SERVER_HOST 修改为实际的服务器 IP 或域名，例如 "192.168.1.100" 或 "mcp.qytang.com"
SERVER_HOST = "196.21.5.218"
SERVER_PORT = 8000

async def main():
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/sse"
    async with sse_client(url=url) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化 MCP 会话（握手）
            await session.initialize()
            print("成功连接到 MCP 服务器！")
            all_resources = await session.list_resources()
            print(f"所有资源: {all_resources}")
            all_prompts = await session.list_prompts()
            print(f"所有提示: {all_prompts}")
            all_tools = await session.list_tools()
            print(f"所有工具: {all_tools}")
            
            # 1. 调用工具： add(2, 3)
            result_tool = await session.call_tool("add", arguments={"a": 2, "b": 3})
            print(f"调用工具 add(2, 3) 的结果: {result_tool}")

            print(result_tool.content[0].text)

            
            # 2. 调用资源：查询用户名 'qinke' 的密码
            username = "qinke"
            result_resource = await session.read_resource(f"psql://{username}")
            print(f"资源 getuserpassword://{username} 的返回结果: {result_resource}")

            print(result_resource.contents[0].text)
            # 3. 调用提示：格式化输出 'qinke' 的密码信息
            result_prompt = await session.get_prompt("query-password", arguments={"username": username})
            print(f"提示 query-password 返回的内容: {result_prompt}")

            print(result_prompt.messages[0].content.text)

if __name__ == "__main__":
    asyncio.run(main())
