import asyncio
import ssl
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx

# 将 SERVER_HOST 修改为实际的服务器 IP 或域名，例如 "192.168.1.100" 或 "mcp.qytang.com"
SERVER_HOST = "196.21.5.218"
SERVER_PORT = 443  # 使用 HTTPS 的标准端口，确保与服务器配置一致

# 是否使用 SSL/TLS 加密连接
USE_SSL = True

# 根证书路径
ROOT_CA_PATH = "./certs/ca.cer"  # 修改为您的根证书路径

# 修改 httpx 的 SSL 验证行为，使用根证书
# 保存原始的 AsyncClient 类
original_async_client = httpx.AsyncClient

# 创建一个新的 AsyncClient 类，使用根证书进行验证
class CustomCAAsyncClient(original_async_client):
    def __init__(self, *args, **kwargs):
        if 'verify' not in kwargs and os.path.isfile(ROOT_CA_PATH):
            kwargs['verify'] = ROOT_CA_PATH
        super().__init__(*args, **kwargs)

# 替换 httpx 的 AsyncClient 类
if USE_SSL and os.path.isfile(ROOT_CA_PATH):
    httpx.AsyncClient = CustomCAAsyncClient
    print(f"使用根证书进行 SSL 验证: {ROOT_CA_PATH}")
else:
    print("警告: 未找到根证书，将使用系统默认的证书验证")

async def main():
    # 根据是否使用 SSL 构建不同的 URL 前缀
    protocol = "https" if USE_SSL else "http"
    url = f"{protocol}://{SERVER_HOST}:{SERVER_PORT}/sse"
    
    print(f"连接到服务器: {url}")
    
    try:
        # 使用 sse_client 连接到服务器
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
                
                sys_info = await session.read_resource("info://system")
                print(f"资源 info://system 的返回结果: {sys_info}")
                print(sys_info.contents[0].text)    
                
                # 3. 调用提示：格式化输出 'qinke' 的密码信息
                result_prompt = await session.get_prompt("query-password", arguments={"username": username})
                print(f"提示 query-password 返回的内容: {result_prompt}")

                print(result_prompt.messages[0].content.text)
    except Exception as e:
        print(f"连接错误: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
