import asyncio
import ssl
import os
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx

# 将 SERVER_HOST 修改为实际的服务器 IP 或域名，例如 "192.168.1.100" 或 "mcp.qytang.com"
SERVER_HOST = "196.21.5.218"
SERVER_PORT = 443  # 使用 HTTPS 的标准端口，确保与服务器配置一致

# 是否使用 SSL/TLS 加密连接
USE_SSL = True

# 全局变量：是否启用认证（需要与服务器端保持一致）
enable_authen = False

# 根证书路径
ROOT_CA_PATH = "./certs/ca.cer"  # 修改为您的根证书路径

# 认证信息
USERNAME = "qinke"
PASSWORD = "cisco"

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

# 获取认证token
async def get_auth_token(username, password):
    # 如果认证被禁用，返回一个虚拟token
    if not enable_authen:
        print("认证已禁用，使用虚拟token")
        return {
            "token": "disabled_authentication_mode",
            "expiry": 0,
            "expires_in": 0,
            "token_type": "Bearer"
        }
        
    protocol = "https" if USE_SSL else "http"
    token_url = f"{protocol}://{SERVER_HOST}:{SERVER_PORT}/get_token"
    
    print(f"正在获取认证token...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                json={"username": username, "password": password},
                timeout=30.0
            )
            
            if response.status_code == 200:
                token_data = response.json()
                print(f"成功获取token: {token_data['token'][:10]}...")
                return token_data
            else:
                print(f"获取token失败: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"获取token时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 自定义SSE客户端，添加认证头
def authenticated_sse_client(url, token=None):
    """返回带有认证头的SSE客户端"""
    # 如果认证被禁用或没有提供token，不添加认证头
    if not enable_authen or not token:
        return sse_client(url=url)
        
    headers = {"Authorization": f"Bearer {token}"}
    return sse_client(url=url, headers=headers)

async def main():
    # 根据是否使用 SSL 构建不同的 URL 前缀
    protocol = "https" if USE_SSL else "http"
    url = f"{protocol}://{SERVER_HOST}:{SERVER_PORT}/sse"
    
    print(f"连接到服务器: {url}")
    print(f"认证状态: {'启用' if enable_authen else '禁用'}")
    
    try:
        token = None
        
        # 如果启用了认证，获取token
        if enable_authen:
            token_data = await get_auth_token(USERNAME, PASSWORD)
            if not token_data:
                print("无法获取认证token，退出程序")
                return
                
            token = token_data["token"]
        
        # 使用带认证的 sse_client 连接到服务器
        async with authenticated_sse_client(url=url, token=token) as (read, write):
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
                
                # 4. 如果启用了认证，使用MCP工具获取token (演示)
                if enable_authen:
                    result_token = await session.call_tool("get_auth_token", arguments={"username": USERNAME, "password": PASSWORD})
                    print(f"调用工具 get_auth_token 的结果: {result_token}")
                    print(json.dumps(result_token.content[0].text, indent=2))
    except Exception as e:
        print(f"连接错误: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
