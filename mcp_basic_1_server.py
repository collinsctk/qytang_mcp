from mcp.server.fastmcp import FastMCP
import psycopg2
import logging
import platform
import os
import ssl
import uvicorn
import asyncio
import secrets
import time
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServer")

# 全局变量：是否启用认证
enable_authen = False

# 初始化 MCP 服务器，名称为 "FastMCP Server Example"
mcp = FastMCP("FastMCP Server Example")

# 存储有效的token
valid_tokens = {}
TOKEN_EXPIRY = 3600  # Token有效期为1小时

# Token验证中间件
class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # 如果认证被禁用，直接放行所有请求
        if not enable_authen:
            return await call_next(request)
            
        # 跳过对token获取端点的验证
        if request.url.path == "/get_token":
            return await call_next(request)
            
        # 检查Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "未提供有效的认证token"}, status_code=401)
        
        token = auth_header.replace("Bearer ", "")
        
        # 验证token
        if token not in valid_tokens or valid_tokens[token]["expiry"] < time.time():
            return JSONResponse({"error": "token无效或已过期"}, status_code=401)
            
        # 更新token最后使用时间
        valid_tokens[token]["last_used"] = time.time()
        
        return await call_next(request)

# 生成新token
def generate_token(username):
    token = secrets.token_hex(16)
    expiry = time.time() + TOKEN_EXPIRY
    valid_tokens[token] = {
        "username": username,
        "expiry": expiry,
        "last_used": time.time()
    }
    return token, expiry

# 1. 工具：加法计算
@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个数字之和"""
    return a + b

# 2. 工具：获取所有设备的名称
@mcp.tool()
def get_all_devices_name() -> list:
    """ 获取所有设备的名称

    Args: None

    return: 设备名称的列表
    """
    return ['C8Kv1', 'C8Kv2', 'Nexus1']

@mcp.tool()
def get_device_version(device_name: str) -> str:
    """
    获取特定设备的系统版本

    Args:
        device_name: 设备名称

    return: 返回特定设备的系统版本
    """
    device_versions = {
        'C8Kv1': 'version 11.1',
        'C8Kv2': 'version 12.2',
        'Nexus1': 'version 13.0'
    }
    return device_versions.get(device_name, '未知版本')

@mcp.tool()
def get_device_interface_info(device_name: str) -> dict:
    """
    获取特定设备接口信息!  包含IP地址, MAC地址和速率

    Args:
        device_name: 设备名称

    return: 返回特定设备的接口信息! 包含IP地址, MAC地址和速率
    """
    device_interfaces = {
        'C8Kv1': {"ip_address": "10.1.1.1", "mac_address": "00:11:22:33:44:55", "speed": "10G"},
        'C8Kv2': {"ip_address": "10.1.1.2", "mac_address": "00:11:22:66:BA:55", "speed": "40G"},
        'Nexus1': {"ip_address": "10.1.11.1", "mac_address": "00:11:22:78:1A:55", "speed": "100G"},
    }
    return device_interfaces.get(device_name, {device_name: '接口信息未找到'})

# 静态资源：返回系统的基本信息
@mcp.resource("info://system", description="系统信息资源，提供服务器版本和Python版本")
def get_system_info() -> str:
    """返回系统的基本信息，包括Python版本"""
    python_version = platform.python_version()
    return f"FastMCP 服务器版本 1.0，运行状态正常。Python版本：{python_version}"


# 2. 资源：通过 PostgreSQL 查询用户密码
@mcp.resource("psql://{username}", description="用户密码查询资源，通过PostgreSQL数据库获取指定用户的密码")
def get_user_password(username: str) -> str:
    """
    从 PostgreSQL 数据库中查询指定用户名的密码。
    """
    try:
        conn = psycopg2.connect(
            database="qytangdb",
            user="qytangdbuser",
            password="Cisc0123",
            host="196.21.5.218",
            port=5432
        )
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            return result[0]
        else:
            return f"未找到用户 {username}"
    except Exception as e:
        return f"数据库错误: {e}"

@mcp.tool()
def query_user_password(username: str) -> str:
    """通过 PostgreSQL 查询并返回用户密码"""
    return get_user_password(username)

# 3. 提示：格式化返回用户密码信息
@mcp.prompt("query-password")
def query_password_prompt(username: str) -> str:
    """格式化返回指定用户的密码信息"""
    password = get_user_password(username)
    return f"用户 {username} 的密码是：{password}"

# 4. 工具：获取认证token
@mcp.tool()
def get_auth_token(username: str, password: str) -> dict:
    """获取认证token，需要提供有效的用户名和密码"""
    # 如果认证被禁用，返回一个虚拟token
    if not enable_authen:
        return {
            "token": "disabled_authentication_mode",
            "expiry": time.time() + TOKEN_EXPIRY,
            "expires_in": TOKEN_EXPIRY,
            "token_type": "Bearer",
            "message": "认证已禁用，此token仅为占位符"
        }
        
    stored_password = get_user_password(username)
    
    if stored_password == password:
        token, expiry = generate_token(username)
        return {
            "token": token,
            "expiry": expiry,
            "expires_in": TOKEN_EXPIRY,
            "token_type": "Bearer"
        }
    else:
        return {"error": "用户名或密码不正确"}

# 创建 ASGI 应用
async def create_app():
    # 创建 SSE 传输
    sse = SseServerTransport("/messages/")
    
    # 处理 SSE 连接的函数
    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )
    
    # 创建token获取端点
    async def get_token(request):
        # 如果认证被禁用，返回一个虚拟token
        if not enable_authen:
            return JSONResponse({
                "token": "disabled_authentication_mode",
                "expiry": time.time() + TOKEN_EXPIRY,
                "expires_in": TOKEN_EXPIRY,
                "token_type": "Bearer",
                "message": "认证已禁用，此token仅为占位符"
            })
            
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
            
            if not username or not password:
                return JSONResponse({"error": "需要提供用户名和密码"}, status_code=400)
                
            stored_password = get_user_password(username)
            if stored_password == password:
                token, expiry = generate_token(username)
                return JSONResponse({
                    "token": token,
                    "expiry": expiry,
                    "expires_in": TOKEN_EXPIRY,
                    "token_type": "Bearer"
                })
            else:
                return JSONResponse({"error": "用户名或密码不正确"}, status_code=401)
        except Exception as e:
            return JSONResponse({"error": f"请求处理错误: {str(e)}"}, status_code=500)
    
    # 创建中间件列表
    middleware = []
    if enable_authen:
        middleware.append(Middleware(TokenAuthMiddleware))
    
    # 创建 Starlette 应用，添加token认证中间件
    app = Starlette(
        debug=True,
        middleware=middleware,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/get_token", endpoint=get_token, methods=["POST"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    
    return app

if __name__ == "__main__":
    # SSL 配置
    ssl_keyfile = os.environ.get("SSL_KEYFILE", "./certs/server.key")  # SSL 私钥文件路径
    ssl_certfile = os.environ.get("SSL_CERTFILE", "./certs/server.crt")  # SSL 证书文件路径
    
    # 初始化 MCP 服务器
    mcp._setup_handlers()

    logger.info(f"启用 SSL 加密，使用证书: {ssl_certfile}")
    
    # 根据认证状态输出日志
    if enable_authen:
        logger.info("已启用 Token 认证")
    else:
        logger.info("已禁用 Token 认证")
    
    # 创建 SSL 上下文
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
    
    # 使用 uvicorn 直接启动，并配置 SSL
    uvicorn.run(
        asyncio.run(create_app()),
        host="0.0.0.0",
        port=443,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_level="info")

