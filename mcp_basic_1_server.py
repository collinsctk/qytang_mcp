from mcp.server.fastmcp import FastMCP
import psycopg2
import logging
import platform
import os
import ssl
import uvicorn
import asyncio
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServer")

# 初始化 MCP 服务器，名称为 "FastMCP Server Example"
mcp = FastMCP("FastMCP Server Example")

# 1. 工具：加法计算
@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个数字之和"""
    return a + b

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
    
    # 创建 Starlette 应用
    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
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

