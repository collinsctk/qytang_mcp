from mcp.server.fastmcp import FastMCP
import psycopg2
import logging

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
@mcp.resource("info://system")
def get_system_info() -> str:
    """返回系统的基本信息，包括Python版本"""
    python_version = platform.python_version()
    return f"FastMCP 服务器版本 1.0，运行状态正常。Python版本：{python_version}"


# 2. 资源：通过 PostgreSQL 查询用户密码
@mcp.resource("psql://{username}")
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

if __name__ == "__main__":
    mcp.run(transport="sse")
    # 注意：此处 mcp.run() 为阻塞调用，表示服务器正在运行
