import asyncio
import json
import sys
import os
import ssl
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
import argparse
from typing import Dict, Any, Optional, List, Union
import logging
import time
import openai
from openai import OpenAI
import re
from pathlib import Path
from dotenv import load_dotenv

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()

# 获取当前文件所在目录的上一级目录
parent_dir = current_file_path.parent.parent

# 根据上一级目录拼接出 .env 文件的路径
env_file_path = parent_dir / ".env"

# 加载指定路径的 .env 文件，从中获取环境变量配置
load_dotenv(dotenv_path=env_file_path)

# ~~~~~~~~~~~~~~~~~~~~使用OpenAI ~~~~~~~~~~~~~~~~~~~~~
# 从环境变量中获取 OpenAI API Key，这里使用的环境变量名称为 "OPENAI2025"
openai_apikey = os.environ.get("OPENAI2025")


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPLLMClient")

# 直接写死OpenAI API密钥
OPENAI_API_KEY = openai_apikey
if not OPENAI_API_KEY:
    logger.warning("未设置OpenAI API密钥，将无法使用OpenAI功能")

# 默认配置 - 默认不启用认证但保留信息
DEFAULT_CONFIG = {
    "host": "196.21.5.218",
    "port": 443,
    "use_ssl": True,
    "enable_authen": False,  # 默认不启用认证
    "username": "qinke",     # 保留认证信息
    "password": "cisco",     # 保留认证信息
    "ca_cert_path": "./certs/ca.cer",
    "verify_ssl": True,
    "openai_model": "gpt-4o-mini",  # 默认使用的OpenAI模型
    "base_url": None,
    # "openai_model": "qwq:latest",  # 默认使用的OpenAI模型
    # "base_url": "http://127.0.0.1:11434/v1/"
}

# 是否启用OpenAI功能
ENABLE_OPENAI = True  # 设置为True以启用OpenAI功能，使用AI处理用户查询

def parse_value(value: str):
    """尝试将字符串解析为 int、float 或 bool 类型，无法解析则返回原字符串。"""
    val = value.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    # 尝试整数/浮点数转换
    for conv in (int, float):
        try:
            return conv(val)
        except:
            continue
    # 保持原始字符串
    return val

class MCPLLMClient:
    """MCP LLM 客户端，用于连接MCP服务器并处理用户请求"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化MCP LLM客户端
        
        Args:
            config: 配置字典，包含连接参数
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self._client = None  # 保存SSE客户端引用
        self.tools: Dict[str, Any] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self.token: Optional[str] = None
        self.token_expiry: Optional[float] = None
        self.last_refresh_time: float = 0  # 上次刷新能力的时间
        self.refresh_interval: float = 300  # 刷新间隔，默认5分钟
        print(self.config["base_url"])
        # 初始化OpenAI客户端
        if ENABLE_OPENAI and OPENAI_API_KEY:
            try:
                if self.config["base_url"]:
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=self.config["base_url"])
                    logger.info(f"OpenAI客户端初始化成功，使用base_url: {self.config['base_url']}")
                else:
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logger.info("OpenAI客户端初始化成功，使用默认base_url")
            except Exception as e:
                logger.error(f"OpenAI客户端初始化失败: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            logger.info("OpenAI功能已禁用，将使用本地规则处理自然语言查询")
        
        # 配置SSL
        if self.config["use_ssl"] and self.config["verify_ssl"]:
            self.setup_ssl()
    
    def setup_ssl(self):
        """配置SSL验证"""
        ca_cert_path = self.config.get("ca_cert_path")
        if ca_cert_path and os.path.isfile(ca_cert_path):
            # 保存原始的 AsyncClient 类
            original_async_client = httpx.AsyncClient
            
            # 创建一个新的 AsyncClient 类，使用根证书进行验证
            class CustomCAAsyncClient(original_async_client):
                def __init__(self, *args, **kwargs):
                    if 'verify' not in kwargs:
                        kwargs['verify'] = ca_cert_path
                    super().__init__(*args, **kwargs)
            
            # 替换 httpx 的 AsyncClient 类
            httpx.AsyncClient = CustomCAAsyncClient
            logger.info(f"使用根证书进行 SSL 验证: {ca_cert_path}")
        else:
            logger.warning("未找到根证书，将使用系统默认的证书验证")
    
    async def get_auth_token(self) -> bool:
        """获取认证token
        
        Returns:
            bool: 是否成功获取token
        """
        # 如果认证被禁用，返回一个虚拟token
        if not self.config["enable_authen"]:
            logger.info("认证已禁用，使用虚拟token")
            self.token = "disabled_authentication_mode"
            self.token_expiry = time.time() + 3600  # 1小时后过期
            return True
        
        protocol = "https" if self.config["use_ssl"] else "http"
        token_url = f"{protocol}://{self.config['host']}:{self.config['port']}/get_token"
        
        logger.info(f"正在获取认证token...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    json={"username": self.config["username"], "password": self.config["password"]},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.token = token_data["token"]
                    self.token_expiry = token_data.get("expiry", time.time() + 3600)
                    logger.info(f"成功获取token: {self.token[:10]}...")
                    return True
                else:
                    logger.error(f"获取token失败: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"获取token时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def connect(self) -> bool:
        """连接到MCP服务器
        
        Returns:
            bool: 是否成功连接
        """
        # 如果启用了认证，先获取token
        if self.config["enable_authen"]:
            if not await self.get_auth_token():
                logger.error("无法获取认证token，连接失败")
                return False
        
        # 构建连接URL
        protocol = "https" if self.config["use_ssl"] else "http"
        url = f"{protocol}://{self.config['host']}:{self.config['port']}/sse"
        
        logger.info(f"连接到MCP服务器: {url}")
        logger.info(f"认证状态: {'启用' if self.config['enable_authen'] else '禁用'}")
        
        try:
            # 准备认证头
            headers = {}
            if self.config["enable_authen"] and self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            # 创建SSE客户端
            self._client = sse_client(url=url, headers=headers)
            read, write = await self._client.__aenter__()
            
            # 创建会话
            session = ClientSession(read, write)
            self.session = await session.__aenter__()
            
            # 初始化会话
            await self.session.initialize()
            logger.info("成功连接到MCP服务器！")
            
            # 获取可用的工具、资源和提示符
            await self.refresh_capabilities()
            
            return True
        except Exception as e:
            logger.error(f"连接MCP服务器失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def refresh_capabilities_if_needed(self):
        """如果需要，刷新服务器能力
        
        如果距离上次刷新的时间超过了刷新间隔，则刷新服务器能力
        """
        current_time = time.time()
        if current_time - self.last_refresh_time > self.refresh_interval:
            logger.info(f"距离上次刷新已经过去 {current_time - self.last_refresh_time:.1f} 秒，开始刷新服务器能力")
            await self.refresh_capabilities()
            self.last_refresh_time = current_time
        else:
            logger.debug(f"距离上次刷新仅过去 {current_time - self.last_refresh_time:.1f} 秒，无需刷新")
    
    async def refresh_capabilities(self):
        """刷新服务器能力（工具、资源、提示符）"""
        if not self.session:
            logger.error("未连接到MCP服务器，无法刷新能力")
            return
        
        # 获取可用工具
        try:
            tools_resp = await self.session.list_tools()
            self.tools = {tool.name: tool for tool in tools_resp.tools}
            logger.info(f"可用工具: {list(self.tools.keys())}")
        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")
            self.tools = {}
        
        # 获取可用资源
        try:
            resources_resp = await self.session.list_resources()
            self.resources = {res.uri: res for res in resources_resp.resources}
            logger.info(f"可用资源: {list(self.resources.keys())}")
        except Exception as e:
            logger.error(f"获取资源列表失败: {e}")
            self.resources = {}
        
        # 获取可用提示符
        try:
            prompts_resp = await self.session.list_prompts()
            self.prompts = {pr.name: pr for pr in prompts_resp.prompts}
            logger.info(f"可用提示符: {list(self.prompts.keys())}")
        except Exception as e:
            logger.error(f"获取提示符列表失败: {e}")
            self.prompts = {}
            
        # 更新最后刷新时间
        self.last_refresh_time = time.time()
    
    async def close(self):
        """关闭连接"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
                
            if self._client:
                await self._client.__aexit__(None, None, None)
                self._client = None
                
            logger.info("已关闭MCP服务器连接")
        except Exception as e:
            logger.error(f"关闭连接时出错: {e}")
            import traceback
            traceback.print_exc()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        if not self.session:
            raise RuntimeError("未连接到MCP服务器")
        
        if tool_name not in self.tools:
            raise ValueError(f"未找到名为 '{tool_name}' 的工具")
        
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # 添加调试信息，查看结果的完整结构
            logger.info(f"工具 '{tool_name}' 原始结果: {result}")
            logger.info(f"结果类型: {type(result)}")
            
            # 通用处理逻辑，适用于所有工具
            if hasattr(result, 'content') and result.content:
                # 检查是否有多个内容项
                if len(result.content) > 1:
                    # 尝试将多个内容项合并为一个列表
                    from types import SimpleNamespace
                    all_items = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            all_items.append(item.text)
                    
                    if all_items:
                        logger.info(f"合并后的列表: {all_items}")
                        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(all_items))])
                
                # 检查第一个内容项是否可能是JSON
                if hasattr(result.content[0], 'text'):
                    content_text = result.content[0].text
                    try:
                        # 尝试解析为JSON
                        parsed_content = json.loads(content_text)
                        # 如果是列表或字典，确保以JSON格式返回
                        if isinstance(parsed_content, (list, dict)):
                            from types import SimpleNamespace
                            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(parsed_content))])
                    except (json.JSONDecodeError, TypeError):
                        # 不是有效的JSON，保持原样
                        pass
            
            # 尝试从原始响应中提取数据
            if hasattr(result, '_raw_response') and hasattr(result._raw_response, 'data'):
                try:
                    raw_data = json.loads(result._raw_response.data)
                    if isinstance(raw_data, (list, dict)):
                        from types import SimpleNamespace
                        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(raw_data))])
                except Exception as e:
                    logger.error(f"解析原始响应数据失败: {e}")
            
            return result
        except Exception as e:
            logger.error(f"调用工具 '{tool_name}' 失败: {e}")
            # 返回一个模拟的结果对象，而不是抛出异常
            from types import SimpleNamespace
            error_content = SimpleNamespace(text=f"调用失败: {str(e)}")
            return SimpleNamespace(content=[error_content])
    
    async def read_resource(self, uri: str) -> Any:
        """读取资源
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容
        """
        if not self.session:
            raise RuntimeError("未连接到MCP服务器")
        
        # 检查资源是否存在，使用字符串比较而不是直接比较对象
        resource_exists = False
        for resource_uri in self.resources.keys():
            if str(resource_uri) == str(uri):
                resource_exists = True
                break
                
        if not resource_exists:
            raise ValueError(f"未找到资源 URI: {uri}")
        
        try:
            result = await self.session.read_resource(uri)
            return result
        except Exception as e:
            logger.error(f"读取资源 '{uri}' 失败: {e}")
            # 返回一个模拟的结果对象，而不是抛出异常
            from types import SimpleNamespace
            error_content = SimpleNamespace(text=f"资源不可用: {str(e)}")
            return SimpleNamespace(contents=[error_content])
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """获取提示符
        
        Args:
            prompt_name: 提示符名称
            arguments: 提示符参数
            
        Returns:
            提示符内容
        """
        if not self.session:
            raise RuntimeError("未连接到MCP服务器")
        
        if prompt_name not in self.prompts:
            raise ValueError(f"未找到名为 '{prompt_name}' 的提示符")
        
        try:
            # 检查必需参数
            prompt_obj = self.prompts.get(prompt_name)
            required_args = []
            if hasattr(prompt_obj, 'arguments'):
                for arg in prompt_obj.arguments:
                    if isinstance(arg, dict) and arg.get('required', False):
                        required_args.append(arg.get('name'))
            
            # 检查是否缺少必需参数
            missing_args = [arg for arg in required_args if arg not in arguments]
            if missing_args:
                logger.warning(f"提示符 '{prompt_name}' 缺少必需参数: {missing_args}")
                from types import SimpleNamespace
                message = SimpleNamespace(content=SimpleNamespace(text=f"缺少必需参数: {missing_args}"))
                return SimpleNamespace(messages=[message])
            
            result = await self.session.get_prompt(prompt_name, arguments=arguments)
            return result
        except Exception as e:
            logger.error(f"获取提示符 '{prompt_name}' 失败: {e}")
            # 返回一个模拟的结果对象，而不是抛出异常
            from types import SimpleNamespace
            message = SimpleNamespace(content=SimpleNamespace(text=f"提示符不可用: {str(e)}"))
            return SimpleNamespace(messages=[message])

    async def process_user_query_with_openai(self, query: str) -> str:
        """使用OpenAI处理用户查询，支持多轮函数调用
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果
        """
        if not self.openai_client:
            return "OpenAI客户端未初始化，无法处理查询。请确保设置了有效的OpenAI API密钥。"
        
        # 在处理查询前刷新服务器能力（如果需要），确保获取最新的工具和资源
        try:
            await self.refresh_capabilities_if_needed()
            logger.info("已检查服务器能力是否需要刷新")
        except Exception as e:
            logger.warning(f"检查服务器能力刷新失败: {e}，将使用现有的工具和资源")
        
        try:
            # 将MCP工具转换为OpenAI工具格式
            openai_tools = []
            
            # 添加MCP工具
            for tool_name, tool in self.tools.items():
                tool_schema = getattr(tool, "inputSchema", {})
                # 直接使用工具的原始描述
                tool_desc = getattr(tool, "description", "")
                
                # 如果没有描述，提供一个基本描述
                if not tool_desc:
                    tool_desc = f"MCP工具: {tool_name}"
                
                # 创建OpenAI工具定义，使用原始工具名称
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{tool_name}",  # 使用双下划线作为前缀，避免命名冲突
                        "description": tool_desc,
                        "parameters": tool_schema if tool_schema else {"type": "object", "properties": {}}
                    }
                }
                openai_tools.append(openai_tool)
            
            # 添加资源访问工具
            if self.resources:
                resource_tool = {
                    "type": "function",
                    "function": {
                        "name": "mcp_resource",
                        "description": "访问MCP资源",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "uri": {
                                    "type": "string",
                                    "description": f"资源URI，可用的资源有: {', '.join([str(uri) for uri in self.resources.keys()])}"
                                }
                            },
                            "required": ["uri"]
                        }
                    }
                }
                openai_tools.append(resource_tool)
            
            # 添加提示符工具
            if self.prompts:
                prompt_tool = {
                    "type": "function",
                    "function": {
                        "name": "mcp_prompt",
                        "description": "使用MCP提示符",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt_name": {
                                    "type": "string",
                                    "description": f"提示符名称，可用的提示符有: {', '.join(list(self.prompts.keys()))}"
                                },
                                "arguments": {
                                    "type": "object",
                                    "description": "提示符参数"
                                }
                            },
                            "required": ["prompt_name"]
                        }
                    }
                }
                openai_tools.append(prompt_tool)
            
            # 系统提示
            system_prompt = """
            你是一个智能助手，可以帮助用户使用MCP服务器上的工具和资源。
            分析用户的查询，选择最合适的工具或资源来满足用户需求。
            
            重要规则：
            1. 只推荐实际存在的工具、资源或提示符
            2. 如果找不到合适的工具或资源，直接告知用户该服务不存在
            3. 不要猜测或推荐不存在的服务
            4. 确保参数完整，不要省略必需参数
            5. 对于返回列表或复杂数据结构的工具，确保完整展示所有结果
            6. 理解工具的功能和用途，选择最适合用户需求的工具
            7. 如果一个问题需要多次调用不同的工具来获取完整信息，请分步骤进行
            8. 当你需要进一步信息时，可以继续调用相关工具
            
            可用的工具包括：
            {tool_list}
            
            可用的资源包括：
            {resource_list}
            
            可用的提示符包括：
            {prompt_list}
            
            请确保正确理解用户的意图，并选择最合适的工具来满足需求。
            """
            
            # 动态生成工具、资源和提示符列表
            tool_list = "\n".join([f"- {name}: {getattr(tool, 'description', '无描述')}" for name, tool in self.tools.items()])
            resource_list = "\n".join([f"- {uri}: {getattr(resource, 'description', '无描述')}" for uri, resource in self.resources.items()])
            prompt_list = "\n".join([f"- {name}: {getattr(prompt, 'description', '无描述')}" for name, prompt in self.prompts.items()])
            
            # 填充系统提示中的占位符
            system_prompt = system_prompt.format(
                tool_list=tool_list if tool_list else "无可用工具",
                resource_list=resource_list if resource_list else "无可用资源",
                prompt_list=prompt_list if prompt_list else "无可用提示符"
            )
            
            # 初始化消息历史
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # 设置最大循环次数，防止无限循环
            max_loop_count = 5
            loop_count = 0
            
            # 最终结果
            final_result = ""
            
            # 循环处理，支持多轮函数调用
            while loop_count < max_loop_count:
                loop_count += 1
                logger.info(f"开始第 {loop_count} 轮对话")
                
                # 调用OpenAI API，使用function calling
                response = self.openai_client.chat.completions.create(
                    model=self.config["openai_model"],
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=1000
                )
                
                # 解析OpenAI回复
                message = response.choices[0].message
                
                # 将AI回复添加到消息历史中
                messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                
                # 检查是否有工具调用
                if message.tool_calls:
                    # 处理所有工具调用
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                        
                        # 执行相应的工具调用
                        if function_name.startswith("mcp__"):
                            # 调用MCP工具
                            mcp_tool_name = function_name[5:]  # 去掉"mcp__"前缀
                            try:
                                # 再次检查工具是否存在
                                if mcp_tool_name not in self.tools:
                                    # 尝试再次刷新工具列表
                                    await self.refresh_capabilities()
                                    if mcp_tool_name not in self.tools:
                                        tool_result = f"很抱歉，'{mcp_tool_name}' 服务不存在。请尝试其他可用的服务。"
                                        messages.append({
                                            "tool_call_id": tool_call_id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps({"error": tool_result})
                                        })
                                        continue
                                
                                # 调用工具
                                result = await self.call_tool(mcp_tool_name, function_args)
                                
                                # 处理结果
                                if hasattr(result, 'content') and result.content:
                                    content_text = result.content[0].text
                                    try:
                                        # 尝试解析为JSON
                                        parsed_content = json.loads(content_text)
                                        # 将结果添加到消息历史
                                        messages.append({
                                            "tool_call_id": tool_call_id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps(parsed_content)
                                        })
                                    except (json.JSONDecodeError, TypeError):
                                        # 如果不是有效的JSON，则直接返回文本
                                        messages.append({
                                            "tool_call_id": tool_call_id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": content_text
                                        })
                                else:
                                    # 没有返回内容
                                    messages.append({
                                        "tool_call_id": tool_call_id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({"result": "无内容"})
                                    })
                            except Exception as e:
                                logger.error(f"执行工具 '{mcp_tool_name}' 时出错: {e}")
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps({"error": f"执行出错: {str(e)}"})
                                })
                        
                        elif function_name == "mcp_resource":
                            # 访问MCP资源
                            uri = function_args.get("uri")
                            if not uri:
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps({"error": "未提供资源URI"})
                                })
                                continue
                            
                            try:
                                # 检查资源是否存在
                                resource_exists = False
                                for resource_uri in self.resources.keys():
                                    if str(resource_uri) == str(uri):
                                        resource_exists = True
                                        break
                                
                                if not resource_exists:
                                    # 尝试再次刷新资源列表
                                    await self.refresh_capabilities()
                                    
                                    # 再次检查资源是否存在
                                    resource_exists = False
                                    for resource_uri in self.resources.keys():
                                        if str(resource_uri) == str(uri):
                                            resource_exists = True
                                            break
                                    
                                    if not resource_exists:
                                        messages.append({
                                            "tool_call_id": tool_call_id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps({"error": f"资源 '{uri}' 不存在"})
                                        })
                                        continue
                                
                                # 读取资源
                                result = await self.read_resource(uri)
                                content = result.contents[0].text if hasattr(result, 'contents') and result.contents else "无内容"
                                
                                # 将结果添加到消息历史
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": content
                                })
                            except Exception as e:
                                logger.error(f"访问资源 '{uri}' 时出错: {e}")
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps({"error": f"访问资源出错: {str(e)}"})
                                })
                        
                        elif function_name == "mcp_prompt":
                            # 使用MCP提示符
                            prompt_name = function_args.get("prompt_name")
                            arguments = function_args.get("arguments", {})
                            
                            if not prompt_name:
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps({"error": "未提供提示符名称"})
                                })
                                continue
                            
                            try:
                                # 检查提示符是否存在
                                if prompt_name not in self.prompts:
                                    # 尝试再次刷新提示符列表
                                    await self.refresh_capabilities()
                                    if prompt_name not in self.prompts:
                                        messages.append({
                                            "tool_call_id": tool_call_id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps({"error": f"提示符 '{prompt_name}' 不存在"})
                                        })
                                        continue
                                
                                # 检查必需参数
                                prompt_obj = self.prompts.get(prompt_name)
                                required_args = []
                                if hasattr(prompt_obj, 'arguments'):
                                    for arg in prompt_obj.arguments:
                                        if isinstance(arg, dict) and arg.get('required', False):
                                            required_args.append(arg.get('name'))
                                
                                # 检查是否缺少必需参数
                                missing_args = [arg for arg in required_args if arg not in arguments]
                                if missing_args:
                                    messages.append({
                                        "tool_call_id": tool_call_id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({"error": f"缺少必需参数: {missing_args}"})
                                    })
                                    continue
                                
                                # 获取提示符
                                result = await self.get_prompt(prompt_name, arguments)
                                if hasattr(result, 'messages') and result.messages:
                                    messages.append({
                                        "tool_call_id": tool_call_id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": result.messages[0].content.text
                                    })
                                else:
                                    messages.append({
                                        "tool_call_id": tool_call_id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({"result": "无内容"})
                                    })
                            except Exception as e:
                                logger.error(f"使用提示符 '{prompt_name}' 时出错: {e}")
                                messages.append({
                                    "tool_call_id": tool_call_id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps({"error": f"使用提示符出错: {str(e)}"})
                                })
                else:
                    # 如果没有工具调用，表示对话已完成
                    final_result = message.content if message.content else "很抱歉，我无法理解您的请求或找不到合适的服务来处理它。"
                    break
            
            # 如果达到最大循环次数但仍未得到最终结果，则请求AI总结
            if not final_result and loop_count >= max_loop_count:
                # 添加一个请求总结的消息
                messages.append({
                    "role": "user", 
                    "content": "请根据已获取的信息，总结并回答我的问题。"
                })
                
                # 再次调用OpenAI API，请求总结
                response = self.openai_client.chat.completions.create(
                    model=self.config["openai_model"],
                    messages=messages,
                    temperature=0,
                    max_tokens=1000
                )
                
                final_result = response.choices[0].message.content
            
            return final_result
                
        except Exception as e:
            logger.error(f"OpenAI处理出错: {e}")
            import traceback
            traceback.print_exc()
            return f"很抱歉，我无法处理您的请求。请稍后再试或尝试其他查询。错误: {str(e)}"
    
    async def process_user_query(self, query: str) -> str:
        """处理用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果
        """
        if not ENABLE_OPENAI:
            return "OpenAI功能已禁用，无法处理自然语言查询。请启用OpenAI功能或使用直接命令格式。"
        
        if not self.openai_client:
            return "OpenAI客户端未初始化，无法处理查询。请确保设置了有效的OpenAI API密钥。"
        
        return await self.process_user_query_with_openai(query)

async def handle_user_input(client: MCPLLMClient, user_input: str) -> Optional[str]:
    """解析并处理用户输入"""
    user_input = user_input.strip()
    if not user_input:
        return "输入为空，请重新输入。"
    
    # 判断退出命令
    if user_input.lower() in ("quit", "exit"):
        return None  # 使用None表示需要退出
    
    # 检测是否是直接命令
    if user_input.lower().startswith(("tool ", "resource ", "prompt ")):
        return await handle_direct_command(client, user_input)
    
    # 否则，将输入视为自然语言查询
    return await client.process_user_query(user_input)

async def handle_direct_command(client: MCPLLMClient, command: str) -> str:
    """处理直接命令（tool/resource/prompt）"""
    if command.lower().startswith("tool "):
        # 提取工具名称和参数串
        try:
            _, rest = command.split(" ", 1)
        except ValueError:
            return "格式错误：未提供工具名称。"
        parts = rest.strip().split(" ", 1)
        tool_name = parts[0]
        args_str = parts[1].strip() if len(parts) > 1 else ""
        
        # 检查工具是否存在
        if tool_name not in client.tools:
            return f"未找到名为 '{tool_name}' 的工具。"
        
        # 解析参数
        tool_args = {}
        if args_str:
            if args_str[0] in "{[":
                # JSON格式参数
                try:
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    return "参数JSON解析错误，请检查格式。"
            else:
                # 使用 空格 分隔的 key=value 列表
                for token in args_str.split():
                    if "=" in token:
                        key, val = token.split("=", 1)
                        tool_args[key] = parse_value(val)
                    else:
                        # 未提供键名的值，仅当工具只有一个参数时使用
                        single_params = []
                        # 获取工具参数定义（如有）
                        tool_schema = getattr(client.tools[tool_name], "inputSchema", None)
                        if tool_schema and "properties" in tool_schema:
                            single_params = list(tool_schema["properties"].keys())
                        if len(single_params) == 1:
                            # 工具只有一个参数
                            tool_args[single_params[0]] = parse_value(token)
                        else:
                            return "参数格式错误，请使用 key=value 形式提供。"
        
        # 执行工具
        try:
            # 移除特殊处理，使用通用逻辑
            result = await client.call_tool(tool_name, tool_args)
            # 将结果转换为字符串输出
            if hasattr(result, 'content') and result.content:
                # 检查返回内容是否为列表类型
                content_text = result.content[0].text
                try:
                    # 尝试解析为JSON，如果成功则可能是列表或字典
                    parsed_content = json.loads(content_text)
                    if isinstance(parsed_content, (list, dict)):
                        return f"工具 `{tool_name}` 执行结果: {json.dumps(parsed_content, ensure_ascii=False, indent=2)}"
                    else:
                        return f"工具 `{tool_name}` 执行结果: {content_text}"
                except (json.JSONDecodeError, TypeError):
                    # 如果不是有效的JSON，则直接返回文本
                    return f"工具 `{tool_name}` 执行结果: {content_text}"
            return f"工具 `{tool_name}` 执行成功，但没有返回内容。"
        except Exception as e:
            return f"调用工具出错: {e}"
    
    elif command.lower().startswith("resource "):
        # 提取资源URI
        try:
            _, uri = command.split(" ", 1)
        except ValueError:
            return "格式错误：未提供资源 URI。"
        uri = uri.strip()
        if uri == "":
            return "格式错误：资源 URI 为空。"
        
        # 读取资源内容
        try:
            result = await client.read_resource(uri)
            content = result.contents[0].text if hasattr(result, 'contents') and result.contents else "无内容"
            return f"资源 `{uri}` 内容:\n{content}"
        except Exception as e:
            return f"读取资源失败: {e}"
    
    elif command.lower().startswith("prompt "):
        # 提取提示符名称和可选参数
        try:
            _, rest = command.split(" ", 1)
        except ValueError:
            return "格式错误：未提供提示符名称。"
        parts = rest.strip().split(" ", 1)
        prompt_name = parts[0]
        args_str = parts[1].strip() if len(parts) > 1 else ""
        
        # 检查提示符是否存在
        if prompt_name not in client.prompts:
            return f"未找到名为 '{prompt_name}' 的提示符。"
        
        # 解析提示符参数（格式同工具参数）
        prompt_args = {}
        if args_str:
            if args_str[0] in "{[":
                try:
                    prompt_args = json.loads(args_str)
                except json.JSONDecodeError:
                    return "提示符参数JSON解析错误，请检查格式。"
            else:
                for token in args_str.split():
                    if "=" in token:
                        key, val = token.split("=", 1)
                        prompt_args[key] = parse_value(val)
                    else:
                        # 若只有一个参数值且提示符只需一个参数
                        params = getattr(client.prompts[prompt_name], "arguments", None)
                        if params and len(params) == 1:
                            # 提示符参数列表只有一个
                            param_name = params[0].get("name") if isinstance(params[0], dict) else None
                            if param_name:
                                prompt_args[param_name] = parse_value(token)
                                continue
                        return "提示符参数格式错误，请使用 key=value 提供。"
        
        # 获取提示符内容
        try:
            result = await client.get_prompt(prompt_name, prompt_args)
            if hasattr(result, 'messages') and result.messages:
                return f"提示符 `{prompt_name}` 内容:\n{result.messages[0].content.text}"
            return f"提示符 `{prompt_name}` 已获取，但没有返回内容。"
        except Exception as e:
            return f"获取提示符失败: {e}"
    
    else:
        return "无法识别指令类型，请以 'tool', 'resource' 或 'prompt' 开头，或直接输入自然语言查询。"

async def run_client(config: Dict[str, Any]):
    """运行MCP LLM客户端"""
    # 使用异步上下文管理器创建客户端
    async with MCPLLMClient(config) as client:
        print("\n" + "="*50)
        print("MCP LLM 客户端已启动")
        print("您可以：")
        print("1. 直接输入自然语言查询，例如：'计算1加2'")
        print("2. 输入 'quit' 或 'exit' 退出程序")
        print("="*50 + "\n")
        
        # 跳过示例查询处理，直接进入交互模式
        print("请输入您的查询:")
        
        # 进入交互循环
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                # 捕捉Ctrl+C或EOF退出
                print("\n退出程序。")
                break
            
            # 处理输入
            try:
                result = await handle_user_input(client, user_input)
                if result is None:
                    # None 用于指示退出
                    print("退出程序。")
                    break
                print(result)
            except Exception as e:
                print(f"处理请求时出错: {e}")
                import traceback
                traceback.print_exc()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MCP LLM 客户端")
    parser.add_argument("--host", default=DEFAULT_CONFIG["host"], help="MCP服务器主机名或IP")
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"], help="MCP服务器端口")
    parser.add_argument("--use-ssl", action="store_true", default=DEFAULT_CONFIG["use_ssl"], help="是否使用SSL连接")
    parser.add_argument("--no-ssl", action="store_false", dest="use_ssl", help="不使用SSL连接")
    parser.add_argument("--enable-authen", action="store_true", default=DEFAULT_CONFIG["enable_authen"], help="启用认证")
    parser.add_argument("--no-authen", action="store_false", dest="enable_authen", help="禁用认证")
    parser.add_argument("--username", default=DEFAULT_CONFIG["username"], help="认证用户名")
    parser.add_argument("--password", default=DEFAULT_CONFIG["password"], help="认证密码")
    parser.add_argument("--ca-cert", default=DEFAULT_CONFIG["ca_cert_path"], help="CA证书路径")
    parser.add_argument("--no-verify", action="store_false", dest="verify_ssl", default=DEFAULT_CONFIG["verify_ssl"], help="不验证SSL证书")
    parser.add_argument("--openai-model", default=DEFAULT_CONFIG["openai_model"], help="OpenAI模型名称")
    if DEFAULT_CONFIG["base_url"]:
        parser.add_argument("--base-url", default=DEFAULT_CONFIG["base_url"], help="模型的base_url")
    else:
        parser.add_argument("--base-url", default=None, help="模型的base_url")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 构建配置
    config = {
        "host": args.host,
        "port": args.port,
        "use_ssl": args.use_ssl,
        "enable_authen": args.enable_authen,
        "username": args.username,
        "password": args.password,
        "ca_cert_path": args.ca_cert,
        "verify_ssl": args.verify_ssl,
        "openai_model": args.openai_model,
        "base_url": args.base_url
    }
    
    # 运行客户端
    asyncio.run(run_client(config))
