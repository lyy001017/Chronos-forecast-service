'''
MCP服务器实现
创建MCP服务器实例并集成到FastAPI
'''
from mcp.server.fastmcp import FastMCP
from app.core.config import settings

mcp = FastMCP(
    name=settings.APP_NAME,
    instructions=f'{settings.APP_NAME}MCP Server',
)
mcp.streamable_http_app

def register_handlers():
    """
    注册 MCP 处理器（工具、资源、提示）
    延迟导入避免循环依赖
    """
    from app.mcp.handlers import tools, resources, prompts
    
    # 注册工具处理器
    tools.register_tools(mcp)
    
    # 注册资源处理器
    resources.register_resources(mcp)
    
    # 注册提示处理器
    prompts.register_prompts(mcp)


# 注册处理器
register_handlers()
