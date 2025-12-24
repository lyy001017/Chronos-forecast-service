#fastapi应用入口（挂 REST，MCP）
from pathlib import Path
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from app.core.exception_handlers import register_exception_handlers


from app.core.config import settings
from app.api.main import api_router
from app.api.routes import health



_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    '''应用生命周期管理'''

    #启动时执行
    logger.info("="*40)
    logging.info(f'Starting {settings.APP_NAME} v{settings.APP_VERSION}')
    logger.info("="*40)

    if settings.ENABLE_MCP:
        from app.mcp.server import mcp

        mcp_app = mcp.streamable_http_app()

        session_manager = mcp.session_manager

        logger.info("MCP server initialized")

        async with session_manager.run():
            logger.info("="*40)
            logging.info("Application startup complete")
            logger.info("="*40)

            #打印访问URL（应用启动成功后）
            print("\n" + "=" * 40)
            print("API docs: http://localhost:5000/docs")
            print("ReDoc: http://localhost:5000/redoc")
            print("OpenAPI: http://localhost:5000/openapi.json")
            print("Health: http://localhost:5000/health")
            print(f"MCP: http://localhost:5000{settings.MCP_PATH}")
            print("=" * 40)
            print("Press CTRL+C to stop the server")
            print("=" * 40 + "\n")

            yield #应用运行期间

            #关闭时执行
            logger.info("Application shutdown")
    else:
        logger.info("=" * 40)
        logger.info("✓ Application startup complete")
        logger.info("=" * 40)
        
        # 打印访问 URL（应用启动成功后）
        print("\n" + "=" * 40)
        print("API docs: http://localhost:5001/docs")
        print("ReDoc: http://localhost:5000/redoc")
        print("OpenAPI: http://localhost:5000/openapi.json")
        print("Health: http://localhost:5000/health")
        print("=" * 40)
        print("Press CTRL+C to stop the server")
        print("=" * 40 + "\n")
        
        yield  # 应用运行期间
        
        # 关闭时执行
        logger.info("Application shutdown")



#创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Chronos时间序列预测微服务",
    docs_url=settings.DOCS_URL,
    openapi_url=settings.OPENAPI_URL,
    # generate_unique_id_function=
    lifespan=lifespan,
    swagger_ui_parameters={
        "presistAuthorizaion":True
    }
)


#注册全局异常处理器
register_exception_handlers(app)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#注册健康检查路由
app.include_router(health.router)

#注册API路由
app.include_router(api_router)

if settings.ENABLE_MCP:
    from app.mcp.server import mcp
    # 获取 FastMCP 的 streamable_http_app（在 lifespan 中已初始化 session_manager）
    mcp_app = mcp.streamable_http_app()
    # 挂载到根路径，因为 FastMCP 应用内部已定义完整路径（如 /mcp）
    app.mount("/", mcp_app)

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=5001,
            reload=settings.DEBUG,
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n" + "=" * 40)
        print("Server stopped by user")
        print("=" * 40)
    except Exception as e:
        print("\n" + "=" * 40)
        print(f"ERROR: Failed to start server: {e}")
        print("=" * 40)
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


