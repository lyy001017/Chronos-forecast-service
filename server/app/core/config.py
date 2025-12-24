# app/core/config.py

# app/core/config.py

import os
from dataclasses import dataclass



class Settings:
    """
    全局配置：
    - 统一从环境变量读取
    - 提供合理的默认值，方便本地开发“一键跑起来”
    - 供 FastAPI、MCP、SDK、服务端代码统一使用
    """

    # ========= 基本应用信息 =========
    # 当前运行环境：dev / staging / prod 等
    ENV: str = os.getenv("ENVIRONMENT", "dev")

    # 是否开启调试模式（影响日志、异常返回等）
    DEBUG: bool = False

    # 日志级别：debug / info / warning / error
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").upper()

    # 应用名称 & 版本（给 FastAPI、MCP、日志用）
    APP_NAME: str = os.getenv("APP_NAME", "Chronos_forecast")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")

    # ========= API 基础配置 =========
    # 统一的 API 前缀，方便将来扩展 /api/v1 等版本管理
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")

    # 文档路径，可用于自定义（例如关闭外网文档）
    DOCS_URL: str = os.getenv("DOCS_URL", "/docs")
    OPENAPI_URL: str = os.getenv("OPENAPI_URL", "/openapi.json")

    # CORS 配置
    BACKEND_CORS_ORIGINS: list[str] = ["*"]


    # ========= Chronos 模型路径配置 =========
    # 注意：这里把“未微调”和“微调后”分开，并使用不同的环境变量名

    # 本地未微调的 GPU 模型路径
    unfinetuned_gpu_model_path: str = os.getenv(
        "CHRONOS_GPU_MODEL_UNFINETUNED_PATH",
        "./server/app/models/chronos_models/unfinetuned_cpu",
    )
    # 本地微调后的 GPU 模型路径
    finetuned_gpu_model_path: str = os.getenv(
        "CHRONOS_GPU_MODEL_FINETUNED_PATH",
        "./server/app/models/chronos_models/finetuned_cpu",
    )

    # S3 上官方/兜底模型路径
    chronos_s3_uri: str = os.getenv(
        "CHRONOS_S3_URI",
        "s3://autogluon/chronos-2/",
    )

    # ========= 预测默认参数配置 =========
    # 默认分位数（如果请求里没传，可以用这个）
    default_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    # 默认预测步长（如果请求没说清楚，可以用这个）
    default_prediction_length: int = int(
        os.getenv("DEFAULT_PREDICTION_LENGTH", "28")
    )

    # 最大允许预测步长（业务安全限制）
    max_prediction_length: int = int(
        os.getenv("MAX_PREDICTION_LENGTH", "365")
    )

    # ========= MCP / Agent 相关配置 =========
    # 是否启用 MCP 服务能力（将来可以用来开关 MCP）
    ENABLE_MCP: bool = os.getenv("ENABLE_MCP", "true").lower() == "true"

     # MCP 端点路径
    MCP_PATH: str = "/mcp"
    
    # MCP 协议版本
    MCP_VERSION: str = "2025-11-15"

    # ========= 帮助属性 =========
    @property
    def is_prod(self) -> bool:
        """是否为生产环境"""
        return self.ENV.lower() == "prod"

    @property
    def is_debug(self) -> bool:
        """是否为调试模式"""
        return self.DEBUG


# 单例配置对象，项目其他地方直接：
# from app.core.config import settings
settings = Settings()