"""
FastAPI 全局异常处理器

目标：
- 为 Chronos 时间序列预测服务提供统一的异常处理和错误响应格式
- 参考行业最佳实践，支持：
  - 业务异常（BaseAppException）
  - 请求参数验证异常（Pydantic）
  - HTTP 标准异常（Starlette HTTPException）
  - 未预期的内部异常（Exception）
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.exceptions import (
    BaseAppException,
    ErrorCode,
)


logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    """
    处理应用自定义业务异常（继承自 BaseAppException）

    典型使用场景：
    - 数据为空 / 缺列 / 格式错误
    - 模型不可用 / 预测失败（可预期的业务错误）
    """
    error_dict = exc.to_dict()

    # 非 DEBUG 环境可以按需裁剪 details（例如不暴露内部敏感信息）
    if not settings.DEBUG and error_dict.get("details"):
        error_dict["details"] = {"summary": error_dict["details"]}

    logger.warning(
        "业务异常 [%s]: %s, details=%s",
        exc.error_code.value,
        exc.message,
        exc.details,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_dict,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    处理请求体 / 查询参数的验证异常（由 FastAPI/Pydantic 抛出）

    典型场景：
    - 请求 JSON 不符合 Pydantic 模型定义
    - 字段缺失 / 类型不匹配
    """
    errors = exc.errors()
    error_messages = []

    for error in errors:
        loc = ".".join(str(l) for l in error.get("loc", []))
        msg = error.get("msg", "验证失败")
        error_messages.append(f"{loc}: {msg}")

    summary = "; ".join(error_messages)

    error_dict: Dict[str, Any] = {
        "success": False,
        "error_code": ErrorCode.VALIDATION_ERROR.value,
        "message": "请求参数验证失败",
        "details": {
            "errors": errors,
            "summary": summary,
        },
    }

    # 生产环境隐藏详细错误结构，只给一个 summary
    if not settings.DEBUG:
        error_dict["details"] = {"summary": summary}

    logger.warning("参数验证失败: %s", summary)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_dict,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """
    处理 FastAPI/Starlette 内部抛出的 HTTPException

    典型场景：
    - 手动 raise HTTPException(status_code=404, detail="xxx")
    - 路由不存在等
    """
    # 默认视为内部错误，再根据状态码微调
    error_code = ErrorCode.INTERNAL_ERROR

    if exc.status_code == status.HTTP_404_NOT_FOUND:
        error_code = ErrorCode.NOT_FOUND
    elif exc.status_code == status.HTTP_401_UNAUTHORIZED:
        error_code = ErrorCode.UNAUTHORIZED
    elif exc.status_code == status.HTTP_403_FORBIDDEN:
        error_code = ErrorCode.FORBIDDEN
    elif exc.status_code == status.HTTP_400_BAD_REQUEST:
        error_code = ErrorCode.BAD_REQUEST

    error_dict: Dict[str, Any] = {
        "success": False,
        "error_code": error_code.value,
        "message": exc.detail or "HTTP 错误",
    }

    logger.warning("HTTP 异常 [%s]: %s", exc.status_code, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_dict,
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    处理所有未被显式捕获的异常（兜底）

    典型场景：
    - 代码 bug
    - 第三方库抛出的未处理异常
    - 意料之外的运行时错误
    """
    error_dict: Dict[str, Any] = {
        "success": False,
        "error_code": ErrorCode.INTERNAL_ERROR.value,
        "message": "服务器内部错误",
    }

    # 调试模式下返回更多细节，便于排查
    if settings.DEBUG:
        import traceback

        error_dict["details"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    # 日志里永远打印完整堆栈
    logger.error(
        "未预期异常: %s: %s",
        type(exc).__name__,
        str(exc),
        exc_info=exc,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_dict,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    将所有异常处理器注册到 FastAPI 应用实例上

    注意：
    - 更具体的异常要优先注册（BaseAppException、RequestValidationError）
    - 最兜底的 Exception 必须最后注册
    """
    # 1. 应用自定义业务异常
    app.add_exception_handler(BaseAppException, app_exception_handler)

    # 2. 请求参数验证异常（Pydantic）
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # 3. HTTP 异常（FastAPI/Starlette 标准异常）
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # 4. 通用异常（兜底）
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("✓ 全局异常处理器已注册")