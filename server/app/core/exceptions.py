"""
应用统一异常与错误码定义

参考行业最佳实践，为 Chronos 时间序列预测服务提供：
- 统一的错误码体系
- 统一的业务异常基类
- 结构化错误响应的数据结构
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from fastapi import status


class ErrorCode(str, Enum):
    """
    错误码枚举定义（按类别分组）

    约定：
    - 前缀 DATA_：输入数据/格式相关问题
    - 前缀 MODEL_：模型加载/预测相关问题
    - 前缀 REQ_：请求级别问题（参数、权限等）
    - 前缀 INTERNAL_：服务器内部错误
    """

    # ---------- 通用 ----------
    SUCCESS = "SUCCESS"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # ---------- 请求 / 参数 ----------
    VALIDATION_ERROR = "VALIDATION_ERROR"        # Pydantic 校验失败或业务级参数校验失败
    BAD_REQUEST = "BAD_REQUEST"                  # 请求不合法的通用错误（语义上 400）

    UNAUTHORIZED = "UNAUTHORIZED"                # 未认证
    FORBIDDEN = "FORBIDDEN"                      # 无权限
    NOT_FOUND = "NOT_FOUND"                      # 路由 / 资源不存在

    # ---------- 数据层（和你的预测路由强相关） ----------
    DATA_EMPTY = "DATA_EMPTY"                    # history_data 为空
    DATA_FORMAT_ERROR = "DATA_FORMAT_ERROR"      # 数据格式错误（类型/字段映射有问题）
    DATA_MISSING_COLUMNS = "DATA_MISSING_COLUMNS"  # 缺少必须列，如 timestamp / id / target
    FUTURE_COV_MISMATCH = "FUTURE_COV_MISMATCH"  # future_cov 与 prediction_length 对不齐

    # ---------- 模型层 ----------
    MODEL_NOT_READY = "MODEL_NOT_READY"          # 模型尚未加载或不可用
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"      # 模型加载失败
    MODEL_PREDICT_FAILED = "MODEL_PREDICT_FAILED"  # 模型预测失败


class BaseAppException(Exception):
    """
    应用统一业务异常基类。

    所有需要走“统一错误响应格式”的业务异常，都应继承自该类或直接使用该类。
    """
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            error_code: 错误码枚举
            message: 对外提示信息（要给客户端看的）
            status_code: HTTP 状态码，默认 400
            details: 可选的补充信息（内部诊断用，不一定全部暴露给前端）
        """
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为标准错误响应结构，便于全局异常处理器直接使用。
        """
        result = {
            "success": False,
            "error_code": self.error_code.value,
            "message": self.message,
        }
        
        # 仅在调试模式下包含详细信息
        if self.details:
            result["details"] = self.details
        
        return result



# 可选：你也可以根据业务再细分子类，方便上层捕获时区分类型
class DataException(BaseAppException):
    """
    输入数据/格式相关异常（和预测路由的数据预处理强相关）
    """
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(error_code, message, status_code, details)


class ModelException(BaseAppException):
    """
    模型加载/预测相关异常
    """
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(error_code, message, status_code, details)