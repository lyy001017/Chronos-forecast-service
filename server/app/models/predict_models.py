

from datetime import datetime
from typing import List,Optional,Dict,Any
from fastapi import Query
from pydantic import BaseModel, Field

# ---------------------------
# 请求体pydantic
# ---------------------------

class HistoryItem(BaseModel):
    """
    单条历史数据记录：
    - timestamp: 时间戳，字符串或 ISO 格式，FastAPI 会自动解析为 datetime
    - id       : 序列 ID
    - target   : 目标值
    """
    timestamp: datetime = Field(..., description="时间戳，例如 '2022-10-18'")
    id: str = Field(..., description="时间序列 ID，每个类别的唯一标识")
    target: float = Field(..., description="要预测的目标值，如售价、库存等")
    
    class Config:
        extra = 'allow'


class FutureItem(BaseModel):
    timestamp: datetime = Field(..., description="未来时间步的时间戳")
    id: str = Field(..., description="时间序列 ID")
    # 未来协变量同样允许任意字段
    class Config:
        extra = "allow"


class PredictRequest(BaseModel):
    """
    预测请求体：
    - history_data      : 历史数据列表
    - prediction_length : 预测步长
    - quantiles         : 预测分位数列表
    - model_path        : 模型路径（可选，默认使用服务内部配置）
    """
    history_data: List[HistoryItem] = Field(..., description="历史时间序列数据列表")
    future_cov: Optional[List[FutureItem]] = Field(
        default=None,
        description="未来协变量数据列表，长度一般等于prediction_length "
    )

    prediction_length: int = Field(
        ...,
        gt=0,
        description="预测步长（未来预测的时间点数量）",
    )
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="预测分位数列表，例如 [0.1, 0.5, 0.9]",
    )
    # use_finetuned: bool = Field(
    #     default=False,
    #     description='是否使用微调后的模型，true=使用微调模型，false=使用基础Chronos-2模型',
    # )
    
    # device : str = Field(
    #     default='cuda',
    #     description= '传入推理环境，默认CPU推理'
    # )
    # mode :str = Field(
    #     default='',
    #     description="可选使用模型针对特定任务：如售价，库存等"
    # )


# ---------------------------
# 响应体pydantic
# ---------------------------

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(
        ..., description="预测结果列表，每一条是一行预测记录"
    )
    prediction_shape: List[int] = Field(
        ..., description="预测结果 DataFrame 的形状 [rows, cols]"
    )
    prediction_length: int = Field(..., description="预测步长")
    quantiles: List[float] = Field(..., description="使用的分位数")
    model_used: str = Field(..., description="模型名称或标识")
    generated_at: str = Field(..., description="预测生成时间 ISO 字符串")