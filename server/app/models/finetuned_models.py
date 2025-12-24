from typing import List, Dict, Optional
from pydantic import BaseModel,Field
from fastapi import Query

class FineTuneSeries(BaseModel):
    '''
    单个时间序列的微调数据'''
    id : str = Field(...,description="时间序列ID")
    target : List[float] = Field(...,description='待预测目标值')
    past_covariates : Optional[Dict[str,List[float]]] = Field(
        default=None,
        description="历史协变量，如客流，销量等"
    )
    future_covariates : Optional[Dict[str,List[float]]] = Field(
        default=None,
        description="未来已知协变量，如促销、节假日等"
    )


class FineTuneRequset(BaseModel):
    series:List[FineTuneSeries] = Field(
        ...,
        description="同一种任务的多个时间序列"
    )
    

class FineTuneReponse(BaseModel):
    model_id : str = Field(...,description='微调好的模型ID')
    