from fastapi import APIRouter
import uuid
import os
from app.models.finetuned_models import *
from app.services.finetune_services import run_finetune
router = APIRouter(tags=["微调接口"])

@router.post('/',response_model=FineTuneReponse)
async def finetune(request:FineTuneRequset,
    prediction_length : int = Query(
        ...,
        gt=0,
        description='预测步长'
    ),
    num_steps : int = Query(
        default=50,gt=0,description='微调步数（默认50，适合小数据量）'
    ),
    learning_rates : float = Query(
        default=1e-5,
        gt=0,
        description='微调学习率'
    ),
    batch_size : int = Query(
        default=32,
        gt=0,
        description='微调批次大小'
    )
):
    
    result = await run_finetune(request,prediction_length,num_steps,learning_rates,batch_size)
    
    #->dict
    return result 