'''
API路由聚合
'''

from fastapi import APIRouter

from app.api.routes import predict
from app.api.routes import finetuned

api_router = APIRouter()

#注册各个路由模块

api_router.include_router(predict.router,prefix="/predict")
api_router.include_router(finetuned.router,prefix='/finetuned')