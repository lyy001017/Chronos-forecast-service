'''
健康检查API路由
'''
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["健康检查"])


@router.get('/health')
def health():
    '''
    健康检查接口
    
    服务健康检查接口，用于服务状态监控
    '''
    return{
        "status":"ok",
        "version":settings.APP_VERSION,
    }