# server/Dockerfile
FROM python:3.12-slim

# 1. 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/server  
    # ^ 关键：把 server 目录加入 python 路径，否则 python 找不到 app 模块

WORKDIR /app

# 2. 安装依赖 (显式指定路径，适配根目录构建上下文)
COPY server/requirements.txt /app/server/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/server/requirements.txt  \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=1000

# 3. 复制源代码 (保持 server/app 的结构)
COPY server /app/server

# 4. 创建模型挂载点 (防止挂载出错)
RUN mkdir -p /app/server/app/models/chronos_models

EXPOSE 5001

# 5. 启动命令

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5001", "--app-dir", "/app/server"]