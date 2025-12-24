# Chronos Forecast Service

基于 Amazon Chronos-2 模型构建的高性能时间序列预测微服务。本项目不仅提供标准的 RESTful API，还集成了 **MCP (Model Context Protocol)**，可作为 LLM (如 Claude Desktop, Cursor) 的直接工具使用。

## ✨ 核心特性

- **基于 Chronos-2**: 利用预训练的大型时间序列模型，支持 Zero-shot (零样本) 预测、以及微调版本模型预测。
- **多模态协变量支持**: 支持传入**历史协变量** (如过去的价格) 和**未来协变量** (如未来的促销计划)。
- **多分位预测**: 支持配置预测分位数 (如 P10, P50, P90)，提供概率预测能力。提供0.01和0.99两个极端分位数预测，增加风险预测功能。
- **模型微调 (Fine-tuning)**: 提供 API 接口针对特定数据集进行num-batch级微调。
- **MCP 集成**: 内置 MCP Server，支持 LLM 直接调用工具读取文档、进行预测分析。
- **高性能架构**: 基于 FastAPI + Uvicorn，支持异步并发与线程池推理。

## 🛠️ 技术栈

- **Python**: 3.12+
- **Web 框架**: FastAPI
- **ML 框架**: PyTorch, Chronos-Forecasting
- **数据处理**: Pandas, Polars
- **MCP**: FastMCP

## 🚀 快速开始

### 1. 环境安装

推荐使用 Conda 或 pip：

```bash
# 安装依赖
pip install -r server/requirements.txt
```

### 2. 启动服务
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

### 3. API文档
启动后访问：
- **Swagger UI** http://localhost:5001/docs
- **Redoc** http://localhost:5001/redoc
- **MCP Endpoint** sse或stdio



