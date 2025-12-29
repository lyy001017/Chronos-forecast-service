# Chronos Forecast Service

基于 Amazon Chronos-2 模型构建的高性能时间序列预测微服务。本项目不仅提供标准的 RESTful API，还集成了 **MCP (Model Context Protocol)**，可作为 LLM (如 Claude Desktop, Cursor) 的直接工具使用。

## ✨ 核心特性

- **基于 Chronos-2**: 利用预训练的大型时间序列模型，支持 Zero-shot (零样本) 预测、以及微调版本模型预测。
- **多模态协变量支持**: 支持传入**历史协变量** (如过去的价格) 和**未来协变量** (如未来的促销计划)。
- **多分位预测**: 支持配置预测分位数 (如 P10, P50, P90)，提供概率预测能力。提供0.01和0.99两个极端分位数预测，增加风险预测功能。
- **MCP 集成**: 内置 MCP Server，支持 LLM 直接调用工具读取文档、进行预测分析。
- **高性能架构**: 基于 FastAPI + Uvicorn，支持异步并发与线程池推理。

## 🛠️ 技术栈

- **Python**: 3.12+
- **Web 框架**: FastAPI
- **ML 框架**: PyTorch, Chronos-Forecasting
- **数据处理**: Pandas, Polars
- **MCP**: FastMCP

## 🔧系统框架

### 核心层（core）
- **`config.py`**：全局配置文件，测试开发变量存储。
- **`exception_handle.py`**：FastAPI全局异常处理。

### 服务层（services）
- **`load_model.py`**:实现lru_cache缓存策略，支持Base模型与Finetuned模型热切换。
- **`predict_service.py`**:处理输入数据格式、协变量对齐、加载模型预测返回结果。

### 接口层

#### API接口（api）
- **`predict.py`**：（post：/）：封装Chronos模型预测服务接口。
- **`health.py`**：（get：/）：健康检查接口，用于K8s的存活探针。

#### MCP服务（mcp）
- **`prompts.py`**：内置chronos_forecast_guide[guide]，规范LLM的预测行为。
- **`resources.py`**：暴露chronos：//sample_request模版，指导LLM构造复杂JSON。
- **`tools.py`**：注册chronos_forecast工具，LLM可以调用完成时间序列预测。

### 模型层
- **`predic_models.py`**:预测接口请求体和响应体格式。
- chronos_models:带权重的Chronos模型存储。

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
- 支持MCP服务和API服务一键启动（app.mount）
### 3. API文档
启动后访问：
- **Swagger UI** http://localhost:5001/docs
- **Redoc** http://localhost:5001/redoc

### 4.MCP服务
启动后访问
- **sse支持**：http://localhost:5001/mcp/sse



