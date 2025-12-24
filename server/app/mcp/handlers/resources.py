# app/mcp/handlers/resources.py

from typing import Any, Dict
from app.core.config import settings


def register_resources(mcp) -> None:
    """
    注册 MCP 资源（提供文档、说明、示例输入等），供 LLM 读取。
    """

    # -------------------------------------------------------
    # 1. 服务总览说明
    # -------------------------------------------------------
    @mcp.resource("chronos://overview")
    async def chronos_overview() -> Dict[str, Any]:
        """
        Chronos 时间序列预测服务的说明文档（供 LLM 阅读）。
        """
        return {
            "app_name": settings.APP_NAME,
            "description": (
                "这是一个基于 Amazon Chronos-2 的时间序列预测服务的 MCP 接入层。\n"
                "LLM 可以通过读取此资源了解工具用途、支持的特性、参数格式等。\n"
                "主要工具为 chronos_forecast，用于多步多分位预测。"
            ),
            "capabilities": [
                "多 id 时间序列预测",
                "多步预测（prediction_length 可配置）",
                "多分位预测（quantiles 可配置）",
                "支持历史/未来协变量",
                "可选择是否使用微调模型（use_finetuned）",
            ],
            "recommended_tools": ["chronos_forecast"],
            "use_cases": [
                "按 SKU + 仓库预测未来 7 天销量",
                "在给定价格/促销计划下估计未来需求",
                "库存与补货优化",
            ],
        }

    # -------------------------------------------------------
    # 2. 示例输入模板
    # -------------------------------------------------------
    @mcp.resource("chronos://sample_request")
    async def chronos_sample_request() -> Dict[str, Any]:
        """
        提供 chronos_forecast 的标准输入格式模板。
        LLM 会读取此资源并自动补全/推断用户缺失字段。
        """
        return {
            "description": "标准调用 chronos_forecast 的示例输入。",
            "history_data": [
                {
                    "timestamp": "2022-09-24",
                    "id": "item_1",
                    "target": 10.0,
                    "price": 1.20,
                    "promo_flag": 0,
                    "weekday": 6,
                },
                {
                    "timestamp": "2022-09-25",
                    "id": "item_1",
                    "target": 11.0,
                    "price": 1.22,
                    "promo_flag": 0,
                    "weekday": 0,
                }
            ],
            "future_cov": [
                {
                    "timestamp": "2022-10-01",
                    "id": "item_1",
                    "price": 1.36,
                    "promo_flag": 0,
                    "weekday": 6,
                }
            ],
            "prediction_length": 7,
            "quantiles": [0.1, 0.5, 0.9],
            "with_cov": True,
            "use_finetuned": False
        }