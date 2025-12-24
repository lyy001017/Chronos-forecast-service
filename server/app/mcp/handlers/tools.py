import json
import asyncio
import logging
from typing import Dict, List, Optional ,Any
import pandas as pd
from mcp.types import TextContent

from app.services.predict_services import make_predictions

logger = logging.getLogger(__name__)

def register_tools(mcp) -> None :
    '''
    注册MCP工具到服务器
    '''

    @mcp.tool()
    async def chronos_forecast(
        history_data: List[Dict[str, Any]],
        prediction_length: int,
        quantiles: List[float],
        future_cov: Optional[List[Dict[str, Any]]] = None,
        with_cov: bool = True,
        use_finetuned: bool = False,
    ) -> str:
        """
        FastMCP 工具实现：包装 make_predictions，直接在本地调用 Chronos 模型。

        注意：FastMCP 的工具函数返回值要是“可序列化”的，
        我们这里直接返回 JSON 字符串，外层由 FastMCP 转成 TextContent。
        
        """

        logger.info(
            "FastMCP chronos_forecast 调用: history_len=%d, prediction_length=%d, "
            "with_cov=%s, use_finetuned=%s, has_future_cov=%s",
            len(history_data),
            prediction_length,
            with_cov,
            use_finetuned,
            future_cov is not None,
        )

        # ---------- 构造 history_df ----------
        if not history_data:
            raise ValueError("history_data 不能为空")

        base_keys = {"timestamp", "id", "target"}

        history_records = [
            {key: rec[key] for key in base_keys}
            for rec in history_data
        ]

        if with_cov:
            history_with_cov_records: List[Dict[str, Any]] = []
            for rec in history_data:
                base_part = {key: rec[key] for key in base_keys}
                covariate_part = {k: v for k, v in rec.items() if k not in base_keys}
                history_with_cov_records.append({**base_part, **covariate_part})
            history_df = pd.DataFrame(history_with_cov_records)
            logger.info("FastMCP: 使用历史协变量, history_shape=%s", history_df.shape)
        else:
            history_df = pd.DataFrame(history_records)
            logger.info("FastMCP: 不使用历史协变量, history_shape=%s", history_df.shape)

        if history_df.empty:
            raise ValueError("构造出的 history_df 为空，请检查输入数据。")
        

        # ---------- 构造 future_cov_df ----------
        future_cov_df: Optional[pd.DataFrame] = None

        if with_cov and future_cov:
            base_future_keys = {"timestamp", "id"}

            future_cov_records: List[Dict[str, Any]] = []
            for rec in future_cov:
                base_part = {k: rec[k] for k in base_future_keys}
                covariate_part = {k: v for k, v in rec.items() if k not in base_future_keys}
                future_cov_records.append({**base_part, **covariate_part})

            tmp_df = pd.DataFrame(future_cov_records)
            if tmp_df.empty:
                logger.info("FastMCP: future_cov 为空，本次预测不使用未来协变量")
            else:
                counts = tmp_df.groupby("id").size()
                if (counts != prediction_length).any():
                    logger.warning(
                        "FastMCP: future_cov_df 按 id 分组的行数与 prediction_length 不一致，"
                        "counts=%s, prediction_length=%d，本次忽略未来协变量",
                        counts.to_dict(),
                        prediction_length,
                    )
                else:
                    future_cov_df = tmp_df
                    logger.info(
                        "FastMCP: 使用未来协变量, future_cov_df_shape=%s, 每个 id 的步长=%s",
                        future_cov_df.shape,
                        counts.to_dict(),
                    )
        else:
            if not with_cov:
                logger.info("FastMCP: with_cov=False，本次预测不使用任何协变量（历史/未来）")
            else:
                logger.info("FastMCP: with_cov=True 但未提供 future_cov，本次仅使用历史协变量")


        loop = asyncio.get_running_loop()

        def _run_forecast() -> tuple[Dict[str, Any], Any]:
            return make_predictions(
                history_df=history_df,
                prediction_length=prediction_length,
                quantile_levels=quantiles,
                future_cov_df=future_cov_df,
                test_df=None,
                use_finetuned=use_finetuned,
            )

        result_dict, _ = await loop.run_in_executor(None, _run_forecast)
        
        # 直接返回 JSON 字符串，由 FastMCP 转成 TextContent
        return json.dumps(result_dict, ensure_ascii=False, indent=2)
