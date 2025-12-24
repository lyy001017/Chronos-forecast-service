'''
加载模型并进行时间序列预测服务
'''
# app/services/forecast_service.py

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import logging
from functools import lru_cache

from chronos import BaseChronosPipeline
from app.core.config import settings
from app.services.load_model import get_pipeline



logger = logging.getLogger(__name__)


   

def make_predictions(
    history_df: pd.DataFrame,
    prediction_length: int,
    quantile_levels: List[float],
    future_cov_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    # model_path: str = "./chronos-2-model",
    # s3_uri: str = "s3://autogluon/chronos-2/",
    mode : str = '',
    device : str = 'cpu',
    use_finetuned : bool =True,
    model_id : str = ''
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    使用 Chronos/Chronos-2 模型进行时间序列预测。

    参数:
        history_df:
            历史数据 DataFrame，至少包含:
                - 'timestamp': 时间戳（字符串或 datetime，形如 '2022-10-18'）
                - 'id'       : 序列 ID（每个类别的唯一标识）
                - 'target'   : 要预测的目标值（如售价、库存等）
        prediction_length:
            预测步长（要向未来预测多少个时间点）
        quantile_levels:
            预测分位数列表，例如 [0.1, 0.5, 0.9]
        future_cov_df:
            未来协变量 DataFrame（没有可以传 None）
        test_df:
            测试数据（如果有，用来对比预测结果）
        
    返回:
        (result_dict, predictions_df)
        - result_dict    : 可直接 JSON 序列化的预测结果
        - predictions_df : pandas DataFrame 形式的预测结果
    """
    logger.info(
        "开始进行预测, history_shape=%s, prediction_length=%d, quantiles=%s",
        history_df.shape,
        prediction_length,
        quantile_levels,
    )

    # if future_cov_df is None:
    #     future_cov_df = pd.DataFrame()
    # if test_df is None:
    #     test_df = pd.DataFrame()

    # logger.info(
    #     "future_cov_shape=%s, test_df_shape=%s",
    #     future_cov_df.shape,
    #     test_df.shape,
    # )

    # 1. 基本校验
    required_cols = {"timestamp", "id", "target"}
    missing_cols = required_cols - set(history_df.columns)
    if missing_cols:
        msg = f"history_df 缺少必要列: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    if prediction_length <= 0:
        msg = f"prediction_length 必须为正整数, 当前值: {prediction_length}"
        logger.error(msg)
        raise ValueError(msg)

    if not quantile_levels:
        msg = "quantile_levels 不能为空"
        logger.error(msg)
        raise ValueError(msg)

    # 2. 数据预处理
    history_df = history_df.copy()
    # 转换时间戳列为 datetime
    try:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        if future_cov_df is not None and "timestamp" in future_cov_df.columns:
            future_cov_df["timestamp"]= pd.to_datetime(future_cov_df["timestamp"])
    except Exception as exc:
        logger.exception("timestamp 列无法转换为 datetime: %s", exc)
        raise

    # 3. 加载模型
    pipeline = get_pipeline(mode=mode,use_finetuned=use_finetuned,device=device,model_id=model_id)

    # 4. 调用模型进行预测
    try:
        predictions = pipeline.predict_df(
            history_df,
            future_df=future_cov_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    except Exception as exc:
        logger.exception("调用 Chronos pipeline 预测失败: %s", exc)
        raise

    # 5. 统一转为 DataFrame
    if hasattr(predictions, "to_pandas"):
        predictions_df = predictions.to_pandas()
    else:
        predictions_df = pd.DataFrame(predictions)
    
    datetime_cols = predictions_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        predictions_df[col] = predictions_df[col].astype(str)
    

    logger.info("预测完成, 结果形状: %s", predictions_df.shape)

    # 6. 构造可 JSON 序列化的结果
    result: Dict[str, Any] = {
        "predictions": predictions_df.to_dict(orient="records"),
        "prediction_shape": list(predictions_df.shape),
        "prediction_length": prediction_length,
        "quantiles": quantile_levels,
        "model_used": "chronos-2",
        "generated_at": pd.Timestamp.now().isoformat(),
        "status" : "success",
    }

    return result, predictions_df