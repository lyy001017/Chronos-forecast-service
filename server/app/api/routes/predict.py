# 预测接口路由

from typing import Any, Dict, List
import asyncio
import functools
import logging

import pandas as pd
from fastapi import APIRouter, status,Query,Request

from app.models.predict_models import PredictRequest, PredictResponse
from app.services.predict_services import make_predictions
from app.core.exceptions import DataException, ModelException, ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["预测接口"])


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest,
                  use_finetuned :bool = Query(
                      default=True,
                      description='是否使用微调模型，如果使用可在mode选择针对特定任务'),
                  mode : str = Query(
                      default='',
                      description='如果使用微调模型，这里可以选测SKU、sales等针对特定任务'),
                  device : str = Query(
                      default='cpu',
                      description='选择模型推理设备，CPU或GPU'),
                  with_cov : bool = Query(
                      default=True,
                      description='传入数据是否含有协变量'),
                  model_id : str = Query(
                    default=None,
                    description='自己微调的模型获取的id')
    ) -> Dict[str, Any]:
                  
    """
    使用 Chronos 模型进行时间序列预测。

    支持json格式输入，输入格式请求示例(JSON)：
    {
      "history_data": [
        {"timestamp": "2022-10-01", "id": "item_1", "target": 10.0},
        {"timestamp": "2022-10-02", "id": "item_1", "target": 12.0}
      ],
      "prediction_length": 28,
      "quantiles": [0.1, 0.5, 0.9],
    }
    如果有协变量，也可以传入历史、未来协变量
    """

    # ---------------------------
    # 1. history_data → DataFrame
    # ---------------------------
    raw_records: List[Dict[str, Any]] = [
        item if isinstance(item, dict) else item.model_dump()
        for item in request.history_data
    ]

    if not raw_records:
        raise DataException(
            error_code=ErrorCode.DATA_EMPTY,
            message="history_data 不能为空",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    base_keys = {"timestamp", "id", "target"}

    # 检查必须字段，防止后面 KeyError
    missing_keys = base_keys - set(raw_records[0].keys())
    if missing_keys:
        raise DataException(
            error_code=ErrorCode.DATA_MISSING_COLUMNS,
            message=f"history_data 缺少必要字段: {missing_keys}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"missing_columns": list(missing_keys)},
        )

    history_records = [
        {key: rec[key] for key in base_keys}
        for rec in raw_records
    ]

    # 是否合入历史协变量
    if with_cov:
        history_with_cov_records: List[Dict[str, Any]] = []
        for rec in raw_records:
            base_part = {key: rec[key] for key in base_keys}
            covariate_part = {k: v for k, v in rec.items() if k not in base_keys}
            history_with_cov_records.append({**base_part, **covariate_part})
        history_df = pd.DataFrame(history_with_cov_records)
        logger.info("使用历史协变量, history_shape=%s", history_df.shape)
    else:
        history_df = pd.DataFrame(history_records)
        logger.info("不使用历史协变量, history_shape=%s", history_df.shape)

    if history_df.empty:
        raise DataException(
            error_code=ErrorCode.DATA_EMPTY,
            message="构造出的 history_df 为空，请检查输入数据。",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # ---------------------------
    # 2. future_cov → DataFrame（可选）
    # ---------------------------
    future_cov_df = None

    if with_cov and request.future_cov:
        raw_future_records: List[Dict[str, Any]] = [
            item if isinstance(item, dict) else item.model_dump()
            for item in request.future_cov
        ]

        base_future_keys = {"timestamp", "id"}
        missing_future_keys = base_future_keys - set(raw_future_records[0].keys())
        if missing_future_keys:
            raise DataException(
                error_code=ErrorCode.DATA_MISSING_COLUMNS,
                message=f"future_cov 缺少必要字段: {missing_future_keys}",
                status_code=status.HTTP_400_BAD_REQUEST,
                details={"missing_columns": list(missing_future_keys)},
            )

        future_cov_records: List[Dict[str, Any]] = []
        for rec in raw_future_records:
            base_part = {k: rec[k] for k in base_future_keys}
            covariate_part = {
                k: v
                for k, v in rec.items()
                if k not in base_future_keys
            }
            future_cov_records.append({**base_part, **covariate_part})

        tmp_df = pd.DataFrame(future_cov_records)

        if tmp_df.empty:
            logger.info("future_cov 为空，本次忽略未来协变量")
        else:
            counts = tmp_df.groupby("id").size()
            if (counts != request.prediction_length).any():
                # 这里直接视为参数错误，抛给调用方修数据
                raise DataException(
                    error_code=ErrorCode.FUTURE_COV_MISMATCH,
                    message=(
                        "future_cov 每个 id 的行数必须等于 prediction_length，"
                        f"当前统计: {counts.to_dict()}, "
                        f"prediction_length={request.prediction_length}"
                    ),
                    status_code=status.HTTP_400_BAD_REQUEST,
                    details={
                        "group_counts": counts.to_dict(),
                        "expected_length": request.prediction_length,
                    },
                )
            future_cov_df = tmp_df
            logger.info(
                "使用未来协变量, future_cov_shape=%s, 每个 id 步长=%s",
                future_cov_df.shape,
                counts.to_dict(),
            )
    else:
        if not with_cov:
            logger.info("with_cov=False，本次预测不使用任何协变量（历史/未来）")
        else:
            logger.info("with_cov=True 但未提供 future_cov，本次仅使用历史协变量")

    # ---------------------------
    # 3. 在线程池中调用 Chronos 进行预测
    # ---------------------------
    loop = asyncio.get_running_loop()
    func = functools.partial(
        make_predictions,
        history_df=history_df,
        prediction_length=request.prediction_length,
        quantile_levels=request.quantiles,
        future_cov_df=future_cov_df,
        test_df=None,
        use_finetuned=use_finetuned,
        mode=mode,
        device = device,
        model_id = model_id
    )

    try:
        result_dict, _ = await loop.run_in_executor(None, func)
    except Exception as exc:
        logger.exception("调用 Chronos 模型预测失败")
        raise ModelException(
            error_code=ErrorCode.MODEL_PREDICT_FAILED,
            message="调用 Chronos 模型预测失败",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"reason": str(exc)},
        )
    logger.info('模型预测成功')
    return result_dict