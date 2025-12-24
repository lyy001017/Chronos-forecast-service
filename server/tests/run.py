

import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from chronos import Chronos2Pipeline, BaseChronosPipeline


# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# 1. CSV -> history_df / future_df
# =========================

def build_history_and_future_from_csv(
    csv_path: str,
    horizon: int = 28,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 supply_chain_dataset1.csv 构造：
      - history_df: 作为 predict_df 的 history_df
      - future_df : 作为 predict_df 的 future_df（未来协变量，不含 target）
      - full_df   : 用于评估时拿真实 label（包含 target）

    规则：
      - Date           -> timestamp
      - SKU_ID + Warehouse_ID + Supplier_ID 合并为 id
      - Demand_Forecast -> target
      - 其他数值列作为协变量
      - 每个 id 最后 horizon 条作为“未来时间段”，既用于：
          * future_df（去掉 target）
          * 评估时的真实 y_true
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "Date",
        "SKU_ID",
        "Warehouse_ID",
        "Supplier_ID",
        "Demand_Forecast",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df = df.copy()

    # 生成 id
    df["id"] = (
        df["SKU_ID"].astype(str)
        + "_"
        + df["Warehouse_ID"].astype(str)
        + "_"
        + df["Supplier_ID"].astype(str)
    )

    # 重命名为统一字段
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Demand_Forecast": "target",
        }
    )

    # 删除构成 id 的原始列
    df = df.drop(columns=["SKU_ID", "Warehouse_ID", "Supplier_ID"])

    # timestamp 转 datetime，方便排序
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 只保留数值型协变量（除了 timestamp / id / target）
    base_cols = ["timestamp", "id", "target"]
    candidate_cov_cols = [c for c in df.columns if c not in base_cols]

    covariate_cols = []
    drop_cols = []

    for col in candidate_cov_cols:
        if is_numeric_dtype(df[col]):
            covariate_cols.append(col)
        else:
            # 尝试转成数值，转不动（全 NaN）的丢弃
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any():
                df[col] = coerced
                covariate_cols.append(col)
            else:
                drop_cols.append(col)

    if drop_cols:
        logger.info("丢弃非数值协变量列: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # 再排序
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # ========= 按 id 切分 history / future =========
    history_parts = []
    future_parts = []

    for gid, g in df.groupby("id", sort=False):
        g = g.sort_values("timestamp")

        if len(g) <= horizon:
            logger.warning(
                "id=%s 样本数 %d <= horizon(%d)，该 id 无法切 history/future，跳过预测",
                gid, len(g), horizon,
            )
            continue

        history_part = g.iloc[:-horizon].copy()
        future_part = g.iloc[-horizon:].copy()

        history_parts.append(history_part)

        # future_df 不需要 target，只保留 timestamp/id + 协变量
        future_cov_part = future_part.drop(columns=["target"])
        future_parts.append(future_cov_part)

    if not history_parts or not future_parts:
        raise ValueError("切分后 history_df 或 future_df 为空，请检查 horizon 和数据量")

    history_df = pd.concat(history_parts, ignore_index=True)
    future_df = pd.concat(future_parts, ignore_index=True)

    logger.info("history_df shape=%s, future_df shape=%s", history_df.shape, future_df.shape)

    # full_df 用于评估时拿真实标签（全量，包含 target）
    full_df = df.copy()

    return history_df, future_df, full_df


# =========================
# 2. 评估指标函数
# =========================

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray) -> float:
    if len(insample) < 2:
        return float("nan")

    naive_errors = np.abs(insample[1:] - insample[:-1])
    denom = float(np.mean(naive_errors))
    if denom == 0:
        return float("nan")

    return _mae(y_true, y_pred) / denom

def _crps_quantile(
    y_true: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
) -> float:
    """
    基于分位数预测的 CRPS 近似计算（Quantile-based CRPS）

    参数
    ----
    y_true : ndarray, shape (N,)
        真实值
    quantile_preds : dict
        {quantile(float): prediction ndarray}

    返回
    ----
    crps : float
    """
    if not quantile_preds:
        return float("nan")

    crps_terms = []

    for tau, q_pred in quantile_preds.items():
        indicator = (y_true <= q_pred).astype(float)
        crps_tau = (indicator - tau) * (q_pred - y_true)
        crps_terms.append(crps_tau)

    # CRPS = 2 * mean over quantiles
    crps = 2.0 * np.mean(np.stack(crps_terms, axis=0))
    return float(crps)

# =========================
# 3. 用 pipeline.predict_df 做预测并评估
# =========================

def run_predict_and_eval(
    pipeline: Chronos2Pipeline,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    full_df: pd.DataFrame,
    prediction_length: int,
    quantiles: List[float],
) -> Dict[str, float]:
    """
    使用 Chronos2Pipeline：
      - predict_df 预测
      - 对齐真实值
      - 计算 MAE / RMSE / MAPE / sMAPE / MASE / CRPS
    """
    # ========= 1. 构造 train / test =========
    full_df = full_df.sort_values(["id", "timestamp"])
    train_parts, test_parts = [], []

    for gid, g in full_df.groupby("id", sort=False):
        g = g.sort_values("timestamp")
        if len(g) <= prediction_length:
            continue
        train_parts.append(g.iloc[:-prediction_length].copy())
        test_parts.append(g.iloc[-prediction_length:].copy())

    if not train_parts or not test_parts:
        raise ValueError("无法切分 train/test，请检查 prediction_length")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    test_df = test_df[["id", "timestamp", "target"]].copy()

    # ========= 2. 调用 Chronos 预测 =========
    logger.info("开始调用 pipeline.predict_df ...")
    predictions = pipeline.predict_df(
        history_df,
        future_df=future_df,
        prediction_length=prediction_length,
        quantile_levels=quantiles,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    preds_df = predictions.to_pandas() if hasattr(predictions, "to_pandas") else pd.DataFrame(predictions)

    # 时间字段对齐
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # ========= 3. 对齐预测 & 真实 =========
    merged = preds_df.merge(
        test_df,
        on=["id", "timestamp"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("预测结果无法与真实值对齐")

    y_true = merged["target"].to_numpy(dtype=float)

    # 点预测（优先 0.5 分位）
    if "0.5" in merged.columns:
        y_pred = merged["0.5"].to_numpy(dtype=float)
    elif "mean" in merged.columns:
        y_pred = merged["mean"].to_numpy(dtype=float)
    else:
        numeric_cols = merged.select_dtypes(include=["number"]).columns
        y_pred = merged[numeric_cols[0]].to_numpy(dtype=float)

    # ========= 4. 构造分位数预测字典（CRPS 用） =========
    quantile_preds = {}
    for q in quantiles:
        q_col = str(q)
        if q_col in merged.columns:
            quantile_preds[q] = merged[q_col].to_numpy(dtype=float)

    # ========= 5. in-sample（MASE 用） =========
    insample = train_df["target"].to_numpy(dtype=float)

    # ========= 6. 指标计算 =========
    metrics = {
        "MAE":   _mae(y_true, y_pred),
        "RMSE":  _rmse(y_true, y_pred),
        "MAPE":  _mape(y_true, y_pred),
        "sMAPE": _smape(y_true, y_pred),
        "MASE":  _mase(y_true, y_pred, insample),
        "CRPS":  _crps_quantile(y_true, quantile_preds),
    }

    return metrics


# =========================
# 4. 主函数：加载模型 + 评估 base vs finetuned
# =========================

def main():
    # —— 1) CSV 路径（根据你实际路径改，不改也能跑你现在的布局）——
    csv_path = "/Users/lyy/Documents/work_timeSequence/Chronos_forecast/server/tests/supply_chain_dataset1.csv"
    # —— 2) 模型路径（用你给的那两行）——
    UNFINETUNED_MODEL_PATH = "/Users/lyy/Documents/work_timeSequence/Chronos_forecast/server/app/models/chronos_models/unfinetuned"
    FINETUNED_MODEL_PATH   = "/Users/lyy/Documents/work_timeSequence/Chronos_forecast/server/chronos-2-finetuned/2025-12-19_18-02-32/finetuned-ckpt"

    # —— 3) 加载 CSV -> history_df / future_df / full_df —— 
    history_df, future_df, full_df = build_history_and_future_from_csv(
        csv_path=csv_path,
        horizon=28,
    )

    # —— 4) 加载两个 pipeline —— 
    logger.info("加载未微调模型: %s", UNFINETUNED_MODEL_PATH)
    base_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        UNFINETUNED_MODEL_PATH,
        device_map="cpu",
    )

    logger.info("加载微调模型: %s", FINETUNED_MODEL_PATH)
    finetuned_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        FINETUNED_MODEL_PATH,
        device_map="cpu",
    )

    # —— 5) 评估未微调模型 —— 
    quantiles = [0.1, 0.5, 0.9]
    prediction_length = 28

    logger.info("开始评估未微调模型（base）")
    base_metrics = run_predict_and_eval(
        pipeline=base_pipeline,
        history_df=history_df,
        future_df=future_df,
        full_df=full_df,
        prediction_length=prediction_length,
        quantiles=quantiles,
    )

    logger.info("开始评估微调模型（finetuned）")
    finetuned_metrics = run_predict_and_eval(
        pipeline=finetuned_pipeline,
        history_df=history_df,
        future_df=future_df,
        full_df=full_df,
        prediction_length=prediction_length,
        quantiles=quantiles,
    )

    print("=" * 60)
    print("未微调模型 (base) 指标：")
    for k, v in base_metrics.items():
        print(f"  {k:6s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:6s}: {v}")

    print("-" * 60)
    print("微调模型 (finetuned) 指标：")
    for k, v in finetuned_metrics.items():
        print(f"  {k:6s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:6s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    # 关键：所有使用 DataLoader 的代码必须放在 main 守卫里，
    # 这样可以避免你之前遇到的 multiprocessing / spawn 报错。
    main()