#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 supply_chain_dataset1.csv 生成两个 DataFrame：

- history_df: 作为 make_predictions 的 history_df 传入
  必含列: timestamp, id, target
  其他列: 只保留数值型协变量列

- future_cov_df: 作为 make_predictions 的 future_cov_df 传入
  必含列: timestamp, id
  其他列: 只保留数值型协变量列（不含 target）

拆分逻辑：
- 对每个 id 按 timestamp 排序
- 每个 id 最后 HORIZON 条数据作为 future_cov 区间
- 其余作为历史区间
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

# 预测步长：最后 HORIZON 天作为 future_cov
HORIZON = 28


def build_dataframes_from_csv(
    csv_path: str,
    horizon: int = HORIZON,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    从 CSV 构造 (history_df, future_cov_df)

    Args:
        csv_path: CSV 文件路径
        horizon: 每个 id 最后多少条作为未来协变量区间

    Returns:
        history_df: DataFrame, 包含 timestamp/id/target + 数值协变量
        future_cov_df: DataFrame 或 None, 包含 timestamp/id + 数值协变量
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    # 1. 读取 CSV
    df = pd.read_csv(csv_path)

    # 2. 基本字段检查
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

    # 3. 构造 id、timestamp、target
    df = df.copy()

    # 合并生成 id
    df["id"] = (
        df["SKU_ID"].astype(str)
        + "_"
        + df["Warehouse_ID"].astype(str)
        + "_"
        + df["Supplier_ID"].astype(str)
    )

    # 重命名字段
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Demand_Forecast": "target",
        }
    )

    # 删除原始 ID 组成列
    drop_cols = [c for c in ["SKU_ID", "Warehouse_ID", "Supplier_ID"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 转换 timestamp 为 datetime，方便排序
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 4. 只保留数值型协变量
    base_cols = ["timestamp", "id", "target"]
    candidate_cov_cols = [c for c in df.columns if c not in base_cols]

    covariate_cols: List[str] = []
    to_drop: List[str] = []

    for col in candidate_cov_cols:
        if is_numeric_dtype(df[col]):
            covariate_cols.append(col)
        else:
            # 尝试强制转为数值；完全转不动（全 NaN）的就丢弃
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any():
                df[col] = coerced
                covariate_cols.append(col)
            else:
                to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)

    # 再确认一次协变量列（此时保证是数值型）
    covariate_cols = [c for c in df.columns if c not in base_cols]

    # 5. 排序
    df = df.sort_values(["id", "timestamp"])

    # 6. 对每个 id 切分：前面历史 → history_df；最后 horizon 条 → future_cov_df
    history_parts = []
    future_parts = []

    for gid, g in df.groupby("id", sort=False):
        g = g.sort_values("timestamp")

        if len(g) <= horizon:
            print(f"[WARN] id={gid} 样本数 {len(g)} <= horizon({horizon})，跳过该 id")
            continue

        # 历史部分（作为 history_df）
        history_part = g.iloc[:-horizon].copy()
        # 未来部分（作为 future_cov_df 的来源）
        future_part = g.iloc[-horizon:].copy()

        history_parts.append(history_part)

        # future_cov 不需要 target 列，只保留 timestamp/id + 协变量
        future_part_cov = future_part.drop(columns=["target"])
        future_parts.append(future_part_cov)

    if not history_parts:
        raise ValueError("切分后 history_df 为空，请检查数据量和 horizon 设置")

    history_df = pd.concat(history_parts, ignore_index=True)

    if not future_parts:
        future_cov_df = None
    else:
        future_cov_df = pd.concat(future_parts, ignore_index=True)

    return history_df, future_cov_df


def main():
    """
    命令行测试入口：
    从 supply_chain_dataset1.csv 生成 history_df 和 future_cov_df，并打印基本信息
    """
    # 根据你的项目结构调整：这里假设脚本在 server/scripts 目录下
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "supply_chain_dataset1.csv"

    history_df, future_cov_df = build_dataframes_from_csv(str(csv_path))

    print("=" * 40)
    print("history_df 预览：")
    print(history_df.head())
    print(f"history_df 形状: {history_df.shape}")
    print("=" * 40)

    if future_cov_df is None:
        print("future_cov_df = None（未生成未来协变量）")
    else:
        print("future_cov_df 预览：")
        print(future_cov_df.head())
        print(f"future_cov_df 形状: {future_cov_df.shape}")
    print("=" * 40)


if __name__ == "__main__":
    main()