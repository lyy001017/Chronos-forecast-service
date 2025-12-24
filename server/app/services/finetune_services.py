import uuid
from pathlib import Path
import os
from typing import Dict,Any,List,Optional

import torch
from app.services.load_model import get_pipeline
from app.models.finetuned_models import FineTuneRequset
from chronos import BaseChronosPipeline,Chronos2Pipeline

'''输入示例{
  "series": [
    {
      "id": "sku_001",
      "target": [10.0, 11.5, 12.0, 11.0, 13.2, 14.0, 13.8],
      "past_covariates": {
        "price": [1.20, 1.20, 1.25, 1.25, 1.30, 1.30, 1.30],
        "promo": [0, 0, 0, 1, 1, 0, 0]
      },
      "future_covariates": {
        "price": [1.35, 1.35, 1.40],
        "promo": [0, 1, 0]
      }
    },
    {
      "id": "sku_002",
      "target": [8.0, 8.5, 9.0, 9.2, 9.8, 10.1, 10.0],
      "past_covariates": {
        "price": [0.95, 0.95, 1.00, 1.00, 1.05, 1.05, 1.05],
        "promo": [0, 0, 1, 1, 0, 0, 0]
      },
      "future_covariates": {
        "price": [1.10, 1.10, 1.15],
        "promo": [0, 0, 1]
      }
    }
  ],
  "prediction_length": 3,
  "num_steps": 30,
  "learning_rate": 0.00001,
  "batch_size": 16,
  "model_name": "chronos-2"
}'''

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "chronos_models"

USER_DIR = MODELS_DIR / "users"

async def run_finetune(request:FineTuneRequset,prediction_length,num_steps,learning_rate,batch_size):
    # -------------------------
    # 1. 构造 Chronos inputs
    # -------------------------
    train_inputs: List[Dict[str, Any]] = []

    for series in request.series:
        item: Dict[str, Any] = {
            "target": torch.tensor(series.target, dtype=torch.float32)
        }

        if series.past_covariates:
            item["past_covariates"] = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in series.past_covariates.items()
            }

        if series.future_covariates:
            # 训练时不使用 future 值，但必须声明名字
            item["future_covariates"] = {
                k: None for k in series.future_covariates.keys()
            }

        train_inputs.append(item)
    # -------------------------
    # 2. 加载基础 Chronos 模型
    # -------------------------
    pipeline = get_pipeline(use_finetuned=True,mode='',device='cpu')

    # -------------------------
    # 3. 开始微调
    # -------------------------
    finetuned_pipeline = pipeline.fit(
        inputs = train_inputs,
        prediction_length=prediction_length,
        num_steps = num_steps,
        learning_rate = learning_rate,
        batch_size = batch_size
    )
    # -------------------------
    # 4. 保存微调模型
    # -------------------------
    USER_DIR.mkdir(parents=True,exist_ok=True)
    model_id =str(uuid.uuid4())
    model_path = USER_DIR/model_id
    model_path.mkdir(parents=False,exist_ok=False)

    try:
        finetuned_pipeline.save_pretrained(str(model_path))
    except Exception :
        if model_path.exists():
            for p in model_path.glob("*"):
                p.unlink()
            model_path.rmdir()
        raise


    return {
        "model_id":model_id,
        "status":"success"
    }
