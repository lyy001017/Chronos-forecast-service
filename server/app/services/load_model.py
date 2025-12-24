from chronos import BaseChronosPipeline,Chronos2Pipeline
from functools import lru_cache

import logging
from pathlib import Path
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "chronos_models"

USER_DIR = MODELS_DIR / "users"
FINETUNED_DIR = MODELS_DIR / "finetuned"
UNFINETUNED_DIR = MODELS_DIR / "unfinetuned"

@lru_cache(maxsize=1)
def get_pipeline(mode:str,use_finetuned:bool,device:str,model_id:str=None):
    if model_id:
        model_path = USER_DIR / model_id
        model_mode = f"用户微调模型{model_id}"
    elif use_finetuned and mode == "SKU":
        
        model_path = FINETUNED_DIR / "SKU_finetuned"
        model_mode = "SKU微调模型"
    elif use_finetuned and mode == 'sales':
        model_path = FINETUNED_DIR / "sales_finetuned"
        model_mode = "售价微调模型"
        
    else:
        model_path = UNFINETUNED_DIR
        model_mode = "未经微调模型"
    if not model_path.exists():
        raise FileExistsError(
            f"模型路径不存在{model_path.resolve()}"
        )
    
    try:
        logger.info("尝试加载：%s",model_mode)

        pipeline :Chronos2Pipeline= BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device
        )

        logger.info("加载%s成功",model_mode)

    
        return pipeline
    except Exception as exc:
        logger.warning(
            "加载模型失败 %s",exc
        )
    return pipeline


