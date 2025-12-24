# 📂 API服务接口说明

## 预测服务（/predict）
- **`predict.py`**
- POST/：核心预测接口。支持传入历史数据（可含协变量）、未来已知协变量、指定预测步长（支持年/月/日/小时级）和分位数（额外支持0.01和0.99两个风险点预测）
- **输入数据格式**：
### 请求体：
{
  "history_data": [
    {
      "timestamp": "2022-09-24",
      "id": "item_1",
      "target": 10.0,
      "price": 1.20,
      "promo_flag": 0,
      "weekday": 6
    },
    {
      "timestamp": "2022-09-25",
      "id": "item_1",
      "target": 11.0,
      "price": 1.22,
      "promo_flag": 0,
      "weekday": 0
    },...
    ...
    {
      "timestamp": "2022-09-29",
      "id": "item_2",
      "target": 9.8,
      "price": 1.02,
      "promo_flag": 0,
      "weekday": 4
    },
    {
      "timestamp": "2022-09-30",
      "id": "item_2",
      "target": 10.0,
      "price": 1.05,
      "promo_flag": 1,
      "weekday": 5
    }
  ],
  "future_cov": [
    {
      "timestamp": "2022-10-01",
      "id": "item_1",
      "price": 1.36,
      "promo_flag": 0,
      "weekday": 6
    },
    {
      "timestamp": "2022-10-02",
      "id": "item_1",
      "price": 1.37,
      "promo_flag": 0,
      "weekday": 0
    },
    ...
    {
      "timestamp": "2022-10-01",
      "id": "item_2",
      "price": 1.06,
      "promo_flag": 0,
      "weekday": 6
    },
    {
      "timestamp": "2022-10-02",
      "id": "item_2",
      "price": 1.07,
      "promo_flag": 0,
      "weekday": 0
    },
    ...]
  "prediction_length": 7,
  "quantiles": [0.1, 0.5, 0.9]
}
### 查询参数
- **use_finetuned**:是否使用微调模型，默认true
- **device**:推理设备选择，默认cuda
- **with_cov**:是否使用协变量预测，默认true



## 健康检查（/health）
- **`health.py`**
- GET/：用于K8s的存活探针