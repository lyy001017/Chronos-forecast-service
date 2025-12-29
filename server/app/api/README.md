# 📂 API服务接口说明

## 预测服务（/predict）
- **`predict.py`**
- POST/：核心预测接口。支持传入历史数据（可含协变量）、未来已知协变量、指定预测步长（支持年/月/日/小时级）和分位数（额外支持0.01和0.99两个风险点预测）

### 输入数据说明：
#### 1.历史数据（前三列必要）history_data
- 固定时间频率（年/月/日/小时级均支持）：缺失值可以填充，略微影响精度，但要保证时间频率连续。列名定义为timestamp
- 序列名称：一个序列对应一个id，通过id来区分不同序列。列名定义为id。
- 待预测目标：要预测的目标变量，可以是销售额、库存等数据。列名定义为target。
- 历史协变量：未来不知道，但是过去已知的协变量（如过去影响能源负荷的气温等）。保证长度与序列长度一致，列名自定义。

#### 2.未来已知协变量 future_cov
- 固定时间频率和序列名称：同历史数据要求。
- 未来已知协变量：未来已知的协变量（如影响价格的未来已知的促销计划、节假日等），要求协变量序列长度与预测步长一致。

#### 3.预测步长 prediction_length
- 输入要预测的长度（年/月/日/小时）

#### 4.分位数输出 quantiles
- 要输出的分位数（支持0.1-0.9，从悲观估计到中位数到乐观估计，额外支持0.01和0.99极端分位数输出，提供风险预测能力）

#### 5.其他查询参数
- **use_finetuned**:是否使用微调模型，默认true
- **device**:推理设备选择，默认cuda
- **with_cov**:是否使用协变量预测，默认false。如果传入未来已知协变量，可以设为true以使用其辅助预测。

### 输入数据格式样例
#### 请求体：
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


## 健康检查（/health）
- **`health.py`**
- GET/：用于K8s的存活探针