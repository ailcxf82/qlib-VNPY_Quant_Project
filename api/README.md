# 股票预测系统 API 使用文档

## 概述

本系统提供基于机器学习的股票预测服务，支持沪深300和中小综指两个股票池的智能选股推荐。

---

## 快速开始

### 1. 启动服务

```bash
# 进入项目目录
cd d:\lianghuatouzi\Qlib1124\project

# 启动 API 服务
python api/predict_api.py
```

服务启动后访问：http://127.0.0.1:5000/

### 2. 生成预测数据

API 服务读取预先生成的预测文件，首次使用或需要最新预测时，请先运行：

```bash
python run_predict.py
```

预测文件将保存在 `data/predictions/` 目录下。

---

## API 接口说明

### 1. 获取预测结果

**请求：**
```
GET /api/predict?pool={股票池}&date={日期}
```

**参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pool | string | 否 | 股票池名称，默认 `csi300`。可选值：`csi300`(沪深300)、`csi101`(中小综指) |
| date | string | 否 | 预测日期，格式 `YYYY-MM-DD`。留空则返回最新日期的预测 |

**响应示例：**
```json
{
  "success": true,
  "data": {
    "pool_name": "csi300",
    "prediction_date": "2026-03-20",
    "generated_at": "2026-03-21T22:17:00",
    "model_weights": {
      "lgb": 0.2,
      "gru": 0.2,
      "stack": 0.2,
      "qlib_ensemble": 0.2,
      "mlp": 0.2
    },
    "predictions": [
      {
        "code": "600519",
        "name": "贵州茅台",
        "prediction": 0.9867,
        "date": "2026-03-20"
      }
    ]
  }
}
```

### 2. 获取可用股票池列表

**请求：**
```
GET /api/pools
```

**响应示例：**
```json
{
  "success": true,
  "data": ["csi300", "csi101"]
}
```

### 3. 获取可用日期列表

**请求：**
```
GET /api/dates?pool={股票池}
```

**响应示例：**
```json
{
  "success": true,
  "data": ["2026-01-05", "2026-01-06", ..., "2026-03-20"]
}
```

### 4. 健康检查

**请求：**
```
GET /api/health
```

**响应示例：**
```json
{
  "status": "ok",
  "timestamp": "2026-03-21T22:17:00"
}
```

---

## Web 界面功能

访问 http://127.0.0.1:5000/ 可使用可视化界面：

1. **股票池选择** - 支持沪深300、中小综指切换
2. **日期选择** - 可查看历史任意日期的预测结果
3. **一键查询** - 点击按钮即可获取推荐股票
4. **结果展示**：
   - 模型权重分布
   - Top 50 推荐股票
   - 预测值可视化进度条
   - 排名高亮显示（前三名金色标识）

---

## 预测值说明

### 预测值含义

预测值范围 `0-1`，表示该股票在未来收益的截面排名百分位：
- **接近 1.0**：预测该股票在股票池中排名靠前，建议关注
- **接近 0.0**：预测该股票在股票池中排名靠后，建议回避

### 模型权重说明

系统使用多模型融合预测，权重基于各模型历史 IC（信息系数）动态计算：

| 模型 | 说明 |
|------|------|
| lgb | LightGBM 梯度提升树模型 |
| gru | GRU 门控循环单元神经网络 |
| mlp | MLP 多层感知机神经网络 |
| stack | Stacking 集成模型 |
| qlib_ensemble | Qlib 内置集成模型 |

---

## 功能依赖来源

### 核心依赖

| 依赖 | 版本要求 | 用途 | 来源 |
|------|----------|------|------|
| Python | >= 3.8 | 运行环境 | python.org |
| Flask | >= 2.0 | Web API 框架 | `pip install flask` |
| Pandas | >= 1.3 | 数据处理 | `pip install pandas` |
| Qlib | >= 0.8 | 量化投资框架 | `pip install pyqlib` |
| LightGBM | >= 3.3 | 梯度提升模型 | `pip install lightgbm` |
| PyTorch | >= 1.10 | 深度学习框架 | `pip install torch` |

### 数据依赖

| 数据 | 来源 | 说明 |
|------|------|------|
| 股票行情数据 | Qlib 数据服务 | 提供股票 OHLCV 等基础行情 |
| 股票池定义 | Qlib 内置 | csi300、csi101 等指数成分股 |
| 股票名称映射 | 本地维护 | 常用股票中文名称字典 |

### 模型依赖

| 模型文件 | 路径 | 说明 |
|----------|------|------|
| LightGBM 模型 | `data/models/{pool}_models/*_lgb.txt` | 训练生成的模型文件 |
| GRU 模型 | `data/models/{pool}_models/*_gru.pth` | 训练生成的模型文件 |
| MLP 模型 | `data/models/{pool}_models/*_mlp.pth` | 训练生成的模型文件 |
| Stack 模型 | `data/models/{pool}_models/*_stack.pth` | 训练生成的模型文件 |

### 预测文件依赖

| 文件 | 路径 | 生成方式 |
|------|------|----------|
| CSI300 预测 | `data/predictions/pred_csi300.csv` | `python run_predict.py` |
| CSI101 预测 | `data/predictions/pred_csi101.csv` | `python run_predict.py` |

---

## 项目结构

```
project/
├── api/
│   ├── __init__.py
│   └── predict_api.py          # API 服务入口
├── config/
│   ├── pipeline.yaml           # CSI300 配置
│   ├── pipeline_csi101.yaml    # CSI101 配置
│   ├── data.yaml               # 数据配置
│   └── rqalpha_config.yaml     # 回测配置
├── data/
│   ├── predictions/            # 预测结果目录
│   │   ├── pred_csi300.csv
│   │   └── pred_csi101.csv
│   ├── models/                 # 模型文件目录
│   │   ├── csi300_models/
│   │   └── csi101_models/
│   └── logs/                   # 训练日志目录
├── feature/
│   └── qlib_feature_pipeline.py # 特征工程
├── predictor/
│   └── predictor.py            # 预测引擎
├── backtest/
│   ├── rqalpha_backtest.py     # 回测执行
│   └── rqalpha_strategy.py     # 交易策略
├── run_predict.py              # 预测入口脚本
├── run_train.py                # 训练入口脚本
└── utils.py                    # 工具函数
```

---

## 使用流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   训练模型       │ ──▶ │   生成预测       │ ──▶ │   查询预测       │
│ run_train.py    │     │ run_predict.py  │     │   API/Web       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   data/models/          data/predictions/        返回推荐股票
   *_lgb.txt             pred_*.csv
   *_gru.pth
   *_mlp.pth
```

### 完整使用流程

1. **训练模型**（首次使用或定期更新）
   ```bash
   python run_train.py
   ```

2. **生成预测**
   ```bash
   python run_predict.py
   ```

3. **查询预测**
   - Web 界面：http://127.0.0.1:5000/
   - API 调用：`GET /api/predict?pool=csi300`

---

## 错误处理

### 常见错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `未找到预测文件` | 预测文件不存在 | 运行 `python run_predict.py` |
| `日期 xxx 无预测数据` | 指定日期没有预测 | 使用 `/api/dates` 查看可用日期 |
| `模型目录不存在` | 模型未训练 | 运行 `python run_train.py` |

### 错误响应格式

```json
{
  "success": false,
  "error": "错误描述信息"
}
```

---

## 生产环境部署

### 使用 Gunicorn（推荐）

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务（4个工作进程）
gunicorn -w 4 -b 0.0.0.0:5000 api.predict_api:app
```

### 使用 uWSGI

```bash
# 安装 uWSGI
pip install uwsgi

# 启动服务
uwsgi --http :5000 --wsgi-file api/predict_api.py --callable app --processes 4
```

### Docker 部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install flask pandas pyqlib lightgbm torch

EXPOSE 5000
CMD ["python", "api/predict_api.py"]
```

---

## 注意事项

1. **预测文件时效性**：预测文件需要定期更新，建议每日收盘后运行 `run_predict.py`

2. **模型更新频率**：建议每周或每月重新训练模型，以适应市场变化

3. **风险提示**：本系统仅供研究参考，不构成投资建议。股市有风险，投资需谨慎。

4. **数据延迟**：Qlib 数据服务可能有 1 天延迟，预测日期应为未来交易日

---

## 联系与支持

如有问题，请检查：
1. 预测文件是否存在：`data/predictions/pred_csi300.csv`
2. 模型文件是否存在：`data/models/csi300_models/`
3. 服务是否正常启动：访问 `/api/health` 检查

---

*文档版本：v1.0*
*最后更新：2026-03-21*
