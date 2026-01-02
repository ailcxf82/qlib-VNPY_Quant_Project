# 行业轮动预测系统使用指南

## 快速开始

### 1. 配置行业指数路径

编辑 `config_data_industry.yaml`，设置 31 个申万一级行业指数的完整路径：

```yaml
data:
  industry_index_path: "sw_l1_801010,sw_l1_801020,sw_l1_801030,sw_l1_801040,sw_l1_801050,sw_l1_801080,sw_l1_801110,sw_l1_801120,sw_l1_801130,sw_l1_801140,sw_l1_801150,sw_l1_801160,sw_l1_801170,sw_l1_801180,sw_l1_801200,sw_l1_801210,sw_l1_801230,sw_l1_801710,sw_l1_801720,sw_l1_801730,sw_l1_801740,sw_l1_801750,sw_l1_801760,sw_l1_801770,sw_l1_801780,sw_l1_801790,sw_l1_801880,sw_l1_801890,sw_l1_801950,sw_l1_801960,sw_l1_801970"
```

**注意**：上述路径仅为示例，请根据实际的 Qlib 数据源中的行业指数代码进行配置。

### 2. 训练模型

使用项目自定义的训练流程：

```bash
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

或使用 Qlib 标准工作流（如果已实现适配器）：

```python
import qlib
from qlib import run

qlib.init(provider_uri="D:/qlib_data/qlib_data", region="cn")
run(config_path="classify/config_qlib_workflow.yaml")
```

### 3. 进行预测

```bash
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml --date 2024-12-31
```

## Alpha 因子表达式说明

### 1. 行业动量 (Momentum)

**20 天收益率 (ROC20)**：
```python
Ref($close, 0) / Ref($close, 20) - 1
```
- **含义**：过去 20 天的累计收益率
- **用途**：捕捉短期动量效应

**60 天收益率 (ROC60)**：
```python
Ref($close, 0) / Ref($close, 60) - 1
```
- **含义**：过去 60 天的累计收益率
- **用途**：捕捉中期动量效应

### 2. 拥挤度 (Crowding)

**标准化成交量偏离度**：
```python
($volume - Mean($volume, 250)) / Std($volume, 250)
```
- **含义**：当前成交量相对于过去 250 天均值的标准化偏离度
- **用途**：衡量交易活跃度，识别过度拥挤的行业

**成交量相对均值比率**：
```python
$volume / Mean($volume, 250) - 1
```
- **含义**：成交量相对均值的百分比变化
- **用途**：备选拥挤度指标

### 3. 利率敏感度 (Macro Beta)

**行业日收益率**：
```python
Ref($close, 0) / Ref($close, 1) - 1
```
- **含义**：单日收益率
- **用途**：用于计算与宏观变量的相关性

**与国债收益率的相关性**（需要数据源支持）：
```python
Correl(Ref($close, 0) / Ref($close, 1) - 1, Ref($bond_yield_10y, 0) - Ref($bond_yield_10y, 1), 60)
```
- **含义**：行业日收益与 10 年期国债收益率变化的 60 天滚动相关性
- **用途**：衡量行业对利率变化的敏感度
- **注意**：需要数据源包含 `$bond_yield_10y` 字段

### 4. 标签 (Label)

**未来 10 天收益率**：
```python
Ref($close, -10) / $close - 1
```
- **含义**：未来 10 天的累计收益率
- **用途**：用于排序预测，预测哪些行业在未来 10 天表现更好

## 模型配置说明

### IndustryGRU 超参数

在 `config_industry_rotation.yaml` 中可以调整以下超参数：

```yaml
industry_gru_config:
  sequence_length: 60      # 时序长度（历史天数）
  hidden_size: 64          # GRU 隐藏层大小
  num_layers: 1            # GRU 层数
  attention_hidden_dim: 64 # Feature Attention 隐藏层维度
  dropout: 0.1             # Dropout 率
  loss: "mse"              # 损失函数：'mse' 或 'ranking'
  ranking_alpha: 0.5       # Ranking Loss 权重（当 loss='ranking' 时）
  batch_size: 32           # 批处理大小
  lr: 0.001               # 学习率
  weight_decay: 1e-4      # 权重衰减
  max_epochs: 50          # 最大训练轮数
  patience: 10            # 早停耐心值
```

### 损失函数选择

- **MSE Loss** (`loss: "mse"`)：
  - 适用于回归任务
  - 直接预测收益率数值

- **Ranking Loss** (`loss: "ranking"`)：
  - 适用于排序任务
  - 更关注预测值的排序关系而非绝对值
  - 通过 `ranking_alpha` 控制排序损失的权重（0-1之间）

## 数据准备

### 1. Qlib 数据源

确保 Qlib 数据源包含以下数据：

- 31 个申万一级行业指数的 OHLCV 数据
- 可选：10 年期国债收益率数据（用于利率敏感度因子）

### 2. 数据格式

数据应遵循 Qlib 标准格式：
- 索引：MultiIndex (datetime, instrument)
- 字段：$close, $open, $high, $low, $volume 等

## 常见问题

### Q1: 如何获取 31 个申万一级行业指数的代码？

A: 可以通过以下方式获取：
1. 查看 Qlib 数据源中的行业指数列表
2. 使用 Tushare、Wind 等数据源查询申万一级行业指数代码
3. 参考申万行业分类标准

### Q2: 模型训练很慢怎么办？

A: 可以尝试：
1. 减小 `sequence_length`（如从 60 降到 30）
2. 增大 `batch_size`（如果 GPU 内存允许）
3. 减少 `max_epochs` 或增加 `patience`（提前停止）

### Q3: 如何添加更多因子？

A: 在 `config_data_industry.yaml` 的 `features` 列表中添加 Qlib 表达式：

```yaml
features:
  - "你的因子表达式"
```

### Q4: 如何使用混合模型（LightGBM + GRU）？

A: 当前版本仅实现了 GRU 模型。要实现混合模型，需要：
1. 训练 LightGBM 模型（使用现有的 LightGBMModelWrapper）
2. 训练 GRU 模型（使用 IndustryGRUWrapper）
3. 在预测时融合两个模型的预测结果

## 性能优化建议

1. **特征选择**：只保留对排序预测有用的因子
2. **时序长度**：根据数据量和计算资源调整 `sequence_length`
3. **批处理大小**：根据 GPU 内存调整 `batch_size`
4. **早停策略**：合理设置 `patience` 避免过拟合

## 后续开发计划

- [ ] 实现完整的滚动训练逻辑
- [ ] 添加 LightGBM + GRU 混合模型支持
- [ ] 实现回测功能
- [ ] 添加更多宏观因子
- [ ] 实现行业轮动策略回测

## 参考资源

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib Expression 语法](https://qlib.readthedocs.io/en/latest/component/data.html#expression)
- [PyTorch GRU 文档](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)






