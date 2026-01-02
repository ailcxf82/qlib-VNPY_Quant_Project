# 行业轮动预测系统 (Industry Rotation System)

基于 Microsoft Qlib 框架的 A 股行业轮动预测系统，使用混合模型（LightGBM + GRU）预测未来 10 天的行业相对收益排名。

## 目录结构

```
classify/
├── config_data_industry.yaml          # 数据配置文件（行业指数路径、Alpha因子表达式）
├── config_industry_rotation.yaml      # 工作流配置文件
├── pytorch_industry_gru.py            # IndustryGRU 模型实现
├── run_industry_train.py              # 训练脚本
├── run_industry_predict.py            # 预测脚本
└── README.md                          # 本文件
```

## 功能特性

### 1. Alpha 因子表达式

系统实现了以下行业轮动专用因子：

- **行业动量 (Momentum)**：
  - 过去 20 天收益率：`Ref($close, 0) / Ref($close, 20) - 1`
  - 过去 60 天收益率：`Ref($close, 0) / Ref($close, 60) - 1`

- **拥挤度 (Crowding)**：
  - 标准化成交量偏离度：`($volume - Mean($volume, 250)) / Std($volume, 250)`
  - 成交量相对均值比率：`$volume / Mean($volume, 250) - 1`

- **利率敏感度 (Macro Beta)**：
  - 行业日收益率：`Ref($close, 0) / Ref($close, 1) - 1`
  - 与国债收益率的相关性（需要数据源支持）

- **标签 (Label)**：
  - 未来 10 天收益率：`Ref($close, -10) / $close - 1`

### 2. IndustryGRU 模型

- **Feature Attention 层**：自动学习特征重要性权重
- **GRU 层**：处理时序信息（默认 60 天历史）
- **输出层**：预测行业相对收益分数
- **损失函数**：支持 MSE 和 Ranking Loss

### 3. 工作流配置

- 数据时间范围：2015-01-01 至 2024-12-31
- 滚动训练：24 个月训练窗口，1 个月验证窗口
- 数据处理器：RobustZScoreNorm（去极值）、Fillna（填充缺失值）
- 标签处理：CSZScoreNorm（截面标准化，用于排序）

## 使用说明

### 1. 配置行业指数路径

编辑 `config_data_industry.yaml`，设置 31 个申万一级行业指数的路径：

```yaml
data:
  industry_index_path: "sw_l1_801010,sw_l1_801020,sw_l1_801030,..."  # 31个行业指数
```

### 2. 训练模型

```bash
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

### 3. 进行预测

```bash
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml --date 2024-12-31
```

## 配置文件说明

### config_data_industry.yaml

- `industry_index_path`: 行业指数路径（待填写）
- `features`: Alpha 因子表达式列表
- `label`: 标签表达式（未来 10 天收益率）
- `label_transform`: 标签转换配置（截面排名）

### config_industry_rotation.yaml

- `industry_gru_config`: IndustryGRU 模型超参数
- `rolling`: 滚动训练配置
- `paths`: 模型、日志、预测结果保存路径
- `portfolio`: 组合配置（最大行业权重、Top-K 选择）

## 模型架构

```
输入: (batch_size, 60, num_features)
  ↓
Feature Attention: 学习特征权重
  ↓
GRU: 单层 GRU，hidden_size=64
  ↓
全连接层: 输出标量分数
  ↓
输出: (batch_size, 1)
```

## 损失函数

- **MSE Loss**：标准均方误差损失
- **Ranking Loss**：Pointwise Ranking Loss，鼓励预测值的排序与标签排序一致

## 注意事项

1. **数据准备**：确保 Qlib 数据源包含 31 个申万一级行业指数的数据
2. **特征数量**：模型会自动从数据中推断特征数量
3. **时序长度**：默认使用 60 天历史数据，可在配置中修改
4. **GPU 支持**：模型自动检测并使用 GPU（如果可用）

## 后续改进

- [ ] 实现完整的滚动训练逻辑
- [ ] 添加回测功能
- [ ] 支持 LightGBM + GRU 混合模型
- [ ] 添加更多宏观因子（利率、通胀等）
- [ ] 实现行业轮动策略回测

## 参考文档

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib Expression 语法](https://qlib.readthedocs.io/en/latest/component/data.html#expression)







