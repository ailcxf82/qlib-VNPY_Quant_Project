# 行业轮动预测模型架构详解

## 一、模型类型概述

行业轮动预测系统使用的是 **IndustryGRU** 模型，这是一个基于 **GRU（门控循环单元）** 的深度学习模型，专门为行业轮动预测任务设计。

### 模型分类
- **模型类型**：时序深度学习模型（Time Series Deep Learning Model）
- **基础架构**：GRU（Gated Recurrent Unit，门控循环单元）
- **任务类型**：排序预测（Ranking Prediction）
- **预测目标**：未来 10 天的行业相对收益排名

---

## 二、模型架构详解

### 2.1 整体架构

IndustryGRU 模型采用三层架构设计：

```
输入层 (Input Layer)
    ↓
特征注意力层 (Feature Attention Layer)
    ↓
GRU 时序处理层 (GRU Temporal Processing Layer)
    ↓
全连接输出层 (Fully Connected Output Layer)
    ↓
预测分数 (Prediction Score)
```

### 2.2 输入数据格式

- **输入形状**：`(batch_size, sequence_length, num_features)`
  - `batch_size`：批次大小（默认 32）
  - `sequence_length`：时序长度（默认 60 天）
  - `num_features`：特征数量（自动从数据中推断）

- **特征类型**：
  - 行业动量因子（Momentum）
  - 拥挤度因子（Crowding）
  - 利率敏感度因子（Macro Beta）
  - 基础价格特征（开高低收量）
  - 技术指标特征（移动平均、波动率等）

---

## 三、核心组件详解

### 3.1 Feature Attention 层（特征注意力层）

**作用**：自动学习每个特征的重要性权重，提高模型对关键特征的关注度。

**实现原理**：
```python
class FeatureAttention(nn.Module):
    - 输入：特征张量 (batch_size, sequence_length, num_features)
    - 处理流程：
      1. 提取最后一个时间步的特征 (batch_size, num_features)
      2. 通过两层全连接网络计算注意力权重
         - Linear(num_features → hidden_dim) + ReLU
         - Linear(hidden_dim → num_features) + Sigmoid
      3. 将权重应用到所有时间步的特征上
    - 输出：加权后的特征张量
```

**关键特点**：
- 使用 **Sigmoid** 激活函数，输出 0-1 之间的权重
- 基于最后一个时间步的特征计算全局权重
- 权重应用到整个序列的所有时间步

**配置参数**：
- `attention_hidden_dim`：注意力网络隐藏层维度（默认 64）

---

### 3.2 GRU 层（门控循环单元层）

**作用**：处理时序信息，捕捉行业轮动的时序模式。

**GRU 简介**：
- GRU 是 LSTM 的简化版本，具有更少的参数，训练速度更快
- 通过门控机制（更新门和重置门）控制信息的流动
- 适合处理中等长度的时序数据（60 天）

**模型配置**：
```python
nn.GRU(
    input_size=num_features,      # 输入特征维度
    hidden_size=64,               # 隐藏层大小
    num_layers=1,                 # GRU 层数（单层）
    batch_first=True,             # 批次维度在前
    dropout=0.1                   # Dropout 率（仅在多层时生效）
)
```

**处理流程**：
1. 接收经过 Feature Attention 加权的特征序列
2. 通过 GRU 处理每个时间步，更新隐藏状态
3. 输出每个时间步的隐藏状态表示

**输出**：
- `gru_out`：形状为 `(batch_size, sequence_length, hidden_size)`
- 包含每个时间步的隐藏状态表示

---

### 3.3 全连接输出层

**作用**：将 GRU 的隐藏状态映射为预测分数。

**架构**：
```python
nn.Sequential(
    Linear(hidden_size → hidden_size//2),  # 64 → 32
    ReLU(),
    Dropout(0.1),
    Linear(hidden_size//2 → 1)            # 32 → 1
)
```

**处理流程**：
1. 取 GRU 输出的最后一个时间步的隐藏状态
2. 通过两层全连接网络降维
3. 输出标量预测分数

**输出**：
- 形状：`(batch_size, 1)`
- 含义：行业相对收益的预测分数（用于排序）

---

## 四、损失函数

### 4.1 支持的损失函数类型

模型支持两种损失函数：

#### 1. MSE Loss（均方误差损失）
- **类型**：回归损失
- **适用场景**：直接预测收益数值
- **公式**：`L = (pred - label)²`

#### 2. Ranking Loss（排序损失）
- **类型**：排序损失 + MSE 损失
- **适用场景**：排序预测任务（行业轮动）
- **公式**：`L = (1-α) × MSE + α × Ranking_Loss`

### 4.2 Ranking Loss 详解

**设计目的**：
- 行业轮动是排序任务，不需要精确预测收益数值
- 只需要正确预测行业的相对排名关系
- 排序损失鼓励模型学习正确的排序关系

**实现原理**：
```python
1. MSE 损失：保证预测值的数值准确性
2. 排序损失：使用 pairwise ranking loss
   - 对于所有样本对 (i, j)
   - 如果 label[i] > label[j]
   - 则希望 pred[i] > pred[j] + margin
   - 使用 hinge loss: max(0, margin - (pred_i - pred_j))
3. 总损失 = (1-α) × MSE + α × Ranking_Loss
```

**关键参数**：
- `alpha`：排序损失的权重（默认 0.5）
  - `alpha=0`：纯 MSE 损失
  - `alpha=1`：纯排序损失
  - `alpha=0.5`：平衡两种损失
- `margin`：排序间隔（默认 0.1）

**当前配置**：
- 损失函数类型：`"mse"`（可切换为 `"ranking"`）
- Ranking Loss 权重：`0.5`

---

## 五、模型超参数配置

### 5.1 架构超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sequence_length` | 60 | 时序长度（60 天历史数据） |
| `hidden_size` | 64 | GRU 隐藏层大小 |
| `num_layers` | 1 | GRU 层数（单层） |
| `attention_hidden_dim` | 64 | Feature Attention 隐藏层维度 |
| `dropout` | 0.1 | Dropout 率（防止过拟合） |

### 5.2 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批处理大小 |
| `lr` | 0.001 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减（L2 正则化） |
| `max_epochs` | 50 | 最大训练轮数 |
| `patience` | 10 | 早停耐心值 |

### 5.3 损失函数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss` | "mse" | 损失函数类型（"mse" 或 "ranking"） |
| `ranking_alpha` | 0.5 | Ranking Loss 权重（当 loss="ranking" 时使用） |

---

## 六、模型特点与优势

### 6.1 时序建模能力

- **GRU 架构**：专门设计用于处理时序数据
- **60 天历史窗口**：捕捉中短期行业轮动趋势
- **门控机制**：有效处理长期依赖关系

### 6.2 特征选择能力

- **Feature Attention**：自动学习特征重要性
- **动态权重**：根据当前市场状态调整特征权重
- **降噪能力**：抑制不重要特征的干扰

### 6.3 排序预测优化

- **Ranking Loss**：专门为排序任务设计
- **相对排名**：不依赖绝对收益数值
- **鲁棒性**：对异常值不敏感

### 6.4 训练稳定性

- **Dropout**：防止过拟合
- **梯度裁剪**：防止梯度爆炸
- **早停机制**：避免过度训练
- **NaN 处理**：自动检测和清理异常值

---

## 七、模型训练流程

### 7.1 数据准备

1. **特征提取**：从 qlib 数据源提取 Alpha 因子
2. **标签构建**：计算未来 10 天收益率作为标签
3. **时序序列构建**：将数据转换为 60 天序列
4. **归一化**：按训练窗口单独归一化（避免数据泄露）

### 7.2 滚动训练

- **训练窗口**：24 个月
- **验证窗口**：1 个月
- **滚动步长**：1 个月
- **窗口数量**：根据数据范围自动生成

### 7.3 训练过程

1. **前向传播**：
   - Feature Attention → GRU → 全连接层 → 预测分数

2. **损失计算**：
   - MSE Loss 或 Ranking Loss

3. **反向传播**：
   - 计算梯度
   - 梯度裁剪（max_norm=1.0）
   - 优化器更新（Adam）

4. **验证评估**：
   - 计算 Rank IC（Spearman 相关系数）
   - 早停判断

---

## 八、模型预测流程

### 8.1 预测数据准备

1. **历史数据获取**：获取最近 60 天的特征数据
2. **数据归一化**：使用训练时的归一化参数
3. **序列构建**：构建 60 天时序序列

### 8.2 预测过程

1. **模型加载**：加载训练好的模型权重
2. **前向传播**：通过模型得到预测分数
3. **排序输出**：按预测分数对行业进行排序

### 8.3 预测结果

- **输出格式**：每个行业的预测分数（Series）
- **使用方式**：按分数排序，选择排名靠前的行业

---

## 九、模型与其他方法的对比

### 9.1 vs. 传统机器学习模型

| 特性 | IndustryGRU | LightGBM/XGBoost |
|------|-------------|------------------|
| 时序建模 | ✅ 专门设计 | ❌ 需要手动特征工程 |
| 特征选择 | ✅ 自动学习 | ⚠️ 需要特征重要性分析 |
| 排序优化 | ✅ Ranking Loss | ⚠️ 需要特殊处理 |
| 可解释性 | ⚠️ 较低 | ✅ 较高 |

### 9.2 vs. 其他深度学习模型

| 特性 | IndustryGRU | LSTM | Transformer |
|------|-------------|------|------------|
| 参数数量 | 较少 | 较多 | 很多 |
| 训练速度 | 快 | 中等 | 慢 |
| 时序建模 | ✅ 优秀 | ✅ 优秀 | ✅ 优秀 |
| 特征注意力 | ✅ 有 | ❌ 无 | ✅ 有（自注意力） |

---

## 十、模型配置示例

### 10.1 当前配置（config_industry_rotation.yaml）

```yaml
industry_gru_config:
  num_features: null              # 自动推断
  sequence_length: 60             # 60 天时序
  hidden_size: 64                 # GRU 隐藏层 64
  num_layers: 1                   # 单层 GRU
  attention_hidden_dim: 64        # Attention 隐藏层 64
  dropout: 0.1                   # Dropout 10%
  loss: "mse"                     # MSE 损失
  ranking_alpha: 0.5              # Ranking Loss 权重
  batch_size: 32                  # 批次大小 32
  lr: 0.001                       # 学习率 0.001
  weight_decay: 1e-4              # 权重衰减
  max_epochs: 50                  # 最大 50 轮
  patience: 10                    # 早停耐心值 10
```

### 10.2 切换到 Ranking Loss

如需使用 Ranking Loss，修改配置：

```yaml
industry_gru_config:
  loss: "ranking"                 # 使用排序损失
  ranking_alpha: 0.5              # 排序损失权重 50%
```

---

## 十一、总结

IndustryGRU 模型是一个专门为行业轮动预测设计的时序深度学习模型，具有以下核心特点：

1. **GRU 架构**：有效处理时序信息，捕捉行业轮动模式
2. **Feature Attention**：自动学习特征重要性，提高模型表现
3. **排序优化**：支持 Ranking Loss，专门优化排序任务
4. **训练稳定**：包含多种正则化和稳定性机制
5. **滚动训练**：使用滚动窗口训练，模拟真实交易场景

该模型适合处理行业轮动这种时序排序预测任务，能够有效捕捉行业轮动的时序规律和特征重要性。

