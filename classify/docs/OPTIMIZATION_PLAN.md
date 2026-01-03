# 模型优化方案（IC 接近 0 问题）

## 一、问题诊断

### 当前表现

```
训练集 IC: 0.015（接近 0，几乎学不到东西）
验证集 IC: -0.0177（负值，不稳定）
验证集 IC 标准差: 0.1502（波动很大）
```

### 问题分析

1. **训练集 IC 接近 0**：
   - 模型几乎学不到有效信号
   - 可能原因：特征质量差、模型容量不足、训练策略不当

2. **验证集 IC 为负且不稳定**：
   - 模型泛化能力差
   - 可能原因：过拟合、数据分布不一致、标签定义问题

---

## 二、优化方案（按优先级）

### 🔴 优先级 1：特征工程优化（最重要）

#### 1.1 简化特征集，聚焦核心信号

**问题**：特征太多（45+）可能导致噪声淹没信号

**方案**：使用基线特征集，聚焦核心信号

```yaml
# 切换到基线特征集
data_config: "classify/config_data_industry_baseline.yaml"
```

**理由**：
- 基线特征集只有 15 个特征，更聚焦
- 减少噪声，提高信号质量
- 更容易调试和理解

#### 1.2 检查特征质量

**运行诊断脚本**：
```bash
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

**重点关注**：
- 测试 3：单特征基线 IC 是否为正
- 如果单特征 IC 都为负，说明数据本身可能没有信号

#### 1.3 优化截面标准化

**当前配置**：
```yaml
cross_sectional_normalization:
  enabled: true
  method: "zscore"  # 尝试改为 "rank"
```

**建议**：
- 如果 Z-score 效果不好，尝试 Rank 方法
- Rank 方法对异常值更鲁棒，可能更适合

---

### 🟠 优先级 2：模型架构优化

#### 2.1 增加模型容量

**当前配置**：
```yaml
hidden_size: 64
num_layers: 2
attention_hidden_dim: 64
```

**优化方案**：
```yaml
hidden_size: 128  # ⬆️ 增加模型容量
num_layers: 2  # 保持
attention_hidden_dim: 128  # ⬆️ 增强注意力能力
```

**理由**：
- 当前模型可能容量不足，无法学习复杂模式
- 增加容量有助于提升学习能力

#### 2.2 调整 Dropout

**当前配置**：
```yaml
dropout: 0.2
```

**优化方案**：
```yaml
dropout: 0.15  # ⬇️ 减少正则化，给模型更多学习空间
```

**理由**：
- 如果训练集 IC 都接近 0，说明可能欠拟合
- 减少 Dropout 可以让模型学习更多

---

### 🔵 优先级 3：训练策略优化

#### 3.1 优化损失函数

**当前配置**：
```yaml
loss: "ranking"
ranking_alpha: 0.7
```

**优化方案 A**：增加 Ranking Loss 权重
```yaml
ranking_alpha: 0.8  # ⬆️ 更强调排序
```

**优化方案 B**：尝试纯 MSE Loss
```yaml
loss: "mse"  # 如果 Ranking Loss 效果不好，尝试 MSE
```

#### 3.2 调整学习率

**当前配置**：
```yaml
lr: 0.0007
```

**优化方案**：
```yaml
lr: 0.001  # ⬆️ 加快学习速度
# 或
lr: 0.0005  # ⬇️ 如果训练不稳定，降低学习率
```

#### 3.3 调整 Batch Size

**当前配置**：
```yaml
batch_size: 64
```

**优化方案**：
```yaml
batch_size: 32  # ⬇️ 更小的 batch 可能有助于学习
# 或
batch_size: 128  # ⬆️ 更大的 batch 可能更稳定
```

#### 3.4 增加训练轮数

**当前配置**：
```yaml
max_epochs: 80
patience: 12
```

**优化方案**：
```yaml
max_epochs: 100  # ⬆️ 给模型更多学习时间
patience: 15  # ⬆️ 更耐心的早停
```

---

### 🟡 优先级 4：数据质量检查

#### 4.1 检查标签定义

**当前配置**：
```yaml
label: "Ref($close, -10) / $close - 1"
```

**检查**：
- 确认标签计算是否正确
- 确认标签是否有足够的区分度

#### 4.2 检查特征提取

**运行检查**：
```bash
python classify/check_data_source.py
```

**检查项**：
- 特征是否正确提取
- 是否有大量缺失值
- 特征分布是否正常

#### 4.3 检查数据对齐

**运行诊断**：
```bash
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

**重点关注**：
- 测试 1：完美预测自测（IC 应接近 1.0）
- 如果失败，说明数据对齐有问题

---

### 🟢 优先级 5：验证集稳定性优化

#### 5.1 增加验证窗口

**当前配置**：
```yaml
valid_months: 1
```

**优化方案**：
```yaml
valid_months: 2  # ⬆️ 更长的验证窗口，减少波动
```

**理由**：
- 验证窗口太短（1个月）可能导致 IC 波动大
- 增加到 2 个月可以更稳定地评估

---

## 三、推荐优化步骤

### 步骤 1：运行诊断（立即执行）

```bash
# 启用诊断
# 在 config_industry_rotation.yaml 中：
diagnosis:
  enabled: true

# 运行训练
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

**目标**：定位问题根源

---

### 步骤 2：简化特征集（优先级最高）

```yaml
# 切换到基线特征集
data_config: "classify/config_data_industry_baseline.yaml"
```

**理由**：
- 特征太多可能导致噪声淹没信号
- 基线特征集更聚焦，更容易学习

---

### 步骤 3：增加模型容量

```yaml
industry_gru_config:
  hidden_size: 128  # 从 64 增到 128
  attention_hidden_dim: 128  # 从 64 增到 128
  dropout: 0.15  # 从 0.2 降到 0.15
```

**理由**：
- 当前模型可能容量不足
- 增加容量有助于学习复杂模式

---

### 步骤 4：优化训练策略

```yaml
industry_gru_config:
  lr: 0.001  # 从 0.0007 增到 0.001
  batch_size: 32  # 从 64 降到 32
  ranking_alpha: 0.8  # 从 0.7 增到 0.8
  max_epochs: 100  # 从 80 增到 100
  patience: 15  # 从 12 增到 15
```

---

### 步骤 5：增加验证窗口

```yaml
rolling:
  valid_months: 2  # 从 1 增到 2
```

---

## 四、优化配置文件

已创建优化配置文件：`config_industry_rotation_optimized_v2.yaml`

**使用方法**：
```bash
python classify/run_industry_train.py --config classify/config_industry_rotation_optimized_v2.yaml
```

---

## 五、预期效果

### 优化前
```
训练集 IC: 0.015（接近 0）
验证集 IC: -0.0177（负值）
验证集 IC 标准差: 0.1502（波动大）
```

### 优化后（预期）
```
训练集 IC: 0.05-0.10（有学习能力）
验证集 IC: 0.02-0.05（正值，稳定）
验证集 IC 标准差: < 0.10（波动减小）
```

---

## 六、如果优化后仍无效

### 6.1 检查数据本身是否有信号

**运行单特征基线测试**：
```bash
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

**如果单特征 IC 都为负**：
- 数据本身可能没有信号
- 需要重新设计特征或标签

### 6.2 尝试更简单的模型

**方案**：使用线性模型作为基线
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**目的**：验证是否是模型复杂度问题

### 6.3 检查标签定义

**可能问题**：
- 标签计算错误
- 标签噪声太大
- 标签与特征不匹配

**检查方法**：
- 查看标签分布
- 检查标签与特征的相关性

---

## 七、总结

### 立即执行（按优先级）

1. **🔴 运行诊断**：定位问题根源
2. **🔴 简化特征集**：切换到基线特征集
3. **🟠 增加模型容量**：hidden_size=128
4. **🟠 减少 Dropout**：dropout=0.15
5. **🔵 优化训练策略**：调整学习率、batch_size、损失函数
6. **🟡 增加验证窗口**：valid_months=2

### 预期改进

- 训练集 IC：从 0.015 → 0.05-0.10
- 验证集 IC：从 -0.0177 → 0.02-0.05
- 验证集稳定性：从 std=0.1502 → < 0.10

