# 目标变量与损失函数优化实施指南

## 一、概述

本文档详细说明如何实施两个核心优化：
1. **目标变量优化**：将收益转换为排名百分位（Rank）
2. **非对称损失函数**：惩罚低估正向收益的错误

## 二、方案 1：标签 Rank 转换（推荐优先实施）⭐⭐⭐

### 2.1 原理

**问题**：模型预测收益的中位数或均值，对极端的正向收益不敏感。

**解决方案**：将目标变量转换为下一期收益率的**截面排名百分位**，使其分布在 $[0, 1]$ 之间。

**优点**：
- 模型的优化目标变为**最大化高收益股票的排名**，而非精确预测收益值
- 直接解决"捕捉高收益"的需求
- 对异常值更稳健
- 更适合选股任务（我们只需要知道哪些股票排名靠前）

### 2.2 实施步骤

#### 步骤 1：更新配置文件

在 `config/data.yaml` 中添加标签转换配置：

```yaml
data:
  # ... 其他配置 ...
  
  label: "Ref($close, -5)/$close - 1"  # 原始标签（收益）
  
  # 标签转换配置
  label_transform:
    enabled: true              # 启用标签转换
    method: "percentile"       # 转换方法：percentile（百分位）或 rank（排名）
    groupby: "datetime"       # 分组方式：按日期分组（截面排名）
```

**配置说明**：
- `enabled: true`：启用标签转换
- `method: "percentile"`：转换为 [0, 1] 的百分位（推荐）
- `method: "rank"`：转换为排名（1 到 N）
- `groupby: "datetime"`：按日期分组，同一日期的股票进行截面排名（推荐）

#### 步骤 2：验证转换效果

运行训练后，检查日志输出：

```
标签已转换为排名（方法: percentile, 分组: datetime）
```

#### 步骤 3：理解预测结果

**转换后的标签**：
- 范围：$[0, 1]$（百分位）或 $[1, N]$（排名）
- 含义：0.9 表示该股票在当日所有股票中排名前 10%

**预测结果**：
- 预测值也是排名百分位，范围 $[0, 1]$
- **直接用于选股**：选择预测值高的股票（如 > 0.8）
- **不需要逆变换**：排名预测可以直接用于选股，不需要转换回收益

### 2.3 代码实现

已实现的模块：`utils/label_transform.py`

**核心函数**：
```python
from utils.label_transform import transform_to_rank

# 将收益转换为排名百分位
rank_label = transform_to_rank(
    label=original_label,      # 原始收益标签
    method="percentile",        # 转换为百分位
    groupby="datetime"          # 按日期分组
)
```

**转换逻辑**：
```python
# 按日期分组
for each_date:
    # 获取当日所有股票的收益
    returns = label[date]
    
    # 计算排名百分位
    rank_pct = returns.rank(pct=True)  # [0, 1]
    
    # 0.0 = 最低收益（最差）
    # 1.0 = 最高收益（最好）
```

### 2.4 优势分析

| 方面 | 原始收益标签 | Rank 标签 |
|------|------------|-----------|
| **优化目标** | 精确预测收益值 | 最大化高收益股票排名 |
| **异常值处理** | 敏感 | 稳健 |
| **选股适用性** | 需要阈值判断 | 直接可用 |
| **模型学习** | 可能被极端值干扰 | 更关注相对排名 |

## 三、方案 2：非对称损失函数（补充优化）⭐⭐

### 3.1 原理

**问题**：标准 MSE 损失对高估和低估的惩罚相同，但低估正向收益（乐观不足）更严重。

**解决方案**：使用非对称损失函数，对低估正向收益给予更重惩罚。

**损失函数公式**：
$$
L(\hat{y}, y) = \begin{cases}
(\hat{y} - y)^2 \times \gamma & \text{if } y > 0 \text{ and } \hat{y} < y \quad (\text{乐观不足}) \\
(\hat{y} - y)^2 & \text{otherwise}
\end{cases}
$$

其中：
- $\gamma > 1$：惩罚系数，通常取 2.0-3.0
- $y > 0$：实际收益为正
- $\hat{y} < y$：预测值低于实际值（低估）

**意义**：当模型预测错误地低估了实际的上涨幅度时，给予更重的惩罚，迫使模型在有上涨信号时更激进地预测高值。

### 3.2 实施步骤

#### 步骤 1：更新 LightGBM 配置

在 `config/model_lgb.yaml` 中：

```yaml
model:
  loss: "asymmetric_mse"      # 使用非对称损失
  loss_params:
    gamma: 2.0                # 惩罚系数（推荐 2.0-3.0）
  # ... 其他参数 ...
```

**注意**：由于 qlib 的 LGBModel 可能不完全支持自定义目标函数，当前实现会给出警告。完整实现需要直接使用 lightgbm 原生接口。

#### 步骤 2：更新 MLP 配置

在 `config/model_mlp.yaml` 中：

```yaml
model:
  loss: "asymmetric_mse"      # 使用非对称损失
  loss_params:
    gamma: 2.0                # 惩罚系数
  # ... 其他参数 ...
```

#### 步骤 3：验证损失函数

训练时查看日志：

```
初始化非对称 MSE 损失，惩罚系数 gamma=2.0
```

### 3.3 代码实现

已实现的模块：`utils/loss_functions.py`

**PyTorch 版本**（用于 MLP）：
```python
from utils.loss_functions import AsymmetricMSELoss

criterion = AsymmetricMSELoss(gamma=2.0)
loss = criterion(pred, target)
```

**LightGBM 版本**（目标函数）：
```python
from utils.loss_functions import asymmetric_mse_objective_lgb

def custom_objective(y_true, y_pred):
    grad, hess = asymmetric_mse_objective_lgb(y_true, y_pred, gamma=2.0)
    return grad, hess
```

### 3.4 参数调优

**gamma 参数选择**：

| gamma 值 | 效果 | 适用场景 |
|---------|------|---------|
| 1.5 | 轻微惩罚 | 保守策略 |
| 2.0 | 中等惩罚 | **推荐** ⭐ |
| 3.0 | 强惩罚 | 激进策略 |
| 5.0+ | 极强惩罚 | 可能过拟合 |

**调优建议**：
1. 从 2.0 开始
2. 观察验证集表现
3. 如果模型过于激进（预测值普遍偏高），降低 gamma
4. 如果模型仍然保守，提高 gamma

## 四、组合使用（最佳实践）⭐⭐⭐

### 4.1 推荐配置

**方案 A：仅使用 Rank 转换（推荐）**
```yaml
# config/data.yaml
label_transform:
  enabled: true
  method: "percentile"
  groupby: "datetime"

# config/model_lgb.yaml
model:
  loss: "mse"  # 使用标准 MSE（因为标签已经是排名）
```

**方案 B：Rank 转换 + 非对称损失（激进）**
```yaml
# config/data.yaml
label_transform:
  enabled: true
  method: "percentile"
  groupby: "datetime"

# config/model_lgb.yaml
model:
  loss: "asymmetric_mse"
  loss_params:
    gamma: 2.0
```

**方案 C：原始收益 + 非对称损失（保守）**
```yaml
# config/data.yaml
label_transform:
  enabled: false  # 不使用 Rank 转换

# config/model_lgb.yaml
model:
  loss: "asymmetric_mse"
  loss_params:
    gamma: 2.0
```

### 4.2 实施优先级

1. **第一步**：实施 Rank 转换（最简单、最有效）
2. **第二步**：验证效果，如果仍然保守，再添加非对称损失
3. **第三步**：根据回测结果调优参数

## 五、完整配置示例

### 5.1 config/data.yaml

```yaml
data:
  # ... 其他配置 ...
  
  label: "Ref($close, -20)/$close - 1"  # 建议改为 20 日收益
  
  # 标签转换配置
  label_transform:
    enabled: true              # 启用 Rank 转换
    method: "percentile"       # 转换为百分位
    groupby: "datetime"        # 按日期分组（截面排名）
```

### 5.2 config/model_lgb.yaml

```yaml
model:
  # 方案 A：仅使用 Rank 转换（推荐）
  loss: "mse"
  
  # 方案 B：Rank 转换 + 非对称损失
  # loss: "asymmetric_mse"
  # loss_params:
  #   gamma: 2.0
  
  num_boost_round: 2000
  early_stopping_rounds: 100
  params:
    learning_rate: 0.02
    max_depth: 6
    num_leaves: 31
    # ... 其他参数 ...
```

### 5.3 config/model_mlp.yaml

```yaml
model:
  # 方案 A：仅使用 Rank 转换（推荐）
  # 使用标准 MSE（因为标签已经是排名）
  
  # 方案 B：Rank 转换 + 非对称损失
  # loss: "asymmetric_mse"
  # loss_params:
  #   gamma: 2.0
  
  hidden_dims: [128, 64, 32]
  dropout: 0.3
  # ... 其他参数 ...
```

## 六、验证与评估

### 6.1 训练验证

运行训练后，检查：

1. **日志输出**：
   ```
   标签已转换为排名（方法: percentile, 分组: datetime）
   ```

2. **标签分布**：
   ```python
   import pandas as pd
   
   # 检查标签分布
   print(label.describe())
   # 应该看到 min=0.0, max=1.0（如果是百分位）
   ```

3. **预测分布**：
   ```python
   # 检查预测值分布
   print(pred.describe())
   # 应该看到合理的分布，不是全部集中在某个值
   ```

### 6.2 回测验证

重点关注：
1. **快速上涨期间的表现**：策略是否能跟上基准
2. **IC 和 IC-IR**：是否提升
3. **预测值分布**：是否更偏向高值（在上涨期间）

### 6.3 对比实验

建议进行对比实验：

| 实验 | 配置 | 预期效果 |
|------|------|---------|
| 基线 | 原始收益 + MSE | 基准 |
| 实验1 | Rank 转换 + MSE | 应该提升 |
| 实验2 | 原始收益 + 非对称损失 | 应该提升 |
| 实验3 | Rank 转换 + 非对称损失 | 最佳（预期） |

## 七、常见问题

### Q1: Rank 转换后，预测值如何解释？

**A**: 预测值是排名百分位，范围 [0, 1]：
- 0.9 = 该股票排名前 10%
- 0.5 = 该股票排名中等
- 0.1 = 该股票排名后 10%

**选股策略**：选择预测值 > 0.8 的股票（排名前 20%）

### Q2: 需要将预测值转换回收益吗？

**A**: **不需要**。排名预测直接用于选股，不需要转换回收益。

### Q3: Rank 转换和非对称损失可以同时使用吗？

**A**: 可以，但通常**只使用 Rank 转换就足够了**。如果仍然保守，再添加非对称损失。

### Q4: gamma 参数如何选择？

**A**: 
- 从 2.0 开始
- 观察验证集表现
- 如果过于激进，降低到 1.5
- 如果仍然保守，提高到 3.0

### Q5: 使用 Rank 转换后，IC 计算需要修改吗？

**A**: **不需要**。IC 计算使用的是预测值和标签的排名相关性，Rank 转换后仍然有效。

## 八、实施检查清单

- [ ] 更新 `config/data.yaml`，添加 `label_transform` 配置
- [ ] 更新 `config/model_lgb.yaml`，选择损失函数
- [ ] 更新 `config/model_mlp.yaml`，选择损失函数
- [ ] 运行训练，验证标签转换是否生效
- [ ] 检查日志输出，确认配置正确
- [ ] 运行回测，对比改进前后效果
- [ ] 分析预测值分布，确认模型不再保守
- [ ] 调优参数（gamma、method 等）

## 九、预期效果

实施后预期：
- ✅ 模型预测值在快速上涨期间更大
- ✅ 策略能更好地捕捉快速上涨趋势
- ✅ IC 和 IC-IR 提升
- ✅ 回测曲线在上涨期间能跟上基准

## 十、注意事项

1. **Rank 转换的优势**：更适合选股任务，对异常值更稳健
2. **非对称损失的局限**：可能增加过拟合风险，需要监控验证集
3. **参数调优**：需要根据实际数据调整 gamma 等参数
4. **回测验证**：改进后必须重新回测，验证效果


