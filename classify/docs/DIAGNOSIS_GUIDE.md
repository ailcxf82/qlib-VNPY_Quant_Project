# 预测问题诊断指南

## 一、诊断测试优先级

按优先级从高到低进行诊断：

### 1. 完美预测自测（最高优先级）

**目的**：验证 IC 计算和数据对齐是否正确

**测试方法**：
```python
# 令 pred=label，IC 应该接近 1.0
train_ic = rank_ic(train_label, train_label)  # 应该 ≈ 1.0
valid_ic = rank_ic(valid_label, valid_label)  # 应该 ≈ 1.0
```

**判断标准**：
- ✅ IC ≈ 1.0：IC 计算正常
- ❌ IC ≠ 1.0：IC 计算或数据对齐有问题

**如果失败**：
- 检查 `rank_ic` 函数实现
- 检查数据对齐逻辑（`align` 函数）
- 检查是否有 NaN 值影响

---

### 2. 取负号测试

**目的**：判断模型预测方向是否正确

**测试方法**：
```python
# 原始 IC
original_ic = rank_ic(pred, label)

# 取负号后的 IC
negative_ic = rank_ic(-pred, label)
```

**判断标准**：
- ✅ `abs(original_ic) > abs(negative_ic)`：方向正确
- ❌ `abs(negative_ic) > abs(original_ic)`：方向相反

**如果失败**：
- 检查模型输出是否需要取负号
- 检查标签定义是否正确（正数表示好还是坏）
- 检查损失函数是否优化了正确的方向

---

### 3. 单特征基线

**目的**：验证数据本身是否有基本信号

**测试方法**：
```python
# 使用单个特征（如 ret20）进行预测
feature = train_feat["ret_20"]  # 或 "$close / Ref($close, 20) - 1"

# 方法 1：直接使用特征值排序
ic_direct = rank_ic(feature, label)

# 方法 2：线性回归
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(feature.values.reshape(-1, 1), label.values)
pred = model.predict(feature.values.reshape(-1, 1))
ic_lr = rank_ic(pd.Series(pred, index=feature.index), label)
```

**判断标准**：
- ✅ IC > 0：数据有基本信号
- ❌ IC ≤ 0：数据可能没有信号，或特征选择有问题

**如果失败**：
- 检查特征是否正确提取
- 检查标签是否正确计算
- 尝试其他特征（如 pct_change, vol_chg_20 等）

---

### 4. 截面 rank 特征 vs 原始特征

**目的**：判断标准化方式是否重要

**测试方法**：
```python
# 原始特征
ic_raw = rank_ic(feature_raw, label)

# 截面 rank 特征（按日期分组排名）
feature_rank = feature_raw.groupby(level="datetime").transform(lambda x: x.rank(pct=True))
ic_rank = rank_ic(feature_rank, label)
```

**判断标准**：
- ✅ `abs(ic_rank - ic_raw) < 0.05`：标准化影响小
- ⚠️ `abs(ic_rank - ic_raw) > 0.05`：标准化影响大，需要考虑使用截面标准化

**如果差异大**：
- 启用截面标准化（cross_sectional_normalization）
- 使用 rank 方法而非 zscore

---

### 5. 按日截面 batch + ranking loss

**目的**：判断训练组织方式是否影响结果

**测试方法**：
- 修改训练代码，按日期分组进行 batch 训练
- 使用 ranking loss 而非 MSE loss
- 对比 IC 变化

**判断标准**：
- IC 提升：说明训练组织方式重要
- IC 不变：说明训练组织方式影响不大

---

## 二、使用方法

### 方法 1：使用诊断脚本

```bash
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

### 方法 2：在训练代码中添加诊断

在 `run_industry_train.py` 的训练循环中添加：

```python
# 在每个窗口训练后，进行诊断测试
from classify.scripts.diagnose_prediction_issues import (
    test_1_perfect_prediction,
    test_2_negative_prediction,
    test_3_single_feature_baseline,
)

# 测试 1：完美预测自测
test_1_perfect_prediction(train_lbl, valid_lbl)

# 测试 2：取负号测试
if has_valid:
    valid_pred = model.predict(valid_feat_norm, history_feat=train_feat_norm)
    test_2_negative_prediction(valid_pred, valid_lbl)

# 测试 3：单特征基线
test_3_single_feature_baseline(train_feat, train_lbl, valid_feat, valid_lbl)
```

---

## 三、常见问题及解决方案

### 问题 1：测试 1 失败（IC ≠ 1.0）

**可能原因**：
1. IC 计算函数有 bug
2. 数据对齐有问题
3. 有 NaN 值影响

**解决方案**：
```python
# 检查数据对齐
pred, label = pred.align(label, join="inner")
print(f"对齐后样本数: {len(pred)}")

# 检查 NaN 值
print(f"预测 NaN 数: {pred.isna().sum()}")
print(f"标签 NaN 数: {label.isna().sum()}")

# 清理 NaN
pred = pred.dropna()
label = label.dropna()
pred, label = pred.align(label, join="inner")
```

### 问题 2：测试 2 显示方向问题

**可能原因**：
1. 模型预测了相反的方向
2. 标签定义错误

**解决方案**：
```python
# 方案 1：在预测时取负号
pred = -model.predict(feat)

# 方案 2：修改损失函数
# 在 RankingLoss 中，确保优化方向正确

# 方案 3：检查标签定义
# 确认正数表示"好"还是"坏"
```

### 问题 3：测试 3 IC 为负

**可能原因**：
1. 数据本身没有信号
2. 特征选择有问题
3. 标签计算错误

**解决方案**：
```python
# 尝试不同的特征
features_to_try = [
    "$close / Ref($close, 20) - 1",  # ret20
    "$pct_change",  # 涨跌幅
    "$vol / Mean($vol, 20) - 1",  # 成交量变化
]

# 检查标签
label_expr = "Ref($close, -10) / $close - 1"  # 未来10天收益率
# 确认标签计算是否正确
```

### 问题 4：测试 4 显示标准化影响大

**解决方案**：
```yaml
# 在配置文件中启用截面标准化
cross_sectional_normalization:
  enabled: true
  method: "rank"  # 或 "zscore"
  groupby: "datetime"
```

---

## 四、诊断结果解读

### 理想情况

```
测试 1：完美预测自测
  ✅ 训练集 IC (pred=label): 1.000000
  ✅ 验证集 IC (pred=label): 1.000000

测试 2：取负号测试
  ✅ 原始 IC: 0.05
  ✅ 取负号后 IC: -0.05
  ✅ 方向正确

测试 3：单特征基线
  ✅ 验证集 IC: 0.03
  ✅ 数据有基本信号

测试 4：截面 rank vs 原始特征
  ✅ 标准化影响较小
```

### 有问题的情况

```
测试 1：完美预测自测
  ❌ 训练集 IC (pred=label): 0.95  # 应该接近 1.0
  ❌ 验证集 IC (pred=label): 0.92   # 应该接近 1.0
  → 需要检查 IC 计算或数据对齐

测试 2：取负号测试
  ⚠️  原始 IC: -0.02
  ⚠️  取负号后 IC: 0.03
  → 方向可能相反，需要取负号

测试 3：单特征基线
  ❌ 验证集 IC: -0.01
  → 数据可能没有信号，需要检查特征和标签

测试 4：截面 rank vs 原始特征
  ⚠️  标准化影响很大 (差异: 0.08)
  → 需要启用截面标准化
```

---

## 五、快速诊断命令

```bash
# 运行完整诊断
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml

# 查看诊断结果
# 根据结果判断问题所在，然后采取相应的修复措施
```

---

## 六、修复优先级

根据诊断结果，按以下优先级修复：

1. **测试 1 失败** → 立即修复 IC 计算或数据对齐
2. **测试 2 显示方向问题** → 修改预测方向或损失函数
3. **测试 3 IC 为负** → 检查特征和标签，可能需要重新设计
4. **测试 4 显示标准化影响大** → 启用截面标准化
5. **测试 5** → 优化训练组织方式

---

## 七、注意事项

1. **数据质量**：确保数据没有缺失值或异常值
2. **时间对齐**：确保预测和标签的时间对齐正确
3. **样本数量**：确保有足够的样本进行统计测试
4. **多次测试**：在不同窗口上重复测试，确保结果稳定

