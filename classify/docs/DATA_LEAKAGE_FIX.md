# 数据泄露修复方案

## 一、问题诊断

### 训练结果分析

```
- 平均验证集 IC: 0.0162（正常，行业轮动预测难度大）
- 验证集 IC 标准差: 0.1094（波动较大，但可接受）
- 平均训练集 IC: 0.7423（⚠️ 异常高！）
- 训练集 IC 标准差: 0.0383
```

**问题**：训练集 IC 0.74 在行业轮动预测中非常不正常，说明存在数据泄露或评估方法问题。

---

## 二、发现的问题

### 2.1 ❌ 训练集预测使用了训练数据本身（最严重）

**问题代码**：
```python
# 当前代码（错误）
train_pred = model.predict(train_feat_norm)  # 使用训练数据本身进行预测
train_ic = _rank_ic(train_pred, train_lbl)
```

**问题分析**：
1. 模型在训练时已经见过 `train_feat_norm` 和 `train_lbl`
2. 使用相同的训练数据进行预测，模型会"记住"训练数据
3. 导致训练集 IC 虚高（0.74），不能反映真实泛化能力

**影响**：
- 训练集 IC 虚高，无法判断模型真实性能
- 可能掩盖其他数据泄露问题

### 2.2 ✅ Purge Gap 已实现（但需要验证）

**当前实现**：
```python
purge_gap = sequence_length + label_future_days  # 60 + 10 = 70天
train_end_adjusted = train_end_original - pd.Timedelta(days=purge_gap)
valid_start = train_end_original + pd.Timedelta(days=1)
```

**检查结果**：
- ✅ Purge Gap 计算正确
- ✅ 训练集结束日期提前了 purge_gap 天
- ⚠️ 需要验证实际窗口切分是否正确

### 2.3 ✅ 时间序列归一化已修复

**当前实现**：
```python
# 对每个训练窗口单独计算归一化参数
train_feat_norm, norm_mean, norm_std = QlibFeaturePipeline.normalize_features(train_feat)

# 验证集使用训练集的归一化参数
valid_feat_norm = (valid_feat - norm_mean) / norm_std
```

**检查结果**：
- ✅ 每个训练窗口单独计算归一化参数
- ✅ 验证集使用训练集的归一化参数
- ✅ 无数据泄露

### 2.4 ✅ 截面标准化配置正确

**当前配置**：
```yaml
cross_sectional_normalization:
  enabled: true
  method: "zscore"
  groupby: "datetime"  # 按日期分组，不会使用未来数据
```

**检查结果**：
- ✅ 按日期分组进行截面标准化
- ✅ 每个日期内的标准化只使用当日截面数据
- ✅ 无数据泄露

---

## 三、修复方案

### 3.1 修复训练集预测问题（最重要）

**方案 1：时间序列交叉验证（推荐）**

**原理**：
- 将训练集分为两部分：前 80% 用于训练，后 20% 用于评估
- 使用前 80% 的数据训练模型，在后 20% 的数据上评估
- 这样评估的是模型在"未来"数据上的表现，更接近真实场景

**实现**：
```python
# 将训练集分为两部分
split_idx = int(len(train_feat_norm) * 0.8)
train_feat_train = train_feat_norm.iloc[:split_idx]
train_lbl_train = train_lbl.iloc[:split_idx]
train_feat_eval = train_feat_norm.iloc[split_idx:]
train_lbl_eval = train_lbl.iloc[split_idx:]

# 使用前 80% 的数据训练临时模型
temp_model = IndustryGRUWrapper(model_config)
temp_model.fit(train_feat_train, train_lbl_train, None, None)

# 在后 20% 的数据上评估
train_pred_eval = temp_model.predict(train_feat_eval, history_feat=train_feat_train)
train_ic = _rank_ic(train_pred_eval, train_lbl_eval)
```

**优点**：
- 评估的是模型在"未来"数据上的表现
- 更接近真实交易场景
- 训练集 IC 会更接近验证集 IC

**缺点**：
- 需要额外训练一个临时模型（计算成本增加）
- 训练集 IC 会下降（但更真实）

**方案 2：只报告验证集 IC（简单）**

**原理**：
- 不计算训练集 IC，只报告验证集 IC
- 训练集 IC 本身就不应该用于评估模型性能

**实现**：
```python
# 不计算训练集 IC
train_ic = float("nan")

# 只报告验证集 IC
if has_valid:
    valid_ic = _rank_ic(valid_pred, valid_lbl)
    logger.info("窗口 %d: 验证集 IC=%.4f", idx, valid_ic)
```

**优点**：
- 实现简单
- 避免误导性的训练集 IC

**缺点**：
- 无法监控训练过程（但可以通过验证集 IC 监控）

---

## 四、验证方法

### 4.1 错位标签实验

**原理**：
- 将标签整体向后错位一段（如 200 天）
- 如果训练 IC 仍然很高（>0.5），说明存在数据泄露
- 如果训练 IC 接近 0，说明没有数据泄露

**实现**：
```python
# 在训练前，将标签整体向后错位
labels_shifted = labels.shift(200)  # 向后错位 200 天

# 使用错位的标签进行训练
model.fit(train_feat_norm, labels_shifted, ...)

# 如果训练 IC 仍然很高，说明有数据泄露
```

**预期结果**：
- 无数据泄露：训练 IC ≈ 0（标签和特征完全错位）
- 有数据泄露：训练 IC > 0.5（说明特征中包含了未来信息）

### 4.2 检查 Purge Gap 实现

**验证方法**：
1. 打印每个窗口的实际日期范围
2. 检查训练集结束日期和验证集开始日期之间的间隔
3. 确认间隔 >= purge_gap

**代码**：
```python
logger.info("窗口 %d 实际训练集日期范围: %s ~ %s", idx, tmin, tmax)
logger.info("窗口 %d 实际验证集日期范围: %s ~ %s", idx, vmin, vmax)
logger.info("训练集结束到验证集开始的间隔: %d 天", (vmin - tmax).days)
```

---

## 五、修复后的预期效果

### 修复前：
- 训练集 IC: 0.74（虚高）
- 验证集 IC: 0.016（正常）

### 修复后（使用时间序列交叉验证）：
- 训练集 IC: 0.01-0.05（更真实，接近验证集 IC）
- 验证集 IC: 0.016（不变）

**说明**：
- 训练集 IC 下降是正常的，因为现在评估的是模型在"未来"数据上的表现
- 训练集 IC 和验证集 IC 应该接近，说明模型泛化能力正常

---

## 六、实施步骤

### 步骤 1：修复训练集预测逻辑

1. 修改 `run_industry_train.py` 中的训练集 IC 计算逻辑
2. 使用时间序列交叉验证（前 80% 训练，后 20% 评估）

### 步骤 2：运行数据泄露检查脚本

```bash
python classify/scripts/check_data_leakage_industry.py --config classify/config_industry_rotation.yaml
```

### 步骤 3：重新训练并验证

1. 重新训练模型
2. 检查训练集 IC 是否下降到合理范围（0.01-0.05）
3. 验证集 IC 应该保持不变（0.016）

### 步骤 4：运行错位标签实验（可选）

1. 在训练代码中添加错位标签选项
2. 运行实验验证是否有其他数据泄露

---

## 七、总结

**主要问题**：
1. ❌ **训练集预测使用了训练数据本身**（导致 IC 虚高 0.74）
2. ✅ Purge Gap 已实现（需要验证）
3. ✅ 时间序列归一化已修复
4. ✅ 截面标准化配置正确

**修复优先级**：
1. **【高优先级】** 修复训练集预测逻辑（使用时间序列交叉验证）
2. **【中优先级】** 验证 Purge Gap 实现
3. **【低优先级】** 运行错位标签实验

**预期效果**：
- 训练集 IC 下降到 0.01-0.05（更真实）
- 训练集 IC 和验证集 IC 接近（说明模型泛化能力正常）

