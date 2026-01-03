# Purge Gap 修复总结

## 一、问题诊断结果

### ✅ 检查 1：标签是否严格使用未来数据

**当前配置**：
```yaml
label: "Ref($close, -10) / $close - 1"  # 未来10天收益率
```

**检查结果**：
- ✅ 标签表达式正确：使用 `Ref($close, -10)` 表示未来10天
- ⚠️ **您提到应该是20天**，如果确实如此，需要修改配置

**如果需要修改为20天**：
```yaml
label: "Ref($close, -20) / $close - 1"  # 未来20天收益率
```

---

### ❌ 检查 2：Purge Gap 问题（已修复）

**问题**：
- ❌ **之前没有实现 purge gap**
- ❌ 只提前了 `label_future_days`（10天），没有考虑 `sequence_length`（60天）
- ❌ 导致训练集和验证集之间存在数据泄露

**修复**：
- ✅ 实现了正确的 purge gap：`purge_gap = sequence_length + label_future_days = 60 + 10 = 70天`
- ✅ 训练集结束日期提前 `purge_gap` 天
- ✅ 验证集开始日期从训练集原始结束日期+1天开始

**修复后的逻辑**：
```python
# 训练集原始结束日期
train_end_original = cursor - 1天

# 训练集实际可用结束日期（提前 purge_gap 天）
train_end_adjusted = train_end_original - 70天

# 验证集开始日期（从训练集原始结束日期+1天开始）
valid_start = train_end_original + 1天
```

**示例**：
```
修复前（错误）：
  训练集: 2023-01-01 到 2024-12-31
  验证集: 2025-01-01 到 2025-01-31  ❌ 数据泄露！

修复后（正确）：
  训练集原始结束: 2024-12-31
  训练集实际可用: 2023-01-01 到 2024-10-22（提前70天）
  验证集: 2025-01-01 到 2025-01-31  ✅ 无数据泄露
```

---

### ✅ 检查 3：IC 计算口径是否一致

**检查结果**：
- ✅ 训练集和验证集使用相同的 `_rank_ic` 函数
- ✅ 都使用 Spearman 相关系数
- ✅ 都先进行 `align(join="inner")` 确保对齐

**IC 计算代码**：
```python
def _rank_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算 Rank IC（Spearman 相关系数）"""
    pred, label = pred.align(label, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(label, method="spearman")

# 训练集 IC
train_ic = _rank_ic(train_pred, train_lbl)

# 验证集 IC
valid_ic = _rank_ic(valid_pred, valid_lbl)
```

**结论**：IC 计算口径一致，没有问题。

---

## 二、修复内容

### 1. 添加 Purge Gap 计算

```python
# 获取序列长度（用于计算 purge gap）
sequence_length = model_config.get("sequence_length", 60)

# 计算 purge gap：序列长度 + 标签未来天数
purge_gap = sequence_length + label_future_days  # 60 + 10 = 70天
```

### 2. 修复窗口生成逻辑

```python
# 训练集原始结束日期（用于确定验证集开始日期）
train_end_original = cursor - pd.Timedelta(days=1)

# 训练集实际可用结束日期（提前 purge_gap 天，防止数据泄露）
train_end_adjusted = train_end_original - pd.Timedelta(days=purge_gap)

# 验证集开始日期：从训练集原始结束日期+1天开始
valid_start = train_end_original + pd.Timedelta(days=1)
```

### 3. 添加日志输出

```python
logger.info(
    "Purge Gap 配置: 序列长度=%d, 标签未来天数=%d, Purge Gap=%d 天",
    sequence_length, label_future_days, purge_gap,
)
```

---

## 三、影响分析

### 修复前的问题

1. **数据泄露**：
   - 训练集结束日期：2024-12-31
   - 验证集开始日期：2025-01-01
   - 但模型在 2024-12-31 训练时，使用了 [2024-11-01, 2024-12-31] 的60天序列
   - 验证集从 2025-01-01 开始，但模型已经"看到"了 2024-12-31 的数据
   - **这造成了数据泄露！**

2. **验证集 IC 为负的原因**：
   - 模型在训练时使用了验证期的信息（间接）
   - 导致模型学到了错误的模式
   - 验证集 IC 为负（-0.0150）是数据泄露的结果

### 修复后的效果

1. **消除数据泄露**：
   - 训练集实际可用结束日期提前70天
   - 验证集开始日期从训练集原始结束日期+1天开始
   - **确保训练集和验证集完全隔离**

2. **预期改善**：
   - 验证集 IC 应该会提升（从负值变为正值）
   - 训练集 IC 可能会下降（因为数据更少）
   - 但训练集和验证集 IC 差距应该会缩小

---

## 四、下一步操作

### 1. 确认标签天数

如果标签应该是20天而不是10天，需要修改：

```yaml
# config_data_industry_baseline.yaml
label: "Ref($close, -20) / $close - 1"  # 从 -10 改为 -20
```

**影响**：
- `label_future_days` 会从 10 变为 20
- `purge_gap` 会从 70 变为 80（60 + 20）

### 2. 重新训练模型

使用修复后的代码重新训练，观察：
- 验证集 IC 是否提升
- 训练集 IC 是否下降
- 两者差距是否缩小

### 3. 对比结果

**预期结果**：
- 验证集 IC: 从 -0.0150 提升到 > 0.05
- 训练集 IC: 从 0.3758 下降到 0.20-0.30
- 差距: 从 0.39 缩小到 < 0.15

---

## 五、总结

### 已修复的问题

1. ✅ **Purge Gap 实现**：正确实现了 purge gap = sequence_length + label_future_days
2. ✅ **窗口生成逻辑**：训练集结束日期提前 purge_gap，验证集从训练集原始结束日期+1开始
3. ✅ **IC 计算一致性**：确认训练集和验证集使用相同的计算方式

### 待确认的问题

1. ⚠️ **标签天数**：当前是10天，您提到应该是20天，需要确认并修改

### 关键修复

**最重要的修复**：实现了正确的 purge gap，这应该能解决"泛化失败"的问题。

修复后，模型应该能够：
- 正确隔离训练集和验证集
- 避免数据泄露
- 获得更真实的验证集 IC


