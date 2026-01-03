# 评估性能优化指南

## 一、性能瓶颈分析

### 问题发现

评估模型性能比训练还耗时，主要原因是：

### 1. 训练集 IC 计算（最严重）

**位置**：`run_industry_train.py`, line 364-365

**问题**：
```python
# 重新训练一个临时模型用于评估
temp_model = IndustryGRUWrapper(model_config)
temp_model.fit(train_feat_train, train_lbl_train, None, None)  # ❌ 又训练一次！
```

**耗时原因**：
- 相当于训练了两次模型（主模型 + 临时模型）
- 如果训练需要 10 分钟，评估就需要 20 分钟！

**解决方案**：
- **推荐**：禁用训练集 IC 计算（`train_ic.enabled: false`）
- 或者：使用更快的评估方法（不重新训练）

---

### 2. 验证集预测（次严重）

**位置**：`run_industry_train.py`, line 394

**问题**：
```python
valid_pred = model.predict(valid_feat_norm, history_feat=train_feat_norm)
```

**耗时原因**：
1. **合并数据**：`pd.concat([history_feat, feat]).sort_index()` - 如果训练集很大，很耗时
2. **序列构建**：双重循环（instrument × datetime），效率低
3. **重复操作**：每个验证集样本都要 `xs()`, `sort_index()`, `get_loc()`

**优化方案**：
- 缓存序列构建结果
- 优化序列构建逻辑（向量化）
- 减少重复的数据操作

---

### 3. 序列构建方法（中等）

**位置**：`pytorch_industry_gru.py`, `_prepare_sequences()` 和 `_prepare_sequences_for_prediction()`

**问题**：
```python
# 双重循环，效率低
for instrument in instruments:
    inst_data = feat.xs(instrument, level=1, drop_level=False)  # 每次都要 xs
    inst_data = inst_data.sort_index()  # 每次都要排序
    for i in range(self.sequence_length, len(inst_data)):
        seq = inst_data.iloc[i - self.sequence_length:i].values  # 每次都要切片
```

**优化方案**：
- 预先排序和分组
- 使用向量化操作
- 批量构建序列

---

## 二、优化方案

### 方案 1：禁用训练集 IC 计算（推荐，立即生效）

**配置**：
```yaml
train_ic:
  enabled: false  # 禁用训练集 IC 计算
```

**效果**：
- ✅ 立即减少 50% 的评估时间
- ✅ 避免重新训练临时模型
- ✅ 训练集 IC 本身就不应该用于评估模型性能

**理由**：
- 训练集 IC 虚高，不能反映真实性能
- 验证集 IC 才是真实性能指标
- 节省大量时间

---

### 方案 2：优化验证集预测

**优化点**：
1. **缓存合并数据**：如果训练集不变，可以缓存合并结果
2. **优化序列构建**：使用向量化操作
3. **减少重复操作**：预先排序和分组

**实现**：
```python
# 优化前：每次都合并和排序
combined_feat = pd.concat([history_feat, feat]).sort_index()

# 优化后：如果数据未变化，使用缓存
if not hasattr(self, '_cached_combined_feat') or self._cached_combined_feat is None:
    self._cached_combined_feat = pd.concat([history_feat, feat]).sort_index()
combined_feat = self._cached_combined_feat
```

---

### 方案 3：优化序列构建（需要重构）

**优化思路**：
1. 预先按 instrument 分组和排序
2. 使用向量化操作构建序列
3. 批量处理

**实现**（需要较大改动）：
```python
# 优化前：双重循环
for instrument in instruments:
    for i in range(sequence_length, len(inst_data)):
        seq = inst_data.iloc[i - sequence_length:i].values

# 优化后：向量化
# 使用 numpy 的滑动窗口函数
from numpy.lib.stride_tricks import sliding_window_view
sequences = sliding_window_view(inst_data.values, (sequence_length, num_features))
```

---

## 三、立即优化（推荐）

### 步骤 1：禁用训练集 IC 计算

在 `config_industry_rotation.yaml` 中：

```yaml
train_ic:
  enabled: false  # 禁用训练集 IC 计算
```

**效果**：立即减少 50% 的评估时间

### 步骤 2：优化验证集预测（可选）

如果验证集预测仍然很慢，可以考虑：
1. 减少验证集样本数（用于快速评估）
2. 使用更小的 batch_size 进行预测
3. 优化序列构建逻辑

---

## 四、性能对比

### 优化前

```
训练：10 分钟
评估：
  - 训练集 IC（重新训练）：10 分钟 ❌
  - 验证集预测：5 分钟
  - 总计：15 分钟（比训练还慢！）
```

### 优化后（禁用训练集 IC）

```
训练：10 分钟
评估：
  - 训练集 IC：跳过 ✅
  - 验证集预测：5 分钟
  - 总计：5 分钟（减少 67%）
```

---

## 五、建议

### 立即执行

1. **禁用训练集 IC 计算**：
   ```yaml
   train_ic:
     enabled: false
   ```

2. **只关注验证集 IC**：
   - 验证集 IC 才是真实性能指标
   - 训练集 IC 虚高，没有参考价值

### 后续优化

1. **优化序列构建**：使用向量化操作
2. **缓存机制**：缓存合并数据
3. **批量处理**：优化批处理逻辑

---

## 六、总结

**主要问题**：
1. ❌ 训练集 IC 计算重新训练模型（最严重）
2. ⚠️ 验证集预测序列构建效率低（次严重）

**立即解决方案**：
- ✅ 禁用训练集 IC 计算（`train_ic.enabled: false`）
- ✅ 可以立即减少 50% 的评估时间

**长期优化**：
- 优化序列构建逻辑（向量化）
- 添加缓存机制
- 优化批处理

