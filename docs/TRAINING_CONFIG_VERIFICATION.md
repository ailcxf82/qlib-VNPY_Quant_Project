# 训练配置验证指南：如何确认训练使用了 Rank 转换

## 一、验证方法

### 方法 1：检查训练日志（最直接）⭐⭐⭐

运行训练时，观察日志输出：

```bash
python run_train.py --config config/pipeline.yaml
```

**应该看到的日志**：

1. **特征管线构建时**：
   ```
   INFO - 标签已转换为排名（方法: percentile, 分组: datetime）
   ```

2. **训练开始时**（已添加）：
   ```
   INFO - 训练使用 Rank 转换后的标签（范围应在 [0, 1] 之间）
   INFO - 标签值统计: min=0.000000, max=1.000000, mean=0.500000
   ```

**如果没有看到这些日志**：
- ❌ 说明配置未生效
- 检查 `config/data.yaml` 中 `label_transform.enabled` 是否为 `true`

### 方法 2：运行验证脚本

```bash
# 检查配置和代码逻辑
python scripts/check_training_config_simple.py

# 检查预测数据来源
python scripts/check_prediction_source.py

# 综合验证
python scripts/verify_training_label_transform.py
```

### 方法 3：测试转换函数

```bash
# 测试转换函数本身
python scripts/test_label_transform_function.py

# 模拟训练过程
python scripts/test_training_with_sample_data.py
```

## 二、代码检查点

### 2.1 配置文件

**文件**: `config/data.yaml`

```yaml
data:
  label_transform:
    enabled: true              # 必须为 true
    method: percentile         # 推荐使用 percentile
    groupby: datetime          # 推荐按日期分组
```

### 2.2 特征管线代码

**文件**: `feature/qlib_feature_pipeline.py`

**关键代码**（第 349-359 行）：
```python
# 标签转换：支持转换为排名百分位
label_transform = self.feature_cfg.get("label_transform", {})
if label_transform.get("enabled", False):
    from utils.label_transform import transform_to_rank
    method = label_transform.get("method", "percentile")
    groupby = label_transform.get("groupby", "datetime")
    label = transform_to_rank(label, method=method, groupby=groupby)
    logger.info("标签已转换为排名（方法: %s, 分组: %s）", method, groupby)
    self._label_is_rank = True
else:
    self._label_is_rank = False
```

**检查点**：
- ✅ 读取 `label_transform` 配置
- ✅ 检查 `enabled` 标志
- ✅ 调用 `transform_to_rank` 函数
- ✅ 设置 `_label_is_rank` 标志
- ✅ 输出转换日志

### 2.3 训练器代码

**文件**: `trainer/trainer.py`

**关键代码**（第 105-120 行）：
```python
def train(self):
    self.pipeline.build()  # 这里会执行 Rank 转换
    features, labels = self.pipeline.get_all()  # 获取转换后的标签
    
    # 检查标签转换是否生效（已添加）
    label_is_rank = getattr(self.pipeline, "_label_is_rank", False)
    if label_is_rank:
        logger.info("训练使用 Rank 转换后的标签（范围应在 [0, 1] 之间）")
        logger.info("标签值统计: min=%.6f, max=%.6f, mean=%.6f", 
                   labels.min(), labels.max(), labels.mean())
        if labels.min() < 0 or labels.max() > 1:
            logger.warning("标签值不在 [0, 1] 范围内！可能转换未生效")
```

**检查点**：
- ✅ 调用 `pipeline.build()`
- ✅ 调用 `pipeline.get_all()`
- ✅ 检查 `_label_is_rank` 标志
- ✅ 输出标签值统计

## 三、验证清单

运行训练前，确认：

- [ ] **配置已启用**: `config/data.yaml` 中 `label_transform.enabled = true`
- [ ] **代码已更新**: `feature/qlib_feature_pipeline.py` 包含转换逻辑
- [ ] **训练器已更新**: `trainer/trainer.py` 包含标签检查

运行训练后，检查：

- [ ] **日志输出**: 看到 "标签已转换为排名" 的提示
- [ ] **标签值范围**: 日志显示标签值在 [0, 1] 范围内
- [ ] **无警告**: 没有 "标签值不在 [0, 1] 范围内" 的警告

## 四、常见问题排查

### 问题 1：训练日志中没有转换提示

**可能原因**：
1. 配置未启用：`label_transform.enabled = false` 或未配置
2. 配置文件路径错误：使用了错误的配置文件
3. 代码未更新：`feature/qlib_feature_pipeline.py` 未包含转换逻辑

**解决方法**：
1. 检查 `config/data.yaml` 配置
2. 确认使用的配置文件路径
3. 检查代码是否已更新

### 问题 2：日志显示转换了，但标签值不在 [0, 1] 范围内

**可能原因**：
1. 转换函数有 bug
2. 转换后的标签被后续处理覆盖
3. 数据问题（如 NaN 值）

**解决方法**：
1. 运行 `python scripts/test_label_transform_function.py` 测试转换函数
2. 检查 `feature/qlib_feature_pipeline.py` 中转换后的处理逻辑
3. 检查数据质量

### 问题 3：预测值不在 [0, 1] 范围内

**可能原因**：
1. 使用了旧模型（在配置更新之前训练的）
2. 预测时配置未生效
3. 模型输出需要后处理

**解决方法**：
1. 确认模型训练时间在配置更新之后
2. 重新训练模型
3. 检查预测流程

## 五、完整验证流程

### 步骤 1：检查配置

```bash
python scripts/check_training_config_simple.py
```

**预期输出**：
```
[OK] 标签转换已启用
[OK] 特征管线已正确实现 Rank 转换逻辑
[OK] 训练器正确调用了特征管线
```

### 步骤 2：运行训练

```bash
python run_train.py --config config/pipeline.yaml > training.log 2>&1
```

**检查日志文件**：
```bash
# 查找转换提示
grep "标签已转换为排名" training.log

# 查找标签值统计
grep "标签值统计" training.log

# 查找警告
grep "标签值不在" training.log
```

### 步骤 3：验证预测结果

```bash
# 重新预测
python run_predict.py --config config/pipeline.yaml --tag auto

# 检查预测值
python scripts/check_prediction_source.py
```

**预期输出**：
```
[OK] 预测值在 [0, 1] 之间，可能使用了 Rank 转换
```

## 六、调试技巧

### 技巧 1：添加详细日志

在 `feature/qlib_feature_pipeline.py` 的 `build()` 方法中添加：

```python
# 转换前
logger.info("转换前标签统计: min=%.6f, max=%.6f", label.min(), label.max())

# 转换后
logger.info("转换后标签统计: min=%.6f, max=%.6f", label.min(), label.max())
```

### 技巧 2：保存转换后的标签

在 `trainer/trainer.py` 中添加：

```python
# 保存标签样本用于检查
if idx == 0:  # 只保存第一个窗口
    sample_labels = labels.head(1000)
    sample_labels.to_csv("data/logs/sample_labels.csv")
    logger.info("已保存标签样本到 data/logs/sample_labels.csv")
```

### 技巧 3：检查模型元数据

修改 `models/lightgbm_model.py` 的 `save()` 方法，记录标签转换信息：

```python
# 从 pipeline 获取标签转换信息
label_transform_info = {
    "enabled": getattr(self, "_label_transform_enabled", None),
    "method": getattr(self, "_label_transform_method", None),
}

meta = {
    "config": self.config,
    "feature_names": self.feature_names,
    "label_transform": label_transform_info,  # 添加这个
}
```

## 七、总结

✅ **代码逻辑正确**: 训练逻辑已正确实现 Rank 转换  
✅ **配置已启用**: `config/data.yaml` 中 `label_transform.enabled = true`  
✅ **已添加检查**: 训练时会输出标签值统计  

**下一步**：
1. 重新运行训练，观察日志输出
2. 确认看到 "标签已转换为排名" 的提示
3. 确认标签值在 [0, 1] 范围内
4. 如果仍有问题，检查详细日志


