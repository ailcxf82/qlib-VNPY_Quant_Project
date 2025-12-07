# 归一化数据泄露修复总结

## 修复内容

已成功修复非时序归一化导致的数据泄露问题。

## 修改的文件

### 1. `feature/qlib_feature_pipeline.py`

**修改**：
- 移除了 `build()` 方法中的全局归一化
- 保存原始特征，而不是归一化后的特征
- 添加了 `normalize_features()` 静态方法，用于按窗口归一化

**关键代码**：
```python
# 修复前：使用全局归一化
self._fit_norm(features)
norm_feat = self._transform(features)
self.features_df = norm_feat

# 修复后：保存原始特征
self.features_df = features
```

### 2. `trainer/trainer.py`

**修改**：
- 在每个训练窗口单独计算归一化参数
- 验证集使用训练集的归一化参数（不能使用验证集数据计算）
- 保存归一化参数到模型元数据文件

**关键代码**：
```python
# 对每个训练窗口单独计算归一化参数
train_feat_norm, norm_mean, norm_std = self.pipeline.normalize_features(train_feat)

# 验证集使用训练集的归一化参数
if has_valid:
    valid_feat_norm = (valid_feat - norm_mean) / norm_std
    valid_feat_norm = valid_feat_norm.clip(-5, 5)

# 保存归一化参数
norm_meta = {
    "feature_mean": norm_mean.to_dict(),
    "feature_std": norm_std.to_dict(),
    ...
}
```

### 3. `predictor/predictor.py`

**修改**：
- 在 `load_models()` 时加载归一化参数
- 在 `predict()` 时使用训练时的归一化参数对特征进行归一化

**关键代码**：
```python
# 加载归一化参数
norm_meta_path = os.path.join(model_dir, f"{tag}_norm_meta.json")
if os.path.exists(norm_meta_path):
    with open(norm_meta_path, "r", encoding="utf-8") as fp:
        norm_meta = json.load(fp)
    self._norm_mean = pd.Series(norm_meta["feature_mean"])
    self._norm_std = pd.Series(norm_meta["feature_std"])

# 预测时使用归一化参数
features_norm = (features - self._norm_mean) / self._norm_std
```

## 修复效果

### 修复前的问题：
- ❌ 使用整个数据集的均值和标准差进行归一化
- ❌ 训练时使用了未来数据的信息（测试集的统计量）
- ❌ 导致回测结果过于乐观，实际部署时性能下降

### 修复后的改进：
- ✅ 每个训练窗口使用该窗口内的数据计算归一化参数
- ✅ 验证集使用训练集的归一化参数（不泄露验证集信息）
- ✅ 预测时使用训练时的归一化参数（保持一致性）
- ✅ 完全避免数据泄露

## 使用说明

### 训练

训练流程无需修改，系统会自动：
1. 对每个训练窗口单独计算归一化参数
2. 保存归一化参数到 `{tag}_norm_meta.json`

```bash
python run_train.py --config config/pipeline.yaml
```

### 预测

预测流程无需修改，系统会自动：
1. 加载最近训练窗口的归一化参数
2. 使用归一化参数对特征进行归一化

```bash
python run_predict.py --start 2019-10-01 --end 2021-10-01
```

## 注意事项

1. **旧模型兼容性**：
   - 旧模型（修复前训练的）没有归一化参数文件
   - 预测时会警告并使用原始特征（可能导致不准确）
   - 建议重新训练模型

2. **归一化参数文件**：
   - 文件位置：`data/models/{tag}_norm_meta.json`
   - 包含：`feature_mean`、`feature_std`、训练窗口信息

3. **特征列匹配**：
   - 如果预测时的特征与训练时不完全一致，会警告并用0填充缺失特征
   - 建议确保特征配置一致

## 验证方法

修复后，可以通过以下方式验证：

1. **回测对比**：
   - 修复前后的回测结果应该有差异
   - 修复后可能表现下降，但更真实

2. **时间分割测试**：
   - 使用前70%数据训练，后30%数据测试
   - 对比修复前后的测试集表现

3. **滚动窗口IC**：
   - 检查修复后各窗口的IC是否更稳定

## 相关文档

- 详细分析：`docs/DATA_LEAKAGE_ANALYSIS.md`
- 检查脚本：`scripts/check_data_leakage.py`

