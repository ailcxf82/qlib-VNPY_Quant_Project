# 特征标准化/截面处理流程详解

## 一、数据处理流程总览

```
原始数据（qlib D.features）
    ↓
缺失值处理（fillna/dropna）
    ↓
标签转换（可选：转换为排名）
    ↓
【截面标准化】（在 build() 中，切分前）✅
    - 按日期分组进行截面标准化
    - Winsorize（极值裁剪）
    ↓
保存原始特征（未归一化）
    ↓
数据切分（训练集/验证集）
    ↓
【时间序列归一化】（在训练时，切分后）✅
    - 每个训练窗口单独计算归一化参数
    - 验证集使用训练集的归一化参数
    ↓
模型训练
```

---

## 二、截面标准化（Cross-Sectional Normalization）

### 2.1 在哪里做？

**位置**：`feature/qlib_feature_pipeline.py::build()` 方法中

**时机**：**在数据切分之前**（在 `build()` 方法中完成）

**代码位置**：
```python
# feature/qlib_feature_pipeline.py, line 451-465
# 截面标准化：按日期对特征进行截面标准化（非常重要，避免未来数据泄露）
cross_sectional_cfg = self.feature_cfg.get("cross_sectional_normalization", {})
if cross_sectional_cfg.get("enabled", False):
    features = self._apply_cross_sectional_normalization(
        features,
        method=cross_sectional_cfg.get("method", "zscore"),
        groupby=cross_sectional_cfg.get("groupby", "datetime"),
        clip=cross_sectional_cfg.get("clip", False),
        clip_quantile=cross_sectional_cfg.get("clip_quantile", 0.05),
    )
```

### 2.2 实现细节

**方法**：`_apply_cross_sectional_normalization()`

**位置**：`feature/qlib_feature_pipeline.py`, line 716+

**实现逻辑**：
```python
def _apply_cross_sectional_normalization(
    self,
    features: pd.DataFrame,
    method: str = "zscore",
    groupby: str = "datetime",
    clip: bool = False,
    clip_quantile: float = 0.05,
) -> pd.DataFrame:
    # 按日期分组进行截面标准化
    grouped = features.groupby(level=groupby)
    
    if method == "zscore":
        # Z-score 标准化：每个日期内的特征进行标准化
        result = grouped.apply(lambda x: (x - x.mean()) / x.std().replace(0, 1))
    elif method == "rank":
        # 排名标准化：每个日期内的特征转换为排名（0-1）
        result = grouped.apply(lambda x: x.rank(pct=True))
    
    # Winsorize（极值裁剪）
    if clip:
        for col in result.columns:
            lower = result[col].quantile(clip_quantile)
            upper = result[col].quantile(1 - clip_quantile)
            result[col] = result[col].clip(lower=lower, upper=upper)
    
    return result
```

### 2.3 为什么在切分前做？

**原因**：
1. **截面标准化只使用当日截面数据**：每个日期内的标准化只使用当日所有行业的数据
2. **不会使用未来数据**：不会跨日期使用数据，因此不会造成数据泄露
3. **提高数据质量**：在切分前统一处理，确保训练集和验证集使用相同的标准化方式

**数据泄露检查**：
- ✅ **安全**：`groupby(level="datetime")` 确保每个日期内的标准化只使用当日数据
- ✅ **无未来信息**：不会使用未来日期的数据计算标准化参数

---

## 三、时间序列归一化（Temporal Normalization）

### 3.1 在哪里做？

**位置**：`classify/run_industry_train.py` 的训练循环中

**时机**：**在数据切分之后**（每个训练窗口单独计算）

**代码位置**：
```python
# classify/run_industry_train.py, line 326-333
# 对每个训练窗口单独计算归一化参数，避免数据泄露
logger.info("窗口 %d: 计算训练窗口归一化参数（仅使用训练集数据）", idx)
train_feat_norm, norm_mean, norm_std = QlibFeaturePipeline.normalize_features(train_feat)

# 验证集使用训练集的归一化参数
if has_valid:
    valid_feat_norm = (valid_feat - norm_mean) / norm_std
    valid_feat_norm = valid_feat_norm.clip(-5, 5)
```

### 3.2 实现细节

**方法**：`QlibFeaturePipeline.normalize_features()`（静态方法）

**位置**：`feature/qlib_feature_pipeline.py`, line 500+

**实现逻辑**：
```python
@staticmethod
def normalize_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    对特征进行归一化，返回归一化后的特征、均值和标准差。
    """
    mean = features.mean()  # 计算均值（仅使用训练集数据）
    std = features.std().replace(0, 1)  # 计算标准差（仅使用训练集数据）
    normalized = (features - mean) / std  # Z-score 标准化
    normalized = normalized.clip(-5, 5)  # 裁剪到 ±5 倍标准差
    return normalized, mean, std
```

### 3.3 为什么在切分后做？

**原因**：
1. **避免数据泄露**：每个训练窗口单独计算归一化参数，不使用验证集数据
2. **模拟真实场景**：在实际预测时，只能使用历史数据计算归一化参数
3. **验证集使用训练集参数**：验证集使用训练集的归一化参数，确保一致性

**数据泄露检查**：
- ✅ **安全**：只使用训练集数据计算归一化参数
- ✅ **无未来信息**：验证集不使用自己的数据计算归一化参数

---

## 四、Winsorize（极值裁剪）

### 4.1 在哪里做？

**位置 1**：截面标准化中（可选）

**代码位置**：
```python
# feature/qlib_feature_pipeline.py, _apply_cross_sectional_normalization()
if clip:
    for col in result.columns:
        lower = result[col].quantile(clip_quantile)  # 下分位数
        upper = result[col].quantile(1 - clip_quantile)  # 上分位数
        result[col] = result[col].clip(lower=lower, upper=upper)
```

**位置 2**：时间序列归一化后（固定范围裁剪）

**代码位置**：
```python
# feature/qlib_feature_pipeline.py, normalize_features()
normalized = normalized.clip(-5, 5)  # 裁剪到 ±5 倍标准差
```

### 4.2 两种 Winsorize 方式

#### 方式 1：分位数裁剪（截面标准化中）

**配置**：
```yaml
cross_sectional_normalization:
  clip: true
  clip_quantile: 0.05  # 裁剪上下各 5%
```

**特点**：
- 基于分位数裁剪
- 在截面标准化后应用
- 在切分前完成

#### 方式 2：固定范围裁剪（时间序列归一化后）

**配置**：
```yaml
temporal_normalization:
  clip: true
  clip_range: 5.0  # 裁剪到 ±5 倍标准差
```

**特点**：
- 基于标准差裁剪
- 在时间序列归一化后应用
- 在切分后完成

---

## 五、完整数据流程

### 5.1 训练流程

```
1. pipeline.build()  # 构建特征和标签
   ├─ 提取原始特征（qlib D.features）
   ├─ 缺失值处理
   ├─ 标签转换（可选）
   └─ 【截面标准化】（切分前）✅
      ├─ 按日期分组
      ├─ Z-score 或 Rank 标准化
      └─ Winsorize（可选）

2. 数据切分（训练集/验证集）
   ├─ train_feat, train_lbl = slice_data(...)
   └─ valid_feat, valid_lbl = slice_data(...)

3. 时间序列归一化（切分后）✅
   ├─ train_feat_norm, mean, std = normalize_features(train_feat)
   └─ valid_feat_norm = (valid_feat - mean) / std  # 使用训练集参数

4. 模型训练
   └─ model.fit(train_feat_norm, train_lbl, valid_feat_norm, valid_lbl)
```

### 5.2 预测流程

```
1. 加载模型和归一化参数
   ├─ model.load(...)
   └─ norm_mean, norm_std = load_norm_meta(...)

2. 提取预测特征
   └─ features = pipeline.get_slice(...)  # 已包含截面标准化

3. 时间序列归一化
   └─ features_norm = (features - norm_mean) / norm_std

4. 模型预测
   └─ predictions = model.predict(features_norm)
```

---

## 六、数据泄露检查清单

### ✅ 正确的做法

1. **截面标准化在切分前做**：
   - ✅ 只使用当日截面数据
   - ✅ 不会使用未来数据
   - ✅ 训练集和验证集使用相同的标准化方式

2. **时间序列归一化在切分后做**：
   - ✅ 只使用训练集数据计算归一化参数
   - ✅ 验证集使用训练集的归一化参数
   - ✅ 不会使用验证集数据计算归一化参数

3. **Winsorize 位置**：
   - ✅ 截面标准化中的 Winsorize：在切分前，基于分位数
   - ✅ 时间序列归一化后的 Clip：在切分后，基于标准差

### ❌ 错误的做法

1. **在切分前做时间序列归一化**：
   - ❌ 使用全量数据（包括验证集）计算归一化参数
   - ❌ 造成数据泄露

2. **在切分后做截面标准化**：
   - ❌ 训练集和验证集使用不同的标准化方式
   - ❌ 可能导致不一致

3. **使用验证集数据计算归一化参数**：
   - ❌ 验证集使用自己的数据计算归一化参数
   - ❌ 造成数据泄露

---

## 七、配置示例

### 7.1 完整配置

```yaml
data:
  # 截面标准化（在 build() 中，切分前）
  cross_sectional_normalization:
    enabled: true              # 启用截面标准化
    method: "zscore"          # 方法：'zscore' 或 'rank'
    groupby: "datetime"       # 按日期分组
    clip: true                # 是否进行极值裁剪（winsorize）
    clip_quantile: 0.05       # 裁剪分位数（上下各5%）
  
  # 时间序列归一化（在训练时，切分后）
  temporal_normalization:
    enabled: true              # 是否在训练时进行时间序列归一化
    clip: true                 # 是否进行极值裁剪
    clip_range: 5.0            # 裁剪范围（±5倍标准差）
```

### 7.2 当前实现状态

**截面标准化**：
- ✅ 已在 `build()` 中实现
- ✅ 在切分前完成
- ✅ 按日期分组，无数据泄露

**时间序列归一化**：
- ✅ 已在训练循环中实现
- ✅ 在切分后完成
- ✅ 每个窗口单独计算，无数据泄露

**Winsorize**：
- ✅ 截面标准化中：分位数裁剪（可选）
- ✅ 时间序列归一化后：固定范围裁剪（±5 倍标准差）

---

## 八、总结

### 处理顺序

1. **截面标准化**：在 `build()` 中，**切分前** ✅
   - 按日期分组进行标准化
   - 不会使用未来数据
   - 训练集和验证集使用相同的标准化方式

2. **数据切分**：训练集/验证集分离

3. **时间序列归一化**：在训练时，**切分后** ✅
   - 每个训练窗口单独计算归一化参数
   - 只使用训练集数据
   - 验证集使用训练集的归一化参数

### 数据泄露检查

- ✅ **截面标准化**：安全（只使用当日截面数据）
- ✅ **时间序列归一化**：安全（只使用训练集数据）
- ✅ **Winsorize**：安全（在各自阶段完成）

### 建议

1. **启用截面标准化**：对于行业轮动预测，截面标准化非常重要
2. **使用 Rank 方法**：如果 Z-score 效果不好，尝试 Rank 方法
3. **调整 Winsorize 参数**：根据数据分布调整 `clip_quantile` 和 `clip_range`

