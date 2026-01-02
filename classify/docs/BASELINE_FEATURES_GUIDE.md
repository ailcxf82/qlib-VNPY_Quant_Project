# 行业轮动基线特征集应用指南

## 一、概述

根据您的建议，我们实现了一个基于 **仅 OHLCV 数据** 的强基线特征集。这个特征集专注于：

1. **价量基础（多尺度）**：动量、波动、振幅、量能
2. **截面标准化**：按日期对特征进行标准化，避免未来数据泄露
3. **极值处理**：winsorize/clip 减少极端值影响

---

## 二、特征集设计

### 2.1 动量特征（多时间尺度）

```yaml
# 动量：ret_1, ret_5, ret_10, ret_20, ret_60, ret_120
- "$close / Ref($close, 1) - 1"      # ret_1: 1日收益率
- "$close / Ref($close, 5) - 1"       # ret_5: 5日收益率
- "$close / Ref($close, 10) - 1"      # ret_10: 10日收益率
- "$close / Ref($close, 20) - 1"      # ret_20: 20日收益率
- "$close / Ref($close, 60) - 1"      # ret_60: 60日收益率
- "$close / Ref($close, 120) - 1"     # ret_120: 120日收益率
```

**设计思路**：
- 捕捉不同时间尺度的动量效应
- 短期（1-10天）：捕捉短期反转或动量
- 中期（20-60天）：捕捉行业轮动趋势
- 长期（120天）：捕捉长期趋势

### 2.2 波动特征

```yaml
# 波动：std(ret_1, 10/20/60)
- "Std($close / Ref($close, 1) - 1, 10)"   # std(ret_1, 10): 10日波动
- "Std($close / Ref($close, 1) - 1, 20)"   # std(ret_1, 20): 20日波动
- "Std($close / Ref($close, 1) - 1, 60)"   # std(ret_1, 60): 60日波动
```

**设计思路**：
- 衡量收益率的波动性
- 不同窗口捕捉不同周期的波动特征
- 波动率是行业轮动的重要信号

### 2.3 振幅特征

```yaml
# 振幅：(high-low)/close
- "($high - $low) / $close"           # 振幅：日内波动率
```

**设计思路**：
- 衡量日内价格波动幅度
- 反映市场情绪和交易活跃度

### 2.4 量能特征

```yaml
# 量能特征
- "Log($volume + 1)"                  # log(volume): 对数成交量
- "$volume / Ref($volume, 5) - 1"      # vol_chg_5: 5日成交量变化率
- "$volume / Ref($volume, 20) - 1"   # vol_chg_20: 20日成交量变化率

# 价量相关性（滚动相关系数）
- "Correl($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, 20)"  # 20日价量相关性
- "Correl($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, 60)"  # 60日价量相关性
```

**设计思路**：
- 对数成交量：处理成交量的长尾分布
- 成交量变化率：捕捉量能趋势
- 价量相关性：衡量价格和成交量的协同性

---

## 三、截面标准化（非常重要）

### 3.1 为什么需要截面标准化？

1. **消除市场环境影响**：
   - 不同时期的市场环境不同（牛市/熊市）
   - 同一特征在不同时期可能有不同的分布
   - 截面标准化使特征在同一日期内具有可比性

2. **避免未来数据泄露**：
   - 只使用当日截面的数据（不跨日期）
   - 不会使用未来信息进行标准化

3. **提高模型泛化能力**：
   - 使模型关注相对排名而非绝对数值
   - 更适合排序任务（行业轮动本质是排序问题）

### 3.2 实现方式

#### 方法 1：Z-score 标准化（推荐）

```python
# 对每个日期内的特征进行 Z-score 标准化
for date in dates:
    features_date = features[date]
    mean = features_date.mean()
    std = features_date.std()
    features_date_normalized = (features_date - mean) / std
```

**优点**：
- 保持特征的相对关系
- 标准化为均值0、标准差1
- 适合大多数机器学习模型

#### 方法 2：Rank 标准化

```python
# 对每个日期内的特征进行排名转换
for date in dates:
    features_date = features[date]
    features_date_ranked = features_date.rank(pct=True)  # 转换为 0-1 排名
```

**优点**：
- 完全消除分布影响
- 只保留排序信息
- 对异常值更鲁棒

### 3.3 配置方法

在 `config_data_industry_baseline.yaml` 中：

```yaml
cross_sectional_normalization:
  enabled: true              # 是否启用截面标准化
  method: "zscore"          # 方法：'zscore' 或 'rank'
  groupby: "datetime"       # 按日期分组进行截面标准化
  clip: true                # 是否进行极值裁剪（winsorize）
  clip_quantile: 0.05       # 裁剪分位数（0.05 表示裁剪上下各5%的极值）
```

---

## 四、极值处理（Winsorize/Clip）

### 4.1 为什么需要极值处理？

1. **减少异常值影响**：
   - 极端值可能来自数据错误或特殊事件
   - 会影响模型训练和预测

2. **提高模型稳定性**：
   - 防止模型过度关注极端值
   - 提高模型的泛化能力

### 4.2 实现方式

#### Winsorize（分位数裁剪）

```python
# 对每个特征列进行极值裁剪
for col in features.columns:
    lower = features[col].quantile(0.05)  # 下5%分位数
    upper = features[col].quantile(0.95)  # 上95%分位数
    features[col] = features[col].clip(lower=lower, upper=upper)
```

**优点**：
- 保留大部分数据信息
- 只裁剪极端值
- 不改变数据分布的主体部分

#### Clip（固定范围裁剪）

```python
# 在时间序列归一化后，裁剪到固定范围
normalized_features = normalized_features.clip(-5, 5)  # 裁剪到 ±5 倍标准差
```

**优点**：
- 简单直接
- 适合 Z-score 标准化后的数据

### 4.3 配置方法

```yaml
cross_sectional_normalization:
  clip: true                # 是否进行极值裁剪
  clip_quantile: 0.05       # 裁剪分位数（上下各5%）

temporal_normalization:
  clip: true                # 是否在时间序列归一化后进行裁剪
  clip_range: 5.0            # 裁剪范围（±5倍标准差）
```

---

## 五、完整配置示例

### 5.1 基线配置文件

已创建 `config_data_industry_baseline.yaml`，包含：

1. **简化特征集**（仅基于 OHLCV）：
   - 动量：6 个时间尺度
   - 波动：3 个窗口
   - 振幅：1 个
   - 量能：5 个（对数、变化率、相关性）
   - **总计：15 个特征**

2. **截面标准化**：
   - 启用 Z-score 标准化
   - 按日期分组
   - 启用极值裁剪（5%分位数）

3. **时间序列归一化**：
   - 在训练时按窗口归一化
   - 裁剪到 ±5 倍标准差

### 5.2 使用方法

#### 方法 1：使用基线配置

```bash
# 修改训练脚本，使用基线配置
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

在 `config_industry_rotation.yaml` 中修改：

```yaml
data_config: "classify/config_data_industry_baseline.yaml"  # 使用基线配置
```

#### 方法 2：在现有配置中添加

在 `config_data_industry.yaml` 中添加：

```yaml
# 替换 features 部分
features:
  # ... 使用基线特征集 ...

# 添加截面标准化配置
cross_sectional_normalization:
  enabled: true
  method: "zscore"
  groupby: "datetime"
  clip: true
  clip_quantile: 0.05
```

---

## 六、特征对比

### 6.1 原特征集 vs 基线特征集

| 特性 | 原特征集 | 基线特征集 |
|------|---------|-----------|
| 特征数量 | ~20+ | 15 |
| 数据源 | OHLCV + 技术指标 | 仅 OHLCV |
| 截面标准化 | ❌ | ✅ |
| 极值处理 | 简单 clip | Winsorize + clip |
| 复杂度 | 较高 | 较低 |
| 可解释性 | 中等 | 高 |

### 6.2 预期效果

**基线特征集的优势**：
1. **更简单**：特征数量少，易于理解和调试
2. **更稳定**：截面标准化消除市场环境影响
3. **更鲁棒**：极值处理减少异常值影响
4. **更通用**：仅基于 OHLCV，适用于任何市场

**适用场景**：
- ✅ 第一版强基线模型
- ✅ 快速验证模型架构
- ✅ 对比实验的基准
- ✅ 生产环境的稳定版本

---

## 七、实现细节

### 7.1 截面标准化实现

在 `feature/qlib_feature_pipeline.py` 中：

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
        # Z-score 标准化
        result = grouped.apply(lambda x: (x - x.mean()) / x.std().replace(0, 1))
    elif method == "rank":
        # 排名标准化
        result = grouped.apply(lambda x: x.rank(pct=True))
    
    # 极值裁剪
    if clip:
        for col in result.columns:
            lower = result[col].quantile(clip_quantile)
            upper = result[col].quantile(1 - clip_quantile)
            result[col] = result[col].clip(lower=lower, upper=upper)
    
    return result
```

### 7.2 数据流程

```
原始 OHLCV 数据
    ↓
Qlib 特征提取（15个特征）
    ↓
截面标准化（按日期 Z-score）
    ↓
极值裁剪（Winsorize，5%分位数）
    ↓
时间序列归一化（训练时按窗口）
    ↓
模型训练
```

---

## 八、注意事项

### 8.1 数据泄露预防

✅ **正确做法**：
- 截面标准化只使用当日截面数据
- 时间序列归一化只使用训练窗口数据
- 验证集使用训练集的归一化参数

❌ **错误做法**：
- 使用未来数据计算归一化参数
- 跨日期进行标准化
- 使用验证集数据计算归一化参数

### 8.2 特征选择建议

1. **动量特征**：
   - 根据行业轮动周期选择合适的时间尺度
   - 建议：1, 5, 10, 20, 60, 120 天

2. **波动特征**：
   - 选择与动量特征匹配的窗口
   - 建议：10, 20, 60 天

3. **量能特征**：
   - 价量相关性窗口建议与动量窗口匹配
   - 建议：20, 60 天

### 8.3 参数调优建议

1. **截面标准化方法**：
   - 首次使用建议 `zscore`
   - 如果模型表现不佳，尝试 `rank`

2. **极值裁剪分位数**：
   - 建议从 0.05 开始（上下各5%）
   - 如果数据异常值较多，可以调整到 0.01（上下各1%）

3. **时间序列归一化裁剪范围**：
   - 建议从 ±5 倍标准差开始
   - 如果数据分布较窄，可以调整到 ±3

---

## 九、总结

您提出的建议非常实用，我们已经实现了：

1. ✅ **简化特征集**：仅基于 OHLCV，15 个核心特征
2. ✅ **截面标准化**：按日期 Z-score 或 Rank 标准化
3. ✅ **极值处理**：Winsorize + Clip 双重保护
4. ✅ **配置文件**：`config_data_industry_baseline.yaml`
5. ✅ **代码实现**：`_apply_cross_sectional_normalization` 方法

这个基线特征集可以作为：
- 第一版强基线模型
- 后续特征工程的对比基准
- 生产环境的稳定版本

建议您先使用这个基线特征集训练模型，验证效果后再考虑添加更多特征。

