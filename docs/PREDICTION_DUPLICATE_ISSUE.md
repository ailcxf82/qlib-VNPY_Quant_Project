# 预测结果重复值问题分析

## 一、问题现象

从预测文件 `pred_20250930_2023-10-01_2025-10-01.csv` 的分析结果：

1. **唯一值数量极少**：
   - `final`: 只有 1188 个唯一值，但总共有 144904 行（0.82%）
   - 这意味着 99.18% 的值是重复的

2. **按日期分组后**：
   - 平均每日唯一值数量 = 平均每日总数量 = 298.77
   - **每个日期内的预测值都是唯一的**（没有重复）
   - 但是**不同日期之间有很多重复值**

## 二、问题根源

### 1. Rank 转换的逻辑

当前代码在 `predictor/predictor.py` 中：

```python
# 如果训练时使用了 Rank 转换，对预测值也进行截面排名转换
if label_transform.get("enabled", False):
    final_pred = transform_to_rank(final_pred, method=method, groupby=groupby)
```

`transform_to_rank` 函数（`utils/label_transform.py`）的实现：

```python
def transform_to_rank(label: pd.Series, method: str = "percentile", groupby: Optional[str] = "datetime"):
    if method == "percentile":
        if groupby == "datetime" and isinstance(label.index, pd.MultiIndex):
            grouped = label.groupby(level="datetime")
            rank_pct = grouped.transform(lambda x: x.rank(pct=True, method="average"))
            return rank_pct
```

### 2. 为什么会有重复值？

**正常情况**：
- Rank 转换是按日期分组进行的
- 每个日期内的排名是唯一的（0 到 1 之间）
- 但是不同日期之间，如果股票的排名相同（比如都是第 100 名），那么排名百分位也会相同

**问题情况**：
- 如果模型预测过于保守，导致很多股票的原始预测值相同或非常接近
- 经过 Rank 转换后，这些股票的排名百分位也会相同
- 例如：如果某日有 300 只股票，但只有 50 个不同的原始预测值，那么 Rank 转换后也只有 50 个不同的排名百分位

### 3. 具体问题

从分析结果看：
- 每个日期平均有 298.77 只股票
- 但 `final` 列只有 1188 个唯一值
- 这意味着：**不同日期之间，很多股票的排名百分位是相同的**

**可能的原因**：
1. **模型预测过于保守**：原始预测值差异太小，导致 Rank 转换后重复
2. **Rank 转换的副作用**：Rank 转换会丢失原始预测值的绝对大小信息，只保留相对排名
3. **数据质量问题**：特征值可能存在问题，导致模型预测值过于集中

## 三、解决方案

### 方案 1：检查原始预测值（推荐）

在 Rank 转换之前，检查原始预测值的分布：

```python
# 在 predictor/predictor.py 的 predict 方法中
# 在 Rank 转换之前添加检查
if label_transform.get("enabled", False):
    # 检查原始预测值的唯一性
    unique_before = final_pred.nunique()
    total_before = len(final_pred)
    logger.info("Rank 转换前：唯一值 %d / %d (%.2f%%)", 
                unique_before, total_before, unique_before/total_before*100)
    
    # 按日期检查
    if isinstance(final_pred.index, pd.MultiIndex):
        date_groups = final_pred.groupby(level="datetime")
        for date, group in list(date_groups)[:5]:  # 只检查前5个日期
            logger.info("日期 %s: 唯一值 %d / %d", date, group.nunique(), len(group))
    
    final_pred = transform_to_rank(final_pred, method=method, groupby=groupby)
    
    # 检查转换后的唯一性
    unique_after = final_pred.nunique()
    logger.info("Rank 转换后：唯一值 %d / %d (%.2f%%)", 
                unique_after, total_after, unique_after/total_after*100)
```

### 方案 2：改进 Rank 转换逻辑

如果原始预测值本身就有很多重复，可以考虑：

1. **添加微小随机扰动**（不推荐，会引入噪声）：
```python
# 在 Rank 转换之前，对重复值添加微小扰动
def add_tiny_noise(series: pd.Series, noise_scale: float = 1e-10) -> pd.Series:
    """对重复值添加微小随机扰动，确保排名唯一"""
    duplicated = series.duplicated(keep=False)
    if duplicated.any():
        noise = np.random.normal(0, noise_scale, size=len(series))
        noise = pd.Series(noise, index=series.index)
        series = series + noise * duplicated
    return series
```

2. **使用更精细的排名方法**：
```python
# 使用 method="min" 或 "max" 而不是 "average"
rank_pct = grouped.transform(lambda x: x.rank(pct=True, method="min"))
```

### 方案 3：检查模型预测质量

如果原始预测值本身就有问题，需要检查：

1. **特征值是否正常**：检查特征是否有缺失值或异常值
2. **模型是否过拟合**：检查训练集和验证集的 IC 差异
3. **模型是否过于保守**：检查预测值的分布是否过于集中

### 方案 4：不使用 Rank 转换（如果不需要）

如果 Rank 转换导致的问题比收益更大，可以考虑：

1. **训练时不使用 Rank 转换**：直接预测原始收益
2. **预测时也不进行 Rank 转换**：保持原始预测值
3. **在回测时进行排名**：在选股时再对预测值进行排名

## 四、诊断步骤

### 1. 检查原始预测值

运行以下代码检查原始预测值（在 Rank 转换之前）：

```python
# 在 predictor/predictor.py 的 predict 方法中，在 Rank 转换之前添加
logger.info("原始预测值统计:")
logger.info("  唯一值: %d / %d (%.2f%%)", 
            final_pred.nunique(), len(final_pred), 
            final_pred.nunique()/len(final_pred)*100)
logger.info("  最小值: %.6f, 最大值: %.6f, 均值: %.6f, 标准差: %.6f",
            final_pred.min(), final_pred.max(), final_pred.mean(), final_pred.std())
```

### 2. 检查按日期的分布

```python
if isinstance(final_pred.index, pd.MultiIndex):
    date_groups = final_pred.groupby(level="datetime")
    for date, group in list(date_groups)[:10]:  # 检查前10个日期
        logger.info("日期 %s: 唯一值 %d / %d, 范围 [%.6f, %.6f]",
                   date, group.nunique(), len(group), group.min(), group.max())
```

### 3. 检查重复值模式

```python
# 检查是否有明显的重复值模式
value_counts = final_pred.value_counts()
logger.info("最常见的10个预测值:")
for val, count in value_counts.head(10).items():
    logger.info("  %.10f: %d 次 (%.2f%%)", val, count, count/len(final_pred)*100)
```

## 五、建议

1. **首先检查原始预测值**：确认问题是在 Rank 转换之前还是之后
2. **如果原始预测值就有问题**：检查模型训练和特征提取
3. **如果 Rank 转换导致的问题**：考虑改进 Rank 转换逻辑或禁用 Rank 转换
4. **如果这是正常现象**：说明模型预测过于保守，需要改进模型或特征

