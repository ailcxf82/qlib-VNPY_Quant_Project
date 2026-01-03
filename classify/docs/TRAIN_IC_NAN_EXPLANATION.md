# 训练集 IC 为 NaN 的原因说明

## 一、问题现象

```
平均训练集 IC: nan
训练集 IC 标准差: nan
```

## 二、原因分析

### 2.1 配置原因

在配置文件中，训练集 IC 计算被禁用了：

```yaml
# config_industry_rotation_optimized_v2.yaml
train_ic:
  enabled: false  # 禁用训练集 IC 计算（节省时间）
```

### 2.2 代码逻辑

在 `run_industry_train.py` 中：

```python
# 第 112 行：读取配置
compute_train_ic = train_ic_config.get("enabled", True)

# 第 350 行：初始化 train_ic 为 NaN
train_ic = float("nan")

# 第 376-377 行：如果禁用，跳过计算
elif not compute_train_ic:
    logger.info("窗口 %d: 训练集 IC 计算已禁用...", idx)
    # train_ic 保持为 NaN
```

### 2.3 为什么默认禁用？

1. **训练集 IC 虚高**：
   - 使用训练数据本身进行预测会导致 IC 虚高（过拟合）
   - 没有参考价值

2. **计算耗时**：
   - 如果启用时间序列交叉验证（`method: "ts_cv"`），需要重新训练模型
   - 每个窗口都要训练两次，耗时翻倍

3. **验证集 IC 才是真实指标**：
   - 验证集 IC 才是模型真实性能的反映
   - 训练集 IC 只是辅助指标

---

## 三、如何启用训练集 IC？

### 方法 1：启用时间序列交叉验证（推荐）

```yaml
train_ic:
  enabled: true  # 启用训练集 IC 计算
  method: "ts_cv"  # 使用时间序列交叉验证
```

**工作原理**：
- 将训练集分为两部分：前 80% 用于训练，后 20% 用于评估
- 使用前 80% 的数据重新训练一个临时模型
- 用临时模型预测后 20% 的数据，计算 IC

**优点**：
- 避免数据泄露
- 更真实的训练集 IC

**缺点**：
- 耗时（每个窗口需要训练两次）

### 方法 2：直接使用训练数据（不推荐）

如果需要快速查看训练集 IC（但结果会虚高），可以修改代码：

```python
# 在 run_industry_train.py 中
# 直接使用训练数据预测（不推荐，IC 会虚高）
train_pred = model.predict(train_feat_norm)
train_ic = _rank_ic(train_pred, train_lbl)
```

**警告**：这种方法会导致 IC 虚高，没有参考价值。

---

## 四、修复后的汇总统计

修复后的代码会正确处理 NaN：

```python
if df["train_ic"].notna().any():
    logger.info("  - 平均训练集 IC: %.4f", df["train_ic"].mean())
    logger.info("  - 训练集 IC 标准差: %.4f", df["train_ic"].std())
else:
    logger.info("  - 平均训练集 IC: N/A（训练集 IC 计算已禁用，见 train_ic.enabled 配置）")
    logger.info("  - 训练集 IC 标准差: N/A（训练集 IC 计算已禁用）")
```

**输出示例**：
```
训练汇总统计:
  - 总窗口数: 11
  - 平均验证集 IC: -0.0014
  - 验证集 IC 标准差: 0.0953
  - 平均训练集 IC: N/A（训练集 IC 计算已禁用，见 train_ic.enabled 配置）
  - 训练集 IC 标准差: N/A（训练集 IC 计算已禁用）
```

---

## 五、建议

### 5.1 默认配置（推荐）

```yaml
train_ic:
  enabled: false  # 禁用训练集 IC 计算
```

**理由**：
- 节省时间
- 验证集 IC 才是真实指标
- 避免误导性的训练集 IC

### 5.2 如果需要查看训练集 IC

```yaml
train_ic:
  enabled: true
  method: "ts_cv"  # 使用时间序列交叉验证
```

**注意**：
- 训练时间会翻倍
- 训练集 IC 应该高于验证集 IC（但差距不应太大）

---

## 六、总结

1. **NaN 是正常的**：因为训练集 IC 计算被禁用了
2. **这是推荐配置**：避免误导性的训练集 IC，节省时间
3. **验证集 IC 才是关键**：关注验证集 IC 即可
4. **如需启用**：修改 `train_ic.enabled: true` 和 `method: "ts_cv"`

---

## 七、相关文件

- 配置文件：`config_industry_rotation_optimized_v2.yaml`
- 训练脚本：`run_industry_train.py`（第 350-377 行）
- 汇总统计：`run_industry_train.py`（第 455-464 行）

