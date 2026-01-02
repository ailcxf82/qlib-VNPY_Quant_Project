# 快速诊断指南

## 一、快速开始

### 方法 1：使用诊断脚本（推荐）

```bash
# 运行完整诊断
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

这会运行所有诊断测试，并输出详细结果。

### 方法 2：在训练时启用诊断

在 `config_industry_rotation.yaml` 中启用诊断：

```yaml
diagnosis:
  enabled: true  # 启用快速诊断
```

然后运行训练：

```bash
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

诊断结果会在每个训练窗口后自动输出。

---

## 二、诊断测试说明

### 测试 1：完美预测自测（最高优先级）

**目的**：验证 IC 计算和数据对齐是否正确

**判断标准**：
- ✅ IC ≈ 1.0：正常
- ❌ IC ≠ 1.0：有问题

**如果失败**：
```python
# 检查 IC 计算函数
def rank_ic(pred, label):
    pred, label = pred.align(label, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(label, method="spearman")
```

---

### 测试 2：取负号测试

**目的**：判断模型预测方向是否正确

**判断标准**：
- ✅ `abs(original_ic) > abs(negative_ic)`：方向正确
- ❌ `abs(negative_ic) > abs(original_ic)`：方向相反

**如果失败**：
```python
# 方案 1：在预测时取负号
pred = -model.predict(feat)

# 方案 2：检查标签定义
# 确认正数表示"好"还是"坏"
```

---

### 测试 3：单特征基线

**目的**：验证数据本身是否有基本信号

**判断标准**：
- ✅ IC > 0：数据有信号
- ❌ IC ≤ 0：数据可能没有信号

**如果失败**：
- 检查特征是否正确提取
- 检查标签是否正确计算
- 尝试其他特征

---

### 测试 4：截面 rank vs 原始特征

**目的**：判断标准化方式是否重要

**判断标准**：
- ✅ 差异 < 0.05：标准化影响小
- ⚠️ 差异 > 0.05：标准化影响大

**如果差异大**：
```yaml
# 启用截面标准化
cross_sectional_normalization:
  enabled: true
  method: "rank"  # 或 "zscore"
  groupby: "datetime"
```

---

## 三、诊断结果解读

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
```

### 有问题的情况

```
测试 1：完美预测自测
  ❌ 训练集 IC (pred=label): 0.95  # 应该接近 1.0
  → 需要检查 IC 计算或数据对齐

测试 2：取负号测试
  ⚠️  原始 IC: -0.02
  ⚠️  取负号后 IC: 0.03
  → 方向可能相反，需要取负号

测试 3：单特征基线
  ❌ 验证集 IC: -0.01
  → 数据可能没有信号，需要检查特征和标签
```

---

## 四、修复优先级

根据诊断结果，按以下优先级修复：

1. **测试 1 失败** → 立即修复 IC 计算或数据对齐
2. **测试 2 显示方向问题** → 修改预测方向或损失函数
3. **测试 3 IC 为负** → 检查特征和标签，可能需要重新设计
4. **测试 4 显示标准化影响大** → 启用截面标准化

---

## 五、常见问题

### Q1: 测试 1 失败怎么办？

**A**: 检查以下几点：
1. IC 计算函数是否正确
2. 数据对齐是否正确（`align` 函数）
3. 是否有 NaN 值影响

### Q2: 测试 2 显示方向问题怎么办？

**A**: 
1. 检查模型输出是否需要取负号
2. 检查标签定义（正数表示好还是坏）
3. 检查损失函数优化方向

### Q3: 测试 3 IC 为负怎么办？

**A**:
1. 检查特征是否正确提取
2. 检查标签是否正确计算
3. 尝试其他特征（如 pct_change, vol_chg_20 等）

### Q4: 如何启用截面标准化？

**A**: 在配置文件中：
```yaml
cross_sectional_normalization:
  enabled: true
  method: "rank"  # 或 "zscore"
  groupby: "datetime"
```

---

## 六、下一步

完成诊断后，根据结果采取相应措施：

1. **如果所有测试通过**：继续优化模型（调整超参数、增加特征等）
2. **如果测试 1 失败**：修复 IC 计算或数据对齐
3. **如果测试 2 失败**：修复预测方向
4. **如果测试 3 失败**：检查数据和特征
5. **如果测试 4 显示需要**：启用截面标准化

