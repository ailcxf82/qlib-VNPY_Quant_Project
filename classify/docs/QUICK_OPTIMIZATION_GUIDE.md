# 快速优化指南（IC 接近 0 问题）

## 一、问题现状

```
训练集 IC: 0.015（接近 0，几乎学不到东西）
验证集 IC: -0.0177（负值，不稳定）
验证集 IC 标准差: 0.1502（波动很大）
```

---

## 二、快速优化方案（3步）

### 步骤 1：使用优化配置（最简单）

```bash
# 直接使用优化配置
python classify/run_industry_train.py --config classify/config_industry_rotation_optimized_v2.yaml
```

**优化内容**：
- ✅ 简化特征集（20个核心特征）
- ✅ 增加模型容量（hidden_size=128）
- ✅ 优化训练策略（lr=0.001, batch_size=32）
- ✅ 增加验证窗口（valid_months=2）
- ✅ 优化截面标准化（method="rank"）

---

### 步骤 2：运行诊断（定位问题）

```bash
# 启用诊断
python classify/scripts/diagnose_prediction_issues.py --config classify/config_industry_rotation.yaml
```

**重点关注**：
- 测试 3：单特征基线 IC 是否为正
- 如果单特征 IC 都为负，说明数据本身可能没有信号

---

### 步骤 3：根据诊断结果调整

**如果单特征 IC 为正**：
- 继续优化模型架构和训练策略
- 使用优化配置

**如果单特征 IC 为负**：
- 检查特征提取是否正确
- 检查标签定义是否正确
- 可能需要重新设计特征

---

## 三、优化配置说明

### 配置文件

**主配置**：`config_industry_rotation_optimized_v2.yaml`
- 模型架构优化
- 训练策略优化

**数据配置**：`config_data_industry_optimized.yaml`
- 简化特征集（20个核心特征）
- 优化截面标准化（rank 方法）

### 关键优化点

1. **特征集**：从 45+ 个减少到 20 个核心特征
2. **模型容量**：hidden_size 从 64 增到 128
3. **正则化**：dropout 从 0.2 降到 0.15
4. **训练策略**：lr=0.001, batch_size=32, ranking_alpha=0.8
5. **验证窗口**：从 1 个月增到 2 个月

---

## 四、预期效果

### 优化前
```
训练集 IC: 0.015
验证集 IC: -0.0177
验证集 IC 标准差: 0.1502
```

### 优化后（预期）
```
训练集 IC: 0.05-0.10
验证集 IC: 0.02-0.05
验证集 IC 标准差: < 0.10
```

---

## 五、如果仍无效

### 检查清单

1. **数据质量**：
   - 运行诊断脚本
   - 检查单特征基线 IC

2. **标签定义**：
   - 检查标签计算是否正确
   - 检查标签分布是否正常

3. **特征提取**：
   - 检查特征是否正确提取
   - 检查是否有大量缺失值

4. **模型复杂度**：
   - 尝试更简单的模型（线性回归）
   - 验证是否是模型复杂度问题

---

## 六、总结

### 立即执行

```bash
# 使用优化配置
python classify/run_industry_train.py --config classify/config_industry_rotation_optimized_v2.yaml
```

### 预期改进

- 训练集 IC：从 0.015 → 0.05-0.10
- 验证集 IC：从 -0.0177 → 0.02-0.05
- 验证集稳定性：从 std=0.1502 → < 0.10

