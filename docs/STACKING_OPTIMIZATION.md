# Stacking 集成策略优化

本文档说明如何使用优化后的 Stacking 集成策略，包括基于验证集 ICIR 的加权平均和 Meta-Learner。

## 一、概述

原有的 `aggregator: "average"` 使用简单平均，在模型有偏差时效果不佳。现在提供了两种优化方案：

1. **加权平均（Weighted Average）**：基于验证集 ICIR 表现分配权重
2. **Meta-Learner**：使用线性回归或 Ridge 回归学习如何动态组合模型预测

## 二、配置选项

在 `config/pipeline.yaml` 中配置：

```yaml
ensemble:
  # 聚合策略选项：
  # - "average": 简单平均（Qlib 标准化后平均）
  # - "weighted_average": 基于验证集 ICIR 的加权平均（推荐）
  # - "meta_learner": Meta-Learner（Ridge 回归，默认 alpha=1.0）
  # - "meta_learner_linear": Meta-Learner（线性回归）
  # - "meta_learner_ridge": Meta-Learner（Ridge 回归，alpha=1.0）
  aggregator: "weighted_average"
  
  # Meta-Learner 参数（仅当 aggregator="meta_learner" 时使用）
  aggregator_params:
    model_type: "ridge"  # "linear" 或 "ridge"
    alpha: 1.0  # Ridge 回归的正则化系数
  
  models:
    - name: "lgb"
      type: "lightgbm"
      config_key: "lightgbm_config"
    - name: "mlp"
      type: "mlp"
      config_key: "mlp_config"
```

## 三、加权平均（Weighted Average）

### 3.1 原理

在训练时，计算各模型在验证集上的 IC（信息系数），然后使用 `RankICDynamicWeighter` 计算 IC-IR（IC 信息比率），将 IC-IR 转换为权重。

**优点**：
- 性能好的模型（通常是 LGBM）会被赋予更高的权重
- 自动适应模型表现变化
- 简单高效，计算开销小

### 3.2 权重计算流程

1. **计算验证集 IC**：
   ```python
   ic_lgb = rank_correlation(valid_pred_lgb, valid_label)
   ic_mlp = rank_correlation(valid_pred_mlp, valid_label)
   ```

2. **计算 IC-IR**：
   - 由于只有一期验证集，IC-IR 直接使用 IC 值
   - 负 IC 会被裁剪为 0（如果 `clip_negative=True`）

3. **归一化和约束**：
   - 权重归一化（和为 1）
   - 应用 min/max 约束（默认 min=0.1, max=0.8）

4. **加权融合**：
   ```python
   final_pred = weight_lgb * pred_lgb + weight_mlp * pred_mlp
   ```

### 3.3 使用示例

```yaml
ensemble:
  aggregator: "weighted_average"
  models:
    - name: "lgb"
      type: "lightgbm"
      config_key: "lightgbm_config"
    - name: "mlp"
      type: "mlp"
      config_key: "mlp_config"
```

**日志输出示例**：
```
IC-IR 权重: {'lgb': 0.65, 'mlp': 0.35}
```

## 四、Meta-Learner

### 4.1 原理

Meta-Learner 是一个简单的线性模型（线性回归或 Ridge 回归），学习如何组合各模型的预测结果：

```
label = intercept + coef_lgb * pred_lgb + coef_mlp * pred_mlp + ...
```

**优点**：
- 可以学习模型之间的非线性交互
- 自动学习最优组合方式
- Ridge 回归可以防止过拟合

**缺点**：
- 需要足够的验证集样本（建议至少 100+ 样本）
- 计算开销略大于加权平均

### 4.2 训练流程

1. **在验证集上获取各模型预测**：
   ```python
   valid_preds = {
       "lgb": lgb_model.predict(valid_feat),
       "mlp": mlp_model.predict(valid_feat),
   }
   ```

2. **构建特征矩阵**：
   ```python
   X = pd.DataFrame(valid_preds)  # 每列是一个模型的预测
   y = valid_label
   ```

3. **标准化特征**：
   ```python
   X_scaled = StandardScaler().fit_transform(X)
   ```

4. **训练 Meta-Learner**：
   ```python
   meta_model = Ridge(alpha=1.0)
   meta_model.fit(X_scaled, y)
   ```

### 4.3 使用示例

#### 4.3.1 Ridge 回归（推荐）

```yaml
ensemble:
  aggregator: "meta_learner"
  aggregator_params:
    model_type: "ridge"
    alpha: 1.0  # 正则化系数，越大越保守
  models:
    - name: "lgb"
      type: "lightgbm"
      config_key: "lightgbm_config"
    - name: "mlp"
      type: "mlp"
      config_key: "mlp_config"
```

#### 4.3.2 线性回归

```yaml
ensemble:
  aggregator: "meta_learner"
  aggregator_params:
    model_type: "linear"
  models:
    - name: "lgb"
      type: "lightgbm"
      config_key: "lightgbm_config"
    - name: "mlp"
      type: "mlp"
      config_key: "mlp_config"
```

#### 4.3.3 快捷方式

```yaml
# 使用 Ridge 回归（alpha=1.0）
ensemble:
  aggregator: "meta_learner_ridge"

# 使用线性回归
ensemble:
  aggregator: "meta_learner_linear"
```

**日志输出示例**：
```
Meta-Learner (ridge) 训练完成:
  截距: 0.000123
  lgb: 0.623456
  mlp: 0.376544
```

## 五、策略选择建议

### 5.1 何时使用加权平均

- ✅ 验证集样本较少（< 100）
- ✅ 需要快速计算
- ✅ 模型表现差异明显
- ✅ 希望权重可解释

### 5.2 何时使用 Meta-Learner

- ✅ 验证集样本充足（≥ 100）
- ✅ 模型表现相近，需要学习最优组合
- ✅ 希望捕捉模型间的交互
- ✅ 可以接受稍高的计算开销

### 5.3 推荐配置

**默认推荐**：`weighted_average`
- 简单高效
- 适应性强
- 通常效果优于简单平均

**进阶推荐**：`meta_learner` (Ridge)
- 如果验证集样本充足
- 如果简单平均效果不佳
- 如果希望学习更复杂的组合方式

## 六、与现有 IC 动态加权的关系

**重要**：`ensemble.aggregator` 和 `predictor` 中的 IC 动态加权是**两个不同层面**的优化：

1. **`ensemble.aggregator`**（训练时）：
   - 在验证集上计算权重
   - 生成 `qlib_ensemble` 或 `weighted_ensemble` 或 `meta_ensemble`
   - 这个结果会作为**一个模型**参与后续的 IC 动态加权

2. **IC 动态加权**（预测时）：
   - 基于历史 IC 表现动态调整权重
   - 融合所有模型（包括 `lgb`, `mlp`, `stack`, `qlib_ensemble`/`weighted_ensemble`/`meta_ensemble`）
   - 生成最终的 `final` 预测

**工作流程**：
```
训练阶段:
  LGB + MLP → [aggregator] → weighted_ensemble (或 meta_ensemble)
  
预测阶段:
  LGB + MLP + Stack + weighted_ensemble → [IC 动态加权] → final
```

## 七、性能对比

### 7.1 简单平均 vs 加权平均

| 指标 | 简单平均 | 加权平均 |
|------|----------|----------|
| 计算开销 | 低 | 低 |
| 适应能力 | 弱 | 强 |
| 可解释性 | 高 | 中 |
| 推荐场景 | 模型表现相近 | 模型表现差异明显 |

### 7.2 加权平均 vs Meta-Learner

| 指标 | 加权平均 | Meta-Learner |
|------|----------|--------------|
| 计算开销 | 低 | 中 |
| 学习能力 | 弱 | 强 |
| 样本需求 | 低 | 高 |
| 过拟合风险 | 低 | 中（Ridge 可缓解） |
| 推荐场景 | 默认选择 | 验证集充足时 |

## 八、故障排查

### 8.1 加权平均权重为 0

**问题**：所有模型的 IC 为负或 NaN

**解决**：
- 检查验证集是否有效
- 检查模型预测是否正常
- 系统会自动回退为等权

### 8.2 Meta-Learner 训练失败

**问题**：`有效样本数太少`

**解决**：
- 增加验证集大小（`valid_months`）
- 或改用 `weighted_average`

### 8.3 权重不合理

**问题**：某个模型权重过高或过低

**解决**：
- 检查验证集 IC 值
- 调整 `min_weight` 和 `max_weight`（在 `weighted_ensemble.py` 中）

## 九、代码位置

- **加权平均实现**：`models/weighted_ensemble.py::ICIRWeightedAverageAdapter`
- **Meta-Learner 实现**：`models/weighted_ensemble.py::MetaLearnerAdapter`
- **聚合器管理**：`models/ensemble_manager.py::EnsembleAggregator`
- **训练集成**：`models/ensemble_manager.py::EnsembleModelManager.fit()`

## 十、参考

- Qlib Ensemble 文档：https://qlib.readthedocs.io/
- Scikit-learn Ridge 回归：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html



方案 1：加权平均（推荐）
ensemble:
  aggregator: "weighted_average"
  models:
      - name: "lgb"
      type: "lightgbm"      
      config_key: "lightgbm_config"    
      - name: "mlp"      
      type: "mlp"      
      config_key: "mlp_config"
方案 2：Meta-Learner (Ridge)
ensemble:
  aggregator: "meta_learner"  
  aggregator_params: 
     model_type: "ridge"    
     alpha: 1.0 
  models:  
    - name: "lgb"   
      type: "lightgbm"      
      config_key: "lightgbm_config"    
    - name: "mlp"      
      type: "mlp"      
      config_key: "mlp_config"
方案 3：Meta-Learner (Linear)
ensemble:
  aggregator: "meta_learner_linear"  
  models:  
    - name: "lgb"      
      type: "lightgbm"      
      config_key: "lightgbm_config"    
    - name: "mlp"   
       type: "mlp"      
       config_key: "mlp_config

