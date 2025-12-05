# 策略改进建议：应对快速上涨市场

## 问题诊断

根据回测曲线分析，策略在 **2024 年底到 2025 年 10 月** 期间表现不佳：
- **基准快速上涨**：从 0.85 涨到 1.25（+47%）
- **策略表现平平**：净值在 1.03-1.04 徘徊（几乎无增长）
- **核心问题**：模型没有捕捉到快速上涨的趋势

## 一、标签定义问题（最重要）

### 当前问题

```yaml
# config/data.yaml:124
label: "Ref($close, -5)/$close - 1"  # 预测未来 5 日收益
```

**问题分析**：
1. **预测周期过短**：5 日收益无法捕捉中长期趋势
2. **信号噪声大**：短期波动干扰大，模型难以学习趋势
3. **不适合趋势市场**：在快速上涨时，5 日收益可能滞后

### 改进建议

#### 方案 1：多周期标签（推荐）⭐

```yaml
# config/data.yaml
label: "Ref($close, -20)/$close - 1"  # 预测未来 20 日收益（约 1 个月）
# 或者
label: "Ref($close, -60)/$close - 1"   # 预测未来 60 日收益（约 3 个月）
```

**优点**：
- 更符合中长期投资逻辑
- 能捕捉趋势性机会
- 减少短期噪声干扰

**缺点**：
- 预测周期变长，信号可能滞后
- 需要更长的回测周期验证

#### 方案 2：多标签融合（最佳）⭐⭐

训练多个模型，分别预测不同周期的收益，然后融合：

```yaml
# 可以创建多个配置文件
# config/data_short.yaml: label: "Ref($close, -5)/$close - 1"   # 短期
# config/data_medium.yaml: label: "Ref($close, -20)/$close - 1" # 中期
# config/data_long.yaml: label: "Ref($close, -60)/$close - 1"   # 长期
```

然后在预测时融合：
```python
final_pred = w_short * pred_short + w_medium * pred_medium + w_long * pred_long
```

## 二、特征工程改进

### 2.1 添加趋势加速特征

当前特征缺少**趋势加速度**指标，无法捕捉快速上涨：

```yaml
# config/data.yaml - 在 features 中添加
features:
  # ... 现有特征 ...
  
  # ========== 趋势加速特征 ==========
  # 价格变化加速度（二阶导数）
  - "(Ref($close, 1) - Ref($close, 2)) - (Ref($close, 2) - Ref($close, 3))"  # 价格变化率的变化
  - "(Mean($close, 5) - Mean($close, 10)) - (Mean($close, 10) - Mean($close, 20))"  # 均线加速度
  
  # 动量强度
  - "Mean($close, 5) / Mean($close, 20) - 1"  # 短期相对长期均线的偏离
  - "Mean($close, 10) / Mean($close, 60) - 1"  # 中期相对长期均线的偏离
  
  # 趋势持续性
  - "Sum($close > Ref($close, 1), 5) / 5"  # 最近 5 日上涨天数占比
  - "Sum($close > Ref($close, 1), 20) / 20"  # 最近 20 日上涨天数占比
  
  # 突破特征
  - "($close - Max($high, 20)) / Max($high, 20)"  # 是否突破 20 日最高
  - "($close - Max($high, 60)) / Max($high, 60)"  # 是否突破 60 日最高
  
  # 成交量趋势
  - "Mean($volume, 5) / Mean($volume, 20) - 1"  # 成交量短期相对长期
  - "($volume - Mean($volume, 20)) / Std($volume, 20)"  # 成交量 Z-score
```

### 2.2 添加市场状态特征

快速上涨往往伴随特定的市场状态：

```yaml
features:
  # ... 现有特征 ...
  
  # ========== 市场状态特征 ==========
  # 市场整体趋势
  - "Mean($close, 20) / Mean($close, 60) - 1"  # 市场中期趋势
  - "Std($close, 20) / Mean($close, 20)"  # 市场波动率
  
  # 相对强度
  - "($close - Mean($close, 60)) / Std($close, 60)"  # 价格 Z-score（相对 60 日均值）
  - "Rank($close, 60)"  # 当前价格在 60 日内的排名百分位
  
  # 趋势强度
  - "Slope($close, 20) / Mean($close, 20)"  # 20 日线性回归斜率（趋势强度）
  - "Slope($close, 60) / Mean($close, 60)"  # 60 日线性回归斜率
```

### 2.3 使用 Alpha158 中的趋势特征

确保 Alpha158 配置中包含趋势相关操作符：

```yaml
# config/data.yaml
alpha158_config:
  rolling:
    windows: [5, 10, 20, 30, 60]
    include: [
      "ROC",      # 价格变化率（动量）
      "MA",       # 移动平均（趋势）
      "BETA",     # 线性回归斜率（趋势强度）⭐ 重要
      "RSV",      # 随机指标（相对位置）
      "RANK",     # 排名（相对强度）
      "SUMP",     # 上涨幅度占比（趋势持续性）⭐ 重要
      "CORR",     # 价量相关性
      "CORD",     # 价量变化相关性
      # ... 其他
    ]
```

## 三、模型训练改进

### 3.1 缩短训练窗口（适应快速变化）

当前配置：
```yaml
# config/pipeline.yaml
rolling:
  train_months: 48  # 4 年，可能包含太多历史，不适应快速变化
```

**改进建议**：
```yaml
rolling:
  train_months: 24  # 改为 2 年，更适应市场变化
  valid_months: 1
  step_months: 1
```

**理由**：
- 快速上涨的市场环境可能只在最近 1-2 年出现过
- 4 年的历史数据可能包含太多不相关的市场环境
- 更短的训练窗口能更快适应市场变化

### 3.2 调整模型参数（提升信号强度）

#### LightGBM 参数调整

```yaml
# config/model_lgb.yaml
model:
  params:
    learning_rate: 0.03        # 从 0.02 提高到 0.03，增强信号
    max_depth: 7               # 从 6 提高到 7，允许更复杂的模式
    num_leaves: 63            # 从 31 提高到 63，增加模型容量
    min_data_in_leaf: 20      # 从 30 降低到 20，允许更细粒度的分裂
    lambda_l1: 0.1            # 从 0.2 降低到 0.1，减少正则化，允许更强信号
    lambda_l2: 0.5            # 从 1.0 降低到 0.5
```

**注意**：这些调整可能增加过拟合风险，需要配合验证集监控。

### 3.3 增加模型多样性

考虑添加专门捕捉趋势的模型：

```yaml
# config/pipeline.yaml
ensemble:
  models:
    - name: "lgb"
      type: "lightgbm"
    - name: "mlp"
      type: "mlp"
    - name: "lgb_trend"      # 新增：专门训练趋势特征的 LGB
      type: "lightgbm"
      config_key: "lightgbm_trend_config"  # 使用更强的参数
```

## 四、预测逻辑改进

### 4.1 调整 IC 动态加权参数

当前配置可能过于保守：

```yaml
# config/pipeline.yaml
ic_logging:
  window: 60        # 使用 60 个窗口的 IC
  half_life: 20     # 半衰期 20 天
```

**改进建议**：
```yaml
ic_logging:
  window: 30        # 缩短到 30，更重视最近表现
  half_life: 10     # 缩短到 10，更快适应市场变化
  min_weight: 0.1   # 从 0.05 提高到 0.1，保证模型参与度
  max_weight: 0.5   # 从 0.7 降低到 0.5，防止单一模型主导
```

### 4.2 添加趋势信号增强

在预测时，如果检测到趋势加速，可以增强信号：

```python
# 在 predictor/predictor.py 中添加
def enhance_trend_signal(pred: pd.Series, features: pd.DataFrame) -> pd.Series:
    """如果检测到趋势加速，增强预测信号"""
    # 计算趋势加速度
    trend_accel = features["trend_acceleration"]  # 假设有这个特征
    
    # 如果趋势加速 > 阈值，增强信号
    mask = trend_accel > 0.02  # 阈值可调
    enhanced_pred = pred.copy()
    enhanced_pred[mask] = pred[mask] * 1.2  # 增强 20%
    
    return enhanced_pred
```

## 五、组合构建改进

### 5.1 增加持仓集中度（在趋势明确时）

当前配置：
```yaml
# config/pipeline.yaml
portfolio:
  top_k: 50           # 持仓 50 只股票
  max_stock_weight: 0.05  # 单股最大 5%
```

**改进建议**：在趋势明确时，可以增加集中度

```python
# 在 portfolio/portfolio_builder.py 中
def build(self, scores, industry_map, top_k=None, trend_strength=None):
    if trend_strength is not None and trend_strength > 0.1:
        # 趋势强时，增加集中度
        top_k = top_k or 30  # 减少到 30 只
        max_stock_weight = 0.08  # 提高到 8%
    else:
        top_k = top_k or 50
        max_stock_weight = 0.05
    # ... 原有逻辑
```

### 5.2 动态调整仓位

在快速上涨时，可以增加仓位：

```yaml
# config/pipeline.yaml
portfolio:
  max_position: 0.3    # 当前最大仓位 30%
  
  # 可以添加动态仓位
  dynamic_position:
    enabled: true
    base_position: 0.3
    max_position: 0.5   # 趋势强时最高 50%
    trend_threshold: 0.1  # 趋势强度阈值
```

## 六、实施优先级

### 高优先级（立即实施）⭐⭐⭐

1. **修改标签定义**：从 5 日改为 20 日或 60 日收益
2. **添加趋势加速特征**：价格加速度、动量强度等
3. **缩短训练窗口**：从 48 个月改为 24 个月

### 中优先级（逐步实施）⭐⭐

4. **调整 IC 动态加权参数**：缩短 window 和 half_life
5. **优化模型参数**：适当提高学习率，降低正则化
6. **确保 Alpha158 包含趋势特征**：BETA、SUMP 等

### 低优先级（长期优化）⭐

7. **多标签融合**：训练多个周期的模型
8. **趋势信号增强**：在预测时检测趋势并增强信号
9. **动态仓位调整**：根据趋势强度调整仓位

## 七、验证方法

### 7.1 回测验证

实施改进后，重新回测，关注：
- **2024 年底到 2025 年 10 月**的表现
- 策略是否能跟上基准的快速上涨
- IC 和 IC-IR 是否提升

### 7.2 特征重要性分析

运行特征重要性分析，确认趋势特征是否被模型使用：

```bash
python scripts/analyze_feature_importance.py \
    --config config/pipeline.yaml \
    --top_k 100
```

检查：
- 趋势相关特征（BETA、SUMP、加速度等）是否进入前 100
- 如果不在，说明特征可能不够强，需要调整

### 7.3 模型预测分析

对比改进前后的预测：
- 在快速上涨期间，预测值是否更大
- 预测分布是否更偏向正值
- 与基准的相关性是否提升

## 八、快速实施步骤

### 步骤 1：修改标签（最重要）

```yaml
# config/data.yaml
label: "Ref($close, -20)/$close - 1"  # 改为 20 日
```

### 步骤 2：添加趋势特征

在 `config/data.yaml` 的 `features` 中添加：
```yaml
- "Mean($close, 5) / Mean($close, 20) - 1"
- "Slope($close, 20) / Mean($close, 20)"
- "Sum($close > Ref($close, 1), 20) / 20"
```

### 步骤 3：缩短训练窗口

```yaml
# config/pipeline.yaml
rolling:
  train_months: 24  # 从 48 改为 24
```

### 步骤 4：重新训练和回测

```bash
# 重新训练
python run_train.py --config config/pipeline.yaml

# 重新预测
python run_predict.py --config config/pipeline.yaml

# 重新回测
python run_backtest.py --config config/pipeline.yaml
```

## 九、预期效果

实施上述改进后，预期：
- ✅ 策略能更好地捕捉快速上涨趋势
- ✅ 在 2024 年底到 2025 年 10 月期间，策略净值能跟上基准
- ✅ IC 和 IC-IR 提升，特别是趋势特征的重要性提升
- ✅ 模型预测值在上涨期间更大，信号强度增强

## 十、注意事项

1. **过拟合风险**：降低正则化、提高模型复杂度可能增加过拟合，需要监控验证集表现
2. **标签滞后**：使用 20 日或 60 日标签，预测信号可能滞后，需要权衡
3. **市场环境变化**：快速上涨可能只是特定时期，需要确保策略在其他市场环境下也能工作
4. **回测偏差**：改进后需要重新回测，但要注意避免过度拟合回测数据

