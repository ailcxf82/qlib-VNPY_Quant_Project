# 预测逻辑详解：run_predict.py 如何解释和使用训练结果

## 一、整体流程概览

```
训练阶段 (run_train.py)
    ↓
生成模型文件 + 训练指标
    ↓
预测阶段 (run_predict.py)
    ↓
加载模型 + 读取历史 IC → 动态加权 → 最终预测
```

## 二、预测流程详细步骤

### 步骤 1: 确定模型标识 (Tag)

```python
# run_predict.py:87-93
tag = _latest_tag(log_path, cfg["paths"]["model_dir"]) if args.tag == "auto" else args.tag
```

**逻辑**：
1. 如果 `--tag auto`（默认）：自动查找最新模型
   - 优先从 `training_metrics.csv` 读取最新的 `valid_end` 日期
   - 如果日志不存在，从模型目录中查找最新的 `*_lgb.txt` 文件
2. 如果指定了 `--tag`：直接使用指定的 tag（格式：`YYYYMMDD`）

**示例**：
```python
# 自动模式：找到最新的验证窗口结束日期，如 "20250930"
tag = "20250930"

# 手动指定：使用特定日期的模型
tag = "20241031"
```

### 步骤 2: 加载历史 IC 指标

```python
# run_predict.py:45-60
def _load_ic_histories(log_path: str) -> Dict[str, pd.Series]:
    df = pd.read_csv(log_path, parse_dates=["valid_end"])
    histories = {
        "lgb": pd.Series(df["ic_lgb"].values, index=df["valid_end"]),
        "mlp": pd.Series(df["ic_mlp"].values, index=df["valid_end"]),
        "stack": pd.Series(df["ic_stack"].values, index=df["valid_end"]),
        "qlib_ensemble": pd.Series(df["ic_qlib_ensemble"].values, index=df["valid_end"]),
    }
    return histories
```

**训练日志格式** (`data/logs/training_metrics.csv`):
```csv
window,train_start,train_end,valid_start,valid_end,ic_lgb,ic_mlp,ic_stack,ic_qlib_ensemble
0,2021-10-01,2023-09-30,2023-10-01,2023-10-31,0.0523,0.0481,0.0567,0.0545
1,2021-11-01,2023-10-31,2023-11-01,2023-11-30,0.0489,0.0512,0.0534,0.0521
2,2021-12-01,2023-11-30,2023-12-01,2023-12-31,0.0515,0.0498,0.0551,0.0532
...
```

**IC 指标含义**：
- **IC (Information Coefficient)**: 预测值与真实值的秩相关系数（Spearman）
- **IC > 0**: 预测方向正确，值越大预测能力越强
- **IC < 0**: 预测方向错误，可能模型失效
- **IC ≈ 0**: 预测无方向性，接近随机

**为什么需要历史 IC？**
- 评估各模型在不同时期的表现
- 计算 IC-IR（IC 信息比率），衡量模型的稳定性和预测能力
- 用于动态加权，给表现好的模型更高权重

### 步骤 3: 加载训练好的模型

```python
# run_predict.py:99-100
predictor = PredictorEngine(args.config)
predictor.load_models(tag)
```

**加载内容**：
1. **LightGBM 模型** (`{tag}_lgb.txt`): 树模型结构
2. **MLP 模型** (`{tag}_mlp.pth`): 神经网络权重
3. **Stack 模型** (`{tag}_stack.pth`): 残差学习模型

**模型文件位置**：
```
data/models/
  ├── 20231031_lgb.txt          # LightGBM 模型
  ├── 20231031_lgb_meta.json    # 模型元数据（特征名等）
  ├── 20231031_mlp.pth          # MLP 模型
  ├── 20231031_stack.pth        # Stack 模型
  └── ...
```

### 步骤 4: 提取特征并生成预测

```python
# run_predict.py:96-101
pipeline = QlibFeaturePipeline(cfg["data_config"])
pipeline.build()
features, _ = pipeline.get_slice(args.start, args.end)
final_pred, preds, weights = predictor.predict(features, ic_histories)
```

**特征提取**：
- 根据 `config/data.yaml` 配置提取因子
- 时间范围：`args.start` 到 `args.end`
- 返回 `pd.DataFrame`，索引为 `(datetime, instrument)`

**预测生成** (`predictor/predictor.py:45-65`):

#### 4.1 基础模型预测

```python
# 1. 基础模型预测（LGB、MLP）
blend_pred, base_preds, aux = self.ensemble.predict(features)
# base_preds = {"lgb": Series, "mlp": Series}
# aux = {"lgb": leaf_index}  # LGB 的叶子索引，用于 Stack
```

**LGB 预测**：
- 使用训练好的树模型对特征进行预测
- 同时输出叶子索引（每棵树到达的叶子节点编号）

**MLP 预测**：
- 使用训练好的神经网络对特征进行预测

#### 4.2 Stack 模型预测（残差学习）

```python
# 2. Stack 模型：学习 LGB 的残差
lgb_pred = base_preds.get("lgb")
lgb_leaf = aux.get("lgb")
residual_pred = self.stack.predict_residual(lgb_leaf, features.index)
stack_pred = self.stack.fuse(lgb_pred, residual_pred)
```

**Stack 模型逻辑**：
1. **残差计算**（训练时）: `residual = label - lgb_pred`
2. **残差学习**（训练时）: 使用 MLP 学习 `residual = f(lgb_leaf_index)`
3. **残差预测**（预测时）: `residual_pred = stack_model.predict(lgb_leaf)`
4. **融合预测**: `stack_pred = lgb_pred + alpha * residual_pred`

**为什么需要 Stack？**
- LGB 擅长捕捉非线性关系，但可能遗漏某些结构化模式
- Stack 通过学习 LGB 的残差，补充 LGB 的不足
- 通常 `stack_pred` 的 IC 会略高于 `lgb_pred`

#### 4.3 Qlib Ensemble（可选）

```python
# 3. Qlib 标准化平均（可选）
if blend_pred is not None:
    preds["qlib_ensemble"] = blend_pred
```

**Qlib Ensemble**：
- 对所有模型预测进行标准化（Z-score）
- 然后求平均
- 用于观察"标准化平均"的效果

### 步骤 5: IC 动态加权

```python
# predictor/predictor.py:63-64
weights = self.weighter.get_weights(ic_histories)
final_pred = self.weighter.blend(preds, weights)
```

**这是核心逻辑！** 根据历史表现动态分配权重。

#### 5.1 计算 IC-IR（IC 信息比率）

```python
# predictor/weight_dynamic.py:43-55
def _ic_ir(self, ic_series: pd.Series) -> float:
    # 1. 取最近 window 个 IC 值（默认 60 个）
    ic_series = ic_series.dropna().tail(self.window)
    
    # 2. 按半衰期生成指数衰减权重（越近的 IC 权重越大）
    weights = np.array([0.5 ** (i / max(1, self.half_life - 1)) 
                        for i in range(len(ic_series))])[::-1]
    weights /= weights.sum()
    
    # 3. 计算加权均值和标准差
    mean = np.sum(ic_series.values * weights)
    std = np.sqrt(np.sum(weights * (ic_series.values - mean) ** 2))
    
    # 4. IC-IR = 均值 / 标准差（类似夏普比率）
    return mean / std
```

**IC-IR 含义**：
- **IC-IR = IC 均值 / IC 标准差**
- 衡量模型的**稳定性和预测能力**
- **IC-IR 高**：模型表现稳定且预测能力强
- **IC-IR 低**：模型表现不稳定或预测能力弱

**半衰期权重示例**（`half_life=20`）：
```
最近的 IC: 权重 = 1.0
20 天前: 权重 = 0.5
40 天前: 权重 = 0.25
60 天前: 权重 = 0.125
```

**为什么使用半衰期？**
- 更重视最近的表现（市场环境可能变化）
- 但不会完全忽略历史（保持稳定性）

#### 5.2 将 IC-IR 转换为权重

```python
# predictor/weight_dynamic.py:57-71
def get_weights(self, ic_histories: Dict[str, pd.Series]) -> Dict[str, float]:
    # 1. 计算每个模型的 IC-IR
    scores = {name: self._ic_ir(series) for name, series in ic_histories.items()}
    
    # 2. 裁剪负值（如果 clip_negative=True）
    if self.clip_negative:
        scores = {k: max(0.0, v) for k, v in scores.items()}
    
    # 3. 归一化（使权重和为 1）
    total = sum(scores.values())
    if total == 0:
        return {k: 1.0/len(scores) for k in scores}  # 等权回退
    weights = {k: v / total for k, v in scores.items()}
    
    # 4. 施加 min/max 约束（防止某个模型权重过高或过低）
    weights = {k: np.clip(w, self.min_weight, self.max_weight) 
               for k, w in weights.items()}
    
    # 5. 重新归一化（因为 clip 后和可能不为 1）
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}
```

**权重计算示例**：

假设各模型的 IC-IR：
```
lgb: 0.85
mlp: 0.72
stack: 0.91
qlib_ensemble: 0.78
```

计算过程：
1. **归一化**（和 = 3.26）:
   ```
   lgb: 0.85 / 3.26 = 0.261
   mlp: 0.72 / 3.26 = 0.221
   stack: 0.91 / 3.26 = 0.279
   qlib_ensemble: 0.78 / 3.26 = 0.239
   ```

2. **Clip 约束**（假设 min=0.05, max=0.7）:
   ```
   lgb: 0.261 (在范围内)
   mlp: 0.221 (在范围内)
   stack: 0.279 (在范围内)
   qlib_ensemble: 0.239 (在范围内)
   ```

3. **最终权重**:
   ```
   lgb: 0.261
   mlp: 0.221
   stack: 0.279  ← 最高（IC-IR 最高）
   qlib_ensemble: 0.239
   ```

#### 5.3 融合预测

```python
# predictor/weight_dynamic.py:73-80
def blend(self, preds: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    combined = None
    for name, series in preds.items():
        w = weights.get(name, 0.0)
        contrib = series * w
        combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
    return combined
```

**融合公式**：
```
final_pred = w_lgb * lgb_pred + w_mlp * mlp_pred + w_stack * stack_pred + w_qlib * qlib_pred
```

**示例**（假设权重如上）：
```
final_pred = 0.261 * lgb_pred + 0.221 * mlp_pred + 0.279 * stack_pred + 0.239 * qlib_pred
```

### 步骤 6: 保存预测结果

```python
# run_predict.py:102
predictor.save_predictions(final_pred, preds, f"{tag}_{args.start}_{args.end}")
```

**保存格式** (`data/predictions/pred_{tag}_{start}_{end}.csv`):
```csv
datetime,instrument,final,lgb,mlp,stack,qlib_ensemble
2023-10-01,000001.SZ,0.0234,0.0251,0.0218,0.0245,0.0232
2023-10-01,000002.SZ,-0.0123,-0.0134,-0.0112,-0.0128,-0.0121
...
```

**文件内容**：
- `final`: 最终融合预测（用于交易）
- `lgb`, `mlp`, `stack`, `qlib_ensemble`: 各模型单独预测（用于分析）

## 三、关键设计思想

### 1. 为什么使用 IC 动态加权？

**问题**：不同模型在不同市场环境下表现不同
- LGB 可能在趋势市场表现好
- MLP 可能在震荡市场表现好
- Stack 可能在某些时期表现最好

**解决方案**：根据历史 IC 动态调整权重
- 表现好的模型获得更高权重
- 表现差的模型权重降低
- 保持模型多样性（min_weight 保证）

### 2. 为什么使用半衰期权重？

**问题**：市场环境会变化，历史表现可能不适用

**解决方案**：指数衰减权重
- 更重视最近的表现
- 但不会完全忽略历史
- 平衡"适应性"和"稳定性"

### 3. 为什么需要 Stack 模型？

**问题**：单一模型（如 LGB）可能遗漏某些模式

**解决方案**：残差学习
- LGB 捕捉主要模式
- Stack 学习 LGB 的残差（LGB 遗漏的部分）
- 两者融合，提升预测能力

## 四、配置参数说明

### 4.1 IC 动态加权配置 (`config/pipeline.yaml`)

```yaml
ic_logging:
  window: 60        # 使用最近 60 个 IC 值
  half_life: 20     # 半衰期 20 天（权重衰减速度）
  min_weight: 0.05  # 最小权重（保证模型多样性）
  max_weight: 0.7   # 最大权重（防止单一模型主导）
  clip_negative: true  # 是否裁剪负 IC-IR
```

**参数调优建议**：

| 参数 | 调大 | 调小 | 说明 |
|------|------|------|------|
| `window` | 更稳定，但反应慢 | 更敏感，但可能不稳定 | 建议 30-90 |
| `half_life` | 更重视历史 | 更重视最近 | 建议 10-30 |
| `min_weight` | 保证更多模型参与 | 允许模型权重为 0 | 建议 0.05-0.15 |
| `max_weight` | 允许单一模型主导 | 强制模型多样性 | 建议 0.5-0.8 |

### 4.2 预测参数 (`run_predict.py`)

```bash
python run_predict.py \
    --config config/pipeline.yaml \
    --start 2023-10-01 \      # 预测起始日期
    --end 2025-10-01 \        # 预测结束日期
    --tag auto                # 模型标识（auto=自动查找最新）
```

## 五、完整示例

### 5.1 训练阶段（生成模型和 IC 历史）

```bash
# 训练多个滚动窗口
python run_train.py --config config/pipeline.yaml
```

**输出**：
- `data/models/20231031_lgb.txt` 等模型文件
- `data/logs/training_metrics.csv`（包含历史 IC）

### 5.2 预测阶段（使用训练结果）

```bash
# 预测未来表现
python run_predict.py \
    --config config/pipeline.yaml \
    --start 2023-10-01 \
    --end 2025-10-01 \
    --tag auto
```

**执行流程**：
1. 读取 `training_metrics.csv`，提取历史 IC
2. 加载 `20231031` 的模型（最新）
3. 提取 2023-10-01 到 2025-10-01 的特征
4. 各模型生成预测
5. 根据历史 IC 计算权重
6. 加权融合，生成最终预测
7. 保存到 `data/predictions/pred_20231031_2023-10-01_2025-10-01.csv`

### 5.3 查看预测结果

```python
import pandas as pd

# 读取预测结果
df = pd.read_csv("data/predictions/pred_20231031_2023-10-01_2025-10-01.csv",
                 index_col=["datetime", "instrument"])

# 查看最终预测
print(df["final"].head())

# 查看各模型预测对比
print(df[["lgb", "mlp", "stack", "final"]].head())

# 查看权重（在日志中）
# "IC 动态权重: {'lgb': 0.261, 'mlp': 0.221, 'stack': 0.279, 'qlib_ensemble': 0.239}"
```

## 六、常见问题

### Q1: 如果某个模型的 IC 一直为负怎么办？

**A**: `clip_negative=True` 会将负 IC-IR 裁剪为 0，该模型权重会很低（接近 min_weight）。

### Q2: 如何判断预测质量？

**A**: 
1. 查看日志中的 IC 动态权重（权重高的模型表现好）
2. 对比各模型预测的差异（差异大说明模型分歧大）
3. 回测验证（使用 `run_backtest.py`）

### Q3: 为什么使用多个模型而不是单一模型？

**A**: 
- **模型多样性**：不同模型捕捉不同模式
- **稳健性**：单一模型可能失效，多模型降低风险
- **动态适应**：通过权重调整适应市场变化

### Q4: Stack 模型一定比 LGB 好吗？

**A**: 不一定。Stack 通过学习残差补充 LGB，但：
- 如果 LGB 已经很好，残差很小，Stack 贡献有限
- 如果残差有规律，Stack 能显著提升
- 最终通过 IC 动态加权，表现好的模型权重更高

## 七、总结

**核心思想**：
1. **多模型协同**：LGB、MLP、Stack 各司其职
2. **历史表现评估**：通过 IC 历史评估模型能力
3. **动态权重调整**：根据 IC-IR 动态分配权重
4. **加权融合**：最终预测 = 各模型预测的加权平均

**优势**：
- ✅ 适应市场变化（动态权重）
- ✅ 降低单一模型风险（多模型）
- ✅ 提升预测稳定性（IC-IR 评估）
- ✅ 保持模型多样性（min_weight 约束）

**关键文件**：
- `run_predict.py`: 预测入口
- `predictor/predictor.py`: 预测引擎
- `predictor/weight_dynamic.py`: IC 动态加权
- `data/logs/training_metrics.csv`: 历史 IC 数据
- `data/models/`: 训练好的模型

