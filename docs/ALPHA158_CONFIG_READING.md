# Alpha158 配置读取说明

本文档说明工程中如何读取 `alpha158_config` 配置，以及 `kbar` 参数的正确配置方式。

## 一、配置读取流程

### 1. 读取位置

配置读取在 `feature/qlib_feature_pipeline.py` 的 `_get_alpha158_factors()` 方法中：

```python
def _get_alpha158_factors(self) -> List[str]:
    """获取 158 因子表达式列表。"""
    # 1. 检查是否有手动指定的因子列表
    manual_factors = self.feature_cfg.get("alpha158_factors", None)
    if manual_factors:
        return manual_factors
    
    # 2. 获取 alpha158_config 配置
    alpha158_config = self.feature_cfg.get("alpha158_config", None)
    
    # 3. 调用 qlib 的 Alpha158DL.get_feature_config() 方法
    if alpha158_config:
        fields, names = Alpha158DL.get_feature_config(alpha158_config)
    else:
        fields, names = Alpha158DL.get_feature_config()
    
    return fields
```

### 2. 配置传递

配置从 `config/data.yaml` 读取后，直接传递给 `Alpha158DL.get_feature_config()`：

```python
# 在 feature/qlib_feature_pipeline.py 中
alpha158_config = self.feature_cfg.get("alpha158_config", None)
if alpha158_config:
    logger.info("使用自定义 158 因子配置: %s", alpha158_config)
    fields, names = Alpha158DL.get_feature_config(alpha158_config)  # 直接传递配置字典
```

### 3. qlib 解析配置

`Alpha158DL.get_feature_config()` 方法会根据配置字典生成因子表达式列表：

- **输入**: 配置字典（如 `{"kbar": {}, "price": {...}, "rolling": {...}}`）
- **输出**: `(fields, names)` 元组
  - `fields`: 因子表达式列表（用于 `D.features()`）
  - `names`: 因子名称列表（用于列名）

---

## 二、kbar 参数配置说明

### 1. 正确的配置方式

**正确配置**：
```yaml
alpha158_config:
  kbar: {}  # 空字典表示启用所有 9 个 K 线特征
```

**说明**：
- `kbar: {}` 表示启用所有 9 个固定的 K 线特征
- K 线特征是固定的 9 个因子，不需要滚动窗口
- 不需要配置 `windows` 和 `include` 参数

### 2. 错误的配置方式

**错误配置 1**（配置了 windows）：
```yaml
alpha158_config:
  kbar:
    windows: [5, 10, 20, 30, 60]  # ❌ 错误：K 线特征不需要滚动窗口
    include: ["KMID", "KLEN", ...]  # ❌ 错误：K 线特征不需要 include
```

**问题**：
- K 线特征是固定的 9 个因子，不需要滚动窗口
- 如果配置了 `windows`，qlib 可能会尝试生成滚动窗口的 K 线特征（如 KMID5, KMID10 等），这不是预期的行为
- `include` 参数对 K 线特征无效，因为它们是固定的 9 个因子

**错误配置 2**（feature 名称错误）：
```yaml
alpha158_config:
  price:
    feature: ["OPEN0", "HIGH0", "LOW0", "VWAP0"]  # ❌ 错误：应该是 ["OPEN", "HIGH", "LOW", "VWAP"]
```

**问题**：
- qlib 会自动为价格特征添加窗口后缀（如 OPEN0, HIGH0 等）
- 配置时应该使用基础名称：`["OPEN", "HIGH", "LOW", "VWAP"]`

---

## 三、完整的正确配置示例

```yaml
data:
  use_alpha158: true
  
  alpha158_config:
    # K线特征（9个因子）- 固定因子，不需要滚动窗口
    kbar: {}  # 空字典表示启用所有 9 个 K 线特征
    
    # 价格特征（4个因子）- 当前价格相对收盘价
    price:
      windows: [0]  # 只使用当前价格
      feature: ["OPEN", "HIGH", "LOW", "VWAP"]  # 基础名称，qlib 会自动添加窗口后缀
    
    # 滚动窗口特征（约 60 个因子）
    rolling:
      windows: [5, 10, 20, 30, 60]  # 滚动窗口
      include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN", "RANK", "RSV", "CORR", "CORD", "SUMP", "VMA"]  # 操作符列表
```

---

## 四、K 线特征说明

### 1. K 线特征列表（9 个）

| 序号 | 因子名称 | 表达式 | 说明 |
|------|---------|--------|------|
| 1 | KMID | `($close-$open)/$open` | 收盘相对开盘的涨跌幅 |
| 2 | KLEN | `($high-$low)/$open` | 振幅（最高-最低）/开盘 |
| 3 | KMID2 | `($close-$open)/($high-$low+1e-12)` | 实体相对振幅的比例 |
| 4 | KUP | `($high-Greater($open, $close))/$open` | 上影线相对开盘 |
| 5 | KUP2 | `($high-Greater($open, $close))/($high-$low+1e-12)` | 上影线相对振幅 |
| 6 | KLOW | `(Less($open, $close)-$low)/$open` | 下影线相对开盘 |
| 7 | KLOW2 | `(Less($open, $close)-$low)/($high-$low+1e-12)` | 下影线相对振幅 |
| 8 | KSFT | `(2*$close-$high-$low)/$open` | 收盘相对高低中点的偏离 |
| 9 | KSFT2 | `(2*$close-$high-$low)/($high-$low+1e-12)` | 收盘相对高低中点的偏离（归一化） |

### 2. 为什么 K 线特征不需要滚动窗口？

- **固定因子**: K 线特征是固定的 9 个因子，基于单日 OHLC 数据计算
- **无历史依赖**: 每个因子只依赖当日的开盘、最高、最低、收盘价
- **不需要窗口**: 不像滚动特征（如 MA, STD）需要历史窗口数据

### 3. 如果需要滚动窗口的 K 线特征怎么办？

如果需要滚动窗口的 K 线特征（如 KMID 的 5 日均值），应该在 `rolling` 配置中使用 `KMID` 作为操作符，而不是在 `kbar` 中配置 `windows`。

---

## 五、验证配置

### 1. 使用测试脚本

运行测试脚本验证配置：

```bash
conda activate qlib_zhengshi
python scripts/test_alpha158_config.py
```

### 2. 查看训练日志

运行训练时，查看日志输出：

```bash
python run_train.py --config config/pipeline.yaml
```

日志中会显示：
```
INFO - 使用自定义 158 因子配置: {'kbar': {}, 'price': {...}, 'rolling': {...}}
INFO - 通过 Alpha158DL.get_feature_config() 成功获取因子列表，共 73 个因子
```

### 3. 检查因子数量

- **K 线特征**: 应该正好 9 个（KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2）
- **价格特征**: 应该正好 4 个（OPEN0, HIGH0, LOW0, VWAP0）
- **滚动特征**: 操作符数量 × 窗口数量（如 12 × 5 = 60 个）

---

## 六、常见问题

### Q1: 为什么配置了 `kbar: {windows: [...]}` 后，K 线特征数量不对？

**A**: K 线特征是固定的 9 个因子，不应该配置 `windows`。如果配置了 `windows`，qlib 可能会尝试生成滚动窗口的 K 线特征，导致因子数量异常。

**解决**: 使用 `kbar: {}` 即可。

### Q2: 为什么 price.feature 配置为 `["OPEN0", "HIGH0", ...]` 会报错？

**A**: qlib 会自动为价格特征添加窗口后缀。配置时应该使用基础名称：`["OPEN", "HIGH", "LOW", "VWAP"]`。

**解决**: 使用 `feature: ["OPEN", "HIGH", "LOW", "VWAP"]`。

### Q3: 如何只使用部分 K 线特征？

**A**: qlib 的 Alpha158 实现中，`kbar` 只支持全部启用（`kbar: {}`）或全部禁用（不配置 `kbar`）。如果需要筛选，可以在 `alpha158_filter` 中配置，或在训练后通过特征重要性筛选。

---

## 七、参考文档

- [Alpha158 因子完整清单](./ALPHA158_FACTOR_LIST.md) - 所有 158 个因子的详细说明
- [Alpha158 使用指南](./ALPHA158_USAGE.md) - 配置和使用方法
- [因子配置总结](./FACTOR_CONFIG_SUMMARY.md) - 当前配置的因子统计

