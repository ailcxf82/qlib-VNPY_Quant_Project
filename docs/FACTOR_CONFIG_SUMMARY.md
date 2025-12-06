# 因子配置总结表

本文档总结当前 `config/data.yaml` 中配置的所有 158 因子。

## 一、配置概览

| 类别 | 因子数量 | 配置方式 | 说明 |
|------|---------|---------|------|
| **K线特征** | 9 个 | `kbar: {}` | 基于 OHLC 的 K 线形态特征 |
| **价格特征** | 4 个 | `price: {windows: [0], feature: [...]}` | 当前价格相对收盘价 |
| **滚动特征** | 60 个 | `rolling: {windows: [...], include: [...]}` | 滚动窗口统计特征 |
| **总计** | **约 73 个** | - | 当前启用的 158 因子 |

---

## 二、详细因子列表

### 1. K线特征 (KBar) - 9 个因子

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

**配置方式**: `kbar: {}`（空字典表示启用所有 9 个因子）

---

### 2. 价格特征 (Price) - 4 个因子

| 序号 | 因子名称 | 表达式 | 说明 |
|------|---------|--------|------|
| 10 | OPEN0 | `$open/$close` | 开盘/收盘 |
| 11 | HIGH0 | `$high/$close` | 最高/收盘 |
| 12 | LOW0 | `$low/$close` | 最低/收盘 |
| 13 | VWAP0 | `$vwap/$close` | 成交量加权均价/收盘 |

**配置方式**: 
```yaml
price:
  windows: [0]  # 只使用当前价格
  feature: ["OPEN", "HIGH", "LOW", "VWAP"]
```

---

### 3. 滚动窗口特征 (Rolling) - 60 个因子

#### 3.1 操作符列表（12 个）

| 操作符 | 说明 | 因子数量 |
|--------|------|---------|
| ROC | 变化率 | 5 个（每个窗口 1 个） |
| MA | 移动平均 | 5 个 |
| STD | 标准差 | 5 个 |
| BETA | 贝塔系数 | 5 个 |
| MAX | 最大值 | 5 个 |
| MIN | 最小值 | 5 个 |
| RANK | 排名 | 5 个 |
| RSV | 相对强弱值 | 5 个 |
| CORR | 相关系数 | 5 个 |
| CORD | 相关系数差值 | 5 个 |
| SUMP | 正数和 | 5 个 |
| VMA | 成交量移动平均 | 5 个 |

#### 3.2 窗口列表（5 个）

- 5 日窗口
- 10 日窗口
- 20 日窗口
- 30 日窗口
- 60 日窗口

#### 3.3 因子命名规则

滚动特征的命名格式为：`{操作符}{窗口}`

例如：
- `ROC5` - 5 日变化率
- `MA20` - 20 日移动平均
- `STD10` - 10 日标准差

**配置方式**:
```yaml
rolling:
  windows: [5, 10, 20, 30, 60]
  include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN", "RANK", "RSV", "CORR", "CORD", "SUMP", "VMA"]
```

**因子数量**: 12 个操作符 × 5 个窗口 = 60 个因子

---

## 三、配置位置

所有因子配置在 `config/data.yaml` 文件的 `alpha158_config` 部分：

```yaml
data:
  use_alpha158: true  # 启用 158 因子
  
  alpha158_config:
    kbar: {}  # K线特征（9个）
    price:
      windows: [0]
      feature: ["OPEN", "HIGH", "LOW", "VWAP"]
    rolling:
      windows: [5, 10, 20, 30, 60]
      include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN", "RANK", "RSV", "CORR", "CORD", "SUMP", "VMA"]
```

---

## 四、因子使用说明

### 4.1 K 线特征

- **特点**: 计算简单，信息量高，不依赖历史数据
- **推荐**: 全部使用（9 个）
- **配置**: `kbar: {}` 即可启用所有

### 4.2 价格特征

- **特点**: 基础价格信息，反映当前价格结构
- **推荐**: 全部使用（4 个）
- **注意**: 如果数据源没有 VWAP，可以去掉 VWAP

### 4.3 滚动特征

- **特点**: 基于历史窗口的统计特征，捕捉趋势和波动
- **推荐**: 根据计算资源选择操作符和窗口
- **当前配置**: 12 个核心操作符 × 5 个窗口 = 60 个因子

---

## 五、因子总数统计

| 类别 | 数量 | 占比 |
|------|------|------|
| K线特征 | 9 | 12.3% |
| 价格特征 | 4 | 5.5% |
| 滚动特征 | 60 | 82.2% |
| **总计** | **73** | **100%** |

---

## 六、修改建议

### 6.1 减少因子数量（如果计算资源有限）

1. **减少滚动窗口**:
   ```yaml
   rolling:
     windows: [5, 10, 20]  # 从 5 个窗口减少到 3 个
   ```
   因子数量：12 × 3 = 36 个（减少 24 个）

2. **减少操作符**:
   ```yaml
   rolling:
     include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN"]  # 从 12 个减少到 6 个
   ```
   因子数量：6 × 5 = 30 个（减少 30 个）

3. **组合优化**:
   ```yaml
   rolling:
     windows: [5, 10, 20]
     include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN"]
   ```
   因子数量：6 × 3 = 18 个（总计：9 + 4 + 18 = 31 个）

### 6.2 增加因子数量（如果需要更多特征）

1. **增加滚动窗口**:
   ```yaml
   rolling:
     windows: [5, 10, 20, 30, 60, 120]  # 增加 120 日窗口
   ```

2. **增加操作符**:
   ```yaml
   rolling:
     include: [..., "RESI", "QTLU", "QTLD", "IMAX", "IMIN", "CNTD", "VSTD"]
   ```

---

## 七、验证配置

运行以下脚本验证配置是否正确：

```bash
conda activate qlib_zhengshi
python scripts/verify_kbar_config.py
```

或者在训练日志中查看实际使用的因子数量：

```bash
python run_train.py --config config/pipeline.yaml
```

训练日志会显示：
- 实际获取的因子数量
- K 线特征数量
- 价格特征数量
- 滚动特征数量

---

## 八、参考文档

- [Alpha158 因子完整清单](./ALPHA158_FACTOR_LIST.md) - 所有 158 个因子的详细说明
- [Alpha158 使用指南](./ALPHA158_USAGE.md) - 配置和使用方法
- [特征重要性指南](./FEATURE_IMPORTANCE_GUIDE.md) - 因子筛选和重要性分析

