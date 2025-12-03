# Alpha158 因子完整清单

本文档列出 qlib Alpha158 的所有因子及其分类，便于选择性使用。

## 因子分类统计

- **K线特征 (KBar)**: 9 个
- **价格特征 (Price)**: 4 个字段 × 1 个窗口 = 4 个（默认配置）
- **成交量特征 (Volume)**: 0 个（默认配置中未启用）
- **滚动特征 (Rolling)**: 约 145 个（5个窗口 × 29个操作符）

**总计**: 约 158 个因子

---

## 一、K线特征 (KBar) - 9 个因子

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

**推荐**: 全部使用（计算简单，信息量高）

---

## 二、价格特征 (Price) - 4 个因子（默认配置）

默认配置只使用窗口 [0]，即当前价格相对收盘价的比率。

| 序号 | 因子名称 | 表达式 | 说明 |
|------|---------|--------|------|
| 10 | OPEN0 | `$open/$close` | 开盘/收盘 |
| 11 | HIGH0 | `$high/$close` | 最高/收盘 |
| 12 | LOW0 | `$low/$close` | 最低/收盘 |
| 13 | VWAP0 | `$vwap/$close` | 成交量加权均价/收盘 |

**推荐**: 全部使用（基础价格信息）

**扩展**: 如果使用 `windows: [0, 1, 2, 3, 4]`，会生成 4×5=20 个因子

---

## 三、成交量特征 (Volume) - 0 个（默认未启用）

默认配置中未启用成交量特征。如果启用，会生成：

| 因子名称模式 | 表达式模式 | 说明 |
|------------|-----------|------|
| VOLUME0 | `$volume/($volume+1e-12)` | 当前成交量（归一化） |
| VOLUME1 | `Ref($volume, 1)/($volume+1e-12)` | 前1日成交量相对当前 |
| VOLUME2 | `Ref($volume, 2)/($volume+1e-12)` | 前2日成交量相对当前 |
| ... | ... | ... |

**推荐**: 根据需要启用，通常使用窗口 [0, 1, 2, 3, 4] 生成 5 个因子

---

## 四、滚动特征 (Rolling) - 145 个因子

滚动特征使用 5 个窗口 [5, 10, 20, 30, 60] 和 29 个操作符，共生成约 145 个因子。

### 4.1 基础统计类（推荐使用）

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **ROC** | ROC5, ROC10, ROC20, ROC30, ROC60 | `Ref($close, d)/$close` | 价格变化率 | ⭐⭐⭐⭐⭐ |
| **MA** | MA5, MA10, MA20, MA30, MA60 | `Mean($close, d)/$close` | 移动平均相对当前价格 | ⭐⭐⭐⭐⭐ |
| **STD** | STD5, STD10, STD20, STD30, STD60 | `Std($close, d)/$close` | 波动率 | ⭐⭐⭐⭐⭐ |
| **BETA** | BETA5, BETA10, BETA20, BETA30, BETA60 | `Slope($close, d)/$close` | 价格趋势斜率 | ⭐⭐⭐⭐ |
| **MAX** | MAX5, MAX10, MAX20, MAX30, MAX60 | `Max($high, d)/$close` | 最高价相对当前 | ⭐⭐⭐⭐ |
| **MIN** | MIN5, MIN10, MIN20, MIN30, MIN60 | `Min($low, d)/$close` | 最低价相对当前 | ⭐⭐⭐⭐ |

**推荐**: 使用这 6 个操作符，共 6×5=30 个因子

### 4.2 回归分析类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **RSQR** | RSQR5, RSQR10, RSQR20, RSQR30, RSQR60 | `Rsquare($close, d)` | 线性回归R² | ⭐⭐⭐ |
| **RESI** | RESI5, RESI10, RESI20, RESI30, RESI60 | `Resi($close, d)/$close` | 线性回归残差 | ⭐⭐⭐ |

**推荐**: 可选，共 2×5=10 个因子

### 4.3 分位数类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **QTLU** | QTLU5, QTLU10, QTLU20, QTLU30, QTLU60 | `Quantile($close, d, 0.8)/$close` | 80%分位数 | ⭐⭐⭐ |
| **QTLD** | QTLD5, QTLD10, QTLD20, QTLD30, QTLD60 | `Quantile($close, d, 0.2)/$close` | 20%分位数 | ⭐⭐⭐ |
| **RANK** | RANK5, RANK10, RANK20, RANK30, RANK60 | `Rank($close, d)` | 当前价格排名百分位 | ⭐⭐⭐⭐ |
| **RSV** | RSV5, RSV10, RSV20, RSV30, RSV60 | `($close-Min($low, d))/(Max($high, d)-Min($low, d)+1e-12)` | 随机指标（KDJ中的RSV） | ⭐⭐⭐⭐ |

**推荐**: 使用 RANK 和 RSV，共 2×5=10 个因子

### 4.4 时间位置类（Aroon 指标相关）

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **IMAX** | IMAX5, IMAX10, IMAX20, IMAX30, IMAX60 | `IdxMax($high, d)/d` | 距离最高价的天数 | ⭐⭐⭐ |
| **IMIN** | IMIN5, IMIN10, IMIN20, IMIN30, IMIN60 | `IdxMin($low, d)/d` | 距离最低价的天数 | ⭐⭐⭐ |
| **IMXD** | IMXD5, IMXD10, IMXD20, IMXD30, IMXD60 | `(IdxMax($high, d)-IdxMin($low, d))/d` | 最高最低价时间差 | ⭐⭐ |

**推荐**: 可选，共 3×5=15 个因子

### 4.5 相关性类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **CORR** | CORR5, CORR10, CORR20, CORR30, CORR60 | `Corr($close, Log($volume+1), d)` | 价格与成交量对数相关性 | ⭐⭐⭐⭐ |
| **CORD** | CORD5, CORD10, CORD20, CORD30, CORD60 | `Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), d)` | 价格变化与成交量变化相关性 | ⭐⭐⭐⭐ |

**推荐**: 全部使用，共 2×5=10 个因子

### 4.6 涨跌统计类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **CNTP** | CNTP5, CNTP10, CNTP20, CNTP30, CNTP60 | `Mean($close>Ref($close, 1), d)` | 上涨天数占比 | ⭐⭐⭐ |
| **CNTN** | CNTN5, CNTN10, CNTN20, CNTN30, CNTN60 | `Mean($close<Ref($close, 1), d)` | 下跌天数占比 | ⭐⭐⭐ |
| **CNTD** | CNTD5, CNTD10, CNTD20, CNTD30, CNTD60 | `Mean($close>Ref($close, 1), d)-Mean($close<Ref($close, 1), d)` | 涨跌天数差 | ⭐⭐⭐ |

**推荐**: 可选，共 3×5=15 个因子

### 4.7 RSI 相关类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **SUMP** | SUMP5, SUMP10, SUMP20, SUMP30, SUMP60 | `Sum(Greater($close-Ref($close, 1), 0), d)/(Sum(Abs($close-Ref($close, 1)), d)+1e-12)` | 上涨幅度占比（类似RSI） | ⭐⭐⭐⭐ |
| **SUMN** | SUMN5, SUMN10, SUMN20, SUMN30, SUMN60 | `Sum(Greater(Ref($close, 1)-$close, 0), d)/(Sum(Abs($close-Ref($close, 1)), d)+1e-12)` | 下跌幅度占比 | ⭐⭐⭐ |
| **SUMD** | SUMD5, SUMD10, SUMD20, SUMD30, SUMD60 | `(Sum(Greater($close-Ref($close, 1), 0), d)-Sum(Greater(Ref($close, 1)-$close, 0), d))/(Sum(Abs($close-Ref($close, 1)), d)+1e-12)` | 涨跌幅度差 | ⭐⭐⭐ |

**推荐**: 使用 SUMP（类似 RSI），共 1×5=5 个因子

### 4.8 成交量统计类

| 操作符 | 因子名称模式 | 表达式模式 | 说明 | 推荐度 |
|--------|------------|-----------|------|--------|
| **VMA** | VMA5, VMA10, VMA20, VMA30, VMA60 | `Mean($volume, d)/($volume+1e-12)` | 成交量移动平均 | ⭐⭐⭐⭐ |
| **VSTD** | VSTD5, VSTD10, VSTD20, VSTD30, VSTD60 | `Std($volume, d)/($volume+1e-12)` | 成交量波动率 | ⭐⭐⭐ |
| **WVMA** | WVMA5, WVMA10, WVMA20, WVMA30, WVMA60 | `Std(Abs($close/Ref($close, 1)-1)*$volume, d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, d)+1e-12)` | 成交量加权价格波动 | ⭐⭐⭐ |
| **VSUMP** | VSUMP5, VSUMP10, VSUMP20, VSUMP30, VSUMP60 | `Sum(Greater($volume-Ref($volume, 1), 0), d)/(Sum(Abs($volume-Ref($volume, 1)), d)+1e-12)` | 成交量上涨占比 | ⭐⭐⭐ |
| **VSUMN** | VSUMN5, VSUMN10, VSUMN20, VSUMN30, VSUMN60 | `Sum(Greater(Ref($volume, 1)-$volume, 0), d)/(Sum(Abs($volume-Ref($volume, 1)), d)+1e-12)` | 成交量下跌占比 | ⭐⭐ |
| **VSUMD** | VSUMD5, VSUMD10, VSUMD20, VSUMD30, VSUMD60 | `(Sum(Greater($volume-Ref($volume, 1), 0), d)-Sum(Greater(Ref($volume, 1)-$volume, 0), d))/(Sum(Abs($volume-Ref($volume, 1)), d)+1e-12)` | 成交量涨跌差 | ⭐⭐ |

**推荐**: 使用 VMA，共 1×5=5 个因子

---

## 五、推荐配置方案

### 方案1：精简版（约 50 个因子）

```yaml
alpha158_config:
  kbar: {}  # 9 个因子
  price:
    windows: [0]
    feature: ["OPEN", "HIGH", "LOW", "VWAP"]  # 4 个因子
  rolling:
    windows: [5, 10, 20, 30, 60]
    include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN", "RANK", "RSV", "CORR", "CORD", "SUMP", "VMA"]  # 12×5=60 个因子
```

**总计**: 9 + 4 + 60 = 73 个因子

### 方案2：标准版（约 80 个因子）

在方案1基础上，增加：
- RSQR, RESI（回归分析）
- QTLU, QTLD（分位数）
- IMAX, IMIN（时间位置）
- CNTP, CNTN, CNTD（涨跌统计）
- VSTD（成交量波动）

**总计**: 约 100 个因子

### 方案3：完整版（158 个因子）

使用所有因子（默认配置）

---

## 六、因子筛选建议

1. **优先使用**: KBar (9) + Price (4) + 基础统计 (30) + 相关性 (10) + RSI (5) + 成交量 (5) = **63 个核心因子**

2. **根据数据质量选择**:
   - 如果数据源不包含 VWAP，去掉 VWAP0
   - 如果数据质量较差，减少滚动窗口数量（如只用 [5, 10, 20]）

3. **根据计算资源选择**:
   - 资源充足：使用完整 158 因子
   - 资源有限：使用精简版 50-80 个因子
   - 资源紧张：只使用核心 30-40 个因子

4. **特征重要性筛选**:
   - 训练后使用 LightGBM 的 `feature_importances_` 筛选重要特征
   - 保留重要性前 50-100 个因子


