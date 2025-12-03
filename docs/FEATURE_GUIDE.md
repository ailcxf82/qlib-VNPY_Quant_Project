# 特征配置指南

## 一、当前特征参数说明

### 技术面特征（已配置）

| 特征表达式 | 含义 | 说明 |
|-----------|------|------|
| `$open` | 开盘价 | 基础行情数据 |
| `$high` | 最高价 | 基础行情数据 |
| `$low` | 最低价 | 基础行情数据 |
| `$close` | 收盘价 | 基础行情数据 |
| `$volume` | 成交量 | 基础行情数据 |
| `Ref($close, 1)/$close - 1` | 前一日收益率 | 昨日收盘相对今日收盘的涨跌幅 |
| `Ref($volume, 1)/$volume - 1` | 成交量变化率 | 昨日成交量相对今日的变化 |
| `Mean($close, 5)/$close - 1` | 5日均价偏离度 | 当前价格相对5日均价的偏离 |
| `Std($close, 10)` | 10日波动率 | 收盘价的10日标准差 |

### 标签说明

- `Ref($close, -5)/$close - 1`：未来5日收益率（用于监督学习）

## 二、添加基本面数据

### 1. 检查 qlib 数据源可用字段

在 Python 中运行以下代码检查可用字段：

```python
import qlib
from qlib.data import D

# 初始化 qlib
qlib.init(provider_uri="D:/qlib_data/qlib_data", region="cn")

# 查看所有可用字段
instruments = D.instruments("csi300")
all_fields = D.list_instruments(instruments, start_time="2020-01-01", end_time="2020-01-10")
print("可用字段示例：", D.features(instruments[:1], fields=["$close"], start_time="2020-01-01", end_time="2020-01-10").columns.tolist())

# 或者直接尝试提取某个字段看是否报错
try:
    test_data = D.features(instruments[:1], fields=["$pe"], start_time="2020-01-01", end_time="2020-01-10")
    print("$pe 字段可用")
except Exception as e:
    print(f"$pe 字段不可用: {e}")
```

### 2. 常见基本面字段（qlib 标准字段）

如果您的 qlib 数据源包含基本面数据，以下字段通常可用：

**估值指标：**
- `$pe` - 市盈率
- `$pb` - 市净率
- `$ps` - 市销率
- `$pcf` - 市现率

**盈利能力：**
- `$roe` - 净资产收益率
- `$roa` - 总资产收益率
- `$roic` - 投入资本回报率
- `$gross_profit_margin` - 毛利率
- `$net_profit_margin` - 净利率

**成长性：**
- `$revenue_growth` - 营收增长率
- `$net_profit_growth` - 净利润增长率
- `$eps_growth` - 每股收益增长率

**财务质量：**
- `$current_ratio` - 流动比率
- `$quick_ratio` - 速动比率
- `$debt_to_equity` - 资产负债率
- `$asset_turnover` - 资产周转率

### 3. 如果字段不存在，如何添加自定义数据

#### 方法一：导入自定义数据到 qlib

如果您的 qlib 数据源不包含某些基本面数据，需要先将数据导入 qlib：

```python
import qlib
from qlib.data import D
import pandas as pd

# 1. 准备自定义数据（示例：财务指标）
# 数据格式：MultiIndex (datetime, instrument)，列为指标名
custom_data = pd.DataFrame({
    'custom_pe': [...],  # 自定义PE数据
    'custom_roe': [...], # 自定义ROE数据
}, index=pd.MultiIndex.from_tuples([...], names=['datetime', 'instrument']))

# 2. 使用 qlib 的数据导入工具（需要根据 qlib 版本调整）
# 参考 qlib 文档：https://qlib.readthedocs.io/
```

#### 方法二：在特征管道中合并外部数据

修改 `feature/qlib_feature_pipeline.py`，在 `build()` 方法中合并外部数据：

```python
def build(self):
    # ... 原有代码 ...
    
    # 加载外部基本面数据
    fundamental_df = self._load_fundamental_data()  # 自定义方法
    if fundamental_df is not None:
        # 合并到特征中
        feature_panel = feature_panel.join(fundamental_df, how="left")
    
    # ... 后续处理 ...
```

## 三、添加情绪面数据

### 1. 常见情绪面字段

**资金流向（如果 qlib 数据源包含）：**
- `$net_inflow` - 净流入资金
- `$main_net_inflow` - 主力净流入
- `$large_net_inflow` - 大单净流入
- `$medium_net_inflow` - 中单净流入
- `$small_net_inflow` - 小单净流入

**换手率：**
- `$turnover` 或 `$turnover_ratio` - 换手率

**涨跌停：**
- `$is_limit_up` - 是否涨停（0/1）
- `$is_limit_down` - 是否跌停（0/1）

**融资融券：**
- `$margin_balance` - 融资余额
- `$short_balance` - 融券余额

**北向资金：**
- `$north_net_inflow` - 北向资金净流入

### 2. 情绪面衍生特征

即使没有直接的情绪数据字段，也可以通过价格和成交量构造情绪指标：

```yaml
# 情绪指标衍生特征
- "($close - $open) / $open"  # 日内涨跌幅（情绪强度）
- "($high - $low) / $close"   # 振幅（波动情绪）
- "($close - Mean($close, 20)) / Std($close, 20)"  # 价格Z-score（偏离度）
- "$volume / Mean($volume, 20) - 1"  # 成交量相对20日均值的偏离度
- "($close - Ref($close, 1)) * $volume"  # 价量配合度
```

### 3. 添加外部情绪数据（如新闻情绪）

如果需要添加外部情绪数据（如新闻情绪、社交媒体情绪），需要：

1. **准备数据文件**：CSV 格式，包含 `datetime`, `instrument`, `sentiment_score` 等列

2. **修改特征管道**：在 `feature/qlib_feature_pipeline.py` 中添加加载逻辑：

```python
def _load_sentiment_data(self) -> pd.DataFrame:
    """加载外部情绪数据"""
    sentiment_path = self.config.get("sentiment_data_path")
    if sentiment_path and os.path.exists(sentiment_path):
        df = pd.read_csv(sentiment_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['datetime', 'instrument'])
        return df[['sentiment_score']]  # 返回情绪分数列
    return None

def build(self):
    # ... 原有代码 ...
    
    # 加载情绪数据
    sentiment_df = self._load_sentiment_data()
    if sentiment_df is not None:
        feature_panel = feature_panel.join(sentiment_df, how="left")
    
    # ... 后续处理 ...
```

3. **在配置文件中添加路径**：

```yaml
data:
  sentiment_data_path: "data/sentiment/sentiment_scores.csv"  # 情绪数据路径
  # ... 其他配置 ...
```

## 四、验证特征是否可用

在添加新特征后，建议先测试特征提取是否成功：

```python
from feature.qlib_feature_pipeline import QlibFeaturePipeline

# 初始化管道
pipeline = QlibFeaturePipeline("config/data.yaml")

# 构建特征
pipeline.build()

# 获取特征
features, labels = pipeline.get_all()

# 检查特征列
print("特征列：", features.columns.tolist())
print("特征形状：", features.shape)
print("缺失值统计：", features.isnull().sum())

# 检查是否有特征全为 NaN（说明字段不存在）
invalid_features = features.columns[features.isnull().all()].tolist()
if invalid_features:
    print(f"警告：以下特征全为 NaN（可能字段不存在）：{invalid_features}")
```

## 五、注意事项

1. **字段名称可能不同**：不同版本的 qlib 数据源，字段名称可能略有差异。建议先用小样本测试。

2. **数据频率**：基本面数据通常是季度或年度更新，需要处理缺失值。可以使用 `ffill()` 前向填充。

3. **数据对齐**：确保外部数据的时间索引和股票代码与 qlib 数据对齐。

4. **特征数量**：添加过多特征可能导致过拟合，建议：
   - 先用少量特征测试
   - 使用特征重要性分析（LightGBM 提供）
   - 逐步添加特征并观察模型性能

5. **计算成本**：复杂表达式（如多日滚动计算）会增加特征提取时间。

## 六、常用表达式参考

```yaml
# 时间序列函数
- "Ref($close, N)"        # N 天前的值
- "Mean($close, N)"        # N 日均值
- "Std($close, N)"         # N 日标准差
- "Max($high, N)"          # N 日最大值
- "Min($low, N)"           # N 日最小值
- "Sum($volume, N)"        # N 日成交量总和

# 比率计算
- "$close / $open - 1"     # 日内收益率
- "$high / $low - 1"       # 日内振幅
- "$volume / Mean($volume, 20)"  # 成交量相对强度

# 技术指标（如果 qlib 支持）
- "RSI($close, 14)"        # RSI 指标（如果支持）
- "MACD($close)"           # MACD 指标（如果支持）
```

## 七、推荐的特征组合策略

1. **技术面 + 基本面**：结合价格趋势和公司基本面
2. **短期 + 长期**：同时使用短期（5日）和长期（20日、60日）指标
3. **绝对 + 相对**：既有绝对值（如 PE），也有相对值（如 PE 相对历史均值）
4. **价量结合**：价格特征配合成交量特征


