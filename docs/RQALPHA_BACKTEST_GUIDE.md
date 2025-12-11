# RQAlpha 回测系统说明文档

## 一、整体架构

```
run_backtest.py
    ↓
rqalpha_backtest.py (回测执行器)
    ↓
    ├─ 预处理预测文件（代码格式转换）
    ├─ 构建 RQAlpha 配置
    ├─ 执行回测（调用 RQAlpha）
    └─ 提取和保存结果
        ↓
rqalpha_strategy.py (策略脚本，在 RQAlpha 框架内运行)
    ├─ init()：初始化，加载预测信号
    ├─ before_trading()：每日开盘前，构建目标组合
    ├─ handle_bar()：每根K线更新时，执行调仓
    └─ after_trading()：收盘后记录信息
```

## 二、rqalpha_backtest.py 详解

### 2.1 核心功能

`rqalpha_backtest.py` 是回测系统的**执行器**，负责：
1. **配置加载**：读取 RQAlpha 配置文件
2. **预测文件预处理**：将 Qlib 格式的股票代码转换为 RQAlpha 格式
3. **回测执行**：调用 RQAlpha 框架执行策略
4. **结果提取**：从回测结果中提取详细数据
5. **结果保存**：保存持仓、交易、盈亏、效率指标等

### 2.2 主要函数

#### `run_rqalpha_backtest()` - 主函数

**功能**：执行完整的回测流程

**流程**：
```
1. 加载配置文件（config/rqalpha_config.yaml）
2. 读取预测文件，确定回测日期范围
3. 预处理预测文件（代码格式转换）
   - Qlib 格式：SH600000, SZ000001
   - RQAlpha 格式：600000.XSHG, 000001.XSHE
4. 加载行业映射文件（可选）
5. 构建 RQAlpha 配置字典
   - base：回测日期、初始资金、基准
   - mod：账户、进度、分析器、模拟器
   - extra：策略参数（通过 context_vars 传递）
6. 执行回测（调用 run_file()）
7. 提取详细结果（持仓、交易、盈亏、效率指标）
8. 生成图表（净值曲线）
```

**关键配置传递**：
```python
# 策略参数通过 extra.context_vars 传递
strategy_params = {
    "prediction_file": temp_prediction_path,      # 预测文件路径
    "max_position": 0.3,                          # 最大仓位（30%）
    "max_stock_weight": 0.05,                     # 单股最大权重（5%）
    "max_industry_weight": 0.2,                   # 行业最大权重（20%）
    "top_k": 50,                                  # Top-K 选股数量
    "rebalance_interval": 3,                      # 调仓频率（3天）
    "industry_map": {...}                         # 行业映射（可选）
}
```

#### `convert_qlib_code_to_rqalpha()` - 代码格式转换

**功能**：将 Qlib 格式的股票代码转换为 RQAlpha 格式

**转换规则**：
- `SH600000` → `600000.XSHG`（上海）
- `SZ000001` → `000001.XSHE`（深圳）
- `600000` → `600000.XSHG`（自动判断交易所）
- `000001` → `000001.XSHE`（自动判断交易所）

**判断逻辑**：
- 6、9 开头 → 上海（XSHG）
- 0、3 开头 → 深圳（XSHE）
- 688、689 开头 → 科创板（上海，XSHG）
- 300 开头 → 创业板（深圳，XSHE）

#### `prepare_prediction_file()` - 预测文件预处理

**功能**：读取预测文件，转换代码格式，保存为 RQAlpha 可用的格式

**输入格式**：
```csv
datetime,instrument,final
2021-10-01,SH600000,0.85
2021-10-01,SZ000001,0.72
```

**输出格式**：
```csv
datetime,rq_code,final
2021-10-01,600000.XSHG,0.85
2021-10-01,000001.XSHE,0.72
```

#### `extract_and_save_detailed_results()` - 结果提取

**功能**：从 RQAlpha 回测结果中提取详细数据并保存

**提取内容**：
1. **持仓明细**（`positions_detail.csv`）
   - 每日持仓记录
   - 包含股票代码、数量、市值等

2. **交易明细**（`trades_detail.csv`）
   - 每笔交易记录
   - 包含买入/卖出、价格、数量、手续费等
   - **增强信息**：预测值、成本、收益、盈亏

3. **盈亏状态**（`detailed_results.json`）
   - 初始资金
   - 最终资金
   - 总收益
   - 总收益率

4. **效率指标**（`detailed_results.json`）
   - 年化收益率
   - 夏普比率
   - 最大回撤
   - 胜率等

#### `enhance_trades_with_prediction_and_pnl()` - 交易明细增强

**功能**：为交易明细添加预测值、成本和收益信息

**添加字段**：
- `prediction_value`：买入时的预测值
- `cost`：买入成本（价格 × 数量 + 手续费 + 税费）
- `profit`：卖出收益（价格 × 数量 - 手续费 - 税费）
- `pnl`：盈亏（profit - 成本）

**计算逻辑**：
- **买入**：`cost = quantity × price + commission + tax`
- **卖出**：使用平均成本法计算盈亏
  - `cost_basis = quantity × avg_cost`（平均成本）
  - `profit = net_amount - cost_basis`
  - `pnl = profit`

#### `generate_and_save_plot()` - 图表生成

**功能**：从回测结果中提取净值数据，生成净值曲线图

**图表内容**：
- 策略净值曲线
- 基准净值曲线（如沪深300）
- 对比分析

**保存位置**：`data/backtest/rqalpha/rqalpha_strategy_plot.png`

## 三、rqalpha_strategy.py 详解

### 3.1 核心功能

`rqalpha_strategy.py` 是**策略脚本**，在 RQAlpha 框架内运行，实现：
1. **信号加载**：从预测文件中读取每日预测信号
2. **组合构建**：根据预测信号构建目标组合（Top-K + 权重分配）
3. **调仓执行**：根据目标组合执行买卖操作
4. **T+1 交易**：遵守 T+1 交易规则（当日买入，次日可卖出）

### 3.2 策略生命周期

RQAlpha 策略有四个关键函数，按时间顺序执行：

```
每日交易流程：
    ↓
init() [仅执行一次]
    ↓
before_trading() [每日开盘前]
    ↓
handle_bar() [每根K线更新时，可能多次]
    ↓
after_trading() [每日收盘后]
```

### 3.3 主要函数

#### `init(context)` - 策略初始化

**执行时机**：回测开始时，仅执行一次

**功能**：
1. **加载预测信号**
   ```python
   # 从配置中读取预测文件路径
   prediction_file = context.config.extra.context_vars["prediction_file"]
   # 读取并解析预测文件
   df = pd.read_csv(prediction_file)
   # 按日期索引存储信号
   context.prediction_signals = {
       date: {code: signal_value, ...}
   }
   ```

2. **加载组合配置**
   ```python
   context.portfolio_config = {
       "max_position": 0.3,        # 最大仓位 30%
       "max_stock_weight": 0.05,   # 单股最大权重 5%
       "max_industry_weight": 0.2, # 行业最大权重 20%
       "top_k": 50                 # Top-K 选股数量
   }
   ```

3. **加载行业映射**（可选）
   ```python
   context.industry_map = {
       "600000.XSHG": "银行",
       "000001.XSHE": "银行",
       ...
   }
   ```

4. **订阅股票**
   ```python
   # 订阅所有可能交易的股票
   subscribe(list(all_codes))
   ```

5. **初始化调仓参数**
   ```python
   context.rebalance_interval = 3  # 每3天调仓一次
   context.last_rebalance_date = None
   ```

#### `before_trading(context)` - 开盘前处理

**执行时机**：每日开盘前（9:30 之前）

**功能**：
1. **获取当日预测信号**
   ```python
   current_date = context.now.date()
   signals = context.prediction_signals.get(current_date, {})
   ```

2. **构建目标组合**
   ```python
   target_weights = build_portfolio(
       signals,
       context.portfolio_config,
       context.industry_map
   )
   ```

3. **调仓频率控制**
   ```python
   # 检查是否需要调仓
   last_date = context.last_rebalance_date
   delta_days = (current_date - last_date).days if last_date else None
   should_rebalance = (last_date is None) or (delta_days >= rebalance_interval)
   
   if should_rebalance:
       context.target_weights = target_weights
       context.need_rebalance = True
   else:
       context.target_weights = {}
       context.need_rebalance = False
   ```

#### `handle_bar(context, bar_dict)` - K线更新处理

**执行时机**：每根K线更新时（可能多次，如分钟线、日线）

**功能**：
1. **绘制净值曲线**
   ```python
   # 计算策略净值
   nav = portfolio.unit_net_value
   plot("strategy_nav", nav)
   
   # 计算基准净值
   benchmark_nav = benchmark_portfolio.unit_net_value
   plot("benchmark_nav", benchmark_nav)
   ```

2. **执行调仓**（仅在满足条件时）
   ```python
   if context.need_rebalance and not context.rebalanced:
       rebalance_portfolio(context, bar_dict)
       context.rebalanced = True
       context.last_rebalance_date = current_date
   ```

#### `after_trading(context)` - 收盘后处理

**执行时机**：每日收盘后（15:00 之后）

**功能**：
1. **记录持仓信息**
   ```python
   positions = get_positions()
   total_value = portfolio.total_value
   cash = account.cash
   ```

2. **检查未成交订单**（T+1 交易）
   ```python
   open_orders = get_open_orders()
   # T+1 交易：当日下单，次日成交
   ```

### 3.4 组合构建逻辑

#### `build_portfolio()` - 组合构建

**功能**：根据预测信号构建目标组合权重

**流程**：
```
1. 排序并取 Top-K
   - 按预测值从高到低排序
   - 取前 top_k 只股票

2. 线性衰减权重
   - 第1名权重 = top_k / top_k = 1.0
   - 第2名权重 = (top_k-1) / top_k
   - 第k名权重 = 1 / top_k
   - 归一化：weights = weights / sum(weights) × max_position

3. 单股权重裁剪
   - 限制单股最大权重：min(weight, max_stock_weight)
   - 重新归一化

4. 行业权重裁剪（如果提供行业映射）
   - 限制行业最大权重：min(industry_weight, max_industry_weight)
   - 重新归一化
```

**示例**：
```python
# 假设 top_k=5, max_position=0.3
signals = {
    "600000.XSHG": 0.9,
    "000001.XSHE": 0.8,
    "600036.XSHG": 0.7,
    "000002.XSHE": 0.6,
    "600519.XSHG": 0.5
}

# 1. 排序并取 Top-5
filtered = {
    "600000.XSHG": 0.9,
    "000001.XSHE": 0.8,
    "600036.XSHG": 0.7,
    "000002.XSHE": 0.6,
    "600519.XSHG": 0.5
}

# 2. 线性衰减权重
weights = {
    "600000.XSHG": 5/5 = 1.0,
    "000001.XSHE": 4/5 = 0.8,
    "600036.XSHG": 3/5 = 0.6,
    "000002.XSHE": 2/5 = 0.4,
    "600519.XSHG": 1/5 = 0.2
}
# 归一化：sum = 3.0
weights = {k: v/3.0 * 0.3 for k, v in weights.items()}
# 结果：{0.1, 0.08, 0.06, 0.04, 0.02}

# 3. 单股权重裁剪（假设 max_stock_weight=0.05）
weights = {k: min(v, 0.05) for k, v in weights.items()}
# 结果：{0.05, 0.05, 0.05, 0.04, 0.02}
# 重新归一化：sum = 0.21，归一化到 0.3
weights = {k: v/0.21 * 0.3 for k, v in weights.items()}
```

#### `apply_industry_constraint()` - 行业约束

**功能**：限制单个行业的权重上限

**流程**：
```
1. 按行业分组，计算行业总权重
2. 如果行业权重 > max_industry_weight，按比例缩放
3. 重新归一化
```

### 3.5 调仓执行逻辑

#### `rebalance_portfolio()` - 调仓执行

**功能**：根据目标权重执行买卖操作

**流程**：
```
1. 获取当前持仓
   positions = get_positions()
   current_positions = {code for code in positions if quantity > 0}

2. 计算需要调仓的股票
   all_codes = target_weights.keys() | current_positions

3. 对每只股票执行调仓
   for code in all_codes:
       target_weight = target_weights.get(code, 0.0)
       order_target_percent(code, target_weight)
       # RQAlpha 会自动处理：
       # - 如果目标权重 > 当前权重：买入
       # - 如果目标权重 < 当前权重：卖出
       # - 如果目标权重 = 0：清仓
       # - 停牌、涨跌停等不可交易情况：订单被拒绝
```

**关键点**：
- 使用 `order_target_percent()` 下单，目标权重是相对于总资产的比例
- RQAlpha 会自动计算需要买入/卖出的数量
- 不需要手动检查停牌、涨跌停等，RQAlpha 会处理

## 四、T+1 交易规则

### 4.1 规则说明

**T+1 交易**：当日买入的股票，次日才能卖出

**实现方式**：
```yaml
# config/rqalpha_config.yaml
trading:
  day_trade: false  # false 表示 T+1 交易
```

**代码实现**：
```python
# rqalpha_backtest.py
if not trading_config.get("day_trade", False):
    config_dict["mod"]["sys_simulation"]["matching_type"] = "next_bar"
    # "next_bar" 表示当日下单，次日成交
```

### 4.2 交易流程

```
第1天（T日）：
  - before_trading：构建目标组合
  - handle_bar：执行买入订单
  - after_trading：订单未成交（T+1 规则）

第2天（T+1日）：
  - before_trading：订单成交，持仓更新
  - handle_bar：可以卖出（已持有1天）
  - after_trading：持仓已建立
```

## 五、调仓频率控制

### 5.1 配置方式

```yaml
# config/rqalpha_config.yaml
extra:
  context_vars:
    rebalance_interval: 3  # 每3天调仓一次
```

### 5.2 实现逻辑

```python
# rqalpha_strategy.py
def before_trading(context):
    current_date = context.now.date()
    last_date = context.last_rebalance_date
    delta_days = (current_date - last_date).days if last_date else None
    
    # 检查是否需要调仓
    should_rebalance = (last_date is None) or (delta_days >= rebalance_interval)
    
    if should_rebalance:
        context.target_weights = target_weights
        context.need_rebalance = True
    else:
        context.target_weights = {}
        context.need_rebalance = False
```

## 六、输出文件说明

### 6.1 回测结果文件

回测完成后，在 `data/backtest/rqalpha/` 目录下生成：

1. **`report.json`**
   - RQAlpha 标准回测报告
   - 包含收益率、夏普比率、最大回撤等指标

2. **`summary.json`**
   - 回测摘要（从 result.summary 提取）

3. **`positions_detail.csv`**
   - 持仓明细
   - 包含每日持仓记录

4. **`trades_detail.csv`**
   - 交易明细（增强版）
   - 包含：买入/卖出、价格、数量、手续费、**预测值**、**成本**、**收益**、**盈亏**

5. **`detailed_results.json`**
   - 详细回测结果
   - 包含：投资明细、盈亏状态、效率指标

6. **`rqalpha_strategy_plot.png`**
   - 净值曲线图
   - 策略净值 vs 基准净值

7. **`rqalpha_pred_*.csv`**
   - 转换后的预测文件（供后续分析使用）

### 6.2 文件格式示例

**`trades_detail.csv`**（增强版）：
```csv
datetime,order_book_id,side,last_price,last_quantity,commission,tax,
prediction_value,cost,profit,pnl
2021-10-01,600000.XSHG,BUY,10.50,1000,5.25,0.00,0.85,10505.25,,
2021-10-02,600000.XSHG,SELL,11.00,1000,5.50,11.00,0.85,,10983.50,478.25
```

**`detailed_results.json`**：
```json
{
  "盈亏状态": {
    "初始资金": 10000000.0,
    "最终资金": 12000000.0,
    "总收益": 2000000.0,
    "总收益率": 20.0
  },
  "效率指标": {
    "annual_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    ...
  }
}
```

## 七、使用流程

### 7.1 准备预测文件

```bash
# 运行预测，生成预测文件
python run_predict.py --start 2021-10-01 --end 2023-10-31

# 预测文件保存在：data/predictions/pred_*.csv
```

### 7.2 配置回测参数

编辑 `config/rqalpha_config.yaml`：
```yaml
base:
  start_date: "2021-10-01"
  end_date: "2023-10-31"
  initial_cash: 10000000

risk:
  max_position: 0.3
  max_stock_weight: 0.05
  max_industry_weight: 0.2
  top_k: 50

extra:
  context_vars:
    rebalance_interval: 3
```

### 7.3 执行回测

```bash
# 方式1：使用 run_backtest.py（推荐）
python run_backtest.py \
    --rqalpha-config config/rqalpha_config.yaml \
    --prediction data/predictions/pred_20230930_2021-10-01_2023-10-31.csv

# 方式2：直接调用 rqalpha_backtest.py
python -m backtest.rqalpha_backtest \
    --rqalpha-config config/rqalpha_config.yaml \
    --prediction data/predictions/pred_*.csv
```

### 7.4 查看结果

```bash
# 查看详细回测结果
python scripts/analyze_backtest_results.py --output-dir data/backtest/rqalpha

# 查看净值曲线图
# 打开：data/backtest/rqalpha/rqalpha_strategy_plot.png
```

## 八、关键概念总结

### 8.1 代码格式转换

- **Qlib 格式**：`SH600000`, `SZ000001`
- **RQAlpha 格式**：`600000.XSHG`, `000001.XSHE`
- **自动转换**：`rqalpha_backtest.py` 会自动处理

### 8.2 组合构建流程

```
预测信号 → Top-K 选股 → 线性衰减权重 → 归一化 → 单股权重裁剪 → 行业权重裁剪 → 目标组合
```

### 8.3 调仓执行流程

```
目标组合 → 计算目标权重 → order_target_percent() → RQAlpha 自动撮合 → 持仓更新
```

### 8.4 T+1 交易

- **当日下单**：`handle_bar()` 中执行 `order_target_percent()`
- **次日成交**：订单在次日开盘时成交
- **持仓建立**：T+1 日后才能卖出

### 8.5 调仓频率

- **默认**：每日调仓（`rebalance_interval=1`）
- **可配置**：每 N 天调仓一次（`rebalance_interval=N`）
- **实现**：在 `before_trading()` 中检查距离上次调仓的天数

## 九、常见问题

### Q1: 为什么交易明细中没有持仓？

**A**: T+1 交易规则下，当日下单次日成交。如果查看当日的交易明细，可能看到订单但无持仓，这是正常的。

### Q2: 如何调整组合参数？

**A**: 修改 `config/rqalpha_config.yaml` 中的 `risk` 部分：
```yaml
risk:
  max_position: 0.3        # 最大仓位
  max_stock_weight: 0.05   # 单股最大权重
  max_industry_weight: 0.2 # 行业最大权重
  top_k: 50                # Top-K 选股数量
```

### Q3: 如何查看详细的交易记录？

**A**: 查看 `data/backtest/rqalpha/trades_detail.csv`，包含：
- 买入/卖出
- 价格、数量
- 手续费、税费
- **预测值**（买入时的预测值）
- **成本**（买入成本）
- **收益**（卖出收益）
- **盈亏**（profit - cost）

### Q4: 调仓频率如何设置？

**A**: 在 `config/rqalpha_config.yaml` 中设置：
```yaml
extra:
  context_vars:
    rebalance_interval: 3  # 每3天调仓一次
```

### Q5: 如何添加行业约束？

**A**: 准备行业映射文件（CSV格式）：
```csv
instrument,industry
SH600000,银行
SZ000001,银行
...
```

然后在 `run_backtest.py` 中指定：
```bash
python run_backtest.py \
    --rqalpha-config config/rqalpha_config.yaml \
    --prediction data/predictions/pred_*.csv \
    --industry data/industry_map.csv
```

