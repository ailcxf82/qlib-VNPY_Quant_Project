### `run_msa_signal.py` 使用与实现说明（MSA 次日交易建议生成器）

本文件说明脚本 `backtest/msa/run_msa_signal.py` 的用途、输入输出、对齐规则、策略逻辑与可扩展点，方便你后续优化。

---

## 1. 脚本定位：它“不会预测”，只会“把预测变成交易建议”

`run_msa_signal.py` **不训练模型**、**不计算特征**、**不生成预测值**。  
它只做一件事：

- 读取你已经生成好的预测 CSV（CSI101 与 CSI300）
- 按 MSA 两个子策略的规则（TopK → 过滤 → 选 N 只）
- 按资金占比 allocation 合并为一个目标权重
- 输出“下一交易日”的交易清单：`MSA_YYYYMMDD.csv`

一句话：**信号（score）→ 选股/风控/权重 → 次日可执行的买入清单**。

---

## 2. 输入：预测文件格式与含义

### 2.1 输入文件

脚本需要两份预测文件：

- `--pred-csi101`：CSI101 股票池预测文件
- `--pred-csi300`：CSI300 股票池预测文件

不传时，会自动在 `data/predictions/` 下按 `pred_{pool}_*.csv` 找最新文件。

### 2.2 CSV 必要字段

至少包含：

- `datetime`：日期（见第 3 节的日期语义）
- `instrument` 或 `rq_code`：股票代码
- `final`：模型评分（越大越看好）

可选字段：

- `_meta_shifted_next_day`：若为 1，表示该预测文件的 `datetime` 已经被 shift 成 “trade_date”（次日交易建议日）。

### 2.3 预测信息如何进入脚本？

通过 `load_prediction_csv()` 将 CSV 转成 `PredictionBook`：

- 规范化 `datetime`（只保留日期，去掉时分秒）
- 统一代码到 RQAlpha 格式（`000001.XSHE / 600000.XSHG`）
- 按日期分组存为：
  - `PredictionBook.by_date = {date: {rq_code: score}}`

之后脚本按天取信号：

- `signals = book.get(signal_date)`  → 得到当天所有股票的分数 `Dict[str, float]`

---

## 3. 核心对齐：`signal_date` vs `trade_date`

这是脚本里最关键的概念：你想要“**用截至今天的数据，给明天的买入建议**”，因此必须区分：

- **signal_date**：用来做决策的那一天（当日可观测数据对应的信号）
- **trade_date**：建议你实际下单/执行的那一天（通常是下一交易日）

脚本通过参数 `--pred-dates-are` 解释预测文件里的 `datetime`：

### 3.1 `--pred-dates-are signal_date`（最推荐用于策略生产）

含义：预测文件的 `datetime` 就是 **signal_date**。  
脚本会自动计算：

- `trade_date = next_trading_day(signal_date)`

### 3.2 `--pred-dates-are trade_date`

含义：预测文件的 `datetime` 已经是 **trade_date**（预测阶段就 shift 到次日了）。  
脚本会反推：

- `signal_date = prev_trading_day(trade_date)`

### 3.3 `--pred-dates-are auto`（默认）

逻辑：

- 若预测文件里存在 `_meta_shifted_next_day=1` → 当作 `trade_date`
- 否则 → 当作 `signal_date`

### 3.4 `--asof` 参数

用于指定你想生成哪一天的建议：

- 当 `pred_dates_are=signal_date` 时：`--asof` 表示 signal_date
- 当 `pred_dates_are=trade_date` 时：`--asof` 表示 trade_date

若不传 `--asof`：脚本用两份预测文件的“共同可用最大日期”作为 asof，避免两份文件日期不一致导致空信号。

---

## 4. MSA 两子策略的逻辑（选股 → 过滤 → 目标持仓）

脚本内部将 MSA 拆为两条子策略：

- **策略1（CSI101 小市值）**：`small_cap_csi101`
- **策略2（CSI300 低估值）**：`value_csi300`

### 4.1 选股流程（每个子策略一致）

1) **取 TopK 候选**（按 `final` 分数降序）
2) **过滤**（见 4.2）
3) **取前 N 只**作为目标持仓（默认：策略1 6 只，策略2 4 只）

这些参数都可以从命令行调整：

- `--s1-topk / --s2-topk`
- `--s1-hold / --s2-hold`

### 4.2 过滤规则（当前版本）

过滤由 `apply_basic_filters()` 完成：

- **基础过滤（不依赖 Tushare）**
  - 科创/北交（按代码前缀）
- **需要 Tushare（可选）**
  - ST / 退市（通过 stock_basic 的 name 判断）
  - 上市天数 `min_list_days`
  - PB 范围（策略2 默认 `0 < pb < 1`）
  - 近 N 日涨停剔除（策略2 默认 5 日）

如果本机未安装 tushare 或未配置 token，脚本会告警并跳过 Tushare 相关过滤（基础过滤仍生效）。

---

## 5. 组合层：allocation 合并与权重生成

### 5.1 allocation（资金占比）

通过参数：

- `--alloc1`：策略1（csi101）占比
- `--alloc2`：策略2（csi300）占比

脚本会自动归一化为和为 1（即使你传了 30/70 或 1/1 也可）。

### 5.2 子策略内部权重

当前为 **等权**：

- 每只股票权重 = `sub_allocation / 持仓数`

### 5.3 合并权重

两子策略的权重会按股票代码合并相加，再整体归一化到 1（满仓）。

最终输出列：

- `sub_target_weight`：子策略内部权重
- `target_weight_total`：合并后的最终权重（推荐用于下单）

---

## 6. 输出：年度持仓台账 `MSA_YYYY.csv` 字段解释

默认输出目录：`data/trade_plans/`

文件名：`MSA_{year}.csv`（例如 `MSA_2025.csv`）

该文件会被**持续追加/覆盖更新**：
- 每次检测到“发生调仓”，会写入本次 `trade_date` 对应的一组记录
- 若重复运行同一天（同一 `trade_date`），会覆盖当天旧记录（保证幂等）

文件为“长表（long format）”，包含两类行：
- **SUMMARY**：每次调仓一行，记录上一期到本期的收益/成本/净值/盈亏
- **POSITION**：本次调仓后的持仓明细（每只股票一行）

字段含义（核心列）：

- `row_type`：`SUMMARY` / `POSITION`
- `trade_date`：调仓执行日（建议你执行买入/调仓的日期）
- `signal_date`：信号生成日（模型分数对应的日期）
- `prev_trade_date`：上一期调仓日（用于收益计算）

SUMMARY 行：
- `period_return`：上一期持仓从 `prev_trade_date` 到 `trade_date` 的组合收益（close-to-close）
- `turnover`：换手（\(0.5*\sum|\Delta w|\)）
- `cost_rate`：成本率（你可用 `--cost-rate` 配置）
- `cost`：估算成本（`turnover * cost_rate`）
- `nav_before` / `nav_after`：净值（以 1.0 为起点滚动）
- `pnl_amount`：把净值变化换算为金额的盈亏（`--initial-cash`）

POSITION 行：
- `sub_strategy`：来自哪个子策略
- `rq_code` / `ts_code`：代码
- `score` / `rank_in_topk`：模型分数与排名
- `sub_alloc`：子策略 allocation
- `sub_target_weight`：子策略内部目标权重
- `target_weight_total`：合并后的最终目标权重
- `prev_weight` / `weight_change`：上期权重与变化量
- `close_prev` / `close_cur` / `stock_return` / `stock_contrib`：用于解释收益的价格与贡献（来自 RQAlpha bundle）

---

## 7. 你要“优化”的最佳入口（扩展点）

如果你想提升真实表现，通常按优先级建议改这些地方：

1) **过滤器**（风险控制/可交易性）
   - 例如：涨跌停不可买、停牌剔除、成交额/换手率门槛、财务异常剔除、黑名单、行业/主题分散等。
2) **排序规则**
   - 目前只按 `score` 排序，你可以改为：`score - 波动惩罚 - 拥挤惩罚 + 流动性奖励` 等。
3) **权重分配**
   - 目前等权，可改成 softmax、风险平价、按分数分层等权、单股上限等。
4) **交易成本与约束**
   - `run_msa_signal.py` 目前只输出目标权重，不模拟成交与成本；若要更贴近实盘，可以把交易成本/最小成交单位/资金占用等逻辑加入输出阶段。

---

## 8. 常见问题（FAQ）

### Q1：为什么 trade_date 不是 asof+1？
因为 A 股存在周末/节假日，脚本会按交易日历（优先 Qlib 日历，否则工作日）计算下一交易日。

### Q2：为什么输出股票数不是我设定的 N？
原因通常是过滤条件把候选股剔除了，导致最终可选数不足。你可以：

- 增大 `topk`（TopK 候选更多）
- 放宽过滤
- 或在代码里增加“不足则回填”的逻辑（例如回填到未过滤前的候选）

### Q3：我想输出卖出清单怎么办？
当前脚本只生成目标持仓（BUY list）。要生成 SELL list，需要你提供“当前持仓”输入，然后比较：

- `sell = current_holdings - target_holdings`
- `buy = target_holdings - current_holdings`

---

## 9. 运行示例

### 9.1 预测文件的 datetime 是 signal_date（推荐）

```bash
python backtest/msa/run_msa_signal.py ^
  --pred-csi101 data/predictions/pred_csi101_xxx.csv ^
  --pred-csi300 data/predictions/pred_csi300_xxx.csv ^
  --pred-dates-are signal_date ^
  --output-dir data/trade_plans
```

生成/更新：`data/trade_plans/MSA_2025.csv`

### 9.2 预测文件的 datetime 已经是 trade_date（预测阶段 shift 过）

```bash
python backtest/msa/run_msa_signal.py ^
  --pred-csi101 data/predictions/pred_csi101_xxx.csv ^
  --pred-csi300 data/predictions/pred_csi300_xxx.csv ^
  --pred-dates-are trade_date
```


