## MSA 多策略回测：执行逻辑与修改指南（工程视角）

本文档解释 `backtest/msa/` 目录下的 **MSA（Multi-Strategy Allocation，多策略资金分配）** 回测是如何执行的，并告诉你**从哪里改参数、怎么加过滤、怎么加第3个子策略**。

---

## 1. 你要跑的是什么

MSA 回测的目标是：把**两套子策略**在同一个账户里同时运行，并用**资金占比**把总仓位拆成两份：

- **子策略1（CSI101 小市值）**：预测 TopK → 过滤 → 选 N 只 → 等权 → 占用 allocation1 仓位
- **子策略2（CSI300 低估值）**：预测 TopK → 过滤（含 PB/近N日涨停等）→ 选 N 只 → 等权 → 占用 allocation2 仓位
- **组合层**：把两份权重合并后再归一化到 1（默认满仓）
- **通用风控**：收盘前触发回撤止损；卖出后立即按策略补仓

---

## 2. 入口到底从哪里开始（非常关键）

### 2.1 主入口：`backtest/msa/run_msa_backtest.py`

你运行的入口是：

- `python backtest/msa/run_msa_backtest.py ...`

它做 3 件事：

- **固定工作目录为项目根目录**（避免相对路径乱跳）
- **解析两份预测文件路径**（csi101/csi300），如果不传就自动从 `data/predictions/` 选最新
- 把参数写入 RQAlpha 配置的 `extra.context_vars`，然后复用现有 runner `backtest/rqalpha_backtest.py` 启动 RQAlpha

你最常用的命令：

```bash
python backtest/msa/run_msa_backtest.py ^
  --rqalpha-config config/rqalpha_config.yaml ^
  --pred-csi101 data/predictions/pred_csi101_*.csv ^
  --pred-csi300 data/predictions/pred_csi300_*.csv
```

### 2.2 RQAlpha 策略脚本：`backtest/msa/rqalpha_msa_strategy.py`

RQAlpha 会加载策略文件并依次调用生命周期函数：

- `init(context)`：初始化
- `before_trading(context)`：每日开盘前
- `handle_bar(context, bar_dict)`：每个 bar 调用（本策略主要在收盘时执行一次风控/补仓）

---

## 3. 参数从哪里来（你应该改哪里）

MSA 的参数读取统一来自 RQAlpha 配置里的：

- `config/rqalpha_config.yaml` → `extra.context_vars`

`run_msa_backtest.py` 会把命令行参数写进去；你也可以手工在 yaml 里写。

目前支持的关键参数（都在策略 `init()` 里读取）：

- **预测文件**
  - `pred_csi101`: CSI101 预测文件路径（csv）
  - `pred_csi300`: CSI300 预测文件路径（csv）
- **资金分配**
  - `alloc_strategy1` / `alloc_strategy2`：两策略占比（会自动归一化）
- **风控**
  - `drawdown_stop`：个股回撤止损阈值（默认 0.08=8%）
  - `close_check_minute`：收盘前 N 分钟触发风控（默认 30）
- **子策略1可调参数**
  - `s1_topk_pred`：从预测里取 TopK（默认20）
  - `s1_target_holdings`：最终持仓数（默认6）
  - `s1_rebalance_interval_days`：调仓间隔（默认5）
  - `s1_min_list_days`：上市最少天数（默认360）
- **子策略2可调参数**
  - `s2_topk_pred`：默认20
  - `s2_target_holdings`：默认4（注意你原文写过 Top2；这里可调成2）
  - `s2_rebalance_interval_days`：默认5
  - `s2_min_list_days`：默认360
  - `s2_pb_min/s2_pb_max`：PB 范围（默认 0~1）
  - `s2_exclude_recent_limitup_days`：近N日涨停剔除（默认5）

---

## 4. 数据怎么流动（预测 → 候选 → 过滤 → 目标权重 → 下单）

### 4.1 预测文件读取：`backtest/msa/prediction_loader.py`

MSA 支持两种 CSV：

- `datetime, instrument, final`
- `datetime, rq_code, final`

读取后会转换为：

- `PredictionBook.by_date: {交易日 -> {rqalpha_code -> score}}`

这让策略在某天 `today` 时可以直接：

- `signals = book.get(today)`

### 4.2 选股：先 TopK 再过滤 再截断到 N

流程在策略里是：

- `topk(signals, sub.topk_pred)`：先取 TopK
- `apply_basic_filters(...)`：再过滤（Tushare 可选）
- 最后按 score 排序，取 `sub.target_holdings`

### 4.3 过滤：`backtest/msa/filters.py`

过滤逻辑是“尽量做，做不了就跳过”：

- **不依赖 Tushare**：科创/北交（按代码前缀）
- **依赖 Tushare（若 token/依赖可用）**：
  - `stock_basic()`：ST（通过 name 粗过滤）、上市天数
  - `daily_basic()`：PB 过滤
  - `limit_list()`：近 N 日涨停过滤

### 4.4 权重合成（组合层）

组合层是两步：

- 子策略内等权，权重和为 `allocation`
- 两个子策略合并后再 `normalize` 到总仓位=1（满仓）

---

## 5. 交易与风控怎么执行（RQAlpha bar 行为）

### 5.1 调仓触发

策略用 `_should_rebalance()` 判断是否到调仓日（默认 5 天一次，取两个子策略的最小间隔）。

触发后：

- 计算目标权重 `target = _build_target_weights(...)`
- 执行 `_rebalance_to_target(target)`

### 5.2 收盘前风控

每个 bar 会更新持仓高水位（`high_watermark`），并在接近收盘（默认 15:00 前 30 分钟）时：

- 计算回撤超过阈值的持仓 → 卖出
- 卖出后再跑一次 `_rebalance_to_target(target)` 补仓

工程细节：

- RQAlpha 的 `bar_dict` 是 `BarMap`，不能用 `.get()`，策略内部用 `bar_dict[code]` 的方式兼容。

---

## 6. Tushare 接入与缓存（如何稳定）

### 6.1 token 从哪里读

`backtest/msa/tushare_client.py` 的 token 读取顺序：

- 环境变量 `TUSHARE_TOKEN`
- 本地 `config/secrets.yaml`（已加入 `.gitignore`，不会提交）

### 6.2 缓存目录

默认缓存目录：

- `data/tushare_cache/`

每个接口请求会生成一个 cache key，优先读缓存，减少网络波动。

### 6.3 财务接口已封装

客户端已封装：

- `income`
- `fina_indicator`
- `fina_audit`

你后续如果要实现“盈利 + ROA 排序 + 审计正常”过滤，推荐把逻辑加在 `filters.py` 里（而不是散落在策略里）。

另外有一个验证脚本：

- `python -m backtest.msa.verify_tushare_fundamentals --ts-code 000001.SZ`

用来检查接口是否为空、字段是否存在。

参考文档入口：`https://tushare.pro/document`

---

## 7. 常见修改场景（怎么改最省事）

### 7.1 改持仓数 / TopK / PB 范围

优先方式：在 `run_msa_backtest.py` 里加命令行参数并写入 `context_vars`，或者直接在 `rqalpha_config.yaml` 的 `extra.context_vars` 里写：

- `s1_target_holdings`
- `s2_target_holdings`
- `s2_pb_min/s2_pb_max`
- `s2_exclude_recent_limitup_days`

### 7.2 加“盈利/ROA/审计”过滤

推荐做法：

- 在 `backtest/msa/filters.py` 里新增一个 `apply_fundamental_filters(...)`
- 内部调用 `TushareClient.income/fina_indicator/fina_audit`
- 把过滤结果接到 `apply_basic_filters()` 后面

### 7.3 增加第3个子策略

工程上最干净的方式：

- `SubStrategyConfig` 增加一个字段 `book_key`（如 `csi101/csi300/...`），避免靠 `name` 判断
- `context.pred_books` 增加第三份预测 book
- `context.sub_strategies.append(s3)`

---

## 8. 输出结果去哪了

输出目录由 `config/rqalpha_config.yaml` 决定：

- `output.output_dir`（默认 `data/backtest/rqalpha/`）

你会看到：

- `report.json`
- `trades_detail.csv`
- `rqalpha_strategy_plot.png`
- 以及 runner 生成的其他分析文件


