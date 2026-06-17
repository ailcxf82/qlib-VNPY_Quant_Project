# 日线股票投资监听系统

> 基于 Qlib 量化框架的智能投资监听系统，集成信号生成、持仓追踪、舆情分析、LLM 金融分析和量化回测能力。

## 目录

- [系统概述](#系统概述)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [配置说明](#配置说明)
- [模块详解](#模块详解)
- [回测框架](#回测框架)
- [API 接口](#api-接口)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

---

## 系统概述

### 功能特性

| 功能 | 描述 |
|------|------|
| **定时调度** | 每日收盘后自动运行预测和信号生成 |
| **信号引擎** | 基于预测分数和策略规则生成买卖信号 |
| **持仓追踪** | 维护虚拟持仓、计算盈亏、追踪收益 |
| **舆情分析** | 抓取财经新闻，分析市场情绪 |
| **LLM 分析** | 使用 DeepSeek API 进行大盘走势和交易建议分析 |
| **消息推送** | 通过企业微信机器人推送信号和报告 |
| **Web API** | 提供 REST API 接口供外部系统调用 |
| **量化回测** | 集成式回测引擎，支持多策略回测和绩效分析 |
| **风控系统** | 多层止损机制，Kelly 仓位管理 |
| **市场环境** | 自动检测牛熊市、波动率，动态调整策略权重 |

### 系统要求

- Python 3.8+
- Qlib 数据环境
- Akshare 数据源（免费，无需注册）
- Tushare Pro API（可选，用于实时行情）
- DeepSeek API（可选，用于 LLM 分析）
- 企业微信机器人（可选，用于消息推送）

---

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy pyyaml flask requests akshare
```

### 2. 配置环境变量

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"
$env:WECHAT_WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"

# Linux/Mac
export DEEPSEEK_API_KEY="sk-xxxxxxxx"
export WECHAT_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
```

### 3. 运行系统

```bash
# 查看当前状态
python run_monitor.py --mode status

# 手动运行一次
python run_monitor.py --mode once
        
# 启动定时调度器
python run_monitor.py --mode scheduler

# 启动 Web API
python run_monitor.py --mode api --port 5000
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MonitorScheduler                              │
│  (协调器 - 负责调度和编排所有组件)                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │SignalEngine  │  │PositionTracker│ │SentimentAnalyzer│            │
│  │  信号生成     │  │  持仓追踪     │  │   舆情分析      │            │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                  │                      │
│         └─────────────────┼──────────────────┘                      │
│                           │                                          │
│                    ┌──────▼──────┐                                   │
│                    │ LLMAnalyzer │                                   │
│                    │  LLM 分析    │                                   │
│                    └──────┬──────┘                                   │
│                           │                                          │
│                    ┌──────▼──────┐                                   │
│                    │WeChatNotifier│                                  │
│                    │   消息推送    │                                  │
│                    └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     量化回测与策略系统                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    DataSourceAdapter                          │   │
│  │           (Akshare 数据源适配器 - 统一数据接口)                  │   │
│  └───────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │StrategyManager│ │MarketRegime  │  │Integrated    │              │
│  │  策略管理器   │ │  Detector    │  │BacktestEngine│              │
│  │              │ │  市场环境检测  │  │  集成回测引擎 │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                    │                    │
│         └─────────────────┼────────────────────┘                    │
│                           │                                          │
│                    ┌──────▼──────┐                                   │
│                    │RiskController│                                  │
│                    │  风控系统     │                                  │
│                    └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 文件结构

```
project/
├── config/
│   └── monitor.yaml          # 监听系统配置文件
├── monitor/
│   ├── __init__.py           # 模块导出
│   ├── ts_client.py          # TushareClient 单例管理器
│   ├── exceptions.py         # 自定义异常类
│   ├── signal_engine.py      # 信号引擎
│   ├── position_tracker.py   # 持仓追踪器
│   ├── sentiment_analyzer.py # 舆情分析模块
│   ├── llm_analyzer.py       # LLM 金融分析模块
│   ├── scheduler.py          # 定时调度器
│   ├── data_source.py        # Akshare 数据源适配器
│   ├── strategy_manager.py   # 多策略管理器
│   ├── market_regime.py      # 市场环境检测器
│   ├── risk_controller.py    # 风控系统
│   ├── unified_strategy.py   # 统一策略接口
│   ├── integrated_backtest.py# 集成式回测引擎
│   ├── performance.py        # 绩效分析模块
│   ├── backtest_engine.py    # 基础回测引擎
│   └── notifier/
│       ├── __init__.py
│       └── wechat.py         # 企业微信推送
├── api/
│   └── predict_api.py        # REST API 服务
├── examples/
│   ├── run_backtest.py       # 回测示例脚本
│   └── test_backtest.py      # 回测测试脚本
├── tests/
│   ├── test_unified_strategy.py      # 策略单元测试
│   └── test_integrated_backtest.py   # 回测单元测试
├── run_monitor.py            # 入口脚本
└── data/monitor/             # 数据存储目录
    ├── position.json         # 持仓状态
    ├── trades.csv            # 交易记录
    ├── signals.csv           # 信号记录
    ├── sentiment_cache.json  # 舆情缓存
    ├── llm_cache.json        # LLM 缓存
    └── monitor.log           # 运行日志
```

---

## 配置说明

### 配置文件结构

配置文件位于 `config/monitor.yaml`，主要包含以下部分：

### 1. 调度器配置

```yaml
scheduler:
  enabled: true              # 是否启用定时调度
  schedule_time: "15:05"     # 每日执行时间
  timezone: "Asia/Shanghai"  # 时区
  retry_times: 3             # 重试次数
  retry_interval: 300        # 重试间隔（秒）
```

### 2. 策略配置

```yaml
strategies:
  csi101:
    name: "小市值策略"
    enabled: true
    rules:
      buy:
        - type: "top_k"
          k: 20
          description: "选择预测分数前20名"
        - type: "exclude_board"
          boards: ["科创板", "北交所"]
        - type: "exclude_st"
        - type: "min_list_days"
          days: 60
        - type: "final_top_k"
          k: 6
      sell:
        - type: "holding_days"
          max_days: 5
        - type: "stop_loss"
          threshold: -0.08
        - type: "take_profit"
          threshold: 0.15
    position_size: 0.15      # 单只股票仓位比例
```

### 3. 持仓配置

```yaml
position:
  initial_capital: 1000000   # 初始资金
  max_stocks: 10             # 最大持仓数量
  commission_rate: 0.0003    # 佣金费率
  stamp_duty: 0.001          # 印花税
  slippage: 0.001            # 滑点
  min_trade_amount: 1000     # 最小交易金额
```

### 4. 舆情配置

```yaml
sentiment:
  enabled: true
  sources:
    - name: "sina_finance"
      enabled: true
      url: "https://finance.sina.com.cn/stock/"
      type: "web_scraper"
  keywords:
    - "大盘"
    - "A股"
    - "沪指"
    - "北向资金"
  max_news: 50
  cache_hours: 4
```

### 5. LLM 配置

```yaml
llm:
  enabled: true
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com/v1"
  max_tokens: 2000
  temperature: 0.3
  cache_hours: 4
```

### 6. 通知配置

```yaml
notification:
  wechat:
    enabled: true
    webhook_url: "${WECHAT_WEBHOOK_URL}"
    message_types:
      - "daily_signal"
      - "position_change"
      - "pnl_summary"
      - "market_analysis"
    quiet_hours:
      start: "22:00"
      end: "08:00"
```

### 7. Akshare 数据源配置

```yaml
akshare:
  cache_hours: 4             # 缓存时长（小时）
```

### 8. 回测引擎配置

```yaml
backtest_engine:
  default_initial_capital: 1000000
  default_commission_rate: 0.0003
  default_stamp_duty: 0.001
  default_slippage: 0.001
```

---

## 模块详解

### SignalEngine（信号引擎）

负责根据预测结果和策略规则生成买卖信号。

**支持的过滤规则：**

| 规则类型 | 参数 | 描述 |
|----------|------|------|
| `top_k` | k | 选择预测分数前 k 名 |
| `final_top_k` | k | 最终选择前 k 只股票 |
| `exclude_board` | boards | 排除指定板块（科创板、北交所） |
| `exclude_st` | - | 排除 ST 股票 |
| `min_list_days` | days | 最小上市天数 |
| `pb_range` | min, max | PB 区间过滤 |
| `exclude_recent_limit_up` | days | 排除近期涨停股 |

**使用示例：**

```python
from monitor import SignalEngine

engine = SignalEngine("config/monitor.yaml")

# 生成所有股票池的信号
signals = engine.generate_all_signals()

# 生成指定股票池的信号
signals = engine.generate_signals("csi101")
```

### PositionTracker（持仓追踪器）

负责维护虚拟持仓、执行买卖操作、计算盈亏。

**主要功能：**
- 买入/卖出股票
- 检查卖出信号（持仓天数、止损、止盈）
- 更新持仓价格
- 计算账户收益

**使用示例：**

```python
from monitor import PositionTracker

tracker = PositionTracker("config/monitor.yaml")

# 买入
tracker.buy(
    code="000001",
    name="平安银行",
    price=10.5,
    strategy="csi101",
    reason="预测分数排名 3"
)

# 卖出
tracker.sell("000001", "止盈")

# 获取账户摘要
summary = tracker.get_portfolio_summary()
print(f"总资产: {summary['total_assets']}")
print(f"总收益: {summary['total_return']:.2%}")
```

### DataSourceAdapter（数据源适配器）

统一的数据源接口，支持 Akshare 免费数据源。

**主要功能：**
- 获取日线数据
- 获取股票列表
- 获取财务数据
- 获取指数数据
- 获取股票因子
- 获取市场宽度

**使用示例：**

```python
from monitor.data_source import get_data_source

# 获取 Akshare 数据源
ds = get_data_source("akshare")

# 获取日线数据
df = ds.get_daily_data("000001", "2024-01-01", "2024-06-30")

# 获取股票因子
factors = ds.get_stock_factors("000001")
print(f"PE: {factors['pe']}, PB: {factors['pb']}")

# 获取市场宽度
breadth = ds.get_market_breadth("2024-06-30")
print(f"上涨股票: {breadth['up_count']}, 下跌股票: {breadth['down_count']}")
```

### StrategyManager（策略管理器）

多策略组合管理，支持价值、动量、均值回归、质量四种策略。

**策略类型：**

| 策略 | 描述 | 关键因子 |
|------|------|----------|
| value | 价值策略 | PB、PE、ROE、股息率 |
| momentum | 动量策略 | 5日动量、20日动量、成交量动量 |
| mean_reversion | 均值回归策略 | RSI、MA偏离度、成交量异动 |
| quality | 质量策略 | ROE、ROA、负债率 |

**使用示例：**

```python
from monitor.strategy_manager import StrategyManager

manager = StrategyManager("config/monitor.yaml")

# 获取股票因子
factors = manager.fetch_stock_factors("000001")

# 计算各策略得分
value_score = manager.calculate_value_score(factors)
momentum_score = manager.calculate_momentum_score(factors)

# 组合信号
signals = manager.combine_signals(["000001", "000002"], "csi101")
```

### MarketRegimeDetector（市场环境检测器）

自动检测市场环境，动态调整策略权重。

**市场环境类型：**

| 类型 | 描述 | 策略权重调整 |
|------|------|-------------|
| bull_low_vol | 牛市低波动 | 动量策略权重提高 |
| bull_high_vol | 牛市高波动 | 质量策略权重提高 |
| bear_high_vol | 熊市高波动 | 价值策略权重提高，仓位降低 |
| bear_low_vol | 熊市低波动 | 均值回归策略权重提高 |
| choppy_high_vol | 震荡高波动 | 均值回归策略权重提高 |
| range_bound | 横盘震荡 | 均衡权重 |

**使用示例：**

```python
from monitor.market_regime import MarketRegimeDetector

detector = MarketRegimeDetector("config/monitor.yaml")

# 检测市场环境
regime = detector.detect_regime()
print(f"市场环境: {regime.regime}")
print(f"趋势强度: {regime.trend_strength:.2f}")
print(f"波动率: {regime.volatility:.2%}")

# 获取策略权重
weights = detector.get_strategy_weights(regime)
print(f"动量权重: {weights['momentum']:.2%}")

# 获取仓位倍数
multiplier = detector.get_position_size_multiplier(regime)
print(f"建议仓位倍数: {multiplier:.2f}")
```

### RiskController（风控系统）

多层止损机制和 Kelly 仓位管理。

**止损类型：**

| 类型 | 触发条件 | 优先级 |
|------|----------|--------|
| hard_stop | 亏损达到硬止损线（-8%） | 最高 |
| take_profit | 盈利达到止盈线（+15%） | 高 |
| trailing_stop | 从最高点回撤达到阈值 | 中 |
| time_stop | 持仓超过最大天数 | 低 |

**使用示例：**

```python
from monitor.risk_controller import RiskController

controller = RiskController("config/monitor.yaml")

# 计算止损水平
stop_level = controller.calculate_stop_loss_levels(
    code="000001",
    name="平安银行",
    entry_price=10.0,
    current_price=9.2,
    holding_days=3,
    highest_price=10.5
)

if stop_level.should_stop:
    print(f"触发止损: {stop_level.stop_reason}")

# 计算 Kelly 仓位
position = controller.calculate_position_size(
    code="000001",
    name="平安银行",
    entry_price=10.0,
    stop_price=9.2,
    total_capital=1000000
)
print(f"建议股数: {position.recommended_shares}")
```

---

## 回测框架

### IntegratedBacktestEngine（集成式回测引擎）

集成风控和市场环境的完整回测引擎。

**主要特性：**
- 支持多种策略（动量、均值回归、价值、质量）
- 自动集成风控系统
- 自动集成市场环境检测
- 完整的绩效分析报告

**使用示例：**

```python
from monitor.integrated_backtest import run_integrated_backtest

# 运行回测
results = run_integrated_backtest(
    strategy_config={
        "name": "momentum",
        "lookback_period": 20,
        "top_k": 10,
        "min_momentum": 0.05
    },
    backtest_config={
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "initial_capital": 1000000
    },
    codes=["000001", "000002", "600000", "600036"],
    use_risk_control=True,
    use_market_regime=True
)

# 查看绩效
perf = results["performance"]
print(f"总收益率: {perf['total_return']:.2%}")
print(f"夏普比率: {perf['sharpe_ratio']:.2f}")
print(f"最大回撤: {perf['max_drawdown']:.2%}")
```

### UnifiedStrategyBase（统一策略接口）

所有策略的基类，提供统一的接口和计算逻辑。

**内置策略：**

```python
from monitor.unified_strategy import (
    UnifiedMomentumStrategy,
    UnifiedMeanReversionStrategy,
    UnifiedValueStrategy,
    UnifiedQualityStrategy,
    create_strategy
)

# 使用工厂函数创建策略
strategy = create_strategy("momentum", {
    "lookback_period": 20,
    "top_k": 10,
    "min_momentum": 0.05
})

# 计算策略得分
score = strategy.calculate_score("000001", factors)
```

### PerformanceAnalyzer（绩效分析器）

计算全面的绩效指标。

**绩效指标：**

| 指标 | 描述 |
|------|------|
| total_return | 总收益率 |
| annual_return | 年化收益率 |
| sharpe_ratio | 夏普比率 |
| sortino_ratio | 索提诺比率 |
| max_drawdown | 最大回撤 |
| win_rate | 胜率 |
| profit_factor | 盈亏比 |
| calmar_ratio | 卡玛比率 |

---

## API 接口

### 基础端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/api/health` | GET | 健康检查 |

### 监听系统端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/monitor/status` | GET | 获取系统状态 |
| `/api/monitor/run` | POST | 手动运行监听任务 |
| `/api/monitor/start` | POST | 启动定时调度 |
| `/api/monitor/stop` | POST | 停止定时调度 |
| `/api/monitor/portfolio` | GET | 获取持仓信息 |
| `/api/monitor/signals` | GET | 获取信号列表 |
| `/api/monitor/sentiment` | GET | 获取舆情分析 |
| `/api/monitor/reset` | POST | 重置持仓 |

### API 示例

```bash
# 健康检查
curl http://127.0.0.1:5000/api/health

# 获取系统状态
curl http://127.0.0.1:5000/api/monitor/status

# 手动运行
curl -X POST http://127.0.0.1:5000/api/monitor/run

# 获取持仓
curl http://127.0.0.1:5000/api/monitor/portfolio

# 获取信号
curl http://127.0.0.1:5000/api/monitor/signals

# 重置持仓
curl -X POST http://127.0.0.1:5000/api/monitor/reset
```

### 响应格式

所有 API 返回 JSON 格式：

```json
{
  "success": true,
  "data": { ... }
}
```

错误时：

```json
{
  "success": false,
  "error": "错误信息"
}
```

---

## 使用示例

### 示例 1：每日自动运行

```bash
# 启动定时调度器（每日 15:05 自动运行）
python run_monitor.py --mode scheduler
```

### 示例 2：手动运行一次

```bash
# 运行一次完整的监听流程
python run_monitor.py --mode once

# 指定日期运行
python run_monitor.py --mode once --date 2024-01-15
```

### 示例 3：启动 Web 服务

```bash
# 启动 API 服务
python run_monitor.py --mode api --port 5000

# 访问 Web 界面
# http://127.0.0.1:5000/
```

### 示例 4：运行回测

```python
from monitor.integrated_backtest import IntegratedBacktestEngine, IntegratedBacktestConfig
from monitor.unified_strategy import create_strategy
from monitor.data_source import get_data_source
from monitor.risk_controller import RiskController
from monitor.market_regime import MarketRegimeDetector

# 配置回测
config = IntegratedBacktestConfig(
    start_date="2024-01-01",
    end_date="2024-06-30",
    initial_capital=1000000,
    use_risk_control=True,
    use_market_regime=True
)

# 创建组件
data_source = get_data_source("akshare")
strategy = create_strategy("momentum", {"lookback_period": 20, "top_k": 10})
risk_controller = RiskController()
regime_detector = MarketRegimeDetector()

# 创建引擎
engine = IntegratedBacktestEngine(
    config=config,
    data_source=data_source,
    strategy=strategy,
    risk_controller=risk_controller,
    regime_detector=regime_detector
)

# 运行回测
results = engine.run(["000001", "000002", "600000"])

# 输出结果
print(f"总收益率: {results['performance']['total_return']:.2%}")
print(f"夏普比率: {results['performance']['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['performance']['max_drawdown']:.2%}")
print(f"交易次数: {len(results['trades'])}")
```

### 示例 5：Python 脚本调用

```python
from monitor import MonitorScheduler

# 创建调度器
scheduler = MonitorScheduler("config/monitor.yaml")

# 运行一次
result = scheduler.run_once()
print(f"执行结果: {result['success']}")

# 获取状态
status = scheduler.get_status()
print(f"总资产: {status['portfolio']['total_assets']}")

# 重置持仓
scheduler.reset_positions()
```

---

## 常见问题

### Q1: 如何配置企业微信机器人？

1. 在企业微信群中添加机器人
2. 获取 Webhook URL
3. 设置环境变量：`WECHAT_WEBHOOK_URL`

### Q2: 如何切换到 OpenAI API？

修改 `config/monitor.yaml`：

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  base_url: "${OPENAI_BASE_URL}"
```

### Q3: 为什么舆情分析返回空结果？

可能原因：
- Tushare 新闻接口需要更高权限
- 网络无法访问新闻网站
- 当天没有符合条件的新闻

解决方案：
- 升级 Tushare 积分
- 配置代理
- 添加其他新闻源

### Q4: 如何修改策略规则？

编辑 `config/monitor.yaml` 中的 `strategies` 部分，添加或修改规则：

```yaml
strategies:
  my_strategy:
    name: "我的策略"
    enabled: true
    rules:
      buy:
        - type: "top_k"
          k: 10
        - type: "final_top_k"
          k: 5
      sell:
        - type: "stop_loss"
          threshold: -0.05
```

### Q5: 如何查看运行日志？

日志文件位于 `data/monitor/monitor.log`。

```bash
# 查看最近日志
tail -f data/monitor/monitor.log

# Windows
Get-Content data\monitor\monitor.log -Tail 50
```

### Q6: 如何备份数据？

备份 `data/monitor/` 目录即可：

```bash
# Linux/Mac
tar -czf monitor_backup_$(date +%Y%m%d).tar.gz data/monitor/

# Windows
Compress-Archive -Path data\monitor -DestinationPath monitor_backup.zip
```

### Q7: Akshare 数据源有什么优势？

- **免费使用**：无需注册 API Key
- **数据丰富**：支持股票、指数、基金、期货等多种数据
- **实时性好**：数据更新及时
- **接口简单**：易于集成和使用

### Q8: 如何添加自定义策略？

继承 `UnifiedStrategyBase` 类并实现必要方法：

```python
from monitor.unified_strategy import UnifiedStrategyBase

class MyCustomStrategy(UnifiedStrategyBase):
    STRATEGY_NAME = "my_custom"
    
    def generate_signals(self, date, data, positions):
        # 实现信号生成逻辑
        pass
    
    def calculate_score(self, code, factors):
        # 实现得分计算逻辑
        pass
```

---

## 更新日志

### v1.1.0 (2026-03-24)

- 新增 Akshare 数据源适配器
- 新增统一策略接口（UnifiedStrategyBase）
- 新增集成式回测引擎（IntegratedBacktestEngine）
- 新增市场环境检测器（MarketRegimeDetector）
- 新增多策略管理器（StrategyManager）
- 新增风控系统（RiskController）
- 新增绩效分析模块（PerformanceAnalyzer）
- 修复 holding_days 计算问题
- 统一因子评分阈值
- 修复缓存类型不一致问题
- 添加单元测试覆盖

### v1.0.0 (2026-03-23)

- 初始版本发布
- 支持定时调度、信号生成、持仓追踪
- 集成舆情分析和 LLM 金融分析
- 支持企业微信推送
- 提供 REST API 接口

---

## 许可证

MIT License
