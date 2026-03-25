# 日线股票投资监听系统 - 开发目录

> 开发日期：2026-03-23  
> 版本：v1.0.0  
> 分支：V0.1.0-MSA

---

## 一、项目概述

### 1.1 项目背景

在现有 Qlib 量化投资系统基础上，构建一个日线级别的股票投资监听系统，实现：
- 每日收盘后自动运行预测
- 基于预测结果生成买卖信号
- 追踪虚拟持仓和盈亏
- 结合舆情分析和 LLM 金融分析
- 通过企业微信推送投资报告

### 1.2 功能模块

| 模块 | 功能 | 文件 |
|------|------|------|
| 信号引擎 | 基于预测分数和策略规则生成买卖信号 | `monitor/signal_engine.py` |
| 持仓追踪 | 维护虚拟持仓、计算盈亏、执行交易 | `monitor/position_tracker.py` |
| 舆情分析 | 抓取财经新闻、分析市场情绪 | `monitor/sentiment_analyzer.py` |
| LLM 分析 | 调用 DeepSeek API 进行金融分析 | `monitor/llm_analyzer.py` |
| 消息推送 | 企业微信机器人推送报告 | `monitor/notifier/wechat.py` |
| 定时调度 | 每日自动运行监听任务 | `monitor/scheduler.py` |
| Web API | REST API 接口服务 | `api/predict_api.py` |

---

## 二、文件清单

### 2.1 新增文件

```
project/
├── config/
│   └── monitor.yaml                    # 监听系统配置文件
│
├── monitor/
│   ├── __init__.py                     # 模块导出
│   ├── ts_client.py                    # TushareClient 单例管理器
│   ├── exceptions.py                   # 自定义异常类
│   ├── signal_engine.py                # 信号引擎
│   ├── position_tracker.py             # 持仓追踪器
│   ├── sentiment_analyzer.py           # 舆情分析模块
│   ├── llm_analyzer.py                 # LLM 金融分析模块
│   ├── scheduler.py                    # 定时调度器
│   ├── README.md                       # 使用说明文档
│   └── notifier/
│       ├── __init__.py
│       └── wechat.py                   # 企业微信推送
│
├── run_monitor.py                      # 入口脚本
│
└── data/monitor/                       # 数据存储目录
    ├── position.json                   # 持仓状态
    ├── trades.csv                      # 交易记录
    ├── signals.csv                     # 信号记录
    ├── sentiment_cache.json            # 舆情缓存
    ├── llm_cache.json                  # LLM 缓存
    └── monitor.log                     # 运行日志
```

### 2.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/predict_api.py` | 新增监听系统 API 端点 |
| `api/__init__.py` | 新增模块导出 |

---

## 三、核心模块说明

### 3.1 SignalEngine（信号引擎）

**文件**: `monitor/signal_engine.py`

**功能**:
- 加载预测结果
- 应用策略规则过滤
- 生成买入信号

**支持的过滤规则**:
- `top_k`: 选择预测分数前 k 名
- `final_top_k`: 最终选择前 k 只
- `exclude_board`: 排除指定板块
- `exclude_st`: 排除 ST 股票
- `min_list_days`: 最小上市天数
- `pb_range`: PB 区间过滤
- `exclude_recent_limit_up`: 排除近期涨停股

### 3.2 PositionTracker（持仓追踪器）

**文件**: `monitor/position_tracker.py`

**功能**:
- 维护虚拟持仓状态
- 执行买入/卖出操作
- 检查卖出信号（持仓天数、止损、止盈）
- 更新持仓价格
- 计算账户收益

**线程安全**: 使用 `threading.Lock` 保护共享状态

### 3.3 SentimentAnalyzer（舆情分析器）

**文件**: `monitor/sentiment_analyzer.py`

**功能**:
- 从 Tushare 和网站抓取财经新闻
- 分析新闻情绪（正面/负面/中性）
- 提取关键词热度
- 缓存分析结果

### 3.4 LLMAnalyzer（LLM 分析器）

**文件**: `monitor/llm_analyzer.py`

**功能**:
- 调用 DeepSeek API 进行金融分析
- 大盘走势分析
- 交易机会分析
- 缓存分析结果

**支持的 LLM 提供商**:
- DeepSeek（默认）
- OpenAI

### 3.5 WeChatNotifier（企业微信推送）

**文件**: `monitor/notifier/wechat.py`

**功能**:
- 发送每日投资报告
- 发送持仓变动通知
- 发送盈亏汇总
- 发送市场分析报告
- 支持静默时段

### 3.6 MonitorScheduler（定时调度器）

**文件**: `monitor/scheduler.py`

**功能**:
- 定时调度（每日指定时间运行）
- 协调各模块执行
- 错误处理和重试
- 状态管理

---

## 四、配置说明

### 4.1 配置文件

**文件**: `config/monitor.yaml`

**主要配置项**:

```yaml
# 调度器配置
scheduler:
  enabled: true
  schedule_time: "15:05"
  
# 策略配置
strategies:
  csi101:
    name: "小市值策略"
    rules:
      buy: [...]
      sell: [...]
    position_size: 0.15

# 持仓配置
position:
  initial_capital: 1000000
  max_stocks: 10
  
# LLM 配置
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  
# 通知配置
notification:
  wechat:
    enabled: true
```

### 4.2 环境变量

| 变量名 | 用途 | 必需 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | LLM 分析需要 |
| `WECHAT_WEBHOOK_URL` | 企业微信机器人 Webhook | 消息推送需要 |

---

## 五、API 接口

### 5.1 端点列表

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/monitor/status` | GET | 获取系统状态 |
| `/api/monitor/run` | POST | 手动运行监听任务 |
| `/api/monitor/start` | POST | 启动定时调度 |
| `/api/monitor/stop` | POST | 停止定时调度 |
| `/api/monitor/portfolio` | GET | 获取持仓信息 |
| `/api/monitor/signals` | GET | 获取信号列表 |
| `/api/monitor/sentiment` | GET | 获取舆情分析 |
| `/api/monitor/reset` | POST | 重置持仓 |

### 5.2 使用示例

```bash
# 获取系统状态
curl http://127.0.0.1:5000/api/monitor/status

# 手动运行
curl -X POST http://127.0.0.1:5000/api/monitor/run

# 获取持仓
curl http://127.0.0.1:5000/api/monitor/portfolio
```

---

## 六、运行方式

### 6.1 命令行

```bash
# 查看状态
python run_monitor.py --mode status

# 手动运行一次
python run_monitor.py --mode once

# 启动定时调度器
python run_monitor.py --mode scheduler

# 启动 Web API
python run_monitor.py --mode api --port 5000

# 重置持仓
python run_monitor.py --mode reset
```

### 6.2 Python 调用

```python
from monitor import MonitorScheduler

scheduler = MonitorScheduler("config/monitor.yaml")

# 运行一次
result = scheduler.run_once()

# 获取状态
status = scheduler.get_status()

# 重置持仓
scheduler.reset_positions()
```

---

## 七、架构审查与优化

### 7.1 已修复问题

| 问题 | 优先级 | 状态 |
|------|--------|------|
| 买入价格硬编码 `price=1.0` | P0 | ✅ 已修复 |
| 线程安全问题 | P0 | ✅ 已修复 |
| TushareClient 重复初始化 | P1 | ✅ 已修复 |
| LLM 缓存无过期机制 | P1 | ✅ 已修复 |
| 异常处理过于宽泛 | P1 | ✅ 已修复 |

### 7.2 架构改进

- 新增 `ts_client.py`: TushareClient 单例管理器
- 新增 `exceptions.py`: 自定义异常类层次结构
- 添加线程锁保护共享状态
- 添加缓存过期机制

---

## 八、测试报告

### 8.1 API 测试结果

| 测试项 | 状态 |
|--------|------|
| `/api/health` | ✅ PASS |
| `/api/monitor/status` | ✅ PASS |
| `/api/monitor/portfolio` | ✅ PASS |
| `/api/monitor/signals` | ✅ PASS |
| `/api/monitor/sentiment` | ✅ PASS |

### 8.2 功能测试结果

| 功能 | 状态 |
|------|------|
| 信号生成 | ✅ 正常 |
| 持仓追踪 | ✅ 正常 |
| 价格获取 | ✅ 正常 |
| LLM 分析 | ✅ 正常（需配置 API Key） |
| 消息推送 | ✅ 正常（需配置 Webhook） |

---

## 九、后续优化建议

### 9.1 短期优化

- [ ] 添加单元测试覆盖
- [ ] 优化舆情新闻源
- [ ] 添加更多技术指标过滤规则

### 9.2 中期优化

- [ ] 支持多账户管理
- [ ] 添加回测功能
- [ ] 支持更多消息推送渠道（钉钉、邮件）

### 9.3 长期优化

- [ ] 实盘交易接口对接
- [ ] 风险管理模块
- [ ] 多策略组合优化

---

## 十、文档清单

| 文档 | 路径 | 内容 |
|------|------|------|
| 使用说明 | `monitor/README.md` | 完整的使用文档 |
| 开发目录 | `monitor/DEVLOG.md` | 本文档 |
| 配置示例 | `config/monitor.yaml` | 配置文件模板 |

---

## 十一、版本信息

- **版本**: v1.0.0
- **发布日期**: 2026-03-23
- **分支**: V0.1.0-MSA
- **Python 版本**: 3.8+
- **主要依赖**: pandas, numpy, pyyaml, flask, requests, openai

---

*本文档由开发过程自动生成*
