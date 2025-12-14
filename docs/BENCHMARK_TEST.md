# 基准配置测试文档

## 概述

本文档说明如何测试和验证 RQAlpha 回测中的基准指数配置是否生效。

## 当前配置

根据 `config/rqalpha_config.yaml`，当前基准设置为：
- **基准代码**: `399005.XSHE`（中小板指）
- **回测日期**: 2023-10-01 至 2025-10-01

## 测试步骤

### 方法 1: 使用 RQAlpha API 测试基准支持（推荐）

使用 RQAlpha API 直接测试基准代码是否被支持：

```bash
# 简单查询测试（快速，仅查询合约信息）
python scripts/test_rqalpha_benchmark_support.py --mode simple

# 回测测试（较慢，实际运行最小回测验证基准数据加载）
python scripts/test_rqalpha_benchmark_support.py --mode backtest

# 完整测试（两种模式都测试）
python scripts/test_rqalpha_benchmark_support.py --mode both

# 测试特定基准代码
python scripts/test_rqalpha_benchmark_support.py --mode simple --benchmark 399005.XSHE

# 指定回测日期范围
python scripts/test_rqalpha_benchmark_support.py \
    --mode backtest \
    --start-date 2023-10-01 \
    --end-date 2023-10-10
```

**参考文档**: [RQAlpha API 文档](https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html)

### 方法 2: 使用配置测试脚本

运行测试脚本自动检查基准配置：

```bash
# 基本测试
python scripts/test_benchmark.py

# 指定配置文件和输出目录
python scripts/test_benchmark.py \
    --config config/rqalpha_config.yaml \
    --output-dir data/backtest/rqalpha

# 保存测试报告
python scripts/test_benchmark.py \
    --save-report data/backtest/rqalpha/benchmark_test_report.txt
```

### 方法 3: 手动检查

#### 步骤 1: 检查配置文件

查看 `config/rqalpha_config.yaml` 中的基准设置：

```yaml
base:
  benchmark: "399005.XSHE"  # 中小板指
```

#### 步骤 2: 运行回测

运行回测并观察日志输出：

```bash
python run_backtest.py \
    --use-rqalpha \
    --rqalpha-config config/rqalpha_config.yaml \
    --prediction data/predictions/pred_*.csv
```

在日志中查找以下信息：
- `基准配置信息:` - 显示配置文件中读取的基准代码
- `实际使用的基准代码:` - 显示实际传递给 RQAlpha 的基准代码
- `✓ 基准数据已成功加载` - 表示基准数据加载成功
- `⚠ 基准数据为空或未加载` - 表示基准数据加载失败

#### 步骤 3: 检查回测结果

检查回测结果文件：

1. **查看报告文件** (`data/backtest/rqalpha/report.json`):
   ```bash
   # 查找基准相关指标
   cat data/backtest/rqalpha/report.json | grep -i benchmark
   ```

2. **查看详细结果** (`data/backtest/rqalpha/detailed_results.json`):
   ```bash
   cat data/backtest/rqalpha/detailed_results.json
   ```

3. **查看图表** (`data/backtest/rqalpha/rqalpha_strategy_plot.png`):
   - 如果图表中包含基准净值曲线（橙色虚线），说明基准配置生效
   - 如果只有策略净值曲线，说明基准数据未加载

## 常见基准代码

| 基准代码 | 指数名称 | 交易所 | 说明 |
|---------|---------|--------|------|
| `000300.XSHG` | 沪深300 | 上海 | 大盘指数，最常用 |
| `000905.XSHG` | 中证500 | 上海 | 中盘指数 |
| `399005.XSHE` | 中小板指 | 深圳 | 中小板指数 |
| `399101.XSHE` | 中小综指 | 深圳 | 中小盘综合指数 |
| `399006.XSHE` | 创业板指 | 深圳 | 创业板指数 |

**注意**: 
- 当前配置使用的是 `399005.XSHE`（中小板指）
- 如需使用中小综指，应改为 `399101.XSHE`
- 使用前建议先用测试脚本验证 RQAlpha 是否支持该基准代码

## 问题排查

### 问题 1: 基准数据未加载

**症状**: 日志显示 `⚠ 基准数据为空或未加载`

**可能原因**:
1. 基准代码格式错误
2. 数据源中不包含该基准数据
3. 回测日期范围内无基准数据
4. RQAlpha 版本问题

**解决方法**:
1. 检查基准代码格式是否正确（应为 `代码.交易所` 格式）
2. 尝试使用常见的基准代码（如 `000300.XSHG`）
3. 检查 RQAlpha 数据源是否包含该基准
4. 查看回测日志中的详细错误信息

### 问题 2: 基准代码格式错误

**症状**: 日志显示 `⚠ 基准代码格式可能错误`

**解决方法**:
- 确保基准代码格式为 `代码.交易所`
- 上海交易所使用 `XSHG`，深圳交易所使用 `XSHE`
- 例如：`000300.XSHG`（不是 `000300.SH` 或 `SH000300`）

### 问题 3: 图表中无基准曲线

**症状**: 生成的图表中只有策略净值曲线，没有基准净值曲线

**可能原因**:
1. 基准数据未加载
2. 基准数据为空
3. 图表生成时基准数据提取失败

**解决方法**:
1. 检查回测日志，确认基准数据是否加载
2. 检查 `report.json` 中是否包含基准相关指标
3. 查看图表生成日志中的错误信息

## 验证基准是否生效的标准

基准配置生效的标志：

1. ✅ **日志输出**: 显示 `✓ 基准数据已成功加载，数据点数量: XXX`
2. ✅ **报告文件**: `report.json` 中包含基准相关指标（如 `benchmark_return`）
3. ✅ **图表文件**: `rqalpha_strategy_plot.png` 中包含基准净值曲线（橙色虚线）
4. ✅ **详细结果**: `detailed_results.json` 中包含基准相关数据

## 测试报告示例

运行测试脚本后，会生成类似以下的报告：

```
================================================================================
基准配置测试报告
================================================================================
测试时间: 2025-01-XX XX:XX:XX

【配置信息】
✓ 基准代码: 399005.XSHE
✓ 回测日期: 2023-10-01 至 2025-10-01

【回测结果检查】
✓ 回测报告存在
✓ 基准数据存在（数据点: 500）

【数据分析】
✓ 基准净值数据已找到
✓ 策略净值数据已找到

【测试结论】
✓ 基准配置已生效！
  基准数据已成功加载并在回测中使用。
================================================================================
```

## 相关文件

- 配置文件: `config/rqalpha_config.yaml`
- 回测脚本: `backtest/rqalpha_backtest.py`
- 测试脚本: `scripts/test_benchmark.py`
- 回测结果目录: `data/backtest/rqalpha/`

## 注意事项

1. **数据源**: 确保 RQAlpha 的数据源包含所需的基准数据
2. **日期范围**: 确保回测日期范围内有基准数据
3. **版本兼容**: 不同版本的 RQAlpha 可能对基准代码的支持不同
4. **日志级别**: 建议使用 `INFO` 或 `DEBUG` 级别查看详细日志

