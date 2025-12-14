# 基准指数快速测试指南

## 快速测试 RQAlpha 对中小指数的支持

根据 [RQAlpha API 文档](https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html)，使用 RQAlpha API 直接测试基准代码是否被支持。

## 快速开始

### 1. 简单查询测试（推荐，最快）

测试基准代码是否存在于 RQAlpha 数据源中：

```bash
python scripts/test_rqalpha_benchmark_support.py --mode simple
```

**输出示例**:
```
================================================================================
RQAlpha 中小指数支持测试
================================================================================
测试模式: simple

要测试的基准代码: 399005.XSHE, 399101.XSHE, 399006.XSHE, 000300.XSHG, 000905.XSHG

================================================================================
步骤 1: 简单查询测试（查询合约信息）
================================================================================

测试基准代码: 399005.XSHE
  ✓ 合约存在
    代码: 399005.XSHE
    名称: 中小板指
    类型: Index
    ...
```

### 2. 回测测试（验证基准数据可加载）

实际运行最小回测，验证基准数据是否能被正确加载：

```bash
python scripts/test_rqalpha_benchmark_support.py --mode backtest
```

**注意**: 此测试会运行最小回测，可能需要一些时间。

### 3. 完整测试（两种模式都测试）

```bash
python scripts/test_rqalpha_benchmark_support.py --mode both
```

## 测试特定基准代码

```bash
# 测试中小板指
python scripts/test_rqalpha_benchmark_support.py --mode simple --benchmark 399005.XSHE

# 测试中小综指
python scripts/test_rqalpha_benchmark_support.py --mode simple --benchmark 399101.XSHE

# 测试创业板指
python scripts/test_rqalpha_benchmark_support.py --mode simple --benchmark 399006.XSHE
```

## 测试结果解读

### ✓ 合约存在
- 表示该基准代码在 RQAlpha 数据源中存在
- 可以作为基准使用

### ❌ 合约不存在
- 表示该基准代码在 RQAlpha 数据源中不存在
- 可能原因：
  1. 代码格式错误
  2. 数据源不包含该指数
  3. RQAlpha 版本不支持

### ✓ 基准数据可正常加载
- 表示回测时基准数据可以正常加载
- 该基准代码可以正常使用

### ⚠ 回测成功但基准数据未加载
- 回测可以运行，但基准数据未加载
- 可能原因：
  1. 数据源中该日期范围内无数据
  2. 基准代码格式问题
  3. RQAlpha 配置问题

## 常见中小指数代码

| 代码 | 名称 | 说明 |
|------|------|------|
| `399005.XSHE` | 中小板指 | 中小板指数 |
| `399101.XSHE` | 中小综指 | 中小盘综合指数 |
| `399006.XSHE` | 创业板指 | 创业板指数 |

## 当前配置检查

当前配置文件中使用的是 `399005.XSHE`（中小板指）。

**注意**: 配置文件中注释写的是"中小综指"，但代码实际是"中小板指"。
- 如需使用中小综指，应改为 `399101.XSHE`
- 使用前建议先用测试脚本验证支持情况

## 使用 RQAlpha API

测试脚本使用了以下 RQAlpha API：

1. **`instruments(order_book_id)`**: 查询合约信息
   - 用于验证基准代码是否存在
   - 参考: [RQAlpha API 文档](https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html)

2. **`run_file(strategy_path, config=config)`**: 运行回测
   - 用于验证基准数据是否能被加载
   - 参考: [RQAlpha 运行文档](https://rqalpha.readthedocs.io/zh-cn/latest/intro/run_algorithm.html)

## 故障排查

### 问题 1: RQAlpha 未安装

**错误**: `ModuleNotFoundError: No module named 'rqalpha'`

**解决**:
```bash
pip install rqalpha
```

### 问题 2: 所有基准代码都显示不存在

**可能原因**:
1. RQAlpha 数据源未初始化
2. 数据源路径配置错误

**解决**:
```bash
# 初始化 RQAlpha 数据源
rqalpha update_bundle
```

### 问题 3: 回测测试失败

**可能原因**:
1. 日期范围内无数据
2. 基准代码格式错误
3. RQAlpha 版本问题

**解决**:
1. 尝试不同的日期范围
2. 检查基准代码格式
3. 查看详细错误信息

## 相关文件

- 测试脚本: `scripts/test_rqalpha_benchmark_support.py`
- 配置文件: `config/rqalpha_config.yaml`
- 详细文档: `docs/BENCHMARK_TEST.md`
- RQAlpha API 文档: https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html

