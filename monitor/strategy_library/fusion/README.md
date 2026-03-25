# 融合系统 - Fusion System

将模型预测与策略信号进行智能融合，实现增益效应。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    四层融合架构                          │
├─────────────────────────────────────────────────────────┤
│  Layer 1: 信号预处理                                     │
│  ├─ 标准化：统一到[-1, 1]                               │
│  ├─ 置信度调整                                          │
│  └─ 信号对齐                                            │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 一致性分析                                     │
│  ├─ 方向一致性                                          │
│  ├─ 强度一致性                                          │
│  └─ 时效一致性                                          │
├─────────────────────────────────────────────────────────┤
│  Layer 3: 增益计算                                       │
│  ├─ 基础增益                                            │
│  ├─ 一致性增益 (+30%)                                   │
│  ├─ 市场状态增益                                        │
│  └─ 协同增益                                            │
├─────────────────────────────────────────────────────────┤
│  Layer 4: 决策输出                                       │
│  ├─ 仓位计算                                            │
│  ├─ 止损止盈                                            │
│  └─ 执行优先级                                          │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 运行测试

```bash
cd d:/lianghuatouzi/Qlib1124/project

# 运行所有测试
python -m monitor.strategy_library.fusion.run_tests

# 测试指定层
python -m monitor.strategy_library.fusion.run_tests --layer 1

# 运行演示
python -m monitor.strategy_library.fusion.run_tests --demo

# 运行回测演示
python -m monitor.strategy_library.fusion.run_tests --backtest
```

### 基本使用

```python
from monitor.strategy_library.fusion import (
    LayeredFusionProcessor,
    FusionConfig,
    ModelPrediction,
    StrategySignalWrapper,
    MarketContext,
    MarketRegime,
)

# 创建融合处理器
config = FusionConfig(
    consistency_boost=1.3,    # 一致性增益30%
    conflict_penalty=0.7,     # 冲突惩罚30%
)
processor = LayeredFusionProcessor(config)

# 准备数据
model_predictions = {
    "000001.SZ": ModelPrediction(
        code="000001.SZ",
        name="平安银行",
        score=0.72,
        confidence=0.85,
        horizon=5,
    ),
}

strategy_signals = {
    "000001.SZ": StrategySignalWrapper(
        code="000001.SZ",
        name="平安银行",
        action="buy",
        strength=0.75,
        reason="多因子策略看多",
        strategy_id="multi_factor",
        strategy_name="多因子策略",
    ),
}

market_context = MarketContext(
    index_value=3050,
    index_change=0.015,
    volatility=0.018,
    trend="up",
    regime=MarketRegime.BULL_TRENDING,
    uncertainty=0.35,
)

# 执行融合
decisions = processor.process(model_predictions, strategy_signals, market_context)

# 查看结果
for d in decisions:
    print(f"{d.signal.code}: {d.signal.action}")
    print(f"  融合分数: {d.signal.fused_score:.3f}")
    print(f"  仓位: {d.position_ratio:.2%}")
    print(f"  优先级: {d.execution_priority}/10")
```

## 模块说明

### 核心模块

| 模块 | 说明 |
|------|------|
| `fusion_engine.py` | 基础融合引擎 |
| `layered_fusion.py` | 四层融合处理器 |
| `fusion_backtest.py` | 融合回测系统 |
| `data_provider.py` | 数据支持模块 |

### 主要类

| 类 | 说明 |
|------|------|
| `LayeredFusionProcessor` | 分层融合处理器（主入口） |
| `GainValidator` | 增益效应验证器 |
| `FusionBacktester` | 融合回测器 |
| `FusionDataProvider` | 数据提供者 |

## 增益机制

### 一致性增益

- **一致信号**: 增益 +30%
- **冲突信号**: 惩罚 -30%
- **部分信号**: 无增益

### 市场状态增益

| 市场状态 | 顺势信号 | 逆势信号 |
|----------|----------|----------|
| 牛市趋势 | +15% | -10% |
| 牛市震荡 | +5% | -5% |
| 熊市趋势 | -10% | +15% |
| 熊市震荡 | -5% | +5% |
| 中性 | 0% | 0% |

### 协同增益

当模型和策略一致时：
```
协同增益 = 置信度乘积 × 强度对齐度 × 20%
```

## 增益验证指标

| 指标 | 计算方式 | 增益条件 |
|------|----------|----------|
| 收益增益 | 融合收益 - max(模型, 策略) | > 0 |
| 夏普增益 | 融合夏普 - max(模型, 策略) | > 0 |
| 回撤增益 | 融合回撤 - min(模型, 策略) | < 0 |
| 胜率增益 | 融合胜率 - max(模型, 策略) | > 0 |

## 回测示例

```python
from monitor.strategy_library.fusion import run_fusion_backtest

result = run_fusion_backtest(
    price_data=price_df,
    model_predictions=predictions,
    strategy_signals=signals,
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=1000000,
)

# 查看增益效果
gain = result['gain_metrics']
print(f"收益增益: {gain.return_gain * 100:.2f}%")
print(f"夏普增益: {gain.sharpe_gain:.2f}")
print(f"增益效应: {'存在' if gain.has_gain() else '不存在'}")
```

## 配置参数

```python
config = FusionConfig(
    base_model_weight=0.5,        # 基础模型权重
    base_strategy_weight=0.5,     # 基础策略权重
    consistency_boost=1.3,        # 一致性增益倍数
    conflict_penalty=0.7,         # 冲突惩罚倍数
    min_confidence_threshold=0.3, # 最小置信度阈值
    high_confidence_threshold=0.7,# 高置信度阈值
    enable_dynamic_weights=True,  # 启用动态权重
    enable_regime_adjustment=True,# 启用市场状态调整
)
```

## 测试覆盖

- Layer 1 信号预处理 (4 tests)
- Layer 2 一致性分析 (4 tests)
- Layer 3 增益计算 (3 tests)
- Layer 4 决策输出 (3 tests)
- 增益验证器 (3 tests)
- 分层融合处理器 (3 tests)
- 融合回测系统 (1 test)
- 数据提供者 (3 tests)
- 集成测试 (1 test)

**总计: 25 tests**
