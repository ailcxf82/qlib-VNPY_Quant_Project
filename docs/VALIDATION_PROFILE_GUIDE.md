# ValidationProfile 编写指南

> 本文件指导如何编写 / 调试 / 评审 `factor_validation/profiles/*.yaml` 验证标准。
> 总体架构见 [`docs/ARCHITECTURE_FACTOR_LAB.md`](ARCHITECTURE_FACTOR_LAB.md)。
>
> **版本**：v1.0 — 2026-04-19

---

## 1. 一个 profile 是什么

一个 ValidationProfile = 一组 **可配置的验证标准**：

* 启用哪些 `check`
* 每个 check 的阈值和权重
* 如何把多个 check 聚合成 `overall_score`
* PASS / FAIL / HOLD 的边界

每次验证都把 profile YAML 的 sha256 写进 `CertifiedFactorRecord.profile_hash`，
**证书永久可复现**。改 profile = 旧证书自动失效。

---

## 2. 顶层结构

```yaml
profile_name: strict           # 必须；与文件名（去后缀）一致
description: "..."             # 必须；人类可读说明

universe: csi300               # 必须；该 profile 适用的股票池
oos_window:                    # 必须；OOS 时间窗（验证用，不重叠候选训练窗）
  - "2025-01-01"
  - "2025-10-31"
benchmark: SH000300            # 必须；基准

aggregation:                   # 必须；如何聚合所有 check.score
  method: weighted_sum         # 当前仅支持 weighted_sum
  pass_threshold: 0.6          # >= 此值 → PASS
  fail_threshold: 0.3          # <  此值 → FAIL
                               # 介于二者 → HOLD

checks:                        # 必须；启用的所有 check 配置
  <check_name>:
    enabled: true|false
    weight: 0.0~1.0            # enabled=true 时必填；总和须 == 1.0
    <check 私有字段>: ...
```

**强一致性**（orchestrator 加载时校验）：

* `aggregation.fail_threshold <= aggregation.pass_threshold`
* `sum(weight for c in checks if c.enabled) == 1.0` (容差 1e-6)
* 所有出现的 `<check_name>` 必须在 `factor_validation.checks` 包内有同名注册

---

## 3. 内置 check 清单

| name | 类别 | 主要字段 | 含义 |
|---|---|---|---|
| `coverage` | 离线 | `min_non_null_ratio`, `min_instrument_ratio` | 数据覆盖率（非空率、票数覆盖） |
| `ic` | 离线 | `min_rank_ic`, `min_ic_ir` | OOS Rank IC、IC_IR |
| `stability` | 离线 | `n_subperiods`, `min_ic_positive_ratio` | IC 在子时段间稳定 |
| `orthogonality` | 离线 | `target_feature_set`, `max_abs_corr` | 与 L3 现有因子集相关性 |
| `turnover` | 离线 | `max_daily_rank_turnover`, `min_rank_autocorr` | 因子日换手 / 排名自相关 |
| `lookahead` | 离线 | `n_permutations`, `max_p_value` | permutation test 防前视 |
| `backtest` | 联动 L3 | `engine`, `config_path`, `min_sharpe`, `max_drawdown`, `max_annualized_turnover` | 项目真实回测（RQAlpha） |
| `marginal` | 联动 L3 | `baseline_features`, `min_sharpe_uplift` | 加入新因子后 A/B Sharpe 增益 |

各 check 的字段细节见 `factor_validation/checks/<name>.py` 的 docstring。

---

## 4. 实例：`strict.yaml`

```yaml
profile_name: strict
description: "Production 准入标准；所有 check 必须通过"

universe: csi300
oos_window: ["2025-01-01", "2025-10-31"]
benchmark: SH000300

aggregation:
  method: weighted_sum
  pass_threshold: 0.6
  fail_threshold: 0.3

checks:
  coverage:
    enabled: true
    weight: 0.10
    min_non_null_ratio: 0.85
    min_instrument_ratio: 0.90
  ic:
    enabled: true
    weight: 0.25
    min_rank_ic: 0.015
    min_ic_ir: 0.30
  stability:
    enabled: true
    weight: 0.15
    n_subperiods: 4
    min_ic_positive_ratio: 0.75
  orthogonality:
    enabled: true
    weight: 0.10
    target_feature_set: lgb_short_cycle
    max_abs_corr: 0.50
  turnover:
    enabled: true
    weight: 0.10
    max_daily_rank_turnover: 0.40
    min_rank_autocorr: 0.60
  lookahead:
    enabled: true
    weight: 0.05
    n_permutations: 100
    max_p_value: 0.01
  backtest:
    enabled: true
    weight: 0.20
    engine: rqalpha
    config_path: config/rqalpha_config.yaml
    min_sharpe: 1.5
    max_drawdown: 0.20
    max_annualized_turnover: 5.0
  marginal:
    enabled: true
    weight: 0.05
    baseline_features: lgb_short_cycle
    min_sharpe_uplift: 0.05
```

---

## 5. 新增 check 的步骤

1. 在 `factor_validation/checks/` 新建 `<name>_check.py`，继承 `CheckBase`
2. 实现 `def run(self, candidate: CandidateFactorPackage, context: CheckContext) -> CheckResult`
3. 写 `tests/factor_validation/test_<name>_check.py`：纯单测，不依赖 LLM / 网络
4. 在你的 profile YAML 中启用并指定 weight
5. 在本文件 §3 表格补行

---

## 6. 常见踩坑

| 现象 | 原因 | 解决 |
|---|---|---|
| profile 改了但旧证书还显示 PASS | 证书是 snapshot；`profile_hash` 已变，下次 validate 会重判 | 重跑 `validate_one.py` |
| `weight` 总和不等于 1 | YAML 校验失败 | 重新分配权重，确保总和精确为 1.0 |
| `pass_threshold < fail_threshold` | 配置反了 | 检查上下界 |
| `marginal_check` 一直 FAIL | 候选与 baseline 高相关 | 检查 `orthogonality_check` 是否也 FAIL，可能是同质 |
| 整个 profile 报"check 未注册" | 你写的 check 名 / 文件名 / 注册位置不一致 | 见 `factor_validation/checks/__init__.py` 的注册表 |

---

## 7. 演进规则

* 新增 check：minor 版本；旧 profile 不受影响（默认 `enabled: false`）
* 修改既有 check 的字段语义：major 版本；所有引用该字段的 profile 必须同步更新
* `aggregation.method` 新增方式（如 `geometric_mean`）：minor 版本，旧 profile 默认仍走 `weighted_sum`
