# factor_validation —— L2 因子验证环境

> 本包是三层架构的 L2 层。请先阅读 [`docs/ARCHITECTURE_FACTOR_LAB.md`](../docs/ARCHITECTURE_FACTOR_LAB.md)。

## 子目录

| 路径 | 职责 |
|---|---|
| `profiles/` | **可配置的验证标准**（YAML），如 `default.yaml`、`strict.yaml`、`exploratory.yaml` |
| `checks/` | 单项检查实现（每个一个文件，可独立单测） |
| `orchestrator.py` | 编排：按 profile 串起 checks，聚合产出 `CertifiedFactorRecord` |
| `certifier.py` | 决策：PASS / FAIL / HOLD 判定 + 出准入证书 |
| `reports/` | gitignore；HTML / Markdown 报告产出 |
| `schema.py` | `CertifiedFactorRecord`、`CheckResult` 等契约 schema |

## 对外唯一出口

L2 通过 `CertifiedFactorRecord` (契约 C2) 把验证结果交给 L3 / 持久化。其他模块
**禁止**直接调用 L2 内部 check / orchestrator，必须通过对外暴露的 API。

```python
from factor_validation import CertifiedFactorRecord, CheckResult, Decision
```

## 当前状态

- 阶段 A：仅有契约 schema、profile 目录骨架
- 阶段 C：实现 `CheckBase` + 离线 checks (coverage/ic/orthogonality/turnover)
- 阶段 D：实现 backtest_check + marginal_check（联动 L3）
