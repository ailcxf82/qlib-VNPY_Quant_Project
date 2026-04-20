# factor_lab —— L1 因子试验场

> 本包是三层架构的 L1 层。请先阅读 [`docs/ARCHITECTURE_FACTOR_LAB.md`](../docs/ARCHITECTURE_FACTOR_LAB.md)。

## 子目录

| 路径 | 职责 |
|---|---|
| `runners/` | RD-Agent 循环入口、手工提交入口（阶段 E 实现） |
| `adapters/` | RD-Agent 框架适配层（阶段 E 从 `rdagent_integration/` + `rdagent_overrides/` 重构） |
| `exporters/` | 把 RD-Agent 跑完的产物打包成 `CandidateFactorPackage` (契约 C1) |
| `workspace/` | gitignore；候选因子工作区，每个候选一个目录 |

## 对外唯一出口

L1 通过 `CandidateFactorPackage` (契约 C1) 把候选因子交给 L2。其他模块**禁止**直接读
`factor_lab/workspace/`，必须通过 `factor_validation` 拉取经过校验的契约对象。

```python
from factor_lab import CandidateFactorPackage
```

## 当前状态

- 阶段 A：仅有契约 schema 和包骨架（本目录）
- 阶段 E：会把 `rdagent_integration/` + `rdagent_overrides/` + `scripts/run_fin_quant.py`
  重构进 `runners/` 与 `adapters/`
