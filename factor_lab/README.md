# factor_lab —— L1 因子试验场

> 本包是三层架构的 L1 层。请先阅读 `[docs/ARCHITECTURE_FACTOR_LAB.md](../docs/ARCHITECTURE_FACTOR_LAB.md)`。

## 子目录


| 路径           | 职责                                                                      |
| ------------ | ----------------------------------------------------------------------- |
| `runners/`   | RD-Agent 循环入口、手工提交入口（阶段 E 实现）                                           |
| `adapters/`  | RD-Agent 框架适配层（阶段 E 从 `rdagent_integration/` + `rdagent_overrides/` 重构） |
| `exporters/` | 把 RD-Agent 跑完的产物打包成 `CandidateFactorPackage` (契约 C1)                    |
| `workspace/` | gitignore；候选因子工作区，每个候选一个目录                                              |


## 对外唯一出口

L1 通过 `CandidateFactorPackage` (契约 C1) 把候选因子交给 L2。其他模块**禁止**直接读
`factor_lab/workspace/`，必须通过 `factor_validation` 拉取经过校验的契约对象。

```python
from factor_lab import CandidateFactorPackage
```

## 当前状态

- 阶段 A：契约 schema 和包骨架。
- 阶段 B~D：`exporters/` + L2 校验闭环（见 `factor_validation/`）。
- 阶段 E：**已完成** `rdagent_integration/` → `factor_lab/adapters/` 搬迁，`scripts/run_fin_quant.py`
→ `factor_lab/runners/rdagent_loop.py` + `scripts/lab/run_rdagent_loop.py`。老路径保留
DeprecationWarning shim。新增 `factor_lab/feedback/` 子包：L2 cycle 结论 → C3 FeedbackBundle →
RAG 注入闭环，见 `docs/STAGE_E_REPORT.md`。
- 阶段 F：**已完成** 把家族分类（`factor_families.yaml`）和 RAG 静态宪法（`rag_constitution.yaml`）
从硬编码迁到 `factor_lab/config/`（YAML 主源 + Python fallback，byte-for-byte 一致）；
`FeedbackBundle` 三类 summary 加 `universe` 字段并在 markdown 里渲染 `[universe=...]`；
新增 `tests/e2e/` 与 `@pytest.mark.e2e_rdagent`（默认 skip）的端到端闭环测试。
详见 `docs/STAGE_F_REPORT.md`。
- 阶段 G：**已完成** 四项能力增强 / 清理：
G.1 `FeedbackBundle.by_universe: dict[str, UniverseSubBundle]`（per-universe 子视图 +
`latest_<universe>.json` 落盘）；
G.2 `rdagent_integration.`* 与 `scripts/run_fin_quant.py` 硬下线（import 即
`RuntimeError`，附迁移路径）；
G.3 新增 `factor_lab/feedback/embedding.py`（TF-IDF 检索 + 可插拔 backend），
在 `compose_project_rag` 里按 query opt-in 注入 top-k 相似历史失败；
G.4 `rag_constitution.yaml` 的 `discouraged_families` 支持可选 `penalty` 字段
（`.inf`=硬黑、具体数值=软降权并渲染"需附加理由"），默认仍 byte-for-byte 兼容 F。
详见 `docs/STAGE_G_REPORT.md`。
- 阶段 H：**已完成** H.1 —— `ProjectQlibQuantHypothesisGen.prepare_context` 自动推导
  `retrieval_query`（优先级：`ctx['RETRIEVAL_QUERY']` > trace 语义字段 > RAG 末行），
  G.3 检索从"需手工传参"升级为"默认在线"。配套落地 DEBUG 观测日志、类级回滚开关
  `auto_retrieval_enabled`、异常兜底，以及 5 层测试分层策略
  （`docs/STAGE_H_TEST_STRATEGY.md`）。已用真实 DeepSeek 跑通 hypothesis-only dry loop，
  LLM 实际读懂并规避 `discouraged_families`，详见 `docs/STAGE_H_LIVE_LOOP_LOG.md`。
  报告见 `docs/STAGE_H_REPORT.md`。

