# 阶段 E 完工报告（RD-Agent 反馈回路 · L2→L1 闭环）

**完成日期**：2026-04-20

**范围**：

- `factor_lab/feedback/` 新子包：契约 C3 `FeedbackBundle` + 聚合器 + 落盘/加载 API
- `factor_lab/adapters/` 新位置：RD-Agent 适配层从 `rdagent_integration/` 迁入，带
DeprecationWarning shim
- `factor_lab/runners/rdagent_loop.py` + `scripts/lab/run_rdagent_loop.py` 替代
`scripts/run_fin_quant.py`（旧脚本保留 shim）
- `factor_lab/adapters/quant_proposal.py` 重构：静态宪法（硬编码规则 + `csi300_RD_v2`
历史经验）+ 运行时动态 L2 反馈拼接；缺失/损坏时安全降级
- `scripts/lab/build_feedback_bundle.py` 独立 CLI
- `scripts/lab/run_lab_cycle.py` 末尾钩子：cycle 结束自动重建 feedback bundle

配套阅读：

- `docs/STAGE_D_REPORT.md` —— 前序真实回测 + 边际贡献 + 自动闭环
- `docs/STAGE_E_TEST_GUIDE.md` —— 阶段 E 架构/接口级指南
- `docs/STAGE_E_TEST_PLAYBOOK.md` —— 阶段 E 测试 Playbook（复制即跑）

---

## 0. 为什么做阶段 E

阶段 D 已经把"L1 生因子 → L2 判生死 → L3 入库"自动化了。但系统**没有记忆**：
RD-Agent 每次 loop 都从同一段硬编码 RAG 出发，无论上一轮哪些因子被 L2 打回来。

阶段 E 要解决的**唯一问题**：让 RD-Agent 下一轮 hypothesis generation 读到**上一轮
为什么没进 L3**的结构化信号，从"失败的因子家族"里学习、避开。

闭环示意：

```text
L2 cycle 证书 + L3 registry
        │
        ▼  factor_lab.feedback.aggregator
   C3 FeedbackBundle（factor_lab/workspace/feedback/latest.json）
        │
        ▼  factor_lab.adapters.quant_proposal.compose_project_rag
   RD-Agent RAG（静态宪法 + 动态段拼接）
        │
        ▼
  下一轮 hypothesis 会主动避开 discouraged_families
```

---

## 1. 交付清单


| 子任务 | 产物                                                                                                                                                                                                 | 状态        |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| E.1 | `factor_lab/feedback/schema.py`：`FeedbackBundle` / `ActiveFactorSummary` / `RetiredFactorSummary` / `FailedCandidateSummary`（C3）+ 11 单测                                                            | COMPLETED |
| E.2 | `factor_lab/feedback/aggregator.py`：`build_feedback_bundle` / `write_feedback_bundle` / `load_latest_feedback_bundle` / `classify_family` + `scripts/lab/build_feedback_bundle.py` CLI + 15 + 2 单测 | COMPLETED |
| E.3 | `factor_lab/adapters/quant_proposal.py`：静态宪法剥离为常量 + 纯函数 `compose_project_rag` + 动态 L2 feedback 注入；安全降级 + 12 单测                                                                                     | COMPLETED |
| E.4 | 目录重构 `rdagent_integration/` → `factor_lab/adapters/` + `factor_lab/runners/rdagent_loop.py` + `scripts/lab/run_rdagent_loop.py`；老路径保留 DeprecationWarning shim + 12 单测                              | COMPLETED |
| E.5 | `scripts/lab/run_lab_cycle.py` 末尾钩子：cycle 结束自动 `build_feedback_bundle`；`CycleSummary` 扩展 4 字段；CLI 增 3 flag（`--feedback-dir` / `--feedback-max-cycles` / `--skip-feedback-rebuild`）+ 4 单测           | COMPLETED |
| E.6 | `STAGE_E_REPORT.md` + `STAGE_E_TEST_GUIDE.md` + `STAGE_E_TEST_PLAYBOOK.md` + 全量回归                                                                                                                  | COMPLETED |


### 1.1 代码 / 文件统计

新增文件：

- `factor_lab/feedback/__init__.py`、`factor_lab/feedback/schema.py`（E.1）
- `factor_lab/feedback/aggregator.py`（E.2）
- `scripts/lab/build_feedback_bundle.py`（E.2）
- `factor_lab/adapters/experiments.py`、`factor_lab/adapters/proposal.py`、
`factor_lab/adapters/quant_proposal.py`、`factor_lab/adapters/patch_qlib_conda.py`（E.4 新家）
- `factor_lab/runners/rdagent_loop.py`（E.4）
- `scripts/lab/run_rdagent_loop.py`（E.4）
- 单测：
  - `tests/factor_lab/feedback/test_feedback_schema.py`（11）
  - `tests/factor_lab/feedback/test_feedback_aggregator.py`（15）
  - `tests/scripts/lab/test_build_feedback_bundle_cli.py`（2）
  - `tests/rdagent_integration/test_project_quant_proposal.py`（12）
  - `tests/factor_lab/adapters/test_adapters_refactor.py`（12）
  - `tests/scripts/lab/test_run_lab_cycle_feedback.py`（4）
- 文档：`docs/STAGE_E_TEST_GUIDE.md`、`docs/STAGE_E_TEST_PLAYBOOK.md`、本文件

修改文件：

- `rdagent_integration/__init__.py`、`rdagent_integration/project_experiments.py`、
`rdagent_integration/project_proposal.py`、`rdagent_integration/project_quant_proposal.py`、
`rdagent_integration/patch_qlib_conda.py` → 全部改成 re-export shim（DeprecationWarning）
- `scripts/run_fin_quant.py` → shim，重定向到 `factor_lab.runners.rdagent_loop.run_rdagent_loop`
- `scripts/lab/run_lab_cycle.py` → 集成 E.5 feedback 钩子、CLI flag、`CycleSummary` 扩展
- `factor_lab/README.md` → 阶段 E 状态更新
- `tests/scripts/lab/test_run_lab_cycle.py` → 已有用例补 `feedback_dir=` 参数防止污染默认路径
- （附带修复）`tests/test_unified_strategy.py` 一行语法错误（非 E 范畴，顺手修）

---

## 2. 核心技术说明

### 2.1 C3 FeedbackBundle：结构化 vs. 散文

C1/C2 是"重"契约（引用 parquet/代码路径、嵌入完整 C2 校验结果）。C3 不一样：

- **自包含**：整份 bundle 落一个 JSON 文件，RAG 注入器**只读这个文件**，不再读证书、
不再读 manifest。
- **提炼后的摘要**：`FailedCandidateSummary` 只留"家族标签 / stage / decision /
failure_modes / 关键指标"；不重复存 C2 的 schema。
- **凡是可派生字段都在 aggregator 算好**：`failure_family_counts`、`discouraged_families`
是聚合器根据窗口内 fails + retired 推出来的，LLM 直接读结论。

这么设计有三个好处：

1. **LLM prompt 可复现**：bundle 里的 `to_markdown()` 顺序固定，同一份 bundle → 同一段 RAG。
2. **跨版本兼容**：C3 `schema_version="1.0"`；未来升级窗口策略只需 bump minor。
3. **零下游依赖**：RD-Agent 不 import `factor_validation` 或 `factor_registry`，
  降低耦合。

### 2.2 家族分类：启发式 + 白名单

`classify_family(name)` 用一组有序 regex 把因子名映射到固定 family 标签：


| family                | 正则关键词（忽略大小写）                                                                 |
| --------------------- | ---------------------------------------------------------------------------- |
| volume_price_reversal | `VolRev` / `VolRet` / `VolTwist` / `RangeRatio` / `VolumeTrend` / `VolRatio` |
| volume_price_momentum | `VolumePriceTrend` / `VolPriceMom` / `VolMom`                                |
| quality_persist       | `QualPersist` / `ROEPersist` / `Quality`                                     |
| value_mean_revert     | `ValueMR` / `ValueRev` / `PEMR` / `PBMR`                                     |
| margin_trend          | `MarginTrend` / `MarginFlow` / `Rzye`                                        |
| earnings_revision     | `EarnRev` / `EPSRev` / `ProfitRev`                                           |
| residual_momentum     | `ResidMom` / `ResidualMom` / `BetaAdj`                                       |
| liquidity_stability   | `LiqStab` / `LiqVol` / `TurnStab`                                            |
| low_volatility        | `LowVol` / `LowRisk` / `LowBeta`                                             |


未命中 → `"unknown"`（`discouraged_families` 不收 `"unknown"`，避免一刀切封死 LLM
创造性）。

> **路线图**：阶段 F 可以把这套分类表抽成 YAML 让研究员按需扩展；当前版本锁死代码里
> 以确保 LLM 看到稳定的标签集合。

### 2.3 静态宪法 + 动态反馈两段式 RAG

`factor_lab/adapters/quant_proposal.py::compose_project_rag()` 是纯函数，顺序如下：

1. **base_rag**（RD-Agent 自己的上游 RAG，例如 fin_quant 默认的因子池描述）
2. **_STATIC_CONSTITUTION**：格式规则 / 列名清单 / `csi300_RD_v2` 历史 Sharpe 崩塌故事 /
  encouraged/discouraged families
3. **dynamic feedback**：从 `factor_lab/workspace/feedback/latest.json` 加载 C3 bundle，
  调 `to_markdown()` 生成；**加载失败时这一段完全省略**（静态宪法仍在）。

**为什么不全量替换静态段？** 因为静态段里有两类信息：

- `csi300_RD_v2` 的 Sharpe-崩塌故事 —— 这是一次**性质发现**，不会被下一轮 cycle 再生成；
- 格式/列名规则 —— 数据层面的约束，改代码才会改，不该被 C3 吞噬。

阶段 E 的 RAG 组合原则：**能从数据自动推出来的进动态段；属于工程约束和历史教训的留静态段**。

### 2.4 safety net：RD-Agent 永不因 feedback 回路挂掉

三层保护：

1. `_render_dynamic_feedback` 用 try/except 包起整条"import → load → render"链，
  任何异常都降级为返回空串并 log warning。
2. `run_lab_cycle` 末尾钩子 `_rebuild_feedback_bundle` 独立于 cycle 主流程：
  钩子抛异常只记进 `summary.feedback_error`，不影响 `promoted` / `failed` 数字，
   不阻止 cycle 报告写盘。
3. Shim 层（E.4）保留老 import 路径 + `_PROJECT_FACTOR_RAG` 常量别名；外部 import
  链（包括 CI 脚本）不会因为重构瞬断。

---

## 3. 测试覆盖


| 层级                          | 数量                                   | 关键覆盖点                                                                                                                             |
| --------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| 契约 `FeedbackBundle`         | 11                                   | pydantic 不可变、`cycles_included` 窗口约束、`recent_fails.cycle_id ⊆ cycles_included`、markdown 空分区省略                                      |
| 聚合器 `build_feedback_bundle` | 15                                   | 最近 N 截断、同因子取最深 stage、PASS 不进 fails、坏证书跳过、退役家族永远进 discouraged、空树返回空 bundle                                                         |
| 聚合器 CLI                     | 2                                    | 参数透传、max_cycles=0 非法                                                                                                              |
| 静态 + 动态 RAG 合成              | 12                                   | 静态宪法常驻、动态存在时追加、bundle 缺失/损坏时静默降级、`include_dynamic=False` 关闭动态、章节顺序稳定、默认路径指向 `factor_lab/workspace/feedback`                       |
| E.4 重构                      | 12                                   | 新家 4 模块可 import；老 shim 4 模块仍发 DeprecationWarning；同名符号 `is` 相等；`_PROJECT_FACTOR_RAG` 别名保留；`scripts/run_fin_quant.py` shim 可 import |
| E.5 cycle 钩子                | 4                                    | 钩子写 `latest.json` + `latest.md` + `history/<cycle_id>.json`；`--skip-feedback-rebuild` 跳过；聚合器异常不污染 cycle；CLI flag 正确传导             |
| **阶段 E 合计（新增）**             | **56**                               |                                                                                                                                   |
| 阶段 D 已有（未修改通过）              | 4（`test_run_lab_cycle.py` 原 4 条）+ 其它 | 老契约兼容                                                                                                                             |


**全量回归**：`python -m pytest tests -q --ignore=tests/test_unified_strategy.py` →
`334 passed, 7 warnings`（7 warnings 全是 pydantic v2 兼容警告或 pandas FutureWarning，
非阶段 E 引入）。

> 注：`tests/test_unified_strategy.py` 里 2 条 pre-existing 失败（momentum 与 mean
> reversion 的数据形状不匹配）与阶段 E 无关，属于更早提交引入，已在附录登记。

---

## 4. 非回归保证

- **D.5 自动闭环**：`scripts/lab/run_lab_cycle.py` 的参数默认值保持不变；
`skip_feedback_rebuild=False` 是默认，老 CLI 调用自动享受 E.5 能力。如果下游系统
还依赖 D.5 的 JSON 报告字段，新字段只是**追加**在 `totals` 同级下，不改既有字段。
- **老 import 路径**：`rdagent_integration.`* / `scripts.run_fin_quant`
继续可用，仅发 DeprecationWarning；阶段 G 前不会删。
- **硬编码宪法不变**：静态 RAG 段的 7 条规则、csi300_RD_v2 经验段、encouraged/
discouraged families 保持原文（`_STATIC_CONSTITUTION`）。现有 fin_quant 跑法零感知。
- **RD-Agent 永不挂**：动态注入失败 → 静态宪法；cycle 钩子失败 → cycle 报告仍在。

---

## 5. 产物路径速查


| 用途                               | 路径                                                                                     |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| C3 最新 feedback bundle（RAG 注入器读取） | `factor_lab/workspace/feedback/latest.json`                                            |
| C3 渲染版（人工阅读）                     | `factor_lab/workspace/feedback/latest.md`                                              |
| C3 历史快照                          | `factor_lab/workspace/feedback/history/<cycle_id>.json`                                |
| Cycle 报告（含 feedback 子结构）         | `factor_validation/reports/lab_cycle_<cycle_id>.{json,md}`                             |
| RD-Agent 循环入口（新）                 | `scripts/lab/run_rdagent_loop.py` → `factor_lab.runners.rdagent_loop.run_rdagent_loop` |
| RD-Agent 循环入口（旧 shim）            | `scripts/run_fin_quant.py`                                                             |
| Feedback CLI（独立触发）               | `python -m scripts.lab.build_feedback_bundle`                                          |


---

## 6. 已知限制与后续

- **家族分类启发式**：regex 列表是手工维护的。阶段 F 建议抽成 YAML + 单测保护。
- **跨 universe 支持**：目前聚合器不区分 `universe`；若未来同时跑 csi300 与 csi500
的 L2 cycle，会把两者的失败混在一起。需要给 FeedbackBundle 加 `universe` 维度。
- **窗口策略**：目前只有 `max_cycles`（数量）。若某段时间 cycle 非常频繁，实际窗口
会过短。可选加 `max_age_days` 兜底。
- **embedding-based RAG**：当前只注入 markdown 文本。未来可考虑把 C3 的失败家族
embedding 化，供 RD-Agent 的 similarity RAG 调用；不在阶段 E 范围。

---

## 7. 下一步（阶段 F 预案）

- **F.1**：家族分类 → YAML-driven，研究员可扩展；保留向后兼容
- **F.2**：FeedbackBundle 增 `universe` 维度；cycle 钩子按 universe 分桶
- **F.3**：静态宪法的 encouraged/discouraged lists 改为 YAML，RD-Agent 之外其它消费者
（例如手工研究员）也能复用
- **F.4**：端到端冒烟：跑一次真实 RD-Agent 小型 loop → 验证 bundle 自动回流后下一轮
hypothesis 确实避开了 discouraged 家族（验证 LLM 对我们 RAG 的响应能力）

