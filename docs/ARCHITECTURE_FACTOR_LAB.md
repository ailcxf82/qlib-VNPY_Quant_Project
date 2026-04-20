# 因子试验场三层架构（Factor Lab Architecture）

> **本文件是项目宪法。** 所有涉及 RD-Agent / 因子 / 模型特征 / 回测的新增代码，必须严格遵守本文件定义的分层、目录、契约、纪律。如需偏离，必须先修改本文件并取得评审通过。
>
> **版本**：v1.0 — 2026-04-19 起生效
> **影响范围**：`rdagent_integration/`, `rdagent_overrides/`, `scripts/`, `feature/`, `models/`, `factor_registry/`（新增）, `factor_validation/`（新增）, `factor_lab/`（新增/由 `rdagent_integration` 重构而来）

---

## 0. 设计目标

| 目标 | 必须满足 |
|---|---|
| **G1** RD+Qlib 自动跑出"当下有效"候选因子 | L1 试验场可独立跑、独立报告、独立失败，不污染正式环境 |
| **G2** 候选因子在 Qlib 的标准化测试环境验证 | L2 验证标准**可配置**，验证逻辑与 L1 / L3 完全解耦 |
| **G3** 通过验证的因子可随时被回测和正式投资模型使用 | L3 正式环境只读 L2 颁发的"准入证书"+ 因子物料；L3 内部任何回测 / 训练管线都能直接消费已认证因子 |
| **G4** 三层之间形成可循环、可滚动、可审计的工作流 | 每个候选因子的全生命周期可在一个 `factor_id` 下追溯：提出 → 验证 → 准入 → 退役 |

---

## 1. 三层总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1  FACTOR LAB (试验场)        factor_lab/                          │
│  目标：用 RD-Agent + Qlib 大量产候选因子                             │
│  入：市场数据快照 + 项目"已有特征清单" + 上一轮反馈                  │
│  出：CandidateFactorPackage（候选因子包）                            │
│  失败成本：可任意失败 / 重跑 / 抛弃，不影响 L2 L3                    │
└─────────────────────────────────────────────────────────────────────┘
                            │  CandidateFactorPackage
                            ▼  （契约 C1）
┌─────────────────────────────────────────────────────────────────────┐
│  L2  FACTOR VALIDATION (验证环境)   factor_validation/               │
│  目标：用可配置的"准入标准"对候选因子打分、判定通过/拒绝             │
│  入：CandidateFactorPackage + ValidationProfile（可配置的验证标准）  │
│  出：CertifiedFactorRecord（认证报告）+ 准入决策 (PASS/FAIL/HOLD)    │
│  失败成本：拒绝候选；不影响已在 L3 的因子                            │
└─────────────────────────────────────────────────────────────────────┘
                            │  CertifiedFactorRecord (PASS only)
                            ▼  （契约 C2）
┌─────────────────────────────────────────────────────────────────────┐
│  L3  PRODUCTION (正式环境)         factor_registry/ + 现有项目代码   │
│  目标：被认证的因子可随时被任何 production 管线消费                  │
│  入：CertifiedFactorRecord                                            │
│  出：注册到 FactorRegistry，写入 production parquet，               │
│       可被 EnsembleModelManager / 任何回测脚本无差别消费             │
│  失败成本：不允许失败；任何代码变更必须通过 L2 重新验证              │
└─────────────────────────────────────────────────────────────────────┘
```

**关键纪律**：
1. **L1 → L2 → L3 单向流动**。L3 不能感知 L1 是否在跑；L1 不能直接写 production parquet。
2. **L1 / L2 / L3 之间的所有数据交换必须经过契约（C1, C2）的 schema 校验**。任何跨层直接调函数 = 违反架构。
3. **L2 是唯一的把关者**。L1 跑 1000 个因子，L2 只放过通过验证的几个；L3 永远不会"试试看"。
4. **任何一层可以独立测试**。L1 不依赖 L3；L2 可用 mock CandidateFactorPackage 跑通；L3 可用 mock CertifiedFactorRecord 跑通。

---

## 2. 目录结构（最终态）

```
qlib-VNPY_Quant_Project/
├── factor_lab/                       # ★ L1 (新)，由 rdagent_integration/ + rdagent_overrides/ 重构
│   ├── __init__.py
│   ├── README.md                     # L1 用法 + 与 L2 的契约约定
│   ├── runners/
│   │   ├── rdagent_runner.py         # 现 scripts/run_fin_quant.py 搬进来 + 重命名
│   │   └── manual_runner.py          # 手工注入候选因子（不走 LLM）
│   ├── adapters/
│   │   ├── rdagent_overrides/        # 现 rdagent_overrides/ 整体搬进来
│   │   │   ├── factor_template/
│   │   │   ├── model_template/
│   │   │   └── README.md
│   │   ├── patch_qlib_conda.py       # 现文件搬进来
│   │   ├── project_proposal.py
│   │   ├── project_quant_proposal.py
│   │   └── project_experiments.py
│   ├── exporters/
│   │   ├── candidate_exporter.py     # 把 RD-Agent 跑完的产物打包成 CandidateFactorPackage
│   │   └── schema.py                 # CandidateFactorPackage 的 pydantic schema (契约 C1)
│   └── workspace/                    # gitignore；候选因子工作区
│       └── candidates/
│           └── <factor_id>/          # 每个候选一个目录
│               ├── factor.py         # 因子计算代码
│               ├── meta.json         # 元信息（hypothesis、author=LLM/manual、parent_loop）
│               ├── lab_metrics.json  # L1 内部指标（仅参考，L2 会重新算）
│               └── values.parquet    # 因子值 (datetime, instrument) → float
│
├── factor_validation/                # ★ L2 (新)
│   ├── __init__.py
│   ├── README.md
│   ├── profiles/                     # ★ 验证标准可配置
│   │   ├── default.yaml              # 默认验证标准
│   │   ├── strict.yaml               # 严格（用于 production 准入）
│   │   ├── exploratory.yaml          # 宽松（用于早期探索）
│   │   └── README.md                 # 各 profile 字段说明 + 如何新建 profile
│   ├── checks/                       # 单项检查（每个一个文件，可独立测试）
│   │   ├── __init__.py
│   │   ├── base.py                   # CheckBase 抽象类 + CheckResult dataclass
│   │   ├── coverage_check.py         # 数据覆盖率（非空率、时段覆盖、票数覆盖）
│   │   ├── ic_check.py               # OOS Rank IC、IC_IR
│   │   ├── stability_check.py        # IC 稳定性（分时段、分行业、分市值）
│   │   ├── orthogonality_check.py    # 与 L3 现有因子相关性 |ρ| ≤ 阈值
│   │   ├── turnover_check.py         # 因子日换手 / 排名自相关
│   │   ├── lookahead_check.py        # 前视检查（permutation test）
│   │   ├── backtest_check.py         # 真实回测：项目 RQAlpha 跑一次，要求 Sharpe / MDD / 换手达标
│   │   └── marginal_check.py         # 加入现有特征集后边际增益（A/B 训练）
│   ├── orchestrator.py               # 编排：按 profile 串起 checks，输出 CertifiedFactorRecord
│   ├── certifier.py                  # 决策：PASS / FAIL / HOLD + 出准入证书
│   ├── reports/                      # gitignore；HTML / Markdown 报告产出
│   └── schema.py                     # CertifiedFactorRecord pydantic schema (契约 C2)
│
├── factor_registry/                  # ★ L3 准入注册表 (新)
│   ├── __init__.py
│   ├── README.md
│   ├── registry.py                   # FactorRegistry：增 / 查 / 退役 / 列表
│   ├── store.py                      # 物料存储：把因子值落到 production parquet
│   ├── schema.py                     # ProductionFactorRecord schema
│   ├── data/                         # ★ 不 gitignore：注册表元信息（json）
│   │   ├── manifest.json             # 当前所有 active 因子清单
│   │   ├── certified/                # 每个因子一份证书副本（来自 L2）
│   │   │   └── <factor_id>.json
│   │   └── retired/                  # 已退役因子证书归档
│   └── parquet/                      # gitignore；production 因子值仓库
│       └── factors_v<N>.parquet      # 版本化；EnsembleModelManager 直接读
│
├── feature/                          # 现有；新增 production_factor_loader.py
│   └── production_factor_loader.py   # 唯一允许读 factor_registry/parquet 的接口
│
├── models/                           # 现有；EnsembleModelManager 改为只通过 production_factor_loader 拿因子
├── trainer/                          # 现有；不变
├── predictor/                        # 现有；不变
├── backtest/                         # 现有；不变
│
├── config/                           # 现有
│   ├── pipeline.yaml                 # 不再写死 "rdagent_exported"，改为引用 factor_registry/data/manifest.json
│   └── factor_lab.yaml               # ★ 新：L1 + L2 + L3 间的全局开关 / 默认 profile / 自动注入策略
│
├── scripts/                          # 现有；逐步把散乱脚本归位
│   ├── lab/                          # L1 入口的薄包装
│   │   └── run_lab_loop.py           # 调 factor_lab.runners.rdagent_runner
│   ├── validate/                     # L2 入口
│   │   ├── validate_candidates.py    # 批量验证 workspace/candidates/* 下所有候选
│   │   └── validate_one.py
│   ├── promote/                      # L2 → L3 准入
│   │   └── promote_certified.py      # 把 PASS 的候选自动注册到 registry
│   ├── retire/
│   │   └── retire_factor.py          # 退役（手工或周期任务触发）
│   └── e2e/
│       └── run_full_cycle.py         # L1 → L2 → L3 全自动一键跑
│
└── docs/
    ├── ARCHITECTURE_FACTOR_LAB.md    # ★ 本文件
    ├── CONTRACT_C1_CANDIDATE.md      # 契约 C1 详细 schema 文档
    ├── CONTRACT_C2_CERTIFIED.md      # 契约 C2 详细 schema 文档
    └── VALIDATION_PROFILE_GUIDE.md   # 如何写自定义 ValidationProfile
```

---

## 3. 契约定义（Contracts）

### 3.1 契约 C1：CandidateFactorPackage（L1 → L2）

**位置**：`factor_lab/exporters/schema.py`，pydantic v2 model。

```python
class CandidateFactorPackage(BaseModel):
    factor_id: str                      # 唯一 ID，格式 "<source>_<name>_<short_hash>"
    name: str                           # 人类可读名，例 "QualPersist_60D"
    source: Literal["rdagent", "manual", "external"]
    hypothesis: str                     # LLM 提的假设原文 / 人手填的设计动机
    formulation: str                    # 数学公式 / pseudocode
    code_path: Path                     # factor.py 绝对路径
    values_path: Path                   # values.parquet 绝对路径
    universe: str                       # 例 "csi300", "csi500", "all"
    date_range: tuple[date, date]       # 因子值覆盖的时间窗
    lab_metrics: dict[str, float]       # L1 自报指标（IC, IC_IR 等；仅参考）
    parent_loop: int | None             # RD-Agent loop index；manual 为 None
    created_at: datetime
    lab_run_id: str                     # 一次 lab run 的 UUID，便于审计

    model_config = ConfigDict(frozen=True)
```

**校验规则**：
- `code_path` 必须存在且可执行（subprocess 运行不报错）
- `values_path` 必须 parquet，含两级 MultiIndex `(datetime, instrument)`，单列 float64
- `factor_id` 不可与 L3 manifest 中已 active 的 ID 冲突（除非显式 `--overwrite`）

### 3.2 契约 C2：CertifiedFactorRecord（L2 → L3）

**位置**：`factor_validation/schema.py`。

```python
class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float | None
    threshold: float | None
    detail: dict
    elapsed_ms: int

class CertifiedFactorRecord(BaseModel):
    factor_id: str                      # 与 C1 相同
    candidate: CandidateFactorPackage   # 原始候选包（snapshot）
    profile_name: str                   # 用了哪个 ValidationProfile
    profile_hash: str                   # profile yaml 的 sha256，方便审计
    decision: Literal["PASS", "FAIL", "HOLD"]
    overall_score: float                # 综合分（profile 自己定义聚合方式）
    check_results: list[CheckResult]
    backtest_metrics: dict              # 真实 RQAlpha 跑出的核心指标
    validated_at: datetime
    validator_version: str              # factor_validation 包版本，便于回溯
    notes: str | None
```

**只有 `decision == "PASS"` 的 record 才能被 L3 接收。** `HOLD` 表示需要人工复核（可放进 `factor_registry/data/pending/`）。

---

## 4. 各层职责清单（What 与 Not-What）

### L1 Factor Lab

**做**：
- 接受 `factor_lab.yaml` 中的运行参数（loop_n、universe、time_window、reward formula 等）
- 启动 RD-Agent 循环 / 接收手工提交的因子代码
- 把每一轮跑出的因子打包成 `CandidateFactorPackage` 落到 `workspace/candidates/<factor_id>/`
- 记录 `lab_run_id` 便于审计
- 在 `lab_metrics.json` 写自报指标（仅参考）

**不做**：
- 不跑 production 回测
- 不判断因子"是否够格用"
- 不写 `factor_registry/`
- 不调用 `EnsembleModelManager` 等 L3 组件

### L2 Factor Validation

**做**：
- 读 `CandidateFactorPackage` + `ValidationProfile`
- 按 profile 执行所有 enabled checks
- 计算 overall_score、出 decision、生成 HTML/MD 报告
- 把 `CertifiedFactorRecord` 落到 `factor_validation/reports/<factor_id>/`

**不做**：
- 不修改候选因子代码
- 不直接写 `factor_registry/`（promote 是独立动作）
- 不重新跑 RD-Agent

### L3 Production

**做**：
- `factor_registry/` 维护 manifest，记录每个 active 因子的 `factor_id` + 证书路径 + 物料 parquet 列名
- `production_factor_loader` 是**唯一**允许其它 production 模块读因子值的入口
- `EnsembleModelManager.lgb_short_cycle` 通过 loader 取因子（不再硬编码 `"rdagent_exported"`）
- 任何 production 回测 / 训练 / 预测脚本，按 manifest 拿因子集

**不做**：
- 不接受未经 L2 PASS 的因子
- 不直接读 `factor_lab/workspace/`
- 不"实验性"启用某个因子（实验在 L1，认证在 L2）

---

## 5. ValidationProfile（"验证标准可配置"的具体实现）

### 5.1 Profile YAML 范例（`factor_validation/profiles/strict.yaml`）

```yaml
profile_name: strict
description: "Production 准入标准；所有 checks 都必须 pass"

universe: csi300
oos_window: ["2025-01-01", "2025-10-31"]   # 验证用的 OOS 时间窗
benchmark: SH000300

aggregation:
  method: weighted_sum
  pass_threshold: 0.6
  fail_threshold: 0.3                      # < 0.3 = FAIL，[0.3,0.6) = HOLD，>= 0.6 = PASS

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
    target_feature_set: lgb_short_cycle    # 与谁正交
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
    engine: rqalpha                         # 项目真实回测
    config_path: config/rqalpha_config.yaml
    min_sharpe: 1.5
    max_drawdown: 0.20
    max_annualized_turnover: 5.0
  marginal:
    enabled: true
    weight: 0.05
    baseline_features: lgb_short_cycle
    min_sharpe_uplift: 0.05                 # 加入新因子后 Sharpe 至少提升 0.05
```

### 5.2 Profile 切换

- 命令行：`python scripts/validate/validate_one.py --factor-id ... --profile strict`
- 程序内：`Orchestrator(profile="strict").run(candidate)`
- **每次验证都会把 profile 的 `sha256` 写进 `CertifiedFactorRecord.profile_hash`**，证书永远可复现

### 5.3 新增 check 的开发纪律

1. 在 `factor_validation/checks/` 新建文件，继承 `CheckBase`
2. 必须实现 `def run(self, candidate, context) -> CheckResult`
3. 必须可独立 unit test（不依赖 LLM、不依赖网络）
4. 在某个 profile yaml 里启用并指定权重
5. 在 `docs/VALIDATION_PROFILE_GUIDE.md` 补充字段说明

---

## 6. 与现有代码的对应关系

| 现有路径 / 文件 | 处理 | 目的层 |
|---|---|---|
| `rdagent_integration/` | **重构**到 `factor_lab/adapters/` | L1 |
| `rdagent_overrides/` | **搬迁**到 `factor_lab/adapters/rdagent_overrides/` | L1 |
| `scripts/run_fin_quant.py` | **搬迁**到 `factor_lab/runners/rdagent_runner.py` + 留薄壳在 `scripts/lab/run_lab_loop.py` | L1 |
| `scripts/refresh_rdagent_parquet.py` | **拆分**：因子筛选逻辑 → L2 的 orthogonality/ic checks；parquet 写入逻辑 → L3 `factor_registry/store.py` | L2 + L3 |
| `scripts/factor_diagnostic_p01.py` | **拆分**到 L2 的 `checks/ic_check.py`, `stability_check.py`, `orthogonality_check.py` | L2 |
| `scripts/factor_redundancy_report.py` | **搬迁**到 `factor_validation/checks/orthogonality_check.py` 的辅助报告 | L2 |
| `scripts/validate_combined_factors.py` | **搬迁**到 `factor_validation/checks/coverage_check.py` | L2 |
| `scripts/run_feature_ablation.py` | **搬迁**到 `factor_validation/checks/marginal_check.py` | L2 |
| `git_ignore_folder/combined_factors_df.parquet` | **改名 + 搬位置**到 `factor_registry/parquet/factors_v1.parquet`，由 `store.py` 管理 | L3 |
| `config/pipeline.yaml` 的 `model_features.lgb` | 改为 `["lgb_short_cycle", "factor_registry:active"]` 这种引用形式 | L3 |
| `models/ensemble_manager.py` | 改为通过 `feature/production_factor_loader.py` 拿因子 | L3 |
| `models/`, `trainer/`, `predictor/`, `backtest/` 其它内容 | **不动** | L3 |

**重构原则**：
- 任何"搬迁"必须保留旧位置的 deprecated wrapper（指向新位置 + WarningLog），过渡期 1 个版本
- 任何"拆分"必须保证拆完后单测覆盖现有调用路径

---

## 7. 开发推进阶段（按层落地）

### 阶段 A — 立架构骨架（无功能改动，仅目录 + 契约 + 文档）
- A.1 创建 `factor_lab/`, `factor_validation/`, `factor_registry/` 三个空包及其 `README.md`
- A.2 写 `CandidateFactorPackage` 和 `CertifiedFactorRecord` schema + 单元测试
- A.3 写 `docs/CONTRACT_C1_CANDIDATE.md`, `docs/CONTRACT_C2_CERTIFIED.md`, `docs/VALIDATION_PROFILE_GUIDE.md`
- A.4 写 `config/factor_lab.yaml`（全局开关）
- **验收**：`pytest` 通过，三个新包 `__init__.py` 可 import，schema 校验逻辑覆盖率 ≥ 90%

### 阶段 B — L3 先稳固（注册表 + production loader）
> 先做 L3，是因为 L2 的 backtest_check 和 marginal_check 需要 L3 已经能"无差别消费已认证因子"。

- B.1 实现 `factor_registry/registry.py` + `store.py`（注册 / 查询 / 退役 / 写 parquet）
- B.2 实现 `feature/production_factor_loader.py`（唯一读因子接口）
- B.3 把现有 `combined_factors_df.parquet` 用 `manual` 来源**手工补一份证书** + 注册到 registry（一次性迁移脚本 `scripts/promote/migrate_legacy_factors.py`）
- B.4 改 `models/ensemble_manager.py` 让它通过 loader 取因子；改 `config/pipeline.yaml` 的引用形式
- B.5 跑一次 production 训练 + RQAlpha 回测，确认结果与迁移前一致（容差 < 1e-6）
- **验收**：现有 v2 的 csi300 训练 / 回测在改造后**完全等价**复现

### 阶段 C — L2 验证环境
- C.1 实现 `CheckBase` + `coverage_check.py`, `ic_check.py`, `orthogonality_check.py`, `turnover_check.py`（**离线 checks 优先**）
- C.2 实现 `Orchestrator` + `Certifier`
- C.3 写 `profiles/default.yaml`, `strict.yaml`, `exploratory.yaml`
- C.4 实现 `scripts/validate/validate_one.py` + `validate_candidates.py`
- C.5 用现有 5 个 RD-Agent 因子做端到端验证，输出报告
- **验收**：5 个老因子全部走完 L2，证书 + 报告产出；至少 1 个被判 FAIL（验证 L2 真的在把关）

### 阶段 D — L2 backtest_check + marginal_check（联动 L3）
- D.1 `backtest_check.py`：调用项目 RQAlpha 跑因子（不动 L3 manifest，临时 parquet）
- D.2 `marginal_check.py`：在 baseline + new_factor vs baseline 上跑 A/B
- **验收**：`strict.yaml` 完整运行所有 checks，能识别 v2 那 5 个因子的"高换手陷阱"

### 阶段 E — L1 重构 + 自动 promote
- E.1 把 `rdagent_integration/` + `rdagent_overrides/` 重构到 `factor_lab/`
- E.2 实现 `factor_lab/exporters/candidate_exporter.py`：每个 RD-Agent loop 结束自动出 `CandidateFactorPackage`
- E.3 实现 `scripts/promote/promote_certified.py`：读 L2 PASS 证书 → 调 `FactorRegistry.register()`
- E.4 实现 `scripts/e2e/run_full_cycle.py`：L1 → L2 → L3 一键跑
- **验收**：`run_full_cycle.py --loop_n=2` 完整跑完，至少 1 个新因子被注册

### 阶段 F — 调度 + 退役
- F.1 周期调度（Windows Task / cron），按周自动 `run_full_cycle`
- F.2 滚动退役：每月评估 active 因子的近 N 月 marginal IC，低于阈值自动 `retire`
- F.3 监控告警：注册 / 退役事件推送到 `monitor/notifier/`

---

## 8. 编码纪律（PR 评审 checklist）

任何涉及因子 / 模型特征 / 回测的 PR，必须在描述中勾选：

- [ ] 本 PR 影响哪一层（L1 / L2 / L3）？是否跨层？跨层是否走契约？
- [ ] 是否新增了 `factor_id` 来源？若是，文档化在 `CONTRACT_C1_CANDIDATE.md`
- [ ] 是否改动了任何 ValidationProfile？若是，是否更新了 profile_hash 测试？
- [ ] 是否新增 check？若是，是否提供了 unit test + profile 字段文档？
- [ ] 是否绕开了 `production_factor_loader` 直接读 `factor_registry/parquet/`？（**禁止**）
- [ ] 是否在 L1 / L2 内调用了 L3 的 production 写入接口？（**禁止**）
- [ ] 是否更新了本文件（如有架构层级的偏离）？

**违反纪律的 PR 一律拒绝合并**。

---

## 9. 关键非功能要求

| 维度 | 要求 |
|---|---|
| 可审计 | 每个 production 因子可追溯到原始 `lab_run_id` + `profile_hash` + 当时的代码 commit |
| 可复现 | 给定 `factor_id`，能从证书里完整重跑 L2 验证（数据快照 / profile / 代码 commit 都齐） |
| 可回滚 | 任何 promote 必须可一键 retire；retire 后 production 模型可立即重训 |
| 可隔离 | L1 失败 / 资源耗尽 / LLM 余额耗尽 → L2 L3 完全不受影响 |
| 可观测 | L1 每个 loop、L2 每次验证、L3 每次 promote/retire 都写结构化日志（json line）到 `data/logs/factor_lab/`、`data/logs/factor_validation/`、`data/logs/factor_registry/` |

---

## 10. 与现有 P0-1 / P3a 工作的衔接

- **P0-1 历史教训** → 编入 `default.yaml` 的硬阈值（max_corr ≤ 0.7、OOS IC > 0、turnover 上限）
- **P3a 的 RAG prompt** → 搬到 `factor_lab/adapters/project_quant_proposal.py`（仅 L1 内部，作为 hypothesis 引导）
- **P3a 的 composite_score** → **不再作为准入依据**，仅作为 L1 内部的 LLM 反馈信号；L3 准入完全由 L2 的 ValidationProfile 决定
- **现有 5 个 RD-Agent 因子** → 阶段 B.3 一次性迁移 + 阶段 C.5 重新走 L2，能不能留在 L3 由 L2 决定

---

## 11. 后续相关文档（先列 stub，按阶段补全）

- `docs/CONTRACT_C1_CANDIDATE.md` — C1 schema 详解 + 示例 JSON
- `docs/CONTRACT_C2_CERTIFIED.md` — C2 schema 详解 + 示例 JSON
- `docs/VALIDATION_PROFILE_GUIDE.md` — 如何写 / 调试 / 评审 profile
- `docs/FACTOR_REGISTRY_OPERATIONS.md` — 注册 / 退役 / 紧急回滚操作手册
- `docs/RUNBOOK_LAB_VALIDATE_PROMOTE.md` — 日常运维 runbook（每周一次的全流程）

---

**END OF ARCHITECTURE v1.0**
