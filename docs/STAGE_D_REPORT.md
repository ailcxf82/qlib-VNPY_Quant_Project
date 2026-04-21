# 阶段 D 完工报告（真实回测 + 边际贡献 + 自动闭环）

**完成日期**：2026-04-20

**范围**：`factor_validation/checks/`（新增 4 check）+ 三套 profile 重构 +
`factor_lab/exporters/rdagent_log_exporter.py`（L1 RD-Agent 导出器）+
`scripts/lab/run_lab_cycle.py`（L1→L2→L3 自动闭环）+ `scripts/lab/run_lab_cycle.ps1`
（Windows Task Scheduler 包装器）。

配套阅读：

- `docs/STAGE_C_REPORT.md` —— 前序 L2 离线 4 check + Promote 流水线
- `docs/STAGE_D_TEST_GUIDE.md` —— 阶段 D 架构/接口级指南
- `docs/STAGE_D_TEST_PLAYBOOK.md` —— 阶段 D 测试 Playbook（复制即跑）

---

## 1. 交付清单

| 子任务 | 产物                                                                                                                            | 状态        |
| --- | ----------------------------------------------------------------------------------------------------------------------------- | --------- |
| D.0 | Legacy 因子退役（`VolRet_5D`、`VolumeTrend_10D`）；manifest 更新、parquet 物理列保留                                                          | COMPLETED |
| D.1α | `backtest_check.py` 轻量版（quantile long-short + Sharpe/MDD/win）+ 7 单测                                                           | COMPLETED |
| D.1β | `rqalpha_backtest_check.py` 真实回测（prediction CSV → RQAlpha runner → `report.json`）+ 8 单测                                       | COMPLETED |
| D.2A | `marginal_check.py` baseline-rank 中性化残差 IC 法 + 8 单测                                                                           | COMPLETED |
| D.2B | `marginal_training_check.py` baseline ± candidate 双 LGB A/B IC uplift + 9 单测                                                  | COMPLETED |
| D.3 | 三套 profile 重构（default/strict/exploratory）+ `scripts/validate/prepare_baseline_prediction.py` + 10 单测                          | COMPLETED |
| D.4 | `factor_lab/exporters/rdagent_log_exporter.py` + `scripts/lab/export_rdagent_candidates.py` + 22 单测（内容级 `factor_id` 幂等）       | COMPLETED |
| D.5 | `scripts/lab/run_lab_cycle.py` 自动闭环（export → exploratory → default → auto-promote）+ `run_lab_cycle.ps1` + 4 单测                | COMPLETED |
| D.6 | `STAGE_D_TEST_GUIDE.md` + `STAGE_D_TEST_PLAYBOOK.md` + `STAGE_D_REPORT.md` + cron 配置指南 + 全量回归                                   | COMPLETED |

### 1.1 代码/文件统计

新增文件：

- `factor_validation/checks/backtest_check.py`（D.1α，轻量 long-short）
- `factor_validation/checks/rqalpha_backtest_check.py`（D.1β，真实回测）
- `factor_validation/checks/marginal_check.py`（D.2A，中性化残差 IC）
- `factor_validation/checks/marginal_training_check.py`（D.2B，双 LGB A/B）
- `factor_lab/exporters/rdagent_log_exporter.py`（D.4，RD-Agent → C1 导出核心）
- `scripts/lab/__init__.py`、`scripts/lab/export_rdagent_candidates.py`（D.4，CLI）
- `scripts/lab/run_lab_cycle.py`、`scripts/lab/run_lab_cycle.ps1`（D.5，自动闭环）
- `scripts/validate/prepare_baseline_prediction.py`（D.3，baseline 预测抽取工具）
- 单测：`tests/factor_validation/test_{backtest,rqalpha_backtest,marginal,marginal_training}_check.py`、
  `tests/factor_lab/exporters/test_rdagent_log_exporter.py`、
  `tests/scripts/lab/test_run_lab_cycle.py`、
  `tests/scripts/validate/test_prepare_baseline_prediction.py`
- 文档：`docs/STAGE_D_TEST_GUIDE.md`、`docs/STAGE_D_TEST_PLAYBOOK.md`、本文件

修改文件：

- `factor_validation/profiles/default.yaml`：引入 `backtest` + `marginal`，权重再分配。
- `factor_validation/profiles/strict.yaml`：引入 `backtest_rqalpha` + `marginal` + `marginal_training`，
  关闭轻量 `backtest`；strict 的 PASS 必须有真实回测证据。
- `factor_validation/profiles/exploratory.yaml`：仅引入轻量 `backtest`，不启用 marginal 族
  （exploratory 不吃 baseline_prediction 依赖，秒级判定用）。
- `factor_validation/orchestrator.py`：`CheckContext` 扩展 `baseline_prediction_parquet`、
  `rqalpha_config_path` 字段；profile 里 check 列表由注册表动态派发。
- `factor_registry/data/manifest.json`：D.0 退役操作直接改；证书副本搬到 `data/retired/`。

**未修改**（零回归承诺）：

- L3 `ParquetStore` / `FactorRegistry` / `ProductionFactorLoader` 代码；
- 生产训练链 `run_train.py` / `feature/qlib_feature_pipeline.py` / `config/pipeline.yaml`；
- 阶段 C 的 4 个基础 check（coverage/ic/orthogonality/turnover）核心算法。

### 1.2 测试覆盖

```
$ pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote tests/scripts -q
275 passed, 1 warning in 5.85s
```

相对阶段 C 的 174 项基线，阶段 D 净增 **101 项**新单测（4 新 check ×
7~9 例 / check、D.3 baseline 工具 10 例、D.4 导出器 22 例、D.5 闭环 4 例），全部 PASS。

---

## 2. 核心技术：4 个新 L2 check

### 2.1 D.1α `backtest_check` ——轻量 long-short（秒级）

```
for each day d:
    rank candidate factor values → 取顶 quantile 做多、底 quantile 做空
    equal-weight 组合，计算 long_short_return_d
汇总：annualized_sharpe, max_drawdown, win_ratio, annualized_return
```

- **用途**：exploratory 阶段的"能不能赚钱"初筛，default 阶段做 weight=0.15 的辅助评分。
- **特性**：纯 pandas/numpy，无撮合无费用无涨跌停；只看理想组合的风险调整后收益。
- **阈值**：
  - `exploratory`：`min_long_short_sharpe=0.5`、`max_drawdown=0.60` —— 只排明显反向。
  - `default`：`min_long_short_sharpe=1.0`、`max_drawdown=0.40`。
- **关键实现**：`factor_validation/checks/backtest_check.py`，daily 聚合后调用通用
  Sharpe/DD 工具；空集、全 NaN、单只票分组都有显式失败路径。

### 2.2 D.1β `backtest_rqalpha_check` ——真实回测（5~15 min / 候选）

```
rank(candidate) → prediction CSV (date,code,score)
→ RQAlpha runner（lazy 注入，tests 用 fake runner）
→ report.json 里读 Sharpe / MaxDD / AnnualReturn / TotalReturn
→ 按 strict 阈值 binary PASS/FAIL + 记 detail["metrics"]
```

- **用途**：**strict profile 的回测真源**。生产准入必须跑这一档，吸收交易成本、涨跌停、
  T+1 约束，防止"无摩擦 Sharpe"假阳性。
- **阈值（strict）**：`min_sharpe=1.2`、`max_drawdown=0.30`、`min_annual_return=5%`。
- **工程细节**：
  - Runner 通过 lazy import 注入（`_load_rqalpha_runner`），测试期可替换为 `FakeRunner`
    完全离线；
  - `config/rqalpha_config.yaml` 指定账户、起止日、基准、费率；
  - 每次运行落盘到 `factor_validation/reports/rqalpha/<factor_id>_<ts>/`，
    `report.json` 被 check 解析后原样保留，便于事后复核。

### 2.3 D.2A `marginal_check` —— baseline-rank 中性化残差 IC（秒级）

**问题**：新因子如果与当前 ensemble 预测高度相关，入 L3 后对下游模型毫无增量 ——
L2 必须在秒级能识别这种"看似好、实则冗余"的候选。

**算法**（这是 D 阶段被多轮调优过的核心之一）：

```
rank_label   = rank(label)     # 截面 rank on target
rank_cand    = rank(candidate)
rank_base    = rank(baseline_prediction)

# OLS: rank_cand ~ a + b * rank_base ，求残差 cand_resid
cand_resid = rank_cand − (a + b * rank_base)

residual_rank_ic_d = spearman(cand_resid_d, label_d)   # 每天一个
residual_rank_ic   = mean over days
residual_ic_ir     = residual_rank_ic / std(residual_rank_ic_d)
```

> 早期实现曾直接用 `label_rank − baseline_rank`，对 rank 边界非常敏感、
> 对真正冗余的 redundant 因子给出负 IC；改为"baseline-rank 中性化 + label rank IC"
> 后 `test_redundant_signal_fails` 稳定接近 0，算法统计上正确。

- **阈值（default）**：`min_residual_rank_ic=0.008`、`min_residual_ic_ir=0.15`。
- **阈值（strict）**：`0.012` / `0.25`。
- **数据依赖**：`scripts/validate/prepare_baseline_prediction.py` 产出的
  `baseline_predictions_<tag>.parquet`，MultiIndex=(datetime, instrument)、单列 `prediction`。

### 2.4 D.2B `marginal_training_check` —— 双 LGB A/B（10~30 秒）

**问题**：中性化残差 IC 只能量化线性冗余；非线性 / 交互型贡献得靠下游模型亲自跑一遍。

**算法**：

```
features_A = [baseline_feat_i]                 # 基线特征集（通常等价 baseline_prediction 自身）
features_B = features_A + [candidate_feat]     # A 之上 + 候选

在 OOS 窗 train/test 切分 → 训 LightGBM_A、LightGBM_B（相同超参、相同随机种）
pred_A = LGB_A.predict(OOS), pred_B = LGB_B.predict(OOS)
rank_ic_A, rank_ic_B = spearman(pred_·, label) 日频均值

ic_uplift = rank_ic_B − rank_ic_A              # 候选带来的纯增益
```

- **阈值（strict）**：`min_ic_uplift=0.003`（≈ +0.3pp rank IC）。
- **超参**：`num_boost_round=200` / `learning_rate=0.05` / `num_leaves=31` /
  `min_data_in_leaf=50`，使训练时间可预期。
- **防抖**：固定 `random_seed=42`；train/test 按时间切（默认 60/40）；
  少于 `min_test_days=30` 直接 FAIL。

### 2.5 Profile 权重重分配

三套 profile 重构为"用途优先"：

| 用途            | profile        | 4 基础 check 权重           | 回测族                   | 边际贡献族                            | pass_threshold |
| ------------- | -------------- | ------------------------ | --------------------- | -------------------------------- | -------------- |
| RD-Agent 预筛   | `exploratory`  | cov 0.10 / ic 0.45 / orth 0.10 / to 0.20 | `backtest` 0.15       | 不启用（不吃 baseline pred 依赖）           | 0.40           |
| 提交默认标准        | `default`      | cov 0.10 / ic 0.30 / orth 0.15 / to 0.15 | `backtest` 0.15       | `marginal` 0.15                   | 0.60           |
| Production 准入 | `strict`       | cov 0.10 / ic 0.20 / orth 0.15 / to 0.10 | `backtest_rqalpha` 0.20 | `marginal` 0.10 + `marginal_training` 0.15 | 0.70           |

---

## 3. 核心技术：RD-Agent 自动闭环

### 3.1 D.4 `rdagent_log_exporter` —— L1 唯一源出口

**为什么是反射而不是 import**：直接 `import rdagent.xxx` 会把 rdagent 的重依赖链
（openai、litellm、...）拉进 factor_lab。我们选择把 pickle 当"不透明容器"，
只读 `__dict__` 里的字段名；这样 factor_lab 对 rdagent 包**零 import 依赖**。

**一次 run 的产物**：

```
RD-Agent log/<run-ts>/Loop_N/
  ├─ direct_exp_gen/hypothesis generation/<最新 .pkl>  → Hypothesis
  ├─ direct_exp_gen/experiment generation/<最新 .pkl>  → list[FactorTask]
  ├─ coding/evo_loop_K/evolving code/<最新 .pkl>       → list[FactorFBWorkspace]
  └─ feedback/feedback/<最新 .pkl>                     → HypothesisFeedback

git_ignore_folder/RD-Agent_workspace/<workspace_hash>/
  ├─ factor.py      # RD-Agent 生成的因子代码
  └─ result.h5      # 因子值输出（(date,instrument) → float）
```

→ 导出器把它们 match 起来，每个 (task, workspace) 对组装成一个
`CandidateFactorPackage`，写到 `factor_lab/workspace/candidates/<factor_id>/`。

**`factor_id` 设计**：

```
factor_id = f"rdagent_{sanitized_name}_{workspace_hash[:8]}"
```

- `sanitized_name` = 因子代码规范化名字（字母开头、字母数字下划线、≤64）。
- `workspace_hash[:8]` = RD-Agent 的内容哈希，等价于 `(code, data)` 对。
- **含义**：同一份代码 + 同一份数据 → 同一个 `factor_id` → staging 幂等可跳过。

### 3.2 D.5 `run_lab_cycle` —— 自动闭环

```
┌───────────────────────────────┐
│ export_rdagent_log_tree       │  扫 log/ 下全部 run，按 run_filter 过滤
│ (L1 Loop_N → C1 candidate)    │
└──────────────┬────────────────┘
               │ factor_lab/workspace/candidates/<fid>/c1.json
               ▼
┌───────────────────────────────┐
│ validate_candidate            │  exploratory.yaml → 秒级预筛
│ (L2 exploratory stage)        │
└──────────────┬────────────────┘
               │ PASS only ↓
               ▼
┌───────────────────────────────┐
│ validate_candidate            │  default.yaml → 精验
│ (L2 default stage)            │  （注：strict 留给人工决策，不自动入 L3）
└──────────────┬────────────────┘
               │ PASS only ↓
               ▼
┌───────────────────────────────┐
│ promote (C2 → L3 manifest)    │  allow_promote_overwrite 控制是否覆盖同名因子
└──────────────┬────────────────┘
               ▼
   factor_validation/reports/lab_cycle_<cycle_id>.{json,md}
   factor_validation/certificates/<cycle_id>/<fid>.{exploratory,default}.json
```

**设计边界**：

- 默认 cycle **不碰 strict**：strict 必须跑 RQAlpha 真实回测，耗时不适合周级自动跑；
  strict 定位是"准备上仓位了再人工触发"。
- 闭环**不自动退役**旧因子：保留决策权；用户看 cycle 报告，必要时手动调
  `scripts.promote.retire_factor`。
- `allow_promote_overwrite=False` 为默认：同名因子已存在会 FAIL（而不是静默覆盖），
  避免 id collision 悄悄污染 registry。

### 3.3 Windows Task Scheduler 对接

详见 `docs/STAGE_D_TEST_GUIDE.md` §5 的完整配置表单。简要：

```
Trigger: 每周六 02:30
Action:  powershell -ExecutionPolicy Bypass -File
         D:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\scripts\lab\run_lab_cycle.ps1
```

日志：`factor_validation/reports/scheduler-logs/run_lab_cycle-<ts>.log`。

---

## 4. 回归保障

### 4.1 单测全绿

```
tests/factor_lab/           55 passed        （含 D.4 exporter 22 + 原 schema/C1 pkg）
tests/factor_validation/    97 passed        （4 老 + 4 新 check + orchestrator + profile schema）
tests/factor_registry/      \
tests/feature/               } 109 passed    （L3 存储 + pipeline 桥接 + promote CLI，阶段 C 延续）
tests/promote/              /
tests/scripts/              14 passed        （D.3 prepare_baseline 10 + D.5 run_lab_cycle 4）
─────────────────────────────────────────
总计                         275 passed in 5.85s
```

### 4.2 对生产训练的非回归承诺

- `feature/qlib_feature_pipeline.py`、`config/pipeline.yaml`、`run_train.py` 全未动。
- 阶段 D 新增的 L2 check 只在"是否能进 L3"这个开关点起作用，**不改已 active 因子的值**。
- 训练端的基线 OOS 预测由 `prepare_baseline_prediction.py` 离线抽取，**不触发**训练重跑。

### 4.3 可复现命令

```powershell
conda activate qlib_zhengshi

# 阶段 D 全量回归（5~10 s）
python -m pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote tests/scripts -q

# 手动跑一次自动闭环（扫整个 log 根，不强制覆盖，不覆盖已入库 factor）
python -m scripts.lab.run_lab_cycle --log-level INFO

# 限定一次 run_filter 跑（典型周循环用法）
python -m scripts.lab.run_lab_cycle --run-filter 2026-04-20 --cycle-id labcycle-20260420-weekly
```

---

## 5. 下一步（阶段 E/F 预告）

1. **阶段 E —— RD-Agent 反馈回路**：把 L2 `CertifiedFactorRecord.check_results` 回写
   `factor_lab/workspace/feedback/`，让 RD-Agent 下一轮 hypothesis generation 能读到
   "上一轮为什么没进 L3"的量化信号。
2. **阶段 F —— dashboard & alerting**：把 `factor_registry/data/manifest.json` +
   `factor_validation/reports/lab_cycle_*.md` + RQAlpha report.json 暴露到轻量 Web
   面板；cycle FAIL 超阈值发 alert。

---

## 6. 附录

### 附录 A：涉及的 profile 阈值对照（D 版）

| 字段                                  | strict     | default | exploratory | 说明                               |
| ----------------------------------- | ---------- | ------- | ----------- | -------------------------------- |
| `coverage.min_non_null_ratio`       | 0.90       | 0.85    | 0.70        | 覆盖率下限                            |
| `ic.min_rank_ic`                    | 0.015      | 0.012   | 0.005       | OOS rank IC 均值下限                 |
| `ic.min_ic_ir`                      | 0.30       | 0.25    | 0.10        | IC 信息率下限                         |
| `orthogonality.max_abs_corr`        | 0.50       | 0.60    | 0.80        | 与 reference 集最大 \|ρ\|            |
| `turnover.max_daily_rank_turnover`  | 0.30       | 0.35    | 0.55        | (1-Spearman)/2 日均                |
| `turnover.min_rank_autocorr`        | 0.60       | 0.50    | 0.10        | Lag1 rank 自相关中位数下限              |
| `backtest.min_long_short_sharpe`    | —          | 1.0     | 0.5         | 轻量 long-short Sharpe（strict 不跑） |
| `backtest.max_drawdown`             | —          | 0.40    | 0.60        | 轻量版 MDD                         |
| `backtest_rqalpha.min_sharpe`       | **1.2**    | —       | —           | 真实回测 Sharpe                      |
| `backtest_rqalpha.max_drawdown`     | **0.30**   | —       | —           | 真实回测 MDD                        |
| `backtest_rqalpha.min_annual_return`| **0.05**   | —       | —           | 年化下限                            |
| `marginal.min_residual_rank_ic`     | 0.012      | 0.008   | —           | 中性化残差 IC 均值                    |
| `marginal.min_residual_ic_ir`       | 0.25       | 0.15    | —           | 残差 IC 信息率                       |
| `marginal_training.min_ic_uplift`   | **0.003**  | —       | —           | B-A 双 LGB rank IC 差             |
| `aggregation.pass_threshold`        | 0.70       | 0.60    | 0.40        | 综合分 PASS 门槛                     |
| `aggregation.fail_threshold`        | 0.40       | 0.30    | 0.20        | 综合分 FAIL 门槛                     |

### 附录 B：cycle 产物文件路径速查

| 文件 / 目录                                                                            | 内容                     | 生命周期     |
| ---------------------------------------------------------------------------------- | ---------------------- | -------- |
| `factor_lab/workspace/candidates/<factor_id>/{factor.py,values.parquet,c1.json}`   | 每个候选因子的 L1 产物           | 内容级幂等    |
| `factor_validation/certificates/<cycle_id>/<factor_id>.exploratory.json`           | 预筛证书（PASS 才进 default）  | 每 cycle 一份 |
| `factor_validation/certificates/<cycle_id>/<factor_id>.default.json`               | 精验证书（PASS 才 auto-promote）| 同上       |
| `factor_validation/reports/lab_cycle_<cycle_id>.{json,md}`                         | Cycle 汇总报告              | 永久留存     |
| `factor_validation/reports/scheduler-logs/run_lab_cycle-<ts>.log`                  | PS1 包装器 stdout/stderr   | 看保留策略    |
| `factor_validation/reports/rqalpha/<factor_id>_<ts>/report.json`                   | RQAlpha 原始回测输出          | 永久留存     |
| `factor_registry/data/manifest.json`                                               | L3 注册表                 | 每次 promote 更新 |
