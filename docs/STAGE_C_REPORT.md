# 阶段 C 完工报告（Factor Lab 三层架构 · L2 验证与 Promote 流水线）

**完成日期**：2026-04-20

**范围**：`factor_validation/` (L2) + `scripts/promote/` (CLI) + 三套 profile 阈值升级

- legacy 因子 oracle 复测。

---

## 1. 交付清单


| 子任务 | 产物                                                                                   | 状态        |
| --- | ------------------------------------------------------------------------------------ | --------- |
| C.0 | 探查既有 4 checks / 3 profiles / prepare_oos_labels 脚本                                   | COMPLETED |
| C.1 | 生成 OOS label parquet（`oos_labels_all.parquet`, 1.75M 行 / 5433 只 / 2024-12 → 2026-04） | COMPLETED |
| C.3 | `scripts/promote/validate_candidate.py` + 6 单测                                       | COMPLETED |
| C.4 | `scripts/promote/promote_certified.py` + 7 单测                                        | COMPLETED |
| C.5 | oracle 复测脚本 + 第一版报告（`legacy_oracle_default.md`，揭示旧 turnover bug）                     | COMPLETED |
| C.6 | `turnover_check` 算法升级（截面 Spearman 版），三套 profile 阈值同步                                 | COMPLETED |
| C.7 | `scripts/promote/retire_factor.py` + 6 单测                                            | COMPLETED |
| C.8 | 修复后重跑 oracle，生成 `legacy_oracle_default_v2.md`（本报告附录 A）                               | COMPLETED |
| 文档  | `docs/STAGE_C_TEST_GUIDE.md`（本报告姊妹篇） + 本报告                                           | COMPLETED |


### 1.1 代码/文件统计

新增：

- `scripts/promote/validate_candidate.py` (258 行)
- `scripts/promote/promote_certified.py` (217 行)
- `scripts/promote/retire_factor.py` (182 行)
- `tests/promote/test_validate_candidate_cli.py` (209 行)
- `tests/promote/test_promote_certified_cli.py` (267 行)
- `tests/promote/test_retire_factor_cli.py` (236 行)
- `docs/STAGE_C_TEST_GUIDE.md` + `docs/STAGE_C_REPORT.md`
- `factor_validation/reports/legacy_oracle_default_v2.md`

修改：

- `factor_validation/checks/turnover_check.py`：`daily_rank_turnover` 算法从"rank 变化
占比"替换为 `(1 - 截面 Spearman) / 2`；`test_turnover_check.py` 同步升级（新增反转型
测例、修正随机型断言）。
- `factor_validation/profiles/{default,strict,exploratory}.yaml`：turnover 阈值更新，
配套中文注释同步新语义。

未修改：

- 任何 L1 (`factor_lab/`) 代码。
- 任何生产训练路径（`run_train.py` / `feature/qlib_feature_pipeline.py` 已在 B.4 阶段
接入 L3 loader，本阶段无变更）。

### 1.2 测试覆盖

```
$ pytest tests/factor_validation tests/factor_registry tests/feature tests/promote -q
174 passed in 3.21s
```

新增 19 个 CLI 单测（6 + 7 + 6）+ 3 个 turnover 测例升级（新增反转型、修正随机型
断言、补充慢因子阈值），零真实 qlib 依赖、零外部数据依赖，可在干净 CI 跑通。

---

## 2. 关键技术修复：`turnover_check` 算法升级（C.6）

### 2.1 旧算法缺陷

```
# 旧定义
diff = (ranks_today - ranks_prev).abs()
changed = (diff > 0).sum(axis=1)
daily_rank_turnover = mean(changed / valid_count)
```

- `rank(method='average')` 对稠密连续因子：任一因子值扰动会在其所在同值区间产生
±0.5 rank 漂移，导致"发生变化"的股票占比几乎恒 ≈ 1.0。
- 阶段 C oracle 复测实证：5 个 legacy 因子的 `daily_rank_turnover` 均 ≥ 0.97，与它们
实际慢/快语义无关 —— **指标失效**。
- 连锁后果：C.5 第一次 oracle 下 5 个 legacy 因子全部因 turnover FAIL；如果按老算法
调阈值到 `max ≥ 1.0`，又等于取消本项 check。

### 2.2 新算法（v0.2 语义）

```
turnover_day_d = (1 − Spearman(rank_today, rank_prev)) / 2
daily_rank_turnover = mean_over_valid_days(turnover_day_d)     # ∈ [0, 1]
```

在 `CheckResult.detail.algorithm` 里硬编码了 `"(1 - cross_sectional_spearman) / 2, mean over days"`，便于日后 schema 迁移识别。

### 2.3 实证区分度（重跑 oracle 后）


| 因子                   | 旧 turnover (bugged) | 新 turnover | rank_autocorr | 判定                |
| -------------------- | ------------------- | ---------- | ------------- | ----------------- |
| VolumePriceTrend_10D | ≈ 0.99              | **0.081**  | 0.831         | OK（慢）             |
| VolumeTrend_10D      | ≈ 0.99              | **0.076**  | 0.846         | OK（慢）             |
| VolRet_5D            | ≈ 0.98              | **0.064**  | 0.795         | OK（慢）             |
| VolRatio_20D         | ≈ 0.99              | **0.221**  | 0.547         | OK（偏快）            |
| RangeRatio_10D       | ≈ 0.99              | **0.398**  | 0.190         | FAIL（快；超 0.35 阈值） |


新算法成功把 5 个因子按实际换手率分成两档，第一次给下游"要不要保留 RangeRatio"提供
了数据支撑。

---

## 3. Legacy 因子 Oracle 复测结论（C.8）

完整报告：`[factor_validation/reports/legacy_oracle_default_v2.md](../factor_validation/reports/legacy_oracle_default_v2.md)`

### 3.1 决议汇总


| factor_id                                | name                     | decision | overall | coverage | ic        | orthogonality | turnover  |
| ---------------------------------------- | ------------------------ | -------- | ------- | -------- | --------- | ------------- | --------- |
| `manual_VolumePriceTrend_10D_9f0859e4dc` | **VolumePriceTrend_10D** | **PASS** | 0.9341  | 0.990    | 1.000     | 0.782         | 0.917     |
| `manual_VolumeTrend_10D_5142b584ee`      | VolumeTrend_10D          | FAIL     | 0.8475  | 0.993    | 1.000     | **0.338**     | 0.924     |
| `manual_VolRatio_20D_7439285b70`         | VolRatio_20D             | FAIL     | 0.8090  | 0.982    | 1.000     | **0.338**     | 0.776     |
| `manual_RangeRatio_10D_bb147b539a`       | RangeRatio_10D           | FAIL     | 0.7742  | 0.998    | 1.000     | **0.374**     | **0.598** |
| `manual_VolRet_5D_11d70180fb`            | VolRet_5D                | FAIL     | 0.6392  | 0.997    | **0.283** | 0.738         | 0.917     |


### 3.2 FAIL 原因拆解

- **3 个因子因 orthogonality FAIL**（`VolumeTrend_10D ↔ VolRatio_20D`: 0.662；
`VolRatio_20D ↔ RangeRatio_10D`: 0.626；`VolumeTrend_10D ↔ RangeRatio_10D`: 0.406）：
这不是"质量差"，而是**成交量/波动族因子彼此线性相关度高**。需要"选代表"而非"集体退役"。
- `**RangeRatio_10D` 同时 turnover FAIL**（`daily_rank_turnover=0.398 > 0.35`）：因子
本身换手偏高，但 rank_ic=0.093 / ic_ir=0.693 非常出色 —— 高换手高 IR 型。
- `**VolRet_5D` 因 IC FAIL**（rank_ic=0.006, ic_ir=0.024）：**真·低质**，基本等于噪声，
保留在 L3 只会拖拽下游模型。

### 3.3 三档行动建议（需用户决策）


| 方案    | 动作                                                       | 保留因子数 | 预期影响                                |
| ----- | -------------------------------------------------------- | ----- | ----------------------------------- |
| A（保守） | 仅退役 `VolRet_5D`（确证低质）                                    | 4     | 保留所有强 IC 因子；accept orthogonality 冗余 |
| B（推荐） | 退役 `VolRet_5D` + `VolumeTrend_10D` / `RangeRatio_10D` 之一 | 3     | 打破最强相关对，下游 orthogonality 降至 ≤ 0.5   |
| C（激进） | 只保留 `VolumePriceTrend_10D`（唯一 PASS）+ `VolRatio_20D`      | 2     | 最小依赖集；为阶段 D 新因子留空间                  |


> 本阶段**不自动执行退役**。具体选择请用户确认后，通过 `scripts.promote.retire_factor`
> 批量操作（命令示例见 `docs/STAGE_C_TEST_GUIDE.md` §3.3）。

---

## 4. 回归保障

### 4.1 单测全绿（C.8 完成后）

```
tests/factor_validation/  ─  6 turnover + 其它 4 checks + orchestrator + schema  ─── 全绿
tests/factor_registry/    ─  ParquetStore + FactorRegistry + loader                ─── 全绿
tests/feature/            ─  pipeline_registry_integration（L3 vs legacy 等价）      ─── 全绿
tests/promote/            ─  migrate (+6) + validate (+6) + promote (+7) + retire (+6) ─── 全绿
─────────────────────────────────────────────────────────────────────────────────────
总计 174 项，3.21s
```

### 4.2 对生产训练的非回归承诺

- `feature/qlib_feature_pipeline.py` 的 L3 loader 桥接在 B.5-α 阶段已做过 1.15M 行真
数据 bit-for-bit 等价验证，本阶段没有触碰该文件。
- `config/pipeline.yaml` 与 `run_train.py` 未修改；`csi300_models` 基线模型无影响。
- Turnover 算法升级只影响 L2 验证评分（用于决策"哪个因子能进 L3"），**不改变任何
已 active 因子的值**。

### 4.3 可复现命令

```bash
# 全量 C 阶段回归
python -m pytest tests/factor_validation tests/factor_registry tests/feature tests/promote -q

# 重跑 oracle（当前 5 个 active legacy 因子 against default profile）
python -m scripts.promote.oracle_revalidate_legacy \
    --profile factor_validation/profiles/default.yaml \
    --write-report factor_validation/reports/legacy_oracle_default_v3.md
```

---

## 5. 下一步（阶段 D 预告）

1. **候选退役决策**（本阶段交付的 oracle 支持）：用户在 §3.3 A/B/C 方案中选一个，
  通过 `retire_factor` 批量执行。
2. **阶段 D —— RQAlpha backtest check 接入 L2**：
  - 新增 `factor_validation/checks/backtest_check.py`（IC → 选股 → 组合 → Sharpe/MDD）
  - 扩展 `strict.yaml` 加入 backtest 权重；同时补 `marginal_check`（联合新旧因子跑
  ensemble 模型的边际贡献）
  - 触发 rdagent 第一次"从 L1 自动投稿新因子 → L2 full-stack 验证 → L3 自动 promote"
  的闭环。
3. **阶段 E —— rdagent 回路自动化**：`factor_lab/loops/` 周期性跑 exploratory profile
  预筛 → default profile 精验 → auto-promote；失败因子自动归档。
4. **阶段 F —— dashboard & alerting**：给 `factor_registry/data/manifest.json` +
  oracle 报告接一个轻量 Web 面板。

---

## 6. 附录

### 附录 A：Oracle v2 报告速览

（完整文件：`factor_validation/reports/legacy_oracle_default_v2.md`）

```
profile = default（coverage 0.85 / rank_ic 0.012 / ic_ir 0.25 / orthogonality 0.60 / turnover 0.35）
5 个 legacy 因子 → 1 PASS + 4 FAIL
```

### 附录 B：涉及的 profile 阈值对照（C.6 版）


| 字段                                 | strict   | default  | exploratory | 说明                 |
| ---------------------------------- | -------- | -------- | ----------- | ------------------ |
| `coverage.min_non_null_ratio`      | 0.90     | 0.85     | 0.70        | 覆盖率下限              |
| `ic.min_rank_ic`                   | 0.015    | 0.012    | 0.005       | OOS rank IC 均值下限   |
| `ic.min_ic_ir`                     | 0.30     | 0.25     | 0.10        | IC 信息率下限           |
| `ic.allow_negative`                | false    | true     | true        | 是否接受负向因子           |
| `orthogonality.max_abs_corr`       | 0.50     | 0.60     | 0.80        | 与参照因子集最大           |
| `turnover.max_daily_rank_turnover` | **0.30** | **0.35** | **0.55**    | **C.6 升级后阈值**      |
| `turnover.min_rank_autocorr`       | 0.60     | 0.50     | 0.10        | Lag1 rank 自相关中位数下限 |
| `aggregation.pass_threshold`       | 0.70     | 0.60     | 0.40        | 综合分 PASS 阈值        |
| `aggregation.fail_threshold`       | 0.40     | 0.30     | 0.20        | 综合分 FAIL 阈值        |


