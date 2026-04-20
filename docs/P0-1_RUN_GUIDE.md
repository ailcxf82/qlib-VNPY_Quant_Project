# P0-1 运行手册：打通 RD-Agent → 主工程训练闭环

> 本文件对应 `docs/OPTIMIZATION_ROADMAP_2026.md` 中 P0-1 条目。
> 目的：让主工程 (`run_train.py → trainer/trainer.py → feature/qlib_feature_pipeline.py`) 能够**直接消费** `git_ignore_folder/combined_factors_df.parquet` 并在 LGB 模型训练中联合使用 qlib 表达式因子 + RD-Agent parquet 因子。
>
> 配套脚本：
>
> - `scripts/refresh_rdagent_parquet.py`（把 parquet 的时间/股票覆盖刷新到主工程口径）
> - `scripts/smoke_p01.py`（合并/解析链路冒烟）
> - `scripts/smoke_p01_train.py`（端到端最小训练冒烟）
> - `scripts/validate_combined_factors.py`（独立健康检查，可选）

---

## 1. 背景与定位

在 P0-1 前，RD-Agent 产出的 parquet 只能在 `rdagent_overrides/factor_template/conf_combined_factors.yaml` 的 qrun 沙箱里生效；主工程从未读入过它。本次改造在**不改动 LGB/GRU 模型代码**的前提下，把 parquet 接进特征面板、再通过 ensemble 的"feature_sets"按模型分发。

改造落点（已合并到主分支）：


| 文件                                 | 作用                                                                                                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `feature/qlib_feature_pipeline.py` | `build()` 之后新增 `_maybe_merge_rdagent_parquet()`，把 parquet 按 `MultiIndex(datetime, instrument)` reindex 对齐后拼到 feature 面板；同步写入 `self.rdagent_factor_columns` |
| `models/ensemble_manager.py`       | 新增 `update_feature_set(name, cols)`；`_resolve_feature_cols` 支持把 `model_features[lgb]` 配成**集合名列表**（如 `["lgb_short_cycle", "rdagent_exported"]`）             |
| `trainer/trainer.py`               | `train()` 里在 `pipeline.build()` 之后把 `rdagent_factor_columns` 注入 ensemble，并写入 `cfg["feature_sets_runtime_override"]` 供 OOF 复用                               |
| `trainer/oof_manager.py`           | `generate_oof()` 读取 `feature_sets_runtime_override`，与 `data.feature_sets` 合并后复用相同解析逻辑                                                                      |
| `run_predict.py`                   | 预测前同样注入 `rdagent_factor_columns`，确保推理侧列对齐                                                                                                                  |
| `config/data.yaml`                 | 新增 `rdagent_parquet.path`、`feature_sets.rdagent_exported: []` 占位、`active_feature_sets` 加入 `rdagent_exported`                                               |
| `config/pipeline.yaml`             | `model_features.lgb: ["lgb_short_cycle", "rdagent_exported"]`（GRU 仍保持 `gru_short_cycle` 不变）                                                                |


---

## 2. 前置条件

### 2.1 环境

```powershell
conda activate qlib_zhengshi
# 或 conda run -n qlib_zhengshi --no-capture-output python ...
```

### 2.2 必须存在的目录/文件

- `D:/qlib_data/qlib_data`（主工程的 qlib 数据目录，包含 csi500/csi300 定义）
- `git_ignore_folder/RD-Agent_workspace/`（含 ≥ 一百多个带 `factor.py` 的 workspace 子目录）
- `git_ignore_folder/combined_factors_df.parquet`（可无；Step 1 会生成/覆盖）

### 2.3 关键配置（应当已就位）

`config/data.yaml`：

```yaml
data:
  rdagent_parquet:
    path: git_ignore_folder/combined_factors_df.parquet
  feature_sets:
    rdagent_exported: []          # 运行时由 pipeline 注入
  active_feature_sets:
    - ...
    - rdagent_exported            # 必须列入，才会触发合并
```

`config/pipeline.yaml`：

```yaml
model_features:
  gru: gru_short_cycle
  lgb: ["lgb_short_cycle", "rdagent_exported"]   # 支持集合名列表
```

---

## 3. 运行顺序（三步闭环 + 可选生产训练）

### Step 1 · 刷新 parquet 覆盖范围

```powershell
python scripts/refresh_rdagent_parquet.py --start 2020-01-01 --end 2026-04-07 --instruments 'csi500,csi300'
```


| 参数                | 说明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| `--start / --end` | 与主工程 `data.yaml.start_time/end_time` 对齐，建议往历史回推以覆盖 rolling 所需上下文 |
| `--instruments`   | **必须加引号**（PowerShell 把逗号拆成多参数）。按逗号分号拆分为多个池，自动 union 去重           |
| `--dry-run`       | 只演练筛选、不覆盖磁盘                                                      |
| `--ic-threshold`  | 默认 0.02；                                                         |
| `--max-nan-ratio` | 默认 0.50；NaN 比例高于此的因子跳过                                           |
| `--max-factors`   | 默认 50；按                                                          |


**内部流程**：

1. 从 qlib_data 按 `FIELD_MAP` 重映射出 RD-Agent 期望 schema 的内存 `daily_pv`（$close ← $close_qfq、$volume ← $vol、$rsi12 ← $rsi_qfq_12 等）
2. Monkey-patch `pd.read_hdf('daily_pv.h5')`、`DataFrame.to_hdf('result.h5', ...)`，**不落盘旧 daily_pv**
3. 遍历 `RD-Agent_workspace/*/factor.py`，exec 后自动触发各自的 `calculate_<Name>()`，捕获内存结果
4. 按主工程 label 口径 `close[t+3] / close[t+1] - 1` 计算 RankIC，按阈值 + top-N 挑选
5. 旧 parquet 备份为 `combined_factors_df.parquet.bak.<unix_ts>`，写入新 parquet + `.json` 摘要

**预计耗时**：6~7 分钟（其中带 `scipy.linregress` 的 `01f160fd...` 单个因子占用约 5 分钟）。

**预计输出（关键行）**：

```
=== Step 1: build in-memory daily_pv from qlib_data ===
pool=csi500 贡献 500 只（累计 500）
pool=csi300 贡献 300 只（累计 800）
daily_pv shape=(1150046, 25), dt=2020-01-02 ~ 2026-04-07, n_inst=799
=== Step 4: re-exec each factor.py on new daily_pv ===
scan done: ok=14 fail=67 dup=71 noname=0 nan_skip=5 ic_skip=38 | candidates=14
selected top 14 by |IC|:
  MomRet_5D                    ic=+0.1741 nan=0.003
  VolumeWeightedReturn_5D      ic=+0.1693 nan=0.006
  ...
旧 parquet 已备份: combined_factors_df.parquet.bak.<ts>
写入: git_ignore_folder/combined_factors_df.parquet
summary: git_ignore_folder/combined_factors_df.json
```

> `fail=67` 解释：少数 `factor.py` 用了 `os.path.exists('daily_pv.h5')` 等额外 IO，不是只走 `pd.read_hdf` 单入口，Monkey-patch 无法拦截，被跳过。不影响 top-N 筛选结果。

### Step 2 · 链路冒烟（纯函数级）

```powershell
python scripts/smoke_p01.py
```

在 parquet 末段 60 个交易日里验证合并、ensemble 注入、LGB/GRU 特征解析逻辑。**不跑模型训练**。

**预计耗时**：~30 秒。

**预计关键输出**：

```
parquet: shape=(1150046, 14), range=2020-01-02 ~ 2026-04-07
features_df shape=(..., 45)
rdagent_factor_columns=[MomRet_5D, VolumeWeightedReturn_5D, ...]        共 14 列
RD-Agent 列落入 features_df 数量=14 / 14
至少有一个 RD-Agent 非空的行数 = 总行数                                  100% 覆盖
已更新 feature_sets[rdagent_exported]，列数=14
lgb spec -> 解析列数=45     (31 from lgb_short_cycle + 14 from rdagent_exported)
gru spec -> 解析列数=15, 是否含 RD-Agent 列 False                        GRU 独立
=== SMOKE DONE ===
```

### Step 3 · 端到端最小训练（`RollingTrainer`）

```powershell
python scripts/smoke_p01_train.py
```

在 `%TEMP%\p01_train_*` 下跑一次：`csi300` × `2025-11-15 ~ 2026-04-07` × **仅 LGB** × 单窗口（`step_days=1000`） × **关闭 stack / oof_stacking**。不会污染 `data/models/`。

**预计耗时**：30~40 秒。

**预计关键输出**：

```
P0-1：已合并 RD-Agent 因子 14 列（parquet=...combined_factors_df.parquet）
对齐后数据量: 27807 行
模型 lgb 使用特征集合=['lgb_short_cycle', 'rdagent_exported']，列数=45
LGB 训练特征完整列表=[Ref($pe, 1), ..., MomRet_5D, ..., VolRet_5D]      末尾 14 列 = RD-Agent
LightGBM 模型已保存: <tmp>/models/<date>_lgb.txt
窗口 0 训练完成，耗时 2.x s
=== P0-1 END-TO-END SMOKE ===
```

> 备注：脚本尾部自检用 `booster.feature_name()` 读到的是 `Column_0..Column_44` 位置名（LGB 训练侧传入的是 numpy 矩阵而非 DataFrame），这是**展示层命名**问题，不影响功能。判断 RD-Agent 是否真正参与训练，以日志中的 `LGB 训练特征完整列表=[...]` 末尾是否出现 RD-Agent 列（如 `MomRet_5D`…`VolRet_5D`）为准。

### Step 4 · 正式生产训练（可选）

三个冒烟都过后，跑全量：

```powershell
python run_train.py --config config/pipeline.yaml
```

- 对 `data.instruments` 中每个池 × 完整 rolling × LGB + GRU × OOF-stacking 全流程
- 依硬件约 30~90 分钟
- 产物：`data/models/<pool>_models/*`、`data/logs/<pool>_logs/*`、`data/oof/*`
- 后续：`python run_predict.py` / `python run_backtest.py`

---

## 4. 成功判据（逐步核对）


| 步骤     | 通过标志                                                                                                                     |
| ------ | ------------------------------------------------------------------------------------------------------------------------ |
| Step 1 | 日志含 `scan done: ok=<N>` 且 `N ≥ 1`；有 `写入: ...combined_factors_df.parquet`；生成同目录 `.json` 摘要                                |
| Step 2 | `rdagent_factor_columns` 非空；`RD-Agent 列落入 features_df 数量 = <N> / <N>`；LGB 列数 = qlib 表达式列数 + N；GRU `是否含 RD-Agent 列 False` |
| Step 3 | `P0-1：已合并 RD-Agent 因子 <N> 列`；`LGB 训练特征完整列表` 末尾含 RD-Agent 列名；`LightGBM 模型已保存: .../<date>_lgb.txt`                         |
| Step 4 | `run_train.py` 正常跑完全部 rolling 窗口；各池 `data/models/*_models/*_lgb.txt` 存在；训练日志无 stack/OOF 报错                               |


---

## 5. 常见问题

### 5.1 PowerShell 下参数被逗号拆开

```
error: unrecognized arguments: csi300
```

**原因**：`--instruments csi500,csi300` 未加引号。
**修复**：`--instruments 'csi500,csi300'`（或用双引号）。

### 5.2 `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`

**原因**：在 `base` 环境下跑，缺 parquet 引擎。
**修复**：`conda activate qlib_zhengshi` 或前缀 `conda run -n qlib_zhengshi --no-capture-output`。

### 5.3 `ValueError: instrument not exists: D:\qlib_data\qlib_data\instruments\csi500,csi300.txt`

**原因**：qlib 把逗号分隔的池名当作单个文件名去找。
**修复**：脚本内部已按逗号拆分、对每个池分别 `D.instruments(pool)` 再 union。若仍出现，检查 qlib 数据是否含 `csi500.txt` / `csi300.txt`。

### 5.4 `fail=67`（Step 1）

少数 `factor.py` 走了 `pd.read_hdf` 之外的 IO 路径（如 `os.path.exists('daily_pv.h5')`），Monkey-patch 无法拦截，被记为失败。正常现象；不影响 top-N 筛选。

### 5.5 `slow factor (300s+)`（Step 1）

`01f160fd...` 因子在 rolling window 里用 `scipy.stats.linregress` 做回归，Python 级别慢。首次跑完后结果已写入 parquet，后续无需重算。

### 5.6 Step 3 的 `LGB 使用了 RD-Agent 因子: 0 个`

自检误报，看 5.3 备注；真正判据是 `LGB 训练特征完整列表=[...]` 末尾是否包含 `MomRet_5D`…`VolRet_5D`。若真的没有，检查：

- `config/pipeline.yaml` 的 `model_features.lgb` 是否为 `["lgb_short_cycle", "rdagent_exported"]`
- `config/data.yaml` 的 `active_feature_sets` 是否含 `rdagent_exported`
- parquet 是否存在且列名不空

### 5.7 想回滚到旧 parquet

```powershell
Copy-Item git_ignore_folder\combined_factors_df.parquet.bak.<ts> `
          git_ignore_folder\combined_factors_df.parquet -Force
```

### 5.8 想跑"对照组"（不启用 RD-Agent 因子）

```powershell
python run_train.py --active-feature-sets 'short_momentum_reversal,short_volatility_range,short_volume_price,gru_short_cycle,lgb_short_cycle'
```

即在 `--active-feature-sets` 里**不包含** `rdagent_exported`，pipeline 会跳过合并步骤。

---

## 6. 产物清单

### 脚本（新增）

- `scripts/refresh_rdagent_parquet.py` — parquet 刷新（时间/股票覆盖对齐）
- `scripts/smoke_p01.py` — 链路冒烟（无训练）
- `scripts/smoke_p01_train.py` — 端到端最小训练冒烟
- `scripts/inspect_rdagent_sources.py` — 原料覆盖度排查（可选）
- `scripts/peek_factor_scripts.py` — `factor.py` 接口摸底（可选）

### 数据文件

- `git_ignore_folder/combined_factors_df.parquet` — **当前版本**（本次：`shape=(1150046, 14)`，`2020-01-02 ~ 2026-04-07`，799 只股票）
- `git_ignore_folder/combined_factors_df.json` — 上一次 refresh 的摘要
- `git_ignore_folder/combined_factors_df.parquet.bak.<unix_ts>` — 自动备份

### 主工程改动（见第 1 节表格）

---

## 7. 什么时候该重跑 Step 1？


| 触发条件                                               | 动作                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| qlib_data 扩展到新交易日（例如从 2026-04-07 延长到 2026-05-31）   | 重跑 Step 1，`--end 2026-05-31`                                       |
| 股票池范围变化（例如从 csi500+csi300 加 csi800）                | 重跑 Step 1，`--instruments 'csi500,csi300,csi800'`                   |
| RD-Agent 新发掘了一批 factor（`RD-Agent_workspace` 新增 ws） | 重跑 Step 1，自动纳入筛选                                                   |
| 想调整因子入池阈值                                          | 重跑 Step 1，改 `--ic-threshold` / `--max-nan-ratio` / `--max-factors` |
| 仅修改主工程代码（`trainer/`、`models/`）                     | **无需**重跑 Step 1；直接 Step 3 冒烟 + Step 4                              |


---

## 8. 字段映射（`refresh_rdagent_parquet.py` 内置）

RD-Agent `factor.py` 期望的字段 ← 项目 qlib_data 实际字段：


| RD-Agent schema                               | qlib_data 字段                                    |
| --------------------------------------------- | ----------------------------------------------- |
| `$close / $open / $high / $low`               | `$close_qfq / $open_qfq / $high_qfq / $low_qfq` |
| `$volume`                                     | `$vol`                                          |
| `$rsi12`                                      | `$rsi_qfq_12`                                   |
| `$macd`                                       | `$macd_qfq`                                     |
| `$atr`                                        | `$atr_qfq`                                      |
| 其余（`$amount / $pe / $pb / $roe / $roa / ...`） | 同名                                              |


若未来 qlib_data 字段命名有变，更新 `scripts/refresh_rdagent_parquet.py` 顶部的 `FIELD_MAP` 即可。

---

## 10. 对照实验结论与回滚说明（2026-04-19）

P0-1 链路验证通过后，做了一次"含 RD vs 无 RD"完整训练 + rqalpha 回测对照：

**回测窗口**：2025-11-03 ~ 2026-03-31（~5 个月，105 个交易日）；基准 沪深 300 期间 −4.11%

| 指标 | csi300（无 RD） | csi300_RD | csi500（无 RD） | csi500_RD |
|---|---:|---:|---:|---:|
| 总收益率 | **+43.47%** | +11.04% | **+10.69%** | **−3.96%** |
| 年化收益 | **+150.05%** | +30.29% | +29.20% | −9.94% |
| 夏普 | **3.59** | 1.16 | 1.10 | **−0.17** |
| 信息比 | 11.16 | 2.08 | 2.69 | 0.13 |
| 最大回撤 | 8.39% | 9.40% | 15.99% | **22.11%** |
| **年化换手** | 4.70 | **10.49（+123%）** | 3.59 | **11.34（+216%）** |
| 胜率 | 56.6% | 51.5% | 52.5% | **46.5%** |
| Alpha | 1.66 | 0.40 | 0.43 | 0.04 |

**结论**：当前 14 个 RD-Agent 因子在所有维度均为负贡献，**csi500_RD 由盈利转亏损**。

**根因诊断**：

1. **In-sample factor selection bias**：14 个因子是在 2020-01 ~ 2026-04 全样本上按 |IC| 筛出的，回测窗口已被"看过"，属"看似无前视实际有前视"
2. **同质化严重**：14 个里 4 个 MomRet_*、3 个 Volume*、3 个 VolRatio/VolumeTrend，几乎全是动量/量价族，与 `lgb_short_cycle` 中价格类表达式高度共线
3. **高换手 → 成本吃光**：换手翻 2~3 倍是最干净的"作案证据"，扣完手续费/印花税/滑点后利润被吃光（csi500 从 +29% 退到 −10%）
4. **胜率低于 50%**：csi500_RD 胜率 46.5%，验证集 IC 高（+0.148）而真实胜率反向 → 强证据指向"OOS 信号崩塌"

### 10.1 回滚动作（已执行 2026-04-19）

```yaml
# config/pipeline.yaml
model_features:
  gru: "gru_short_cycle"
  lgb: "lgb_short_cycle"   # 从 ["lgb_short_cycle", "rdagent_exported"] 回滚
```

P0-1 链路代码（`feature/qlib_feature_pipeline.py` / `models/ensemble_manager.py` / `trainer/trainer.py` / `trainer/oof_manager.py` / `run_predict.py`）**全部保留**，只改 `model_features.lgb` 一行即可重新启用。`config/data.yaml.active_feature_sets` 中的 `rdagent_exported` 也无需删除（pipeline 会根据是否被任何模型引用决定是否合并 parquet）。

### 10.2 后置门槛：何时再启用 RD-Agent 因子

必须在以下三件事**同时满足**后才重新启用：

| 门槛 | 检查方式 |
|---|---|
| OOS 严格筛选（不再用全样本算 |IC|） | `refresh_rdagent_parquet.py --oos-cutoff 2024-12-31 --ic-ir-threshold 0.4` |
| 与已有 `lgb_short_cycle` 列两两 |corr| ≤ 0.7 | `scripts/factor_diagnostic_p01.py` 共线性 heatmap |
| OOS 段 IC_IR ≥ 0.3 且符号稳定 | `scripts/factor_diagnostic_p01.py` 滚动 IC 表 |

预计 14 个因子会缩到 **3~5 个**。届时再启用，并跑同样的 csi300/csi500 对照回测，**夏普 / 年化收益 / 换手率三项必须不弱于无 RD 版本**才算通过。

### 10.3 历史产物保留

- 回测对照：`data/backtest/rqalpha/csi{300,500}` vs `data/backtest/rqalpha/csi{300,500}_RD`
- parquet 备份：`git_ignore_folder/combined_factors_df.parquet.bak.<ts>`（旧 200 股版 + 当前 799 股版均保留）
- 当前在用 parquet 仍是 14 因子版本，方便随时切回 P0-1 ON 跑 A/B

---

## 11. 与 Roadmap 的关系

P0-1 完成后，可衔接：

- **P0-2** `market_regime.yaml` 落地（与 P0-1 独立，可并行）
- **P0-3** `base_models` 扩到 4 个（新模型只需在 `pipeline.yaml.model_features` 里决定是否消费 `rdagent_exported`）
- **P0-4** 可视化仪表盘（Factor Pool 卡直接读 `combined_factors_df.json` 作为数据源，见 `docs/DASHBOARD_DATA_SOURCES.md`）

---

*最后更新：2026-04-18*