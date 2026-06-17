# Pipeline Dashboard 数据来源说明

> 对应脚本：`scripts/render_pipeline_dashboard.py` + `scripts/dashboard_collect.py`  
> 生成入口：`data/dashboard/index.html`（双击或浏览器打开）

首页「数据血缘」表格与此文档一致；子页展示各模块的 **JSON 摘要**（便于核对）。

---

## 运行环境（conda：`qlib_zhengshi`）

本工程主环境为 **conda `qlib_zhengshi`**（与 `scripts/run_fin_quant.py`、`trainer`、qlib 一致）。请在该环境中执行渲染脚本，以便使用已安装的 **PyYAML、pandas** 等依赖，完整解析 `config/data.yaml` 与 `training_metrics.csv`。

```powershell
conda activate qlib_zhengshi
cd D:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project
python scripts/render_pipeline_dashboard.py --write-regime
```

或不进入交互式激活：

```powershell
conda run -n qlib_zhengshi python scripts/render_pipeline_dashboard.py --write-regime
```

若用系统自带 Python 且未装 `yaml`，脚本会退化为**正则读取** `data.yaml` 前几行；训练摘要仍依赖 **pandas** 读 CSV，建议在 `qlib_zhengshi` 中运行。

---

## 配置解析说明

- **优先**使用 `PyYAML` 解析完整 `config/data.yaml` 的 `data:` 段（若已 `pip install PyYAML`）。
- **未安装 PyYAML 时**：`scripts/dashboard_collect.py` 用正则从文件前 80 行读取 `instruments`、`start_time`、`end_time`，不依赖第三方库，保证仪表盘仍能显示股票池与区间。

## 总览

| 首页区块 | 数据从哪来 | 谁写入 / 何时有 |
|----------|------------|------------------|
| **市场大卡（窗口/股票池）** | `config/data.yaml` 的 `data.instruments`、`data.start_time`、`data.end_time` | 人工改配置；渲染时合并进 `data/dashboard/regime_snapshot.json`（可选 `--write-regime`） |
| **市场标签文案** | 若未跑 P0-2，显示「配置口径（data.yaml）」；真实 regime 需 `scripts/sync_regime_snapshot.py`（尚未实现） | — |
| **训练 IC 卡** | `data/logs/<pool>_logs/training_metrics.csv`，默认优先 `csi500_logs`；否则取最新修改的 `*_logs/training_metrics.csv` | `trainer` / `run_train.py` 滚动训练后追加行 |
| **因子池卡** | `git_ignore_folder/combined_factors_df.json`（元数据）与可选 `.parquet`（mtime） | `scripts/export_rdagent_factors.py` |
| **RD-Agent 卡** | `git_ignore_folder/RD-Agent_workspace/*/result.h5`（统计有产出的工作区数、最新 mtime） | RD-Agent 因子任务跑通后 |
| **集成卡** | `data/oof/*/`（标签目录）、`data/meta/` 或 `data/models/` 下 `*meta*.json`（取最新 mtime） | `trainer` OOF / meta_stacker |
| **回测卡** | `data/backtest/rqalpha/<csi300|csi500|…>/detailed_results.json`（取**全局最新修改**的一个文件） | `run_backtest` + RQAlpha 导出 |
| **审计卡** | `docs/OPTIMIZATION_ROADMAP_2026.md` 的 mtime | 人工编辑路线图 |
| **事件流** | `data/dashboard/events.log`（每次渲染追加一行） | 本脚本 `append_event` |

---

## 各文件字段含义（摘要）

### `training_metrics.csv`

- 由 `RollingTrainer` 按窗口写入，常见列：`window`, `train_*`, `valid_*`, `ic_lgb`, `ic_gru`, `ic_qlib_ensemble` 等。
- Dashboard 使用：**最后一行**作为「最新窗」；**最后 20 行**的各 `ic_*` 列均值作为「近 20 窗均值」。

### `combined_factors_df.json`

- 由 `export_rdagent_factors.py` 写出：`n_factors`、`factors[].name/ic/nan_ratio`、`created_at` 等。
- Dashboard 使用：因子个数、平均 |IC|（对 factors 列表里 `ic` 取绝对值再平均）。

### `detailed_results.json`（RQAlpha）

- 路径：`data/backtest/rqalpha/<pool>/detailed_results.json`。
- Dashboard 使用：`效率指标` 中的 `annualized_returns`、`max_drawdown`、`sharpe`、`avg_daily_turnover`、`start_date`、`end_date`；`盈亏状态` 中的总收益等作为兜底。

---

## 命令行参数

| 参数 | 含义 |
|------|------|
| `--metrics-pool csi500` | 优先读 `data/logs/csi500_logs/training_metrics.csv` |
| `--write-regime` | 把合并后的 `regime_snapshot.json` 写回磁盘（含 `config_alignment`） |

---

## 未接入 / 后续

- **行情级 regime**（趋势/波动分位）：需 P0-2 `sync_regime_snapshot.py` + qlib，当前仅配置口径。
- **RD-Agent 结构化日志**（`log/<ts>/rdagent.jsonl`）：若存在可再增强「②」子页，当前仅扫 workspace。
