# Pipeline Dashboard 详细设计（P0-4 主线）

> **文档性质**：`data/dashboard/` 仪表盘的详细设计。
> **关联路线图**：[OPTIMIZATION_ROADMAP_2026.md 第十一节](./OPTIMIZATION_ROADMAP_2026.md#十一可视化工作流仪表盘p0-4-主线设计)。
> **制定日期**：2026-04-17（v2 修订同日）。
> **受众**：本项目维护者；一眼看板的终端用户也是本人。

---

## 一、设计目标（硬指标）

> 打开 `data/dashboard/index.html`，**3 秒内**能回答 5 个问题：

1. 当前市场处于什么状态（趋势 / 震荡 / 高波动）？
2. RD-Agent 最近一次跑到第几轮？有几个因子通过 / 失败？
3. 因子池里目前有多少可用因子？平均 |IC| 多少？最近新增哪几个？
4. 训练滚动到第几窗？各基模型最近 20 窗的 IC 均值？
5. 最近一次回测的年化、回撤、换手？

3 秒回答不了任一问题 = P0-4 未达成，必须迭代。

---

## 二、非功能性约束

- **零后端**：所有产物都是静态 HTML + JSON。用户可直接双击打开，不需要 `python -m http.server`。
- **单一入口**：`data/dashboard/index.html`，所有子页都从首页跳转。
- **幂等刷新**：任意时刻重新跑 `render_pipeline_dashboard.py` 产物一致；脏数据提示而非崩溃。
- **弱依赖**：缺任何一个数据源（比如还没跑 RD-Agent），对应卡片显示"暂无数据"而不是报错；dashboard 其他部分仍可看。
- **可追溯**：每张卡右上角有"数据来源: xxx · mtime: yyyy-mm-dd HH:MM"一行小字，让用户知道这块为什么是这个数。
- **色号一致**：`assets/style.css` 统一三色语义：
  - 绿（`#2f9e44`）= 健康 / 通过 / 正超额
  - 黄（`#f59f00`）= 进行中 / 提醒 / 边缘
  - 红（`#e03131`）= 失败 / 负 IC / 超限

---

## 三、信息架构

```
data/dashboard/
├── index.html                   # 首页
├── stage_regime.html            # ① 市场状态详情
├── stage_rdagent.html           # ② RD-Agent 循环详情
├── stage_factor_pool.html       # ③ 因子池详情
├── stage_train.html             # ④ 训练 & OOF
├── stage_ensemble.html          # ⑤ 集成 & Meta
├── stage_backtest.html          # ⑥ 回测 & 组合
├── status_snapshot.json         # 首页数据源（脚本写）
├── regime_snapshot.json         # ① 详情数据源
├── events.log                   # 近 24h 事件流
├── assets/
│   ├── style.css
│   └── sparkline.js             # 纯原生 canvas 小图（无第三方）
└── legacy/                      # 存旧快照（time-series，便于对比）
    └── snapshot_YYYYMMDD_HHMM.json
```

---

## 四、首页 `index.html` 规格

### 4.1 布局（从上到下）

```
┌─── 顶部栏（64px） ───────────────────────────────────┐
│ Logo + Title          Last update: ... · [刷新 ⟲]   │
└─────────────────────────────────────────────────────┘
┌─── 市场状态卡（120px） ─────────────────────────────┐
│ 池 / 区间 / Regime label / Vol 分位 / 胜率 / 目标函数  │
└─────────────────────────────────────────────────────┘
┌─── Pipeline 阶段灯（80px） ─────────────────────────┐
│  ① → ② → ③ → ④ → ⑤ → ⑥                              │
│ 每个灯下方：名称 + 状态图标 + 最后活跃时间               │
└─────────────────────────────────────────────────────┘
┌─── 6 张状态卡（2 行 × 3 列，每张 200px 高）─────────────┐
│ RDAgent      Factor Pool    Training                  │
│ Ensemble     Backtest       Audit Trail               │
└─────────────────────────────────────────────────────┘
┌─── 最近事件流（160px，滚动） ──────────────────────────┐
│ events.log 末 10~20 条，固定行距                       │
└─────────────────────────────────────────────────────┘
```

### 4.2 每个区块字段

#### 4.2.1 市场状态卡（来自 `regime_snapshot.json`）

```json
{
  "universe": ["csi300", "csi500"],
  "window": ["2025-10-01", "2026-04-07"],
  "regime_label": "trend_up",
  "regime_label_cn": "趋势上行",
  "signals": {
    "trend_60d": {"value": 0.082, "quantile": 0.73},
    "vol_20d":   {"value": 0.019, "quantile": 0.38},
    "breadth_60d": {"value": 0.54, "quantile": 0.62}
  },
  "objective": {
    "primary": "rank_ic_ensemble",
    "secondary": "turnover_adj_sharpe"
  },
  "updated_at": "2026-04-17T23:38:00"
}
```

显示规则：

- `regime_label_cn` 作为大字（24pt）。
- 三条 signal 各占一列，值 + 右上角迷你分位条。
- `vol_20d.quantile > 0.8` → 标红（高波动）；`< 0.2` → 标绿（低波动）。

#### 4.2.2 Pipeline 阶段灯

6 个圆点：`① 市场 / ② RDAgent / ③ 因子池 / ④ 训练 / ⑤ 集成 / ⑥ 回测`。

状态枚举：


| 状态        | 图标  | 色   | 触发条件                                                    |
| --------- | --- | --- | ------------------------------------------------------- |
| `ready`   | ✓   | 绿   | 该阶段最新产物 mtime 在 24h 内                                   |
| `running` | ◐   | 黄   | 该阶段有进程在跑（events.log 最后 1h 内有事件）                         |
| `target`  | ●   | 蓝   | 本 Phase/Week 的目标阶段（从 `status_snapshot.current_focus` 读） |
| `stale`   | ⏳   | 灰   | 产物存在但 mtime 超过 7 天                                      |
| `missing` | ○   | 灰   | 从未运行过，产物不存在                                             |
| `failed`  | ✗   | 红   | events.log 最后一次与该阶段相关的事件是 error                         |


#### 4.2.3 6 张状态卡字段规格

见第六节"各子页详情"。首页卡只取每个子页的 Top 3~5 条摘要字段。

### 4.3 `status_snapshot.json` Schema

```json
{
  "version": 1,
  "generated_at": "2026-04-17T23:40:00",
  "generated_by": "scripts/render_pipeline_dashboard.py",
  "current_focus": "stage_rdagent",
  "regime": { "see regime_snapshot.json": true },
  "stages": {
    "rdagent":     { "status": "running", "mtime": "...", "summary": {...} },
    "factor_pool": { "status": "ready",   "mtime": "...", "summary": {...} },
    "train":       { "status": "target",  "mtime": "...", "summary": {...} },
    "ensemble":    { "status": "missing", "mtime": null,  "summary": {} },
    "backtest":    { "status": "stale",   "mtime": "...", "summary": {...} }
  },
  "events_tail_path": "events.log"
}
```

---

## 五、Mermaid 全链路图（贴在首页上方，折叠）

默认折叠为一行按钮 "展开系统全链路图 ▾"；展开后渲染 roadmap 第 11.1 节的 Mermaid 图（客户端 JS 渲染 `mermaid.min.js`，离线版打包到 `assets/`）。

---

## 六、各子页详情

### 6.1 `stage_regime.html`（市场状态）


| 区块        | 字段 / 可视化                                                                                                     | 数据源                                              |
| --------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| 概要卡       | regime_label / updated_at / objective                                                                        | `regime_snapshot.json`                           |
| 趋势 60d 时序 | `Mean($close_qfq,20)/Mean($close_qfq,60)-1` 过去 250 日折线图 + 今日位置标注                                             | `regime_snapshot.json.history`（新增）               |
| 波动 20d 分位 | 过去 250 日分位柱状图 + 今日分位标注                                                                                       | 同上                                               |
| 市场宽度      | 上涨股票占比时序                                                                                                     | 同上                                               |
| 股票池覆盖率    | `csi300` / `csi500` 成份股数量变化                                                                                  | qlib `D.list_instruments`                        |
| 配置对齐检查    | `config/data.yaml.start_time / end_time` 与 `pipeline.yaml.rolling.model_train_days` 的关系是否需要 preflight 回溯说明提醒 | 读两份 YAML + `run_tuning_experiments.py` preflight |


### 6.2 `stage_rdagent.html`（RD-Agent 循环详情）


| 区块         | 字段                                                                                                     | 数据源                                         |
| ---------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| 当前 profile | profile name / feature_sets / env 关键变量（`QLIB_QUANT_EVOLVING_N` 等）                                      | `config/rdagent_profile_matrix.json` + 环境变量 |
| Loop 时间轴   | 每个 loop: `direct_exp_gen` → `coding` → `running` → `feedback` 四段横条                                     | `log/<ts>/rdagent.jsonl`                    |
| 任务表        | 每 task: name / formulation 摘要 / coding 状态 / running 是否产出 `result.h5` / feedback pass/fail / error_tail | `run_artifacts/<loop>/`                     |
| 失败分类统计     | 语法 / 数据缺失 / 超时 / 环境（对应 v1 Phase 5 的失败诊断）                                                               | 聚合 error_tail                               |
| 本次运行环境     | `QLIB_RDAGENT_CONDA_ENV` / `QLIB_FACTOR_TEMPLATE` / `DEEPSEEK_API_KEY` 长度                              | 环境变量                                        |


### 6.3 `stage_factor_pool.html`（因子池详情）


| 区块            | 字段                                                                    | 数据源                                                              |
| ------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 概要卡           | 因子总数 / 平均                                                             | IC                                                               |
| 因子表           | 列：name / IC / ICIR / nan_ratio / workspace_id / profile 来源 / 入池日期；支持按 | IC                                                               |
| 相关矩阵热图        | 按聚类结果重排序（`scipy.cluster.hierarchy.leaves_list`）                       | `scripts/factor_redundancy_report.py` 产物 `corr_heatmap_{ts}.png` |
| 冗余报告摘要        | 总簇数 / 无效因子数 / 推荐保留数（"一眼看板"格子，来自 v1 10.2）                              | `data/factor_analysis/redundancy_report_LATEST.md`               |
| 因子公式预览        | 点击因子名 → 弹层显示 `factor.py` 前 20 行 + `calculate_*` 函数签名                  | `git_ignore_folder/RD-Agent_workspace/<ws>/factor.py`            |
| 与现有集关联        | 每个新因子 vs `active_feature_sets` 内表达式的最大相关（来自去冗余报告）                     | 同上                                                               |
| **与主工程流打通状态** | 当前 `active_feature_sets` 是否包含 `rdagent_exported`（P0-1 DoD 可视化）        | `config/data.yaml`                                               |


### 6.4 `stage_train.html`（训练 & OOF）


| 区块       | 字段                                                         | 数据源                                         |
| -------- | ---------------------------------------------------------- | ------------------------------------------- |
| 窗口进度条    | `n/N` 窗口完成；每窗耗时；失败窗口标红                                     | `data/logs/*_logs/training_metrics.csv`     |
| IC 时序多线图 | 每个基模型一条线：`ic_lgb` / `ic_gru` / `ic_lgbrd` / `ic_mlp_resid` | 同上                                          |
| ICIR 卡   | 每个模型滚动 20 窗 ICIR                                           | 同上                                          |
| OOF 状态   | fold 进度（`n_splits=5`）；每 fold 产物是否齐全                        | `data/oof/{tag}/*.npy`                      |
| 失败窗口     | 最近 5 个失败窗口的 traceback 前 30 行（来自 Phase 1 8.3 修订）            | `data/tuning/runs_*/result.json.error_tail` |
| GPU/环境   | `[env]` 摘要（torch / cuda / device / 显存）                     | `run_train.py` 启动日志                         |


### 6.5 `stage_ensemble.html`（集成 & Meta）


| 区块              | 字段                                                   | 数据源                                                                  |
| --------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| 动态权重时间线         | 堆叠面积图：每天每个基模型的权重                                     | `predictor/weight_dynamic.py` 日志 / `data/predictions/pred_*.csv` 扩展列 |
| Meta-Stacker 系数 | Ridge/ElasticNet 系数 或 LGBM-meta feature_importance   | `data/meta/{tag}_meta_meta.json`                                     |
| 最近 20 窗 IC      | 基模型柱状图 + meta 柱状图对比                                  | `data/logs/*_logs/training_metrics.csv`                              |
| 预测分布            | 每日预测分位数分布直方图；`label_transform=percentile` 之后的分布应接近均匀 | `data/predictions/pred_*.csv`                                        |
| 规则告警            | 任一基模型权重触顶（= `max_weight`）超过 N 日 → 提醒多样性不足            | 脚本侧计算                                                                |


### 6.6 `stage_backtest.html`（回测 & 组合）


| 区块         | 字段                                         | 数据源                                                  |
| ---------- | ------------------------------------------ | ---------------------------------------------------- |
| 净值曲线       | 策略 vs benchmark 两条线                        | `data/backtest/rqalpha/{pool}/detailed_results.json` |
| 指标卡        | AnnRet / MaxDD / Sharpe / Calmar / WinRate | 同上                                                   |
| 换手柱状图      | 每日换手；标红超过阈值的日子                             | 同上                                                   |
| 成本累计       | 手续费 / 滑点 / 冲击（如实现）的累计占比                    | 同上                                                   |
| Top K 持仓行业 | 行业饼图 + 每日 diff（昨天→今天的进 / 出）                | `data/predictions/pred_*.csv` + industry mapping     |
| **成本敏感度**  | P1-3 双倍成本下的同一策略指标对比（左右并排）                  | 重跑 `run_backtest.py --cost-multiplier 2` 的结果（新增参数）   |


### 6.7 `events.log` 格式

```
<ISO8601> [<stage>] <level> <message>
```

例：

```
2026-04-17T23:38:00 [train]     INFO  window 2026-03-01 done, ic_gru=0.033
2026-04-17T23:35:12 [rdagent]   INFO  loop-3 feedback: 3 pass, 1 fail
2026-04-17T23:10:05 [factor_pool] INFO factor F082 merged, ic=0.041
2026-04-17T22:58:41 [backtest]  WARN  turnover 52% exceeds threshold 40%
```

约束：行首固定 ISO8601 时间戳；stage 从固定枚举里取；`WARN`/`ERROR` 行在 dashboard 标黄/标红。

---

## 七、脚本规格

### 7.1 `scripts/render_pipeline_dashboard.py`（P0-4 主脚本）

**输入**：

- `config/data.yaml`、`config/pipeline.yaml`、`config/market_regime.yaml`
- `data/dashboard/regime_snapshot.json`
- `git_ignore_folder/combined_factors_df.parquet` + `.json`
- `data/factor_analysis/redundancy_report_LATEST.md`
- `data/logs/*_logs/training_metrics.csv`
- `data/oof/{tag}/`、`data/meta/{tag}_meta_meta.json`
- `data/backtest/rqalpha/{pool}/detailed_results.json`
- `log/<ts>/rdagent.jsonl`（最近一次）

**输出**：

- `data/dashboard/index.html`
- `data/dashboard/stage_*.html`
- `data/dashboard/status_snapshot.json`
- `data/dashboard/legacy/snapshot_<ts>.json`

**CLI**：

```bash
python scripts/render_pipeline_dashboard.py \
  --out data/dashboard \
  --focus rdagent \
  --keep-legacy 30
```

- `--focus <stage>`：手动指定首页的 `target` 高亮灯（默认自动：最近 24h 内状态变化最频繁的 stage）。
- `--keep-legacy <n>`：legacy 目录保留最近 n 份快照，滚动清理。

**健壮性**：

- 缺任一输入 → 对应卡片渲染 "暂无数据 · 预期来源: `<path>`" 占位；**不 raise**。
- 输入格式异常 → 对应卡片渲染 "数据格式错误 · 文件: `<path>`"；完整 traceback 写 `data/dashboard/render.log`。

### 7.2 `scripts/sync_regime_snapshot.py`（P0-2 新增）

从 qlib 数据计算 regime signals，写入 `regime_snapshot.json` + `history` 字段（过去 250 日时序）。

```bash
python scripts/sync_regime_snapshot.py \
  --regime-config config/market_regime.yaml \
  --out data/dashboard/regime_snapshot.json
```

### 7.3 `scripts/live_events_tailer.py`（可选，半实时模式用）

长驻 watcher，合并多个日志源为统一 `events.log`：

- `data/logs/*_logs/*.log` → `[train]`
- `log/<ts>/*.log` → `[rdagent]`
- `data/backtest/*.log` → `[backtest]`

```bash
python scripts/live_events_tailer.py --out data/dashboard/events.log --tail-lines 200
```

### 7.4 触发时机（建议加到各主脚本尾部）


| 主脚本                                 | 追加                                                                  |
| ----------------------------------- | ------------------------------------------------------------------- |
| `run_train.py`                      | `subprocess.run(["python","scripts/render_pipeline_dashboard.py"])` |
| `run_predict.py`                    | 同上                                                                  |
| `run_backtest.py`                   | 同上                                                                  |
| `scripts/tuning_pipeline.py`        | 每条实验结束后触发一次                                                         |
| `scripts/run_fin_quant.py`          | 每轮 feedback 结束后触发一次                                                 |
| `scripts/export_rdagent_factors.py` | 导出成功后触发一次                                                           |


这些追加可以用 "post-hook" 的形式（统一的 `utils/post_hook.py`）避免每个脚本都 import。

---

## 八、回答力测试（验收清单）

每次 P0-4 交付后执行一次自检：


| 测试项  | 步骤                                                      | 期望                           |
| ---- | ------------------------------------------------------- | ---------------------------- |
| T-1  | 打开 `data/dashboard/index.html`                          | 页面加载 < 1s；不报 404             |
| T-2  | 在首页看 3 秒 → 回答：当前市场是什么状态？                                | 首页市场状态卡明文显示                  |
| T-3  | 在首页看 3 秒 → 回答：RDAgent 跑到第几轮？                            | RDAgent 卡 `loop n/N` 明文      |
| T-4  | 在首页看 3 秒 → 回答：因子池平均                                     | IC                           |
| T-5  | 在首页看 3 秒 → 回答：最近一次回测年化？                                 | Backtest 卡 `AnnRet xx.x%` 明文 |
| T-6  | 点 Factor Pool 卡 → 跳转 `stage_factor_pool.html`           | 因子表可排序；相关热图可查看               |
| T-7  | 删除 `git_ignore_folder/combined_factors_df.parquet` 后重渲染 | Factor Pool 卡显示"暂无数据"而非崩溃    |
| T-8  | 伪造格式错误的 `regime_snapshot.json`                          | 市场状态卡显示"数据格式错误"；其他卡正常        |
| T-9  | 3 个主脚本（train / predict / backtest）结束后自动触发重渲染            | `index.html` mtime 同步更新      |
| T-10 | `--focus rdagent` 时首页 `②` 灯为蓝色 `target`                 | 目视确认                         |


T-2 ~ T-5 对应第一节的 5 个问题，**全部通过才算 P0-4-a/b/c/d 全交付**。

---

## 九、实现顺序（开发期建议）

1. **先搭壳**：`render_pipeline_dashboard.py` 输出固定模板 HTML + 6 张空卡。验证 T-1、T-7、T-8。
2. **接已有数据源**（P0-4-b）：Training / Ensemble / Backtest 三块先接，因为数据已经在。验证 T-5。
3. **接 P0-1 联动**（P0-4-c）：Factor Pool + RDAgent 接上，需要 P0-1 先把 parquet 读入主工程。验证 T-3、T-4、T-6。
4. **接 P0-2 联动**（P0-4-d）：market_regime.yaml + sync_regime_snapshot.py 上线。验证 T-2。
5. **加实时性**（可选）：`live_events_tailer.py` + `<meta http-equiv=refresh>`。
6. **自动化钩子**：改 3 个 `run_*.py` 在尾部调用 renderer。验证 T-9。

---

## 十、非目标（明确不做）

- **不做前后端分离**：本项目是单机研究工具，不需要 React/Vue。
- **不做用户认证**：本地打开，不暴露到公网。
- **不做 DB 落存**：所有状态都在文件系统，文件就是 source of truth。
- **不做实时 WebSocket**：半实时模式用 HTML meta refresh + 30s 轮询足够。
- **不做 Notebook**：Jupyter 对"一眼可见"目标反而是干扰。

---

## 十一、与 roadmap v1 第十节的关系

v1 节 10.2 里的三项（tuning HTML 过滤器 / 因子去冗余一眼看板 / GRU A/B 胜负徽章）都**降级为本 dashboard 的子区块**，不再单独改造对应 HTML；`scripts/render_ic_dashboard.py` 和 `scripts/sync_roadmap_status.py` 保留为 P2-2 的单独脚本，但它们的产物也要在 dashboard 首页 "Audit Trail" 卡里出链接。

---

*文档结束。实现开始前的最后一次校对，请回到 [OPTIMIZATION_ROADMAP_2026.md 第零节](./OPTIMIZATION_ROADMAP_2026.md#零v2-重订总览主线优先级高于-v1-phase-15) 确认本设计与 P0-4 范围一致。*