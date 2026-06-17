# 阶段 D 测试 Playbook（复制即跑版）

> 目的：**让你不用读代码就能把阶段 D 交付的 4 个新 check、RD-Agent 导出器、自动闭环
> 亲自跑一遍**。每一步都给出：命令、预期输出关键字、预期耗时、出错怎么办。
>
> 姊妹篇：
>
> - `docs/STAGE_D_TEST_GUIDE.md` —— 架构 / 接口级指南
> - `docs/STAGE_D_REPORT.md` —— 阶段 D 交付与结论
> - `docs/STAGE_C_TEST_PLAYBOOK.md` —— 前序阶段的对照 playbook
>
> 全文 Shell 是 **PowerShell（Windows 自带）**；所有命令默认在项目根
> `d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\` 下执行，conda 环境
> `qlib_zhengshi`。

---

## 0. 一键冒烟（10 秒，强烈建议第一步跑）

```powershell
conda activate qlib_zhengshi
python -m pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote tests/scripts -q
```

**预期输出**（末行）：

```
275 passed, 1 warning in 5.85s
```

- ✅ `275 passed`：阶段 D 全部契约 / 算法 / 存储 / CLI / 闭环绿灯，可以进后面任何步骤。
- ❌ 若有 FAIL：停下，定位 FAIL 对应的 `tests/...::test_xxx`。通常是你动了：
  - 某个 `factor_validation/checks/*.py` → 先看 §1.2
  - 某个 `factor_validation/profiles/*.yaml` → 先看 §3
  - `factor_lab/exporters/rdagent_log_exporter.py` → 先看 §4

---

## 1. 分层测试矩阵（按需跑）

### 1.1 阶段 D 新增部分

| 想测什么？                                          | 命令                                                                                | 预期耗时    | 覆盖内容                                         |
| ---------------------------------------------- | --------------------------------------------------------------------------------- | ------- | -------------------------------------------- |
| **4 个新 L2 check 全套**                          | `pytest tests/factor_validation -q -k "backtest or marginal"`                     | ~1 s    | 32 例（D.1α 7 + D.1β 8 + D.2A 8 + D.2B 9）     |
| 轻量 backtest（D.1α）                             | `pytest tests/factor_validation/test_backtest_check.py -v`                        | ~0.3 s  | 多空头 Sharpe/DD/win、空集、翻号                     |
| RQAlpha backtest（D.1β）                        | `pytest tests/factor_validation/test_rqalpha_backtest_check.py -v`                | ~0.3 s  | FakeRunner + report.json 解析 + 阈值             |
| marginal 残差 IC（D.2A）                          | `pytest tests/factor_validation/test_marginal_check.py -v`                        | ~0.3 s  | 独立信号 PASS / 冗余信号 FAIL / 数据错误                |
| marginal_training 双 LGB（D.2B）                 | `pytest tests/factor_validation/test_marginal_training_check.py -v`               | ~0.5 s  | uplift>0 PASS / uplift=0 FAIL / 数据不足        |
| L1 RD-Agent 导出器（D.4）                          | `pytest tests/factor_lab/exporters -v`                                            | ~0.5 s  | 22 例：pkl 反射 / h5→parquet / 幂等 / 各种缺失边界      |
| 自动闭环（D.5）                                    | `pytest tests/scripts/lab -v`                                                     | ~0.1 s  | 4 例：happy path / 预筛阻断 / 无效 C1 / main wiring |
| Baseline 预测工具（D.3）                            | `pytest tests/scripts/validate -v`                                                | ~0.2 s  | 10 例：flat / multi-index / synthetic-zero / CLI |

### 1.2 阶段 C 基础部分（D 未变但仍应保持绿）

| 想测什么？                                        | 命令                                         | 预期耗时    |
| -------------------------------------------- | ------------------------------------------ | ------- |
| 阶段 C 4 个 base check                          | `pytest tests/factor_validation -q -k "not backtest and not marginal"` | ~0.5 s  |
| L3 存储 / loader                               | `pytest tests/factor_registry tests/feature -q` | ~0.5 s  |
| Promote / retire CLI                         | `pytest tests/promote -q`                  | ~0.5 s  |

---

## 2. 单独验证 4 个新 L2 check（不需要 RD-Agent、不需要真实回测数据）

> 直接跑单测即可，不用复制代码。每个测试文件都自带 fixture 造最小合成数据，
> 失败时 `-v --tb=short` 一般 30 秒内能看清哪条 assert 断了。

### 2.1 D.1α `backtest_check` —— 理解失败原因

常见 FAIL 模式：

| 日志关键字                                                | 含义                                                   |
| ---------------------------------------------------- | ---------------------------------------------------- |
| `长短组合 Sharpe = -0.xxx < 阈值`                         | 候选方向与 label 反向；若 `allow_negative=true` 会自动翻号重算      |
| `max drawdown = 0.5x > 阈值`                          | 多空组合吃了一段大回撤                                          |
| `win_ratio = 0.4x < 阈值`                             | 日胜率不达标；典型"均值 OK 但稳定性差"                               |
| `quantile long/short 组为空`                           | 候选列在某些交易日 NaN 全光；coverage 也会同时 FAIL，两者互为印证         |

### 2.2 D.1β `backtest_rqalpha_check` —— 只靠单测，不用真装 rqalpha

单测通过 monkeypatch 注入 `FakeRunner`，离线产出 `report.json`；这意味着：

- 你本地**不需要装** rqalpha，也能让 `tests/factor_validation/test_rqalpha_backtest_check.py` 8 例全绿。
- 真正要跑端到端时，需要：装 rqalpha + 配 `config/rqalpha_config.yaml` + 准备 bundle
  数据，详见 §6。

### 2.3 D.2A `marginal_check` —— 为什么冗余信号能被识破

单测 `test_redundant_signal_fails` 构造：

```
label       = 原信号 + 噪声
baseline    = 原信号（强信号本身已经在 baseline 里）
candidate   = baseline + ε          ← 相对 baseline 几乎无增量
```

**算法**：对 candidate_rank 做 baseline_rank 的 OLS 中性化后再求残差 IC。
→ 残差 IC 接近 0、IR 很小 → FAIL。

跑 `-v`：

```powershell
pytest tests/factor_validation/test_marginal_check.py::test_redundant_signal_fails -v
```

能看到 `residual_rank_ic ≈ 0.0xx < 0.008`，说明算法统计上正确。

### 2.4 D.2B `marginal_training_check`

**关键 fixture**：用固定种子造 "baseline 解释 60% 方差、candidate 额外解释 10% 方差"
的合成数据。LGB A vs LGB B → B 的 rank IC 明显高，uplift > `min_ic_uplift`。

若本地 LGB 装得有问题，单测会跳过（通过 `importlib.util.find_spec('lightgbm')` 判定）；
想让它跑，先 `pip install lightgbm`。

---

## 3. Profile 回归（改阈值后必跑）

假设你想把 `default.yaml` 的 `ic.min_rank_ic` 从 0.012 改到 0.010 看看影响：

```powershell
# 1) 改 YAML（不要一次改多个 profile）
# 2) 快速回归 profile 加载 + 聚合逻辑
pytest tests/factor_validation/test_orchestrator.py -v

# 3) 用一个 known-good candidate 跑 validate（借用 C.7 demo candidate，或自己造）
python -m scripts.promote.validate_candidate `
    --candidate-json factor_lab/workspace/candidates/manual_VolumePriceTrend_10D_9f0859e4dc/candidate.json `
    --profile factor_validation/profiles/default.yaml `
    --out-cert tmp_demo/cert_profile_tune.json

# 4) 观察 overall_score 与决议的变化
Get-Content tmp_demo/cert_profile_tune.json | ConvertFrom-Json | Select-Object decision, overall_score
```

> **提醒**：阶段 C 的 `manual_grandfathered` profile 仍然存在，**仅给 legacy 保命**，
> 不要把它用在新因子上。

---

## 4. RD-Agent 导出器端到端（需要本地有 log 目录）

### 4.1 前提

- 项目根有 `log/<run-ts>/Loop_N/...` 结构（RD-Agent 跑过后的产物）；
- 项目根有 `git_ignore_folder/RD-Agent_workspace/<hash>/{factor.py, result.h5}`。

没有这两个 → 跳过 §4，直接用 §5.3 的人造合成数据验证。

### 4.2 扫全部 run（默认幂等）

```powershell
python -m scripts.lab.export_rdagent_candidates --log-level INFO
```

**预期输出**（末几行）：

```
INFO ... 全部完成：runs=N exports=M skipped=K
INFO ...   run=2026-04-20_01-19-42-374811 loops=L exports=... skipped=...
```

- `exports` = 这次新写入 staging 的因子数（首次跑几乎等于 `Loop` 个数）。
- `skipped` = 内容哈希一致，已幂等跳过的数量（重跑时会看到它涨）。

### 4.3 只扫指定 run

```powershell
python -m scripts.lab.export_rdagent_candidates `
    --log-run-dir log/2026-04-20_01-19-42-374811 `
    --summary-json tmp_demo/export_summary.json
```

**预期**：`tmp_demo/export_summary.json` 里每个 loop 一个 `exports` 项，
每项含 `factor_id`、`action=created|unchanged|overwritten`、`target_dir`。

### 4.4 强制覆盖已有 staging

```powershell
python -m scripts.lab.export_rdagent_candidates `
    --log-run-dir log/2026-04-20_01-19-42-374811 `
    --overwrite
```

**什么时候要 `--overwrite`**：你手动删过 `factor_lab/workspace/candidates/<fid>/` 的
部分文件、或改过 exporter 自己、想重建一次。**正常周期跑用法不需要加**。

---

## 5. 端到端：造一个 rdagent 风格的合成候选 → 跑自动闭环（10 分钟）

> 场景：本地没跑过 RD-Agent，但想把"L1 export → L2 exploratory → L2 default →
> L3 promote"的链路跑通一次，保证自动闭环代码在你的机器上可运行。

### 5.1 Step 1：造一个合成 rdagent log tree

```powershell
# 建路径
New-Item -ItemType Directory -Force `
    -Path "tmp_demo/fake_log/2026-04-20_demo-run-12345678/Loop_0/direct_exp_gen/hypothesis generation", `
          "tmp_demo/fake_log/2026-04-20_demo-run-12345678/Loop_0/direct_exp_gen/experiment generation", `
          "tmp_demo/fake_log/2026-04-20_demo-run-12345678/Loop_0/coding/evo_loop_0/evolving code", `
          "tmp_demo/fake_log/2026-04-20_demo-run-12345678/Loop_0/feedback/feedback", `
          "tmp_demo/fake_ws/abcdef0123456789" | Out-Null
```

然后用单测里的 fixture 辅助函数写一段 Python 把 pkl 落进去（直接复用 `tests/factor_lab/exporters/test_rdagent_log_exporter.py` 里的 helpers 最省事）。示例脚本 `tmp_demo/make_fake_rdagent.py`：

```python
"""生成最小合成 rdagent log + workspace 产物，供 Playbook §5 使用。"""
from __future__ import annotations
from pathlib import Path
import pickle, sys
import numpy as np
import pandas as pd

ROOT = Path("tmp_demo/fake_log/2026-04-20_demo-run-12345678/Loop_0")
WS = Path("tmp_demo/fake_ws/abcdef0123456789")

class _FakeHyp:
    hypothesis = "demo slow alpha"
    reason = "playbook synthetic"
    specification = "x_t = x_0 + drift*t"
    score = 0.6

class _FakeTask:
    factor_name = "DemoPlaybookAlpha"
    factor_description = "linear drift factor"
    factor_formulation = "x_t = x_0 + drift*t"
    variables = {"drift": "per-stock"}
    def __init__(self): self.workspace = None

class _FakeFB:
    decision = True
    reason = "looks slow and monotone"
    observations = ""

class _FakeWS:
    def __init__(self, ws_path, task): self.workspace_path, self.target_task = ws_path, task

def _dump(pkl_path: Path, obj):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f: pickle.dump(obj, f)

# 1) Hypothesis / FactorTask / FBWorkspace / Feedback
task = _FakeTask()
ws   = _FakeWS(WS.resolve(), task)
task.workspace = ws
_dump(ROOT / "direct_exp_gen/hypothesis generation" / "000000.pkl", _FakeHyp())
_dump(ROOT / "direct_exp_gen/experiment generation" / "000000.pkl", [task])
_dump(ROOT / "coding/evo_loop_0/evolving code"      / "000000.pkl", [ws])
_dump(ROOT / "feedback/feedback"                    / "000000.pkl", _FakeFB())

# 2) workspace 里的 factor.py + result.h5
WS.mkdir(parents=True, exist_ok=True)
(WS / "factor.py").write_text(
    '"""demo rdagent-like factor stub"""\n'
    'def compute(): return None\n', encoding="utf-8")

dates = pd.date_range("2025-01-02", periods=60, freq="B")
stocks = [f"SH{600000+i:06d}" for i in range(30)]
idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime","instrument"])
rng = np.random.default_rng(7)
init, drift = rng.uniform(-5, 5, 30), rng.uniform(-0.02, 0.02, 30)
t = np.arange(len(dates))
vals = (init[None,:] + drift[None,:]*t[:,None] + rng.normal(0, 0.05, (60, 30))).reshape(-1)
df = pd.DataFrame({"DemoPlaybookAlpha": vals.astype("float64")}, index=idx)
df.to_hdf(WS / "result.h5", key="data", mode="w")

print("fake rdagent tree ready at:", ROOT.parent.resolve())
```

```powershell
python tmp_demo/make_fake_rdagent.py
```

### 5.2 Step 2：用合成 tree 跑导出器

```powershell
python -m scripts.lab.export_rdagent_candidates `
    --log-run-dir tmp_demo/fake_log/2026-04-20_demo-run-12345678 `
    --rdagent-workspace-root tmp_demo/fake_ws `
    --workspace-candidates-dir tmp_demo/candidates `
    --summary-json tmp_demo/export_summary.json
```

**预期**：`tmp_demo/candidates/rdagent_DemoPlaybookAlpha_abcdef01/` 下出现
`factor.py`、`values.parquet`、`c1.json`。`export_summary.json` 里的
`factor_id` 里以 `abcdef01` 结尾（workspace_hash[:8]）。

### 5.3 Step 3：跑自动闭环（dry-like：自定义 workspace 路径，不碰真实 registry）

```powershell
python -m scripts.lab.run_lab_cycle `
    --log-run-dir tmp_demo/fake_log/2026-04-20_demo-run-12345678 `
    --rdagent-workspace-root tmp_demo/fake_ws `
    --workspace-candidates-dir tmp_demo/candidates `
    --report-dir tmp_demo/reports `
    --cert-dir tmp_demo/certs `
    --registry-data-dir tmp_demo/registry_data `
    --registry-parquet-dir tmp_demo/registry_parquet `
    --cycle-id labcycle-playbook-demo
```

**预期**（末行）：

```
... cycle finished: export_candidates=1 exploratory_pass=0 default_pass=0 promoted=0 failed=1
```

`failed=1` **完全正常**：合成因子是随机漂移，过不了 `ic` / `backtest`；但你能看到：

- `tmp_demo/candidates/...` → C1 candidate 成功落盘；
- `tmp_demo/certs/labcycle-playbook-demo/<factor_id>.exploratory.json` → C2 证书落盘
  （HOLD/FAIL 都会产生）；
- `tmp_demo/reports/lab_cycle_labcycle-playbook-demo.json` 与 `.md` → 报告产出。

这就证明自动闭环的**接线**在你的机器上可跑通；真正 PASS 依赖真实因子质量与数据。

### 5.4 Step 4：检查报告是否使用了真换行（避免 D.5 早期 bug 回退）

```powershell
$md = Get-Content tmp_demo/reports/lab_cycle_labcycle-playbook-demo.md -Raw
if ($md -match '`n') { throw "报告里出现了 PowerShell 反引号 n，应为 LF 换行" }
"# ok: markdown newlines are real LF"
```

阶段 D.6 修复了一个把 `` `n `` 当换行符的笔误；回归里这条断言已经加到 `test_run_cycle_happy_path`，
这里是端到端的再保险。

---

## 6. 真跑 RQAlpha 回测（可选；需本地环境就绪）

前提：

- `pip install rqalpha`（及其 bundle 数据）
- `config/rqalpha_config.yaml` 指向本地 bundle
- 一个已落盘的 `factor_registry/parquet/factors_v1.parquet` 列（候选名必须是里面已有
  的列；否则 prediction CSV 构造会缺数据）

```powershell
# 1) 造一份 prediction CSV（如果没有现成的 inference 产出）
python -m scripts.validate.prepare_baseline_prediction `
    --src data/oof/ensemble_v1_20260101_20260407_meta_oof.parquet `
    --prediction-col pred_ensemble `
    --out factor_validation/data/baseline_predictions_ensemble_v1.parquet

# 2) 对一个真实 candidate 跑 strict profile（会真触发 RQAlpha）
python -m scripts.promote.validate_candidate `
    --candidate-json factor_lab/workspace/candidates/manual_VolumePriceTrend_10D_9f0859e4dc/candidate.json `
    --profile factor_validation/profiles/strict.yaml `
    --out-cert tmp_demo/cert_strict_v1.json
```

**预期耗时**：5~15 分钟（RQAlpha 真回测占主导）。

**产物**：`factor_validation/reports/rqalpha/<factor_id>_<ts>/report.json` —— 可以打开看
逐日收益；`tmp_demo/cert_strict_v1.json` 里 `check_results[]` 会有一条
`backtest_rqalpha` 记录，`detail.metrics` 含 `sharpe`、`max_drawdown`、`annual_return`、`total_return`。

**失败排查**：

| 日志                                         | 说明                                              |
| ------------------------------------------ | ----------------------------------------------- |
| `RQAlpha runner not importable`            | rqalpha 没装；或 venv/conda 切错                       |
| `bundle not found at ...`                  | `rqalpha_config.yaml` 里 bundle 路径错；先跑 `rqalpha download-bundle` |
| `prediction CSV 列缺失 score`                  | `score_col` 与 prediction parquet 里实际列名对不上      |

---

## 7. 对着当前 L3 Registry 做观察（无写入、安全）

### 7.1 查看当前 active 因子（D.0 退役后的基线）

```powershell
python -c "from factor_registry.registry import FactorRegistry; r = FactorRegistry('factor_registry/data'); [print(f'{x.factor_id}  status={x.status.value}  v={x.parquet_version}  tags={x.tags}') for x in r.list_all()]"
```

**预期**（D.0 完成后）：两条 `retired` + 三条 `active`；`VolRet_5D` / `VolumeTrend_10D`
在 `retired/` 目录。

### 7.2 过一遍今天的 cycle 报告

```powershell
# 最新一份 lab cycle 报告
Get-ChildItem factor_validation/reports/lab_cycle_*.md -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName -Raw }
```

---

## 8. "我改了 XXX，担心出问题"TL;DR

| 改了什么                                                   | 最小回归集                                                   | 加分项                                                                                   |
| ------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `factor_validation/checks/backtest_check.py`           | `pytest tests/factor_validation -v -k "backtest and not rqalpha"` | 用 §5.3 合成 candidate 跑一次轻量回测，肉眼比对 Sharpe/DD                                          |
| `factor_validation/checks/rqalpha_backtest_check.py`   | `pytest tests/factor_validation/test_rqalpha_backtest_check.py -v` | §6 端到端真跑一次 strict 验证                                                                 |
| `factor_validation/checks/marginal_check.py`           | `pytest tests/factor_validation/test_marginal_check.py -v`        | 拿真数据 + 真 baseline 预测跑一次 `default` validate，观察 `residual_rank_ic`                       |
| `factor_validation/checks/marginal_training_check.py`  | `pytest tests/factor_validation/test_marginal_training_check.py -v` | 装了 lightgbm 再跑一次 strict 证书观察 `ic_uplift`                                              |
| `factor_validation/profiles/*.yaml`                    | `pytest tests/factor_validation/test_orchestrator.py -v`          | 对一个 known-good candidate 跑 validate，确认决议没意外翻转                                         |
| `factor_lab/exporters/rdagent_log_exporter.py`         | `pytest tests/factor_lab/exporters -v`                            | §4.3 真扫一个 run 看 `exports/skipped` 变化                                                  |
| `scripts/lab/run_lab_cycle.py`                         | `pytest tests/scripts/lab -v`                                     | §5 端到端 + 检查 `.md` 报告换行                                                                |
| `scripts/lab/run_lab_cycle.ps1`                        | 手动 Start-ScheduledTask 跑一次，确认日志落盘                         | Task Scheduler 首次配置后必须演练一次                                                            |
| `scripts/validate/prepare_baseline_prediction.py`      | `pytest tests/scripts/validate -v`                                | 换真数据再导一次，对 `marginal` 跑一次 validate 看是否对齐                                              |

---

## 9. 常见故障与排查

### 9.1 `marginal` check 报 `baseline_prediction_parquet 缺失`

profile 里启用了 `marginal` 但 `data_sources.baseline_prediction_parquet` 没指，或文件不存在。
修：

```powershell
python -m scripts.validate.prepare_baseline_prediction `
    --src data/oof/ensemble_v1_20260101_20260407_meta_oof.parquet `
    --prediction-col pred_ensemble `
    --out factor_validation/data/baseline_predictions_ensemble_v1.parquet
```

如果当前 production ensemble 尚未跑出 OOS 预测，可以用**占位 zero baseline**先跑通：

```powershell
python -m scripts.validate.prepare_baseline_prediction `
    --synthetic-zero `
    --label-parquet factor_validation/data/oos_labels_all.parquet `
    --out factor_validation/data/baseline_predictions_zero.parquet
```

然后临时把 profile 里 `baseline_prediction_parquet` 指向它，跑完再切回来。
**不要**长期用 zero baseline，那会把 marginal 退化成 IC 的重复打分。

### 9.2 `marginal_training` 跳过 / 失败：`lightgbm not available`

本地缺 lightgbm。想跑这条：

```powershell
pip install lightgbm
```

不想装但又要临时关：在 strict 里把 `checks.marginal_training.enabled: false`
临时改为 `false`（跑完改回来）。

### 9.3 自动闭环 `promote_exit_code=3`（parquet 列缺失）

自动闭环只写 manifest、不写 parquet；新因子要先把 parquet 列 merge 进
`factor_registry/parquet/factors_v<N>.parquet`。当前阶段 D 还没自动合并新列，
人工路径：

```powershell
# 1) 人工 merge：读当前 factors_v1.parquet + 候选 values.parquet + 合并写 v2
# （阶段 D.5 后续 / 阶段 E 计划把这步也自动化；目前人工挑选入库）
```

在流程完善前，**自动闭环 promote 失败**是**预期的**，不代表链路有 bug；
只表明"L1→L2→manifest" 通畅，"parquet 列合并" 仍待人工。

### 9.4 `Get-ChildItem` 展开 RD-Agent log 卡死

`log/` 目录会膨胀到几万文件。**不要**用 `-Recurse`；用 `-Depth 1` 分层看：

```powershell
Get-ChildItem log -Depth 1 | Select-Object FullName | Select-Object -First 20
```

### 9.5 Task Scheduler 任务执行但无日志

`run_lab_cycle.ps1` 强制把 stdout/stderr Tee 到 log 文件。如果没日志，检查：

- 任务的"起始位置"是否是项目根（或 PS1 里 `Set-Location $ProjectRoot` 已处理）；
- `factor_validation/reports/scheduler-logs/` 目录是否有写权限（LocalSystem 通常没问题，但非管理员账户要 grant）。

---

## 10. 退出前清理（可选）

Playbook §5 的 demo 产物都在 `tmp_demo/`：

```powershell
Remove-Item -Recurse -Force tmp_demo
```

**绝对不要**手动删：

- `factor_lab/workspace/candidates/*`（生产 staging）
- `factor_validation/certificates/*`（历史证书）
- `factor_validation/reports/*`（cycle 报告；归档策略另定）
- `factor_registry/data/manifest.json`（L3 注册表）

如果不小心动了，直接 `git checkout -- <path>` 恢复（上面这几个目录都在 git 管理下）。
