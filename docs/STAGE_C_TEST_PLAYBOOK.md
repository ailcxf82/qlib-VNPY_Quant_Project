# 阶段 C 测试 Playbook（复制即跑版）

> 目的：**让你不用读代码就能把 L2 验证 + Promote 流水线的所有关键点亲自测一遍**。
> 每一步都给出：命令、预期输出关键字、预期耗时、出错时怎么办。
>
> 姊妹篇：
>
> - `docs/STAGE_C_TEST_GUIDE.md` —— 架构/接口级指南
> - `docs/STAGE_C_REPORT.md` —— 阶段 C 交付与 oracle 结论
>
> 全文用的 Shell 是 **PowerShell（Windows 自带）**；所有命令默认在项目根
> `d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\` 下执行，conda 环境
> `qlib_zhengshi`（已在 PATH 中则 `conda activate qlib_zhengshi` 可省）。

---

## 0. 一键冒烟（30 秒，强烈建议第一步跑）

```powershell
conda activate qlib_zhengshi
python -m pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote -q
```

**预期输出**（关键行）：

```
....................................................................... [..%]
...
207 passed in 2.84s
```

- ✅ `207 passed`：C 阶段所有契约 / 算法 / 存储 / CLI 全绿，你可以放心做后面的任何操作。
- ❌ 若有 FAIL：直接停下，打开 FAIL 对应的 `tests/...::test_xxx` 看是谁动了；通常是
你之前改过 `factor_validation/checks/*.py` 或 profile 阈值。

---

## 1. 分层测试矩阵（按需跑）


| 想测什么？                                | 命令                                                         | 预期耗时   | 覆盖内容                                                         |
| ------------------------------------ | ---------------------------------------------------------- | ------ | ------------------------------------------------------------ |
| **所有 4 个 L2 检查**                     | `pytest tests/factor_validation -v`                        | ~0.5 s | coverage / ic / orthogonality / turnover + orchestrator      |
| 只测 turnover（C.6 新算法）                 | `pytest tests/factor_validation/test_turnover_check.py -v` | ~0.2 s | 慢/随机/反转 3 型因子 + 阈值/空窗异常                                      |
| L3 存储（parquet / registry / loader）   | `pytest tests/factor_registry -v`                          | ~0.5 s | `ParquetStore` / `FactorRegistry` / `ProductionFactorLoader` |
| Feature pipeline 桥接（L3 vs legacy 等价） | `pytest tests/feature -v`                                  | ~0.3 s | `_maybe_merge_rdagent_parquet` 7 例                           |
| **三个 CLI（验证 / 入库 / 退役）**             | `pytest tests/promote -v`                                  | ~0.5 s | 19 例 CLI 测试                                                  |
| 契约 schema 层（Pydantic）                | `pytest tests/factor_lab -v`                               | ~0.3 s | C1 / C2 schema 强校验                                           |


**小贴士**：任何一项 `-v` 下都能看到每个测例的名字，失败时用 `--tb=short` 可以只看
核心堆栈，用 `-x` 可以让 pytest 遇到第一个 fail 就停。

---

## 2. 端到端：手动造一个因子并走完 L2 → L3（10 分钟）

> 场景：你写了一个因子，想看它能不能通过 `default` profile；通过了就进 L3；不通过就
> 看看差在哪。这一节全部**在 `tmp/` 下演示，不会污染真实 registry**。

### 2.1 Step 1：准备一个 candidate parquet（纯脚本，无 qlib 依赖）

创建临时脚本 `tmp_demo/make_cand.py`：

```python
# tmp_demo/make_cand.py
from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(exist_ok=True)

# 造一个"缓慢漂移"慢因子：30 只票 × 80 个交易日
dates = pd.date_range("2024-06-03", periods=80, freq="B")
stocks = [f"SH{600000 + i:06d}" for i in range(30)]
idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])

rng = np.random.default_rng(42)
init = rng.uniform(-5.0, 5.0, len(stocks))
drift = rng.uniform(-0.02, 0.02, len(stocks))
t = np.arange(len(dates))
vals = (init[None, :] + drift[None, :] * t[:, None]
        + rng.normal(0, 0.05, (len(dates), len(stocks)))).reshape(-1)

df = pd.DataFrame({"DemoSlowAlpha": vals.astype("float64")}, index=idx)
values_path = ROOT / "values.parquet"
df.to_parquet(values_path)

code_path = ROOT / "factor.py"
code_path.write_text('"""demo stub"""\nLEGACY = False\n', encoding="utf-8")

# 打印 candidate JSON（后续 CLI 会用到）
from factor_lab.exporters.schema import CandidateFactorPackage
cand = CandidateFactorPackage(
    factor_id="manual_DemoSlowAlpha_abcdef0123",
    name="DemoSlowAlpha",
    source="manual",
    hypothesis="线性漂移 + 小噪声，应在 default profile 下通过 4 个 check",
    formulation="x_t = x_0 + drift * t + eps",
    code_path=code_path,
    values_path=values_path,
    universe="csi300",
    date_range=(date(2024, 6, 3), date(2024, 10, 1)),
    lab_metrics={"non_null_ratio": 1.0},
    parent_loop=None,
    created_at=datetime.now(timezone.utc),
    lab_run_id="demo-run-20260420-xxxxxxxx",
)
(ROOT / "candidate.json").write_text(cand.model_dump_json(indent=2), encoding="utf-8")
print("artifacts ready at:", ROOT.resolve())
```

运行：

```powershell
python tmp_demo/make_cand.py
```

**预期**：`tmp_demo/` 下出现 `values.parquet` / `factor.py` / `candidate.json`。

### 2.2 Step 2：用 `manual_grandfathered` profile 跑验证（不需要 OOS label）

```powershell
python -m scripts.promote.validate_candidate `
    --candidate-json tmp_demo/candidate.json `
    --profile factor_validation/profiles/manual_grandfathered.yaml `
    --out-cert tmp_demo/cert.json
```

**预期输出（末尾）**：

```
factor_id      : manual_DemoSlowAlpha_abcdef0123
profile        : manual_grandfathered (hash=XXXXXXXX…)
decision       : PASS
overall_score  : 1.0000
checks:
  [OK] coverage       score= 1.000  passed=True
```

**返回码**：`$LASTEXITCODE` → `0`（PASS）。你也能在 `tmp_demo/cert.json` 看到完整的 C2
证书 JSON（可直接打开查看）。

### 2.3 Step 3：挑战更严格的 `default` profile

```powershell
python -m scripts.promote.validate_candidate `
    --candidate-json tmp_demo/candidate.json `
    --profile factor_validation/profiles/default.yaml `
    --out-cert tmp_demo/cert_default.json
```

**预期**：几乎一定 **FAIL**（返回码 `3`）。理由：

- demo 因子的值域是随机噪声，IC 不会与真实 forward return 对齐 → `ic` FAIL；
- 另外 30 只票 × 80 天的规模太小，orthogonality 与真实 L3 参照集对不齐。

这是**正常现象**；它证明你的 default profile 对"凑数因子"有防守能力。

### 2.4 Step 4：（示意）把 PASS 证书 promote 到 L3（⚠️ 真实 registry 操作，看完演示即停）

```powershell
# 仅展示命令——不要在真实 registry 上执行 DemoSlowAlpha，因为 factors_v1.parquet 不含这列！
# 真实用法：先把你的 parquet 列合并进 factors_v<N>.parquet，再 promote。
python -m scripts.promote.promote_certified `
    --cert tmp_demo/cert.json `
    --parquet-version 1 `
    --tags demo
```

**这一步预期会失败** —— 错误码 `3`（EXIT_PARQUET_MISSING），错误信息类似：

```
ERROR ... promote_certified | parquet 物理校验失败: parquet factors_v1.parquet 不含列 'DemoSlowAlpha'
```

这说明 `promote_certified` 的**护栏工作正常**：不让你把"证书说有、但物理 parquet 没有"
的因子混入 L3。

---

## 3. 对着真实 L3 Registry 做观察（无写入、安全）

### 3.1 查看当前 active 因子

```powershell
python -c "from factor_registry.registry import FactorRegistry; r = FactorRegistry('factor_registry/data'); [print(f'{x.factor_id}  status={x.status.value}  v={x.parquet_version}  tags={x.tags}') for x in r.list_all()]"
```

**预期输出（C.8 时点）**：

```
manual_RangeRatio_10D_bb147b539a        status=active  v=1  tags=['legacy', 'grandfathered']
manual_VolRatio_20D_7439285b70          status=active  v=1  tags=['legacy', 'grandfathered']
manual_VolRet_5D_11d70180fb             status=active  v=1  tags=['legacy', 'grandfathered']
manual_VolumePriceTrend_10D_9f0859e4dc  status=active  v=1  tags=['legacy', 'grandfathered']
manual_VolumeTrend_10D_5142b584ee       status=active  v=1  tags=['legacy', 'grandfathered']
```

### 3.2 看 L3 parquet 头（确认列名 & 索引结构）

```powershell
python -c "import pandas as pd; df = pd.read_parquet('factor_registry/parquet/factors_v1.parquet'); print('shape=', df.shape); print('cols=', list(df.columns)); print(df.head(3))"
```

**预期**：

```
shape= (1150046, 5)
cols= ['RangeRatio_10D', 'VolumePriceTrend_10D', 'VolRatio_20D', 'VolumeTrend_10D', 'VolRet_5D']
                              RangeRatio_10D  ...    VolRet_5D
datetime   instrument                         ...
...
```

### 3.3 对 5 个 legacy 因子跑 Oracle 复测（推荐，耗时 ~20 s）

```powershell
python -m scripts.promote.oracle_revalidate_legacy `
    --profile factor_validation/profiles/default.yaml `
    --write-report factor_validation/reports/legacy_oracle_$(Get-Date -Format yyyyMMdd-HHmm).md
```

**预期**：

1. 控制台 5 条 `INFO ... decision=XXX overall=0.xxxx` 行，中文可能因 GBK 乱码（不影响功能）。
2. 末尾 `INFO ... 报告写入 factor_validation\reports\legacy_oracle_YYYYMMDD-HHMM.md`。
3. 打开那个 .md 文件（UTF-8，不会乱码）看完整报告。
4. 返回码 `1`（因为仍有 FAIL，这是**正常的**—— 阶段 C 结论如此）。

**对比 C.6 之前**：你会看到 `turnover` 一列全部从 `[X] 0.03` 这种 bug 值变成了
`[OK] 0.8~0.9`（慢因子）或 `[X] 0.6` （确实偏快的 RangeRatio_10D），说明算法修复
生效。

### 3.4 （**不会修改 registry**）演练批量退役

```powershell
# 建 ids 文件
@"
# 阶段 C 退役候选（决策后启用）
manual_VolRet_5D_11d70180fb
"@ | Set-Content -Encoding utf8 tmp_demo/to_retire.txt

# --dry-run 不落盘
python -m scripts.promote.retire_factor `
    --ids-file tmp_demo/to_retire.txt `
    --reason "oracle default FAIL: rank_ic=0.006, ic_ir=0.024（真·低质）" `
    --dry-run
```

**预期**（关键输出）：

```
INFO retire_factor | dry-run manual_VolRet_5D_11d70180fb (status=active)
INFO retire_factor | retire: 完成 1/1，失败 0
```

执行完后再看一次 §3.1 的列表 —— `VolRet_5D` 仍然是 `active`（dry-run 不落盘）。

### 3.5 （可选真退役，需你明确许可）

**⚠️ 这一步会真正修改 `factor_registry/data/manifest.json`，但 parquet 物理文件不动。**
只有你在 `STAGE_C_REPORT.md §3.3` 里选定 A/B/C 方案后再执行：

```powershell
python -m scripts.promote.retire_factor `
    --ids-file tmp_demo/to_retire.txt `
    --reason "方案 X 决策：oracle default FAIL"
```

回退方法：直接编辑 `manifest.json` 把对应 `status` 从 `retired` 改回 `active`（并把
`data/retired/<fid>.json` 搬回 `data/certified/<fid>.json`）；或者 `git checkout` 掉
这次改动（manifest.json 在 git 管理下）。

---

## 4. 针对"我改了 XXX，担心出问题"的 TL;DR


| 你改了什么                                       | 最小回归集                                                | 加分项                            |
| ------------------------------------------- | ---------------------------------------------------- | ------------------------------ |
| `factor_validation/checks/xxx_check.py`     | `pytest tests/factor_validation -v`                  | 改算法则跑 §3.3 oracle 对比新老报告       |
| `factor_validation/profiles/*.yaml`         | `pytest tests/factor_validation -v`                  | 必跑 §3.3 oracle，看决议是否意外翻转       |
| `factor_registry/registry.py` or `store.py` | `pytest tests/factor_registry tests/promote -v`      | §3.1 + §3.2 人工看一眼              |
| `feature/qlib_feature_pipeline.py`          | `pytest tests/feature -v`                            | 必要时跑一次 `run_train.py` smoke 训练 |
| `scripts/promote/*.py`                      | `pytest tests/promote -v`                            | §2 端到端演练一遍                     |
| Pydantic schema（C1 / C2）                    | `pytest tests/factor_lab tests/factor_validation -v` | 必然影响 CLI / migrate，跑 §2        |


---

## 5. 常见故障与排查

### 5.1 `EnvironmentNameNotFound` / `No module named pytest`

你没有 `conda activate qlib_zhengshi`。确认：

```powershell
conda info --envs
# 应该看到：qlib_zhengshi   C:\ProgramData\miniconda3\envs\qlib_zhengshi
conda activate qlib_zhengshi
python -c "import pytest, qlib, pydantic; print('OK')"
```

### 5.2 Oracle 控制台输出中文乱码（`[4444:MainThread]`、方块）

Windows GBK stdout 的固有问题。解决：

- **始终给 `--write-report <path>`**：报告文件本身是 UTF-8，打开能看到正常中文；
- 或把 PowerShell 改为 UTF-8：`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
（仅当前 session 有效）。

### 5.3 Oracle 报错：`turnover: OOS 窗内无样本`

你传的 profile 里 `oos_window` 与因子 parquet 的 `date_range` **没有交集**。
`default.yaml` 的窗是 `["2025-01-01", "2026-04-07"]`，需要 parquet 在 2025 之后有数据。

### 5.4 `validate_candidate` 抛 `values.parquet 实际时间窗 ... 超出声明 date_range`

你写 candidate JSON 时 `date_range` 填得比 parquet 实际时间窗小了。把 `date_range`
扩到覆盖整段时间重跑。

### 5.5 `promote_certified` 一直返回 `EXIT_PARQUET_MISSING`

`factor_registry/parquet/factors_v<N>.parquet` 没有你证书里的 `candidate.name` 列。
两种解法：

- **迁入现成数据**：把你的 parquet 列合并进 `factors_v<N>.parquet`，再 promote；
- **开新版本**：用 `ParquetStore.write_version(df, N+1)` 写 `factors_v(N+1).parquet`，
再 `--parquet-version (N+1)` promote。

### 5.6 `retire_factor` 报 `factor_id 不存在`

你的 factor_id 输错了（注意末尾的 10 位 hex 区分大小写敏感）。回到 §3.1 把精确的
`factor_id` 复制下来。

---

## 6. 一张图：阶段 C 流水线 + 你能在哪儿下断点

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      阶段 C  L1 → L3 四位一体                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L1 workspace                                                           │
│  (candidate.json + values.parquet + factor.py)                          │
│          │                                                              │
│          │  ①  python -m scripts.promote.validate_candidate …           │
│          │      → cert.json (CertifiedFactorRecord C2)                  │
│          ▼                                                              │
│  L2 证书 (data dir: factor_validation/certificates/)                    │
│          │                                                              │
│          │  ②  python -m scripts.promote.promote_certified …            │
│          │      (前置：factors_v<N>.parquet 已含 name 列)                 │
│          ▼                                                              │
│  L3 manifest + certified/<fid>.json                                     │
│  + factor_registry/parquet/factors_v<N>.parquet                         │
│          │                                                              │
│          │  ③  生产侧 feature/qlib_feature_pipeline.py                   │
│          │      → ProductionFactorLoader(only_active=True).load()       │
│          ▼                                                              │
│  run_train.py / ensemble models                                         │
│          │                                                              │
│          │  ④  周期性 oracle_revalidate_legacy against 更严 profile      │
│          │      → reports/*.md；若 FAIL 太多：                           │
│          │         python -m scripts.promote.retire_factor …            │
│          ▼                                                              │
│  L3 retired（manifest.status=retired，parquet 保留历史）                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
        测试入口在哪里？
        ├─ 所有 ① 分支 → tests/promote/test_validate_candidate_cli.py  （6 例）
        ├─ 所有 ② 分支 → tests/promote/test_promote_certified_cli.py   （7 例）
        ├─ 所有 ③ 分支 → tests/feature/test_pipeline_registry_integration.py（7 例）
        ├─ 所有 ④ 分支 → tests/promote/test_retire_factor_cli.py       （6 例）
        └─ 每个 check → tests/factor_validation/test_<check>_check.py
```

---

## 7. 附：最常用的 5 条命令速查

```powershell
# A. 一键冒烟
python -m pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote -q

# B. 只跑 CLI 三件套（最快验证 promote 路径没坏）
python -m pytest tests/promote -v

# C. 看当前 registry 状态
python -c "from factor_registry.registry import FactorRegistry; r=FactorRegistry('factor_registry/data'); [print(x.factor_id, x.status.value, x.tags) for x in r.list_all()]"

# D. 跑一次 oracle 复测并落盘
python -m scripts.promote.oracle_revalidate_legacy --profile factor_validation/profiles/default.yaml --write-report factor_validation/reports/legacy_oracle_$(Get-Date -Format yyyyMMdd-HHmm).md

# E. 对某个因子执行 dry-run 退役（看会退役啥，不落盘）
python -m scripts.promote.retire_factor --factor-id manual_VolRet_5D_11d70180fb --reason "probe" --dry-run
```

