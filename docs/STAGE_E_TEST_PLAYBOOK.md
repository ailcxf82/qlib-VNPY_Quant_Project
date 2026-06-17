# 阶段 E 测试 Playbook（复制即跑版）

> 目的：**让你不用读代码就能把阶段 E 交付的反馈回路、RAG 注入、目录重构、cycle 钩子
> 亲自跑一遍**。每一步都给出：命令、预期输出关键字、预期耗时、出错怎么办。
>
> 姊妹篇：
>
> - `docs/STAGE_E_TEST_GUIDE.md` —— 架构 / 接口级指南
> - `docs/STAGE_E_REPORT.md` —— 阶段 E 交付与结论
> - `docs/STAGE_D_TEST_PLAYBOOK.md` —— 前序阶段对照 playbook
>
> 全文 Shell 是 **PowerShell（Windows 自带）**；所有命令默认在项目根
> `d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\` 下执行，conda 环境
> `qlib_zhengshi`。

---

## 0. 一键冒烟（10 秒，强烈建议第一步跑）

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

**预期输出**（末行）：

```
334 passed, 7 warnings in 6.xx s
```

- ✅ `334 passed`：阶段 E 全部契约 / 聚合器 / RAG / shim / cycle 钩子绿灯。
- ❌ 若 FAIL：停下，定位 FAIL 对应的 `tests/...::test_xxx`。通常是：
  - `tests/factor_lab/feedback/...` → 契约或聚合器坏了，去 §2
  - `tests/rdagent_integration/...` → RAG 组合器坏了，去 §3
  - `tests/factor_lab/adapters/...` → 目录 shim 坏了，去 §4
  - `tests/scripts/lab/test_run_lab_cycle_feedback.py` → cycle 钩子坏了，去 §5
- 被忽略的 `tests/test_unified_strategy.py` 是 pre-existing 数据形状问题，与阶段 E 无关
  （见 `STAGE_E_REPORT.md` §6）。

---

## 1. 分层测试矩阵

### 1.1 阶段 E 新增部分

| 想测什么？ | 命令 | 预期耗时 | 覆盖内容 |
| --- | --- | --- | --- |
| **C3 契约** | `pytest tests/factor_lab/feedback -k schema -v` | ~0.3 s | 11 例：frozen/validation/markdown |
| 聚合器 | `pytest tests/factor_lab/feedback -k aggregator -v` | ~0.3 s | 15 例：build_feedback_bundle 分支 + classify_family |
| 聚合器 CLI | `pytest tests/scripts/lab/test_build_feedback_bundle_cli.py -v` | ~0.2 s | 2 例：main() smoke + exit code |
| 静态+动态 RAG | `pytest tests/rdagent_integration -v` | ~0.3 s | 12 例：compose_project_rag 所有分支 |
| 目录 shim | `pytest tests/factor_lab/adapters -v` | ~0.3 s | 12 例：新家可 import + 老路径 DeprecationWarning + 符号 is 相等 |
| cycle 钩子 | `pytest tests/scripts/lab/test_run_lab_cycle_feedback.py -v` | ~0.3 s | 4 例：默认生成 / --skip / 聚合器异常 / CLI 透传 |

### 1.2 阶段 D 基础部分（E 未触但仍应保持绿）

| 想测什么？ | 命令 | 预期耗时 |
| --- | --- | --- |
| 阶段 D 4 个新 L2 check | `pytest tests/factor_validation -q -k "backtest or marginal"` | ~1 s |
| 阶段 D 自动闭环（老 4 条） | `pytest tests/scripts/lab/test_run_lab_cycle.py -v` | ~0.2 s |
| 阶段 D RD-Agent 导出器 | `pytest tests/factor_lab/exporters -v` | ~0.5 s |

---

## 2. C3 FeedbackBundle 契约（E.1）

### 2.1 手工构造一个 bundle（验证 schema 行为）

```powershell
python - <<'PY'
from datetime import datetime, timezone
from factor_lab.feedback import (
    FeedbackBundle, ActiveFactorSummary, RetiredFactorSummary, FailedCandidateSummary,
)

bundle = FeedbackBundle(
    schema_version="1.0",
    generated_at=datetime.now(timezone.utc),
    cycles_included=("cycle_0001", "cycle_0002"),
    window_max_cycles=8,
    active_factors=(
        ActiveFactorSummary(factor_id="f1", name="quality_alpha_v1", family="quality_persist"),
    ),
    retired_factors=(
        RetiredFactorSummary(
            factor_id="r1", name="vol_rev_old", family="volume_price_reversal",
            retired_on="2026-03-10", reason="rolling IC degraded",
        ),
    ),
    recent_fails=(
        FailedCandidateSummary(
            factor_id="c7", name="vol_rev_beta", family="volume_price_reversal",
            cycle_id="cycle_0002", stage="default", decision="FAIL",
            failure_modes=("ic", "orthogonality"),
            rank_ic=0.004, max_abs_corr=0.82,
        ),
    ),
    failure_family_counts={"volume_price_reversal": 3, "unknown": 1},
    discouraged_families=("volume_price_reversal",),
)
print(bundle.to_markdown())
PY
```

**预期**：stdout 打出完整 RAG 片段；任何章节缺失都说明 schema 被动过。

### 2.2 故意制造非法状态

```powershell
python - <<'PY'
from factor_lab.feedback import FeedbackBundle, FailedCandidateSummary
from datetime import datetime, timezone
from pydantic import ValidationError

try:
    FeedbackBundle(
        schema_version="1.0",
        generated_at=datetime.now(timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=1,
        recent_fails=(FailedCandidateSummary(
            factor_id="x", name="x", family="unknown",
            cycle_id="c_ORPHAN", stage="default", decision="FAIL",
        ),),
    )
except ValidationError as e:
    print("OK, rejected orphan:", e.errors()[0]["msg"])
PY
```

**预期**：`OK, rejected orphan: ...cycle_id ... 不在 cycles_included 中`。

---

## 3. 聚合器 + CLI（E.2）

### 3.1 最轻量：对现有 reports/ 跑一次 CLI

```powershell
python -m scripts.lab.build_feedback_bundle --log-level INFO
```

**预期关键字**（stderr / stdout 至少一条）：

```
[build_feedback_bundle] cycles_included=... discouraged=... wrote latest.json
```

产物：

- `factor_lab/workspace/feedback/latest.json`
- `factor_lab/workspace/feedback/latest.md`
- `factor_lab/workspace/feedback/history/<last_cycle_id>.json`（若无 `--no-history`）

### 3.2 隔离目录跑（不污染真实 workspace）

```powershell
$ws = "D:\tmp\feedback_smoke"; if (Test-Path $ws) { Remove-Item -Recurse -Force $ws }
python -m scripts.lab.build_feedback_bundle `
    --feedback-dir $ws `
    --max-cycles 2 `
    --notes "smoke from playbook"
Get-Content "$ws\latest.md" | Select-Object -First 40
```

**预期**：前 40 行是 markdown 片段；顶部有 `generated_at = ... cycles_included = ...`。

### 3.3 如果 `reports/` 里还没有任何 lab_cycle_*.json

- CLI 仍会成功退出（returncode 0），只是 bundle 里 `cycles_included=()`、
  `recent_fails=()`；markdown 只有 "Interpretation rules for LLM" 兜底段。
- 这一行为是刻意的 —— 阶段 E 设计目标"静默降级、不阻塞"。

---

## 4. 静态 + 动态 RAG 拼接（E.3 / E.4）

### 4.1 纯函数层面验证

```powershell
python - <<'PY'
from factor_lab.adapters.quant_proposal import compose_project_rag, _STATIC_CONSTITUTION

rag_empty = compose_project_rag(base_rag="BASE_RAG", feedback_dir=None)
assert "BASE_RAG" in rag_empty
assert _STATIC_CONSTITUTION.strip() in rag_empty
print("OK: static constitution appended")
print("first 400 chars ↓↓↓")
print(rag_empty[:400])
PY
```

**预期**：stdout 首 400 字符里能看到 "ProjectQlib RAG constitution"
/ "encouraged" / "discouraged" 关键字。

### 4.2 把刚才 §3.2 生成的 bundle 喂给 RAG

```powershell
python - <<'PY'
from pathlib import Path
from factor_lab.adapters.quant_proposal import compose_project_rag

rag = compose_project_rag(
    base_rag="BASE",
    feedback_dir=Path(r"D:\tmp\feedback_smoke"),
)
print(rag[-600:])   # 尾部 600 字一般就是动态段
PY
```

**预期**：尾部能看到 `------Feedback from recent L2 cycles (dynamic)------`
这一行；整段可以直接送给 LLM。

### 4.3 动态段加载失败也不崩

```powershell
python - <<'PY'
from pathlib import Path
from factor_lab.adapters.quant_proposal import compose_project_rag

rag = compose_project_rag(base_rag="BASE", feedback_dir=Path(r"D:\nowhere_at_all"))
assert "Feedback from recent L2 cycles" not in rag     # 动态段被静默省略
assert "BASE" in rag
print("OK: graceful degradation")
PY
```

**预期**：`OK: graceful degradation`，不抛异常。若抛异常说明 `_render_dynamic_feedback`
的 try/except 被改坏了。

### 4.4 旧路径 DeprecationWarning

```powershell
python -W error::DeprecationWarning - <<'PY'
try:
    from rdagent_integration.project_quant_proposal import compose_project_rag  # noqa
except DeprecationWarning as exc:
    print("OK, deprecated:", exc)
PY
```

**预期**：能收到一条 DeprecationWarning，内容指向 `factor_lab.adapters.quant_proposal`。
如果没 warning，说明 shim 被改掉了。

---

## 5. cycle 钩子（E.5）

### 5.1 单测层面快速验证

```powershell
pytest tests/scripts/lab/test_run_lab_cycle_feedback.py -v
```

**预期**：4 个用例全绿；每条 <0.1 s。

### 5.2 真实 cycle 级验证（需要 RD-Agent 环境）

仅当你已经配好 RD-Agent、有能跑通的 cycle 时执行。**否则跳过本节**。

```powershell
# 干净 feedback 目录
$fb = "D:\tmp\lab_cycle_feedback"; if (Test-Path $fb) { Remove-Item -Recurse -Force $fb }

python -m scripts.lab.run_lab_cycle `
    --source rdagent_live `
    --feedback-dir $fb `
    --feedback-max-cycles 4 `
    --log-level INFO
```

**预期产物**：

- `factor_validation/reports/lab_cycle_<id>.json` —— 末尾必含 `"feedback": {...}`
- `$fb/latest.json` + `$fb/latest.md` —— 由钩子写入
- `$fb/history/<cycle_id>.json` —— 本轮历史副本

### 5.3 明确跳过 feedback 重建

```powershell
python -m scripts.lab.run_lab_cycle --skip-feedback-rebuild --source rdagent_live
```

**预期**：`$fb` 目录不变；cycle 报告里 `"feedback": {"bundle_path": null, "cycles_included": 0, ...}`
且 `"error": null`（因为是"主动跳过"而不是"失败"）。

---

## 6. 把新 bundle 喂给 RD-Agent，验证 RAG 真的传进了 LLM（手工）

> 这是阶段 E 的**最终验证**，但依赖真实 RD-Agent 运行环境。保守做法是先看
> log 里拼出来的 prompt，不用真花 token。

```powershell
# 强制把 RD-Agent 的 prompt 留档（启动前）
$env:RDAGENT_LOG_DUMP_PROMPT = "1"

python -m scripts.lab.run_rdagent_loop `
    --competition QlibQuantCompetition `
    --max-loop 1 `
    --log-level DEBUG 2>&1 | Tee-Object -FilePath "D:\tmp\rdagent_prompt_dump.log"
```

**在 log 里搜索**：

```powershell
Select-String -Path "D:\tmp\rdagent_prompt_dump.log" `
    -Pattern "Feedback from recent L2 cycles|Discouraged families|ProjectQlib RAG constitution"
```

**预期**：每条关键字至少命中一次；若 "Feedback from recent L2 cycles" 没命中，
说明 `factor_lab/workspace/feedback/latest.json` 还没有（先跑 §3.1）。

---

## 7. 排错表

| 现象 | 首要排查 |
| --- | --- |
| `pytest tests/factor_lab/feedback` 报 `pydantic.ValidationError` | 你最近改了 schema？diff 对比 `factor_lab/feedback/schema.py` 中 `@field_validator` |
| `compose_project_rag` 丢了静态段 | `_STATIC_CONSTITUTION` 是否被误删/更名；snapshot 测试 `test_static_constitution_snapshot_stable` 会抓 |
| `scripts.lab.build_feedback_bundle` 报 `ModuleNotFoundError` | 当前工作目录不在项目根；或 `conda activate qlib_zhengshi` 忘了 |
| DeprecationWarning 都没发 | 测试里可能用了 `warnings.filterwarnings("ignore", ...)`；去掉或用 `pytest.warns(DeprecationWarning)` |
| cycle 结束 `feedback.error` 是 `PermissionError` | `--feedback-dir` 指向了只读目录；或 Windows 下被别的进程占用 |
| RD-Agent 跑出来的 hypothesis 依然提 discouraged 家族 | 正常；阶段 E 只保证 bundle **注入** prompt，模型遵循度由阶段 F.4 的 E2E 冒烟验证 |

---

## 8. 完工口径（一键）

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

看到 `334 passed` 即阶段 E 完工；同时建议手工跑一次 §3.1 + §4.2，确认 RAG 端到端
可视化正常。
