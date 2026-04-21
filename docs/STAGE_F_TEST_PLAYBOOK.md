# 阶段 F 测试 Playbook（复制即跑版）

> 目的：**让你不用读代码就能把阶段 F 交付的 YAML 配置化、universe 字段、E2E 闭环
> 亲自跑一遍**。每一步都给出：命令、预期输出关键字、预期耗时、出错怎么办。
>
> 姊妹篇：
>
> - `docs/STAGE_F_TEST_GUIDE.md` —— 架构 / 接口级指南
> - `docs/STAGE_F_REPORT.md` —— 阶段 F 交付与结论
> - `docs/STAGE_E_TEST_PLAYBOOK.md` —— 前序阶段对照 playbook
>
> 全文 Shell 是 **PowerShell**；命令默认在项目根
> `d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\` 下执行，conda 环境
> `qlib_zhengshi`。

---

## 0. 一键冒烟（10 秒）

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

**预期末行**：

```
389 passed, 3 skipped, 7 warnings in ~10s
```

- ✅ `389 passed, 3 skipped`：阶段 F 全套配置化 / universe / E2E 全绿；3 skipped
就是默认跳过的 `e2e_rdagent` marker。
- ❌ 若有 FAIL，按如下定位：
  - `tests/factor_lab/config/test_factor_families.py` → §2
  - `tests/factor_lab/config/test_constitution.py` → §4
  - `tests/factor_lab/feedback/test_feedback_universe.py` → §3
  - `tests/e2e/test_rdagent_feedback_e2e.py` → §5

然后显式跑 marker：

```powershell
python -m pytest tests/e2e -m e2e_rdagent -v
```

**预期**：

```
3 passed, 2 deselected
```

---

## 1. 分层测试矩阵

### 1.1 阶段 F 新增部分


| 想测什么？       | 命令                                                              | 预期耗时   | 覆盖                   |
| ----------- | --------------------------------------------------------------- | ------ | -------------------- |
| 家族分类 YAML   | `pytest tests/factor_lab/config/test_factor_families.py -v`     | ~0.2 s | 16 例                 |
| universe 字段 | `pytest tests/factor_lab/feedback/test_feedback_universe.py -v` | ~0.2 s | 18 例                 |
| 静态宪法 YAML   | `pytest tests/factor_lab/config/test_constitution.py -v`        | ~0.3 s | 19 例                 |
| E2E（默认）     | `pytest tests/e2e -v`                                           | ~0.2 s | 2 passed + 3 skipped |
| E2E（marker） | `pytest tests/e2e -m e2e_rdagent -v`                            | ~0.2 s | 3 passed             |


### 1.2 阶段 E 基础部分（F 未触但仍应保持绿）


| 想测什么？                                    | 命令                                    | 预期耗时   |
| ---------------------------------------- | ------------------------------------- | ------ |
| FeedbackBundle schema + aggregator + CLI | `pytest tests/factor_lab/feedback -q` | ~0.5 s |
| RAG 组合器                                  | `pytest tests/rdagent_integration -v` | ~0.3 s |
| E.4 shim                                 | `pytest tests/factor_lab/adapters -v` | ~0.3 s |
| E.5 cycle 钩子                             | `pytest tests/scripts/lab -v`         | ~0.3 s |


---

## 2. 家族分类 YAML（F.1）

### 2.1 最短路：验证默认 YAML 被正确加载

```powershell
python - <<'PY'
from factor_lab.config import classify_family, get_family_rules

rules = get_family_rules()
print(f"loaded {len(rules)} rules from YAML:")
for r in rules:
    print(f"  - {r.label}")
print()
print("classify_family('VolRev_5d') =", classify_family("VolRev_5d"))
print("classify_family('QualPersist_60D') =", classify_family("QualPersist_60D"))
print("classify_family('Totally_Unknown') =", classify_family("Totally_Unknown"))
PY
```

**预期**：

- stdout 打出 9 条 label，顺序与 `factor_families.yaml` 一致；
- `VolRev_5d` → `volume_price_reversal`；
- `QualPersist_60D` → `quality_persist`；
- `Totally_Unknown` → `unknown`。

### 2.2 故意用坏 YAML 验证降级

```powershell
$bad = "D:\tmp\bad_families.yaml"
Set-Content -Path $bad -Value "version: ""9.9""`nrules:`n  - {label: x, pattern: ""(""}"  # 正则非法 + schema_version 错

python - <<PY
from pathlib import Path
from factor_lab.config.families import load_family_rules, DEFAULT_FAMILY_RULES
rules = load_family_rules(Path(r"$bad"))
print("fallback ok:", rules == DEFAULT_FAMILY_RULES)
PY
```

**预期**：stdout 打 `fallback ok: True`，同时 log 里有 warning
（schema_version 不匹配或正则非法）。

### 2.3 扩展新规则（研究员日常操作）

```powershell
# 方案：在 YAML 末尾追加新规则；不要改前 9 条顺序。
notepad.exe factor_lab\config\factor_families.yaml

# 同步 Python 默认值
notepad.exe factor_lab\config\families.py

# 回归两份默认一致性
pytest tests/factor_lab/config/test_factor_families.py::test_default_yaml_loads_and_matches_python_default -v
```

**预期**：测试绿才算完工；失败则说明你只改了一侧。

---

## 3. universe 字段（F.2）

### 3.1 最短路：构造 bundle 并看 markdown

```powershell
python - <<'PY'
from datetime import datetime, timezone
from factor_lab.feedback import (
    FeedbackBundle, ActiveFactorSummary, RetiredFactorSummary, FailedCandidateSummary,
)

bundle = FeedbackBundle(
    generated_at=datetime.now(timezone.utc),
    cycles_included=("c1",),
    window_max_cycles=4,
    active_factors=(
        ActiveFactorSummary(
            factor_id="prod_alpha_abcdef01",
            name="Quality_ROE_5d",
            family="quality_persist",
            universe="csi300",
            parquet_version=1,
        ),
    ),
    retired_factors=(
        RetiredFactorSummary(
            factor_id="prod_alpha_abcdef02",
            name="Old_VolRev",
            family="volume_price_reversal",
            universe="csi500",
            reason="rolling IC degraded",
        ),
    ),
    recent_fails=(
        FailedCandidateSummary(
            name="VolRev_5d",
            family="volume_price_reversal",
            universe="csi300",
            cycle_id="c1",
            stage="default",
            decision="FAIL",
            failure_modes=("ic",),
        ),
    ),
    failure_family_counts={"volume_price_reversal": 1},
    discouraged_families=("volume_price_reversal",),
)
print(bundle.to_markdown())
PY
```

**预期**：每条因子行末尾带 `[universe=csi300]` / `[universe=csi500]`，且
数量正好 3 条（active 1 + retired 1 + fail 1）。

### 3.2 真实 aggregator 路径（读工程默认目录）

```powershell
# 若 factor_validation/certificates 里已经有带 universe 的证书：
python -m scripts.lab.build_feedback_bundle --max-cycles 4
Get-Content factor_lab\workspace\feedback\latest.md | Select-String "\[universe="
```

**预期**：`Select-String` 至少命中 1 条；若 0 条，说明 cycle artifacts 里
candidate.universe / manifest.universe 都为空（通常是早期数据；阶段 F 之后
新的 cycle 会自动带 universe）。

### 3.3 backward-compat：老 bundle JSON

```powershell
python - <<'PY'
from factor_lab.feedback import FeedbackBundle
import json
from datetime import datetime, timezone

old = {
    "schema_version": "1.0",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "cycles_included": ["c1"],
    "window_max_cycles": 4,
    "active_factors": [{
        "factor_id": "prod_alpha_abcdef01",
        "name": "Alpha_1",
        "family": "quality_persist",
        "parquet_version": 1,
        "tags": []
    }],
    "retired_factors": [],
    "recent_fails": [],
    "failure_family_counts": {},
    "discouraged_families": []
}
b = FeedbackBundle.model_validate(old)
print("ok, universe=", b.active_factors[0].universe)
PY
```

**预期**：`ok, universe= None`，证明老 JSON 无痛加载。

---

## 4. 静态宪法 YAML（F.3）

### 4.1 最短路：验证 YAML 渲染 = Python fallback byte-for-byte

```powershell
python - <<'PY'
from factor_lab.config.constitution import load_constitution_text, _FALLBACK_CONSTITUTION_TEXT
text = load_constitution_text()
print("len YAML renders:", len(text))
print("len fallback:   ", len(_FALLBACK_CONSTITUTION_TEXT))
print("byte-equal:     ", text == _FALLBACK_CONSTITUTION_TEXT)
PY
```

**预期**：三行分别给出长度 4230、长度 4230、`True`。

### 4.2 肉眼看渲染文本

```powershell
python - <<'PY'
from factor_lab.config.constitution import load_constitution_text
print(load_constitution_text()[:1500])
PY
```

**预期**：前 1500 字节能看到
`------Project factor hypothesis constraints (mandatory)------` 段落和
`5, 10, 20, 30, 60` 窗口列表。

### 4.3 故意破坏 YAML 验证降级

```powershell
python - <<'PY'
from pathlib import Path
from factor_lab.config.constitution import load_constitution_text, _FALLBACK_CONSTITUTION_TEXT

# 用不存在路径
t1 = load_constitution_text(Path(r"D:\tmp\nothing_here.yaml"))
print("missing -> fallback:", t1 == _FALLBACK_CONSTITUTION_TEXT)

import tempfile
p = Path(tempfile.gettempdir()) / "broken_rag.yaml"
p.write_text('version: "9.9"\n', encoding="utf-8")  # schema mismatch
t2 = load_constitution_text(p)
print("schema mismatch -> fallback:", t2 == _FALLBACK_CONSTITUTION_TEXT)
PY
```

**预期**：两行都是 `True`。

### 4.4 研究员扩展 encouraged/discouraged 家族

1. 编辑 `factor_lab/config/rag_constitution.yaml`，在 `encouraged_families` 末尾
  追加新条目（`id` 用不重复的小写字母）；
2. 把等价内容同步到 `factor_lab/config/constitution.py` 的
  `_FALLBACK_CONSTITUTION_TEXT`；
3. 跑 byte-parity：

```powershell
pytest tests/factor_lab/config/test_constitution.py::test_default_yaml_renders_byte_equal_to_fallback -v
pytest tests/factor_lab/config/test_constitution.py::test_hardcoded_static_constitution_equal_to_fallback -v
```

**预期**：两条全绿；任一失败说明两份内容没对齐。

---

## 5. 端到端闭环（F.4）

### 5.1 离线集成（不需要 marker）

```powershell
pytest tests/e2e/test_rdagent_feedback_e2e.py -v -k "not e2e_rdagent"
```

**预期**：

```
tests/e2e/test_rdagent_feedback_e2e.py::test_feedback_pipeline_end_to_end_into_final_rag PASSED
tests/e2e/test_rdagent_feedback_e2e.py::test_pipeline_discouraged_threshold_respected PASSED
```

### 5.2 RD-Agent 子类级（marker）

```powershell
pytest tests/e2e -m e2e_rdagent -v
```

**预期**：

```
tests/e2e/test_rdagent_feedback_e2e.py::test_real_rdagent_base_class_in_mro PASSED
tests/e2e/test_rdagent_feedback_e2e.py::test_prepare_context_chains_compose_project_rag PASSED
tests/e2e/test_rdagent_feedback_e2e.py::test_prepare_context_degrades_when_feedback_missing PASSED
3 passed, 2 deselected
```

### 5.3 手工 live smoke（真调 RD-Agent / LLM；可选）

**只在你持有真实 API key 且愿意花 token 时跑**。步骤：

```powershell
# 1) 先保证有一份 feedback bundle（任何来源）
python -m scripts.lab.build_feedback_bundle

# 2) 把 prompt dump 模式打开
$env:RDAGENT_LOG_DUMP_PROMPT = "1"

# 3) 跑 1 个 RD-Agent loop
python -m scripts.lab.run_rdagent_loop `
    --competition QlibQuantCompetition `
    --max-loop 1 `
    --log-level DEBUG 2>&1 | Tee-Object -FilePath "D:\tmp\f4_live_dump.log"

# 4) 在 log 里搜关键词
Select-String -Path "D:\tmp\f4_live_dump.log" `
    -Pattern "Feedback from recent L2 cycles|Discouraged families|\[universe="
```

**预期**：每条关键字至少命中 1 次；`[universe=...]` 命中说明 F.2 渲染真的传达给
了 LLM。

**验收人工判断**：核对 RD-Agent 生成的下一批 hypothesis，确认它们**不在**
discouraged 家族里（或者至少显式说明"我知道这是 discouraged family 但我的这个
变体不同，理由是 X"）。这是对 LLM 语义遵从的人工验证，**不能被 pytest 自动
替代**。

---

## 6. 排错速查


| 现象                                            | 首要排查                                                        |
| --------------------------------------------- | ----------------------------------------------------------- |
| `get_family_rules()` 返回的条数和 YAML 不符           | 进程 cache 污染，`reload_family_rules()`                         |
| 改 YAML 后测试抱怨 "YAML vs PY 默认不一致"               | 同步修改 `DEFAULT_FAMILY_RULES` / `_FALLBACK_CONSTITUTION_TEXT` |
| `FeedbackBundle(...)` 报 `universe 非法`         | 检查值是否含空格 / 斜杠 / 超长                                          |
| markdown 里没有 `[universe=...]`                 | 数据源 candidate / manifest 里 universe 字段真的缺失；或全 bundle 都是老数据  |
| E2E marker 测试报 `ModuleNotFoundError: rdagent` | 环境里 RD-Agent 没装，换到 `qlib_zhengshi` conda 环境                 |
| 全量回归报 `unknown mark 'e2e_rdagent'`            | `tests/conftest.py` 没被 pytest 识别；检查 pytest 是不是在项目根启动        |


---

## 7. 完工口径

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
python -m pytest tests/e2e -m e2e_rdagent -v
```

两条分别看到 `389 passed, 3 skipped` 与 `3 passed` 即阶段 F 完工。
手工 live smoke（§5.3）作为长期回访项，每次 RD-Agent 重大升级后人工过一次即可。