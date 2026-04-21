# 阶段 E 测试与运维指南（RD-Agent 反馈回路）

> 本指南覆盖阶段 E 交付的运维级产物：
>
> 1. C3 `FeedbackBundle` 契约（`factor_lab/feedback/schema.py`）
> 2. 聚合器 + CLI（`factor_lab/feedback/aggregator.py` + `scripts/lab/build_feedback_bundle.py`）
> 3. 静态 + 动态 RAG 注入（`factor_lab/adapters/quant_proposal.py`）
> 4. 目录重构：`rdagent_integration/*` → `factor_lab/adapters/*`；`scripts/run_fin_quant.py` → `scripts/lab/run_rdagent_loop.py`；带 DeprecationWarning shim
> 5. `run_lab_cycle.py` 末尾 feedback 钩子
>
> 配套阅读：
>
> - `docs/STAGE_D_TEST_GUIDE.md` —— 前序真实回测 + 边际贡献 + 自动闭环
> - `docs/STAGE_E_TEST_PLAYBOOK.md` —— 阶段 E 复制即跑测试
> - `docs/STAGE_E_REPORT.md` —— 阶段 E 交付清单

---

## 1. 测试分层（pytest 快速索引）

| 层 | 目录 | 说明 | 典型命令 |
| --- | --- | --- | --- |
| C3 契约 | `tests/factor_lab/feedback/test_feedback_schema.py` | `FeedbackBundle` 不可变性 / markdown 渲染 / cycles 窗口 / fail cycle ⊆ cycles_included | `pytest tests/factor_lab/feedback -k schema -v` |
| 聚合器 | `tests/factor_lab/feedback/test_feedback_aggregator.py` | `build_feedback_bundle` 各种输入组合 + `classify_family` + IO 读写 | `pytest tests/factor_lab/feedback -k aggregator -v` |
| 聚合器 CLI | `tests/scripts/lab/test_build_feedback_bundle_cli.py` | `scripts.lab.build_feedback_bundle` `main()` 冒烟 | `pytest tests/scripts/lab -k build_feedback -v` |
| 静态 + 动态 RAG | `tests/rdagent_integration/test_project_quant_proposal.py` | `compose_project_rag` 所有分支、snapshot 关键词、默认路径 | `pytest tests/rdagent_integration -v` |
| E.4 重构 | `tests/factor_lab/adapters/test_adapters_refactor.py` | 新家可 import / shim 发 DeprecationWarning / 符号 `is` 相等 | `pytest tests/factor_lab/adapters -v` |
| E.5 cycle 钩子 | `tests/scripts/lab/test_run_lab_cycle_feedback.py` | 默认写 latest + history / `--skip-feedback-rebuild` / 聚合器异常不污染 / CLI flag 透传 | `pytest tests/scripts/lab -k feedback -v` |

### 一键全量

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

阶段 E 交付基准：**334 项全 PASS，整体 <10 s**（新增 56 项；老 `test_unified_strategy.py`
里 2 条 pre-existing 数据形状失败与本阶段无关）。

---

## 2. C3 FeedbackBundle（契约 C3）

### 2.1 schema 关键字段

```python
from factor_lab.feedback import FeedbackBundle
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | str | `"1.0"`；不兼容升级必须 bump major |
| `generated_at` | datetime (UTC) | 聚合器产出时间 |
| `cycles_included` | tuple[str, ...] | 本次聚合纳入的 cycle_id 列表，**时间升序** |
| `window_max_cycles` | int ≥ 1 | 聚合器使用的 `max_cycles` 参数（默认 8） |
| `active_factors` | tuple[ActiveFactorSummary, ...] | L3 manifest 里 status=active |
| `retired_factors` | tuple[RetiredFactorSummary, ...] | L3 manifest 里 status=retired |
| `recent_fails` | tuple[FailedCandidateSummary, ...] | 窗口内证书 FAIL/HOLD（**不含 PASS**） |
| `failure_family_counts` | dict[str,int] | 按 family 计数的失败分布 |
| `discouraged_families` | tuple[str, ...] | `count ≥ min_fail_count` OR `retired` 家族 |
| `notes` | str \| None | 聚合器可留的人工注记 |

### 2.2 不变式（frozen + 校验）

- `recent_fails[i].cycle_id` 必须 ∈ `cycles_included`（orphan 会报错）
- `len(cycles_included) ≤ window_max_cycles`
- `failure_family_counts` 里数值必须非负整数
- `cycles_included` 与 `discouraged_families` 不允许重复

### 2.3 `to_markdown()`

固定章节顺序（便于 prompt-level 复现）：

```text
------Feedback from recent L2 cycles (dynamic)------
generated_at = ... cycles_included = [...]

[L3 active factors — DO NOT propose duplicates]
  - Name1  (family=quality_persist) tags=[...]

[L3 retired factors — these failed in production, avoid similar shape]
  - NameX  (family=volume_price_reversal) retired_on=YYYY-MM-DD
    reason: ...

[Recent L2 failures across last N cycle(s)]
  - NameY  (family=...) stage=default decision=FAIL modes=[ic,orthogonality] rank_ic=0.004 max_abs_corr=0.72

[Failure family counts (last N cycles)]
  - volume_price_reversal: 3
  - unknown: 1

[Discouraged families — empirically blocked by L2; DO NOT propose new factors in these families]
  - volume_price_reversal

[Aggregator notes]
  ...

Interpretation rules for LLM: ...
```

**空章节会被省略**，避免给 LLM 制造"空类别 = 没有限制"的误读。

---

## 3. 聚合器 `build_feedback_bundle`

### 3.1 函数签名

```python
from factor_lab.feedback.aggregator import build_feedback_bundle

bundle = build_feedback_bundle(
    reports_dir=Path("factor_validation/reports"),
    cert_dir=Path("factor_validation/certificates"),
    registry_data_dir=Path("factor_registry/data"),
    max_cycles=8,
    min_fail_count_for_discouraged=2,
    now=None,     # None → datetime.utcnow
    notes=None,
)
```

### 3.2 输入要求

- `reports_dir` 下的 `lab_cycle_<id>.json`：按 **mtime 升序** 取最近 `max_cycles` 个；
  每个文件必须含顶层 `"cycle_id"` 字段。
- `cert_dir/<cycle_id>/<factor_id>.(exploratory|default|strict).json`：标准 C2 JSON。
  文件名后缀决定 `stage`；同一因子的 `exploratory` + `default` 只保留**更深 stage**那条。
- `registry_data_dir/manifest.json`：标准 L3 manifest；`status` 只接收 `active` / `retired`。

### 3.3 容错行为

| 输入异常 | 聚合器行为 |
| --- | --- |
| `reports_dir` 不存在 | `cycles_included=()`，其余字段正常推导 |
| 某个 `lab_cycle_*.json` 坏了 / 缺 `cycle_id` | 跳过该 cycle，其他继续 |
| 某个证书坏了 / JSON 不合法 | 跳过该证书，同目录其他继续 |
| 证书里 `decision=PASS` | 不进 `recent_fails`（只收 FAIL/HOLD） |
| manifest 不存在 | `active_factors=()` / `retired_factors=()` |

### 3.4 CLI

```powershell
# 最常用：默认路径
python -m scripts.lab.build_feedback_bundle

# 仅看最近 4 个 cycle
python -m scripts.lab.build_feedback_bundle --max-cycles 4

# 写进定制目录 + 加注记
python -m scripts.lab.build_feedback_bundle `
    --feedback-dir D:/tmp/feedback_snapshot `
    --notes "手动快照：调试 RAG" `
    --no-history
```

落盘产物：

- `<feedback-dir>/latest.json` —— RAG 注入器读取
- `<feedback-dir>/latest.md`   —— 人工阅读
- `<feedback-dir>/history/<cycles_included[-1]>.json`（默认启用；`--no-history` 关闭）

---

## 4. 静态 + 动态 RAG 注入（E.3 / E.4）

### 4.1 新位置

```python
from factor_lab.adapters.quant_proposal import (
    ProjectQlibQuantHypothesisGen,  # RD-Agent 挂钩类
    compose_project_rag,            # 纯函数，单元可测
    _STATIC_CONSTITUTION,           # 硬编码宪法常量
)
```

旧 path（DeprecationWarning）：

```python
from rdagent_integration.project_quant_proposal import (
    ProjectQlibQuantHypothesisGen,
    compose_project_rag,
    _PROJECT_FACTOR_RAG,  # = _STATIC_CONSTITUTION 的 alias
)
```

### 4.2 运行时路径

`ProjectQlibQuantHypothesisGen.prepare_context(trace)` → 调父类 → 拿到 `ctx["RAG"]` →
调 `compose_project_rag(base_rag, feedback_dir=self._feedback_dir_override)`：

1. 静态宪法 `_STATIC_CONSTITUTION` 被追加（**必定**注入）
2. 动态段从默认位置 `factor_lab/workspace/feedback/latest.json` 加载
   - 加载失败（文件缺失 / JSON 坏 / pydantic 校验失败 / factor_lab.feedback 不可 import）
     → 静默省略，只记 `logger.warning`
3. 最终 `ctx["RAG"] = "<base>\n\n<static>\n\n<dynamic>"`

### 4.3 自定义 feedback 源

覆盖默认路径（通常用于测试 / A/B 对比）：

```python
from factor_lab.adapters.quant_proposal import ProjectQlibQuantHypothesisGen

gen = ProjectQlibQuantHypothesisGen()
gen._feedback_dir_override = Path("D:/tmp/my_custom_feedback")
```

---

## 5. E.4 目录重构清单

### 5.1 新 → 老对照

| 新位置 | 老位置（shim，DeprecationWarning） |
| --- | --- |
| `factor_lab/adapters/experiments.py` | `rdagent_integration/project_experiments.py` |
| `factor_lab/adapters/proposal.py` | `rdagent_integration/project_proposal.py` |
| `factor_lab/adapters/quant_proposal.py` | `rdagent_integration/project_quant_proposal.py` |
| `factor_lab/adapters/patch_qlib_conda.py` | `rdagent_integration/patch_qlib_conda.py` |
| `factor_lab/runners/rdagent_loop.py` | —（新增） |
| `scripts/lab/run_rdagent_loop.py` | `scripts/run_fin_quant.py`（shim） |

### 5.2 迁移建议

- 新代码一律 `from factor_lab.adapters.* import ...`
- 老的 launch 脚本（CI / cron / 文档）可以继续用 `scripts/run_fin_quant.py`，但
  会发 DeprecationWarning；**阶段 G 之前**会删除 shim。
- 任何 `import rdagent_integration` 的代码只需改 import 行就可以升级，符号语义不变
  （`is` 等价）。

---

## 6. E.5 cycle 钩子配置

### 6.1 `scripts/lab/run_lab_cycle.py` 新增 CLI flag

| flag | 含义 | 默认值 |
| --- | --- | --- |
| `--feedback-dir` | feedback bundle 输出目录 | `factor_lab/workspace/feedback` |
| `--feedback-max-cycles` | 聚合器往前看的 cycle 数 | 8 |
| `--skip-feedback-rebuild` | 跳过末尾 bundle 重建 | 关闭 |

### 6.2 `CycleSummary` 新增字段

```python
summary.feedback_bundle_path       # Path | None
summary.feedback_cycles_included   # int（本轮 bundle 覆盖了多少 cycle）
summary.feedback_discouraged_count # int（本轮 bundle 里 discouraged 家族数）
summary.feedback_error             # str | None（钩子失败时记录）
```

Cycle 报告 JSON 里对应结构（与 `totals` 同级）：

```json
"feedback": {
  "bundle_path": "D:\\...\\factor_lab\\workspace\\feedback\\latest.json",
  "cycles_included": 3,
  "discouraged_count": 1,
  "error": null
}
```

### 6.3 Windows Task Scheduler 注意

阶段 D 配置的 `run_lab_cycle.ps1` 不需要修改；E.5 钩子**默认启用**，老 cron
不用动。若确认长期不需要 feedback 回流（非常不建议），可在 ps1 里追加
`--skip-feedback-rebuild`。

---

## 7. 常见问题 / 排错

### 7.1 RD-Agent 启动时没看到 "Feedback from recent L2 cycles"

可能原因（按概率排序）：

1. `factor_lab/workspace/feedback/latest.json` 不存在 —— 先跑一次
   `python -m scripts.lab.build_feedback_bundle` 手工生成。
2. bundle 存在但无内容（`recent_fails=[]` / `active_factors=[]` / `retired_factors=[]`）
   —— markdown 会省略所有空分区，只剩 "Interpretation rules for LLM"。跑一轮真实
   cycle 或人工写一份 mock bundle 即可。
3. `latest.json` JSON 非法 —— 日志会有 `logger.warning` 提示；用
   `python -c "from factor_lab.feedback.aggregator import load_latest_feedback_bundle; print(load_latest_feedback_bundle(Path('factor_lab/workspace/feedback')))"`
   快速验证。

### 7.2 `run_lab_cycle` 报告里 `feedback.error` 不为空

- 如果信息是 `ValueError: max_cycles 必须 >= 1`：检查 `--feedback-max-cycles` 是否 >0；
- 如果信息是 pydantic 校验错：打开 `<feedback-dir>/latest.json` 看具体字段；聚合器
  写 bundle 是原子覆盖的，坏内容不会污染 `<feedback-dir>`（原子 write 其实是先写
  文件再覆盖；如果担心，加 `--feedback-dir` 换路径）。

### 7.3 旧 import 路径突然停止工作

- 如果异常是 `ModuleNotFoundError: factor_lab.adapters.xxx`：检查 `factor_lab/` 下
  是否真的有 `__init__.py`（shim 依赖新家存在）；
- 如果异常只是 DeprecationWarning 被 warning filter 转成了 error（某些 CI 配置）：
  在 conftest 或 pyproject 里对 `DeprecationWarning` 放行。

---

## 8. 回归 checklist（阶段 E 完工口径）

执行顺序：

1. `python -m pytest tests/factor_lab/feedback -v` → 26 绿
2. `python -m pytest tests/rdagent_integration -v` → 12 绿
3. `python -m pytest tests/factor_lab/adapters -v` → 12 绿
4. `python -m pytest tests/scripts/lab -v` → 10 绿（含 E.5 新 4 条 + 老 4 + CLI 2）
5. 全量：`python -m pytest tests -q --ignore=tests/test_unified_strategy.py` → 334 绿
6. 手工 smoke：`python -m scripts.lab.build_feedback_bundle --log-level WARNING` →
   生成 `factor_lab/workspace/feedback/latest.{json,md}`（若 reports 目录里已经有
   真实 cycle 报告）。

全部通过即阶段 E 完工。
