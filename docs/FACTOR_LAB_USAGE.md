# Factor Lab 使用说明

> 适用对象：研究员 / 运维 / 首次上手跑 RD-Agent real loop 的人
> 覆盖范围：Stage F / G / H / I 的所有可用入口、配置点、常见故障
> 最近一次端到端实证：2026-04-21 run #4（Level-2，WSL Ubuntu，exit 0，19m26s，$0.0245）

---

## 1. 系统要求


| 组件             | 版本 / 位置                                                          | 备注                                          |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------- |
| 宿主             | Windows 10/11                                                    | 控制面 / 编辑器 / 单测                              |
| WSL            | Ubuntu 任意近 2 年版                                                  | qrun-bound 执行平面                             |
| Windows Python | `C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe` (3.11) | 跑单测、hypothesis-only dry run                 |
| WSL Python     | `/home/administrator/.local/share/mamba/envs/rdagent/bin/python` | 跑 Level-2（含 CoSTEER + qrun）                 |
| RD-Agent       | 0.8.x                                                            | `patch_qlib_conda.py` 已针对此版本做 monkey-patch  |
| pyqlib         | 0.9.7                                                            | 位于 WSL rdagent env 里                        |
| LLM            | DeepSeek chat（经 litellm）+ 可选 OpenAI embedding                    | `.env` 里配 `CHAT_MODEL` / `DEEPSEEK_API_KEY` |


> **关键跨平台约定**：所有涉及 `qrun` / CoSTEER 真实执行的命令都必须跑在 **WSL Ubuntu** 下。Windows 原生无 POSIX shell，RD-Agent 的 `LocalEnv._run` 会崩（详见 `STAGE_H_LIVE_LOOP_LOG.md::L2-1`）。

## 2. 目录速查

```
factor_lab/
├── config/
│   ├── rag_constitution.yaml        # ★ 静态宪法：研究员主修改点
│   └── constitution.py              # YAML 渲染器 + _FALLBACK_CONSTITUTION_TEXT
├── feedback/                        # C3 动态 bundle
│   ├── schema.py                    # FeedbackBundle / UniverseSubBundle
│   ├── aggregator.py                # L2 判决 → bundle
│   └── embedding.py                 # G.3 TF-IDF 检索
├── adapters/
│   ├── quant_proposal.py            # ★ RD-Agent 入口 HypothesisGen（H.1 自动 retrieval_query）
│   ├── proposal.py                  # Hypothesis2Experiment
│   └── patch_qlib_conda.py          # ★ 所有永久 patch（UTF-8 / Conda / Embedding）
└── workspace/feedback/latest*.json  # 最新 C3 bundle 落盘

scripts/lab/
├── build_feedback_bundle.py         # 从 L2 判决产 C3 bundle
├── dry_loop_hypothesis_only.py      # 安全 dry run：只到 hypothesis
├── dry_loop_prepare_context_only.py # offline smoke test
├── run_rdagent_loop.py              # ★ 完整 RD-Agent 入口
├── _wsl_run_level2.sh               # ★ WSL 下一键跑 Level-2
└── _wsl_sanity*.sh                  # WSL 环境健康检查

docs/
├── STAGE_F/G/H/I_REPORT.md          # 各阶段里程碑报告
├── STAGE_H_LIVE_LOOP_LOG.md         # 真实 loop 诊断 + 修复流水
└── FACTOR_LAB_USAGE.md              # 本文件
```

## 3. 快速上手（Happy Path）

```powershell
# Step 1：在 Windows 下确认 env + 单测绿
cd d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
& 'C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe' -m pytest tests/factor_lab/config -x -q

# Step 2：构建最新 C3 bundle（L2 判决 → bundle）
& 'C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe' -m scripts.lab.build_feedback_bundle

# Step 3：hypothesis-only dry run（安全、<$0.002、~20s，不碰 qrun）
& 'C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe' -m scripts.lab.dry_loop_hypothesis_only

# Step 4：完整 Level-2 run（WSL 下跑，带 qrun；~20 分钟、~$0.03）
wsl -d Ubuntu -- bash /mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project/scripts/lab/_wsl_run_level2.sh
```

Log 会写到 `logs/live_loop/wsl_level2_<TIMESTAMP>.log`。

## 4. 可配置面（不需改代码）

### 4.1 静态宪法 `factor_lab/config/rag_constitution.yaml`


| 字段                             | 作用                            | 示例                                                    |
| ------------------------------ | ----------------------------- | ----------------------------------------------------- |
| `allowed_windows`              | 合法窗口 W                        | `[5, 10, 20, 30, 60]`                                 |
| `allowed_primitives`           | 允许的基元                         | `[pct_change, shift, rolling(W).*, rank, clip]`       |
| `forbidden_patterns`           | 硬禁                            | `- nested rolling correlations across many series`    |
| `columns`                      | `daily_pv.h5` 列分组             | `Valuation: [$pe_ttm, $pb, ...]`                      |
| `columns_nan_note`             | NaN + instrument 格式契约         | I.1 已修正                                               |
| `scoring.formula`              | composite_score 公式            | `1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)` |
| `feature_universe`             | LGB 已有 32 列                   | 防重叠                                                   |
| `orthogonality_rules`          | 6.a/b/c 三条                    | `                                                     |
| `encouraged_families`          | 鼓励家族（正向示范）                    | 7 个                                                   |
| `discouraged_families`         | 不鼓励家族                         | 3 个；每条可选 `penalty: -0.5` 变软降权（G.4）                    |
| `**reference_implementation`** | **I.2 新增：20 行 copy-adapt 骨架** | 见 4.2                                                 |
| `naming_convention`            | 命名规范                          | `QualPersist_60D` 等                                   |


> **byte-parity 契约**：修改本文件后必须同步更新 `constitution.py::_FALLBACK_CONSTITUTION_TEXT`，否则 `tests/factor_lab/config/test_constitution.py::test_default_yaml_renders_byte_equal_to_fallback` 会红。跑一次 `pytest tests/factor_lab/config -x` 即可确认。

### 4.2 `reference_implementation` 的维护（I.2 新增）

当 factor 写法或 qlib 数据约定变化时，**这一段是 LLM 唯一的"样板代码"**，必须保持可直接 copy-adapt：

- 保留 `# EDIT 1/2/3` 三处显式标记
- 严禁包含 `reset_index` / `.str.split` 等破坏 MultiIndex 的操作（否则 LLM 会复制那些操作）
- 保留 `groupby(level="instrument").transform` 范式
- 修改后跑单测确保断言命中：`test_rendered_text_contains_required_markers`

### 4.3 `.env`（运行时开关，不入 git）

```bash
CHAT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-***
EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_API_KEY=                       # 留空 → 触发 I.1 前的 embedding 短路 patch

# 入口模块指向 factor_lab（G.2 后硬下线老 shim）
QLIB_QUANT_QUANT_HYPOTHESIS_GEN=factor_lab.adapters.quant_proposal.ProjectQlibQuantHypothesisGen
QLIB_QUANT_FACTOR_HYPOTHESIS2EXPERIMENT=factor_lab.adapters.proposal.ProjectQlibFactorHypothesis2Experiment
QLIB_QUANT_MODEL_HYPOTHESIS2EXPERIMENT=factor_lab.adapters.proposal.ProjectQlibModelHypothesis2Experiment

# CoSTEER 预算（每次 loop 内评估的 evolving 轮数）
FACTOR_CoSTEER_MAX_LOOP=4
FACTOR_CoSTEER_FAIL_TASK_TRIAL_LIMIT=4
MODEL_CoSTEER_MAX_LOOP=3
MODEL_CoSTEER_FAIL_TASK_TRIAL_LIMIT=3

# action 选择
QLIB_QUANT_ACTION_SELECTION=llm
QLIB_QUANT_EVOLVING_N=3

# Qlib / RD-Agent env 名
QLIB_RDAGENT_CONDA_ENV=qlib_zhengshi
MODEL_CoSTEER_ENV_TYPE=conda
LIVE_OUTPUT=False
```

> 不要手动 `export PATH=`/mamba/envs/rdagent/bin `之外的任何其它激活；WSL 下脚本已处理（`patch_qlib_conda._wrapped_init`填了`conda_env_name="rdagent"` 占位）。

### 4.4 Feedback bundle（研究员无需手动改；`build_feedback_bundle` 会重建）

- `factor_lab/workspace/feedback/latest.json` — 全量 bundle
- `factor_lab/workspace/feedback/latest_<universe>.json` — G.1 按 universe 切分的 sub-bundle（additive view）

## 5. 常用操作

### 5.1 研究员加一个新 encouraged family

1. 编辑 `factor_lab/config/rag_constitution.yaml` 的 `encouraged_families`：
  ```yaml
   - id: h
     text: "Quarterly earnings surprise decay: (q_profit_yoy - rolling_mean(q_profit_yoy, 4Q)) * exp(-days_since_report/30)"
  ```
2. 同步把同一行追加到 `constitution.py::_FALLBACK_CONSTITUTION_TEXT`（位置参照 encouraged 段尾）。
3. 跑 `pytest tests/factor_lab/config -x`。
4. 跑 `scripts.lab.build_feedback_bundle` 刷新 bundle。
5. 跑 `dry_loop_hypothesis_only.py` 抽一次 LLM response，确认它真的引用了 `(h)`。

### 5.2 研究员把 `volume_price_reversal` 从硬禁改成软降权

仅改 YAML 的 `discouraged_families` 对应条目：

```yaml
- id: x
  text: "Short-cycle volume-price reversals on W in {5, 10}"
  penalty: -0.5            # ← 新增，渲染后会追加软降权注解
```

同步更新 fallback；跑 `test_constitution.py` 验证两边一致。

### 5.3 研究员临时禁用 H.1 自动 retrieval

```python
# 在 adapter 处实例化时
gen = ProjectQlibQuantHypothesisGen(...)
gen.auto_retrieval_enabled = False
```

或生产中通过 `prepare_context` 内的 try/except 已做 safe-degrade，无需手动关。

### 5.4 只跑一次链路健康检查（不花 LLM 费用）

```powershell
# 结构检查（YAML + renderer + byte-parity + reference_implementation 段头）
& '<py>' -m pytest tests/factor_lab/config tests/factor_lab/feedback tests/factor_lab/adapters -q
```

### 5.5 在 WSL 里做 sanity check

```bash
wsl -d Ubuntu -- bash /mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project/scripts/lab/_wsl_sanity.sh
wsl -d Ubuntu -- bash /mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project/scripts/lab/_wsl_sanity_patch.sh
```

预期输出：`.env` 能读、`factor_lab` 能 import、embedding 短路生效（OPENAI_API_KEY 为空时不调 backend）。

## 6. 观测 / 审计

每次 Level-2 run 会产出：


| 位置                                           | 内容                                                                |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `logs/live_loop/wsl_level2_<TS>.log`         | 完整 stdout/stderr；含 RAG prompt、LLM response、CoSTEER 决策、feedback 表格 |
| `factor_lab/workspace/feedback/latest*.json` | 下一轮 loop 的输入 C3 bundle                                            |
| `mlruns/` （qrun 正常时）                         | mlflow 实验追踪                                                       |
| `factor_validation/reports/*.json`           | L2 判决结果（`build_feedback_bundle` 的输入）                              |


**快速取关键信号**：

```powershell
$log='logs/live_loop/<文件名>.log'

# 本轮 CoSTEER 结果
Select-String -Path $log -Pattern 'Final decisions:'

# 本轮总成本
Select-String -Path $log -Pattern 'Accumulated Cost:' | Select-Object -Last 1

# 本轮是否触发 qrun
Select-String -Path $log -Pattern 'step_name=running'

# 本轮 feedback 表格（含 PortAnaRecord）
Select-String -Path $log -Pattern '1day\.' -Context 0,6

# 本轮 RAG 是否确实注入了新 constitution
Select-String -Path $log -Pattern 'Reference implementation' | Select-Object -First 1
```

## 7. 故障排查


| 症状                                                                             | 快速诊断                                             | 修复                                                                                  |
| ------------------------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `tests/factor_lab/config` 红 `test_default_yaml_renders_byte_equal_to_fallback` | YAML 和 fallback 不同步                              | 对齐两处文本，跑 pytest 验证                                                                  |
| Windows 下 `qrun` / CoSTEER 崩 (`TypeError: str + NoneType`)                     | 跑错平面（Win 无 `/bin/sh`）                            | 切换到 WSL：`bash _wsl_run_level2.sh`                                                   |
| `UnicodeDecodeError` @ `inject_code_from_folder`                               | 非 UTF-8 locale（Windows GBK）                      | `_patch_inject_code_utf8` 已处理；若日志无 patch 生效记录，确认 `apply_qlib_conda_env_patch()` 被调到 |
| `pydantic_core.ValidationError: CondaConf.conda_env_name`                      | WSL 下 `CONDA_DEFAULT_ENV` 未设                     | `_wrapped_init` 已填占位 `"rdagent"`；检查 patch 是否生效                                      |
| 日志反复出现 `401 Unauthorized` / `embedding`                                        | `EMBEDDING_MODEL=openai/*` 且 `OPENAI_API_KEY=""` | 已自动短路为零向量；若仍出现说明 patch 未加载——检查 `.env` 和 `adapters/patch_qlib_conda.py` 导入顺序         |
| CoSTEER 全轮 `[False, False, ...]` 且 critic 抱怨 instrument 格式                     | Constitution 与 RD-Agent value_feedback 冲突        | 参照 I.1 fix；**核查 YAML 里是否把合法 Qlib 格式误列为反例**                                          |
| CoSTEER 全轮 `[False, False, ...]` 但 critic 说 groupby/index                      | LLM 没 copy 骨架                                    | 查 `Reference implementation` 是否真的进了 prompt（Section 6 快查命令）                          |
| qrun 报 `ValueError: The benchmark ['SH000300'] does not exist`                 | qlib bench 数据缺失（非 pipeline）                      | Stage J 范围：补基准数据或切换到可用基准                                                            |
| qrun 报 `MergeError: Not allowed to merge between different levels`             | factor DataFrame 与 label 的 MultiIndex 级数不匹配      | Stage J 范围：检查 factor output 是否纯 `(datetime, instrument)` 两级                         |
| Workflow hang、长时间无 log 增量                                                      | 最常见是 LLM API 限流                                  | 看 log 末尾最近时间戳；限流时 litellm 会自动 retry，等一下就好                                           |


## 8. 扩展指南

### 8.1 加新的 universe sub-bundle 视图（G.1）

- 在 `build_feedback_bundle` 里新增 universe 聚合逻辑
- `aggregator._derive_by_universe` 会自动写 `latest_<universe>.json`
- RD-Agent 入口若需要：`compose_project_rag(..., feedback_dir=workspace/feedback/latest_<universe>.json)`

### 8.2 换 embedding backend（G.3）

`factor_lab/feedback/embedding.py` 暴露 `EmbeddingBackend` protocol。默认 TF-IDF 零依赖；换成 SentenceTransformer / 第三方 API 时实现 `embed(texts) -> np.ndarray` 并传入 `build_failure_corpus`。

### 8.3 自动 retrieval_query 源切换（H.1）

`adapters/quant_proposal._infer_retrieval_query_with_source` 当前按优先级尝试 `trace_semantic → rag_tail → constitution_tail`。加新来源时在此函数里追加一个 branch，并在单测里加断言 `source="你的新来源"`。

### 8.4 新增 RD-Agent 上游 patch

所有 monkey-patch 都放 `factor_lab/adapters/patch_qlib_conda.py`。原则：

1. 每个 patch 一个带 docstring 的 `_patch_xxx()` 函数
2. `apply_qlib_conda_env_patch()` 里按顺序调
3. 每个 patch 都 `try/except` 兜底，版本变动不能让整个 lab 崩
4. 加对应单测或 sanity 脚本

## 9. 验收 / 首次成功跑通的最小证据

跑完一次完整 `_wsl_run_level2.sh`，在 log 里应该能同时抓到：

```
1. RAG 注入成功
   "------Reference implementation (copy this skeleton; only edit the 3 marked lines)------"

2. CoSTEER 通过一个 factor
   "Final decisions: [True, ...] True count: 1"
   或更高

3. qrun 被触发
   "step_name=running"

4. PortAnaRecord 行回写 feedback
   "1day.composite_score  <Current>  <SOTA>"
```

以上 4 条任一缺失，参考 Section 7 故障排查。

## 10. 参考文档

- `docs/STAGE_F_REPORT.md` — YAML 配置化 + by_universe sub-bundle
- `docs/STAGE_G_REPORT.md` — Shim 下线 + 嵌入式 RAG + 软降权
- `docs/STAGE_H_REPORT.md` — 自动 retrieval_query + 5 层测试金字塔
- `docs/STAGE_H_LIVE_LOOP_LOG.md` — 真实 loop 全量诊断流水（必读）
- `docs/STAGE_I_REPORT.md` — 消 prompt 矛盾 + 注入骨架 + 首次 qrun 实证

如需回顾整个 F → G → H → I 的设计动机和决策链，按顺序读以上 5 份即可。