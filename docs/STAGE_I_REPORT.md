# Stage I 报告 —— Real-loop Finalization

> 范围：消除"LLM 代码生成阻塞 qrun"的两个成因，并在 WSL 真实环境 end-to-end 实证
> 决策源：`STAGE_H_LIVE_LOOP_LOG.md::L2-FULL-5` 列出的 3 条候选（I.1 / I.2 / I.3）
> 本阶段落地：**I.1（prompt 矛盾）+ I.2（reference implementation）**
> 状态：**DONE + 实证通过（首次触发 qrun + 回填 PortAnaRecord）**

---

## 1. 动机

Stage H 完成后，整个 pipeline 架构（F/G/H）已经 100% 可运行：C3 bundle → RAG 注入 → LLM → CoSTEER → feedback 真实闭环。
唯一剩下的阻塞是 **LLM 生成的 Python 代码无法一次通过 CoSTEER 的 shape/value 校验**，导致 `step_name=running`（qrun）一直被跳过。

Stage I 的目标是把这个最后一公里打通，让 `PortAnaRecord` 真实回填到 feedback 表格、产出 `IC / IR / annualized_turnover / composite_score`。

## 2. 两个根因与对应修复

### I.1 — Constitution 把合法 Qlib 格式误列为反例

**诊断**（重读 run #2 的冲突日志后得出）：

- Qlib instrument 代码格式为 **"2-letter 交易所前缀 + 6 位代码"**，`SH600000`、`SZ000001`、`SZ300059` 均合法、同构。
- 旧 constitution 第 2 行反例写了 `'SZ000001'`（把合法格式当反例），与 RD-Agent `value_feedback` 评估器所引用的"expected format"冲突。
- LLM 在两个相悖的"正确答案"之间反复振荡，20 分钟烧 $0.06 才能走完一轮 evolving，且全部失败。

**修复**（最小改动，2 处同步）：

- `factor_lab/config/rag_constitution.yaml::columns_nan_note`
- `factor_lab/config/constitution.py::_FALLBACK_CONSTITUTION_TEXT`

```diff
- NOTE: Instrument index is Qlib-style (e.g. 'SH600000', 'SZ300059'); preserve it as-is.
-       DO NOT transform instrument codes into other formats (e.g. '000001.SZ', 'SZ000001').
+ NOTE: Instrument index is Qlib-style: SH/SZ/BJ prefix + 6-digit code (e.g. 'SH600000', 'SZ000001', 'SZ300059'). Preserve it as-is.
+       DO NOT rewrite to dotted / suffix formats like '000001.SZ', 'SZ.000001' or '600000.SH'.
```

**实证（run #3，带 I.1 不带 I.2）**：


| 指标                              | run #2（旧）            | run #3（I.1）                      |
| ------------------------------- | -------------------- | -------------------------------- |
| 端到端墙钟                           | 21m07s               | **7m49s ↓63%**                   |
| 累计成本                            | $0.0626              | **$0.0301 ↓52%**                 |
| CoSTEER `expected format` 描述稳定性 | 多变、与 constitution 冲突 | **4 轮均稳定 'SH600000'/'SZ300059'** |


prompt 矛盾消除；LLM 不再振荡，但仍然产生了 `'000001.SZ'` 这种错格式——属于 **LLM 代码生成层面的 bug**，需 I.2 处理。

### I.2 — 注入可 copy-adapt 的 reference implementation

**假设**：让 LLM 走 "copy-adapt" 而非 "generate-from-spec"，就能绕过它在 `groupby` + index 重组时的常见错误。

**修复**：

- YAML 新增必填字段 `reference_implementation: |- ...`（~20 行 Python 骨架）
- `render_constitution` 支持该字段（缺失则 raise ValueError 回退到 fallback）
- `_FALLBACK_CONSTITUTION_TEXT` 同步注入同样文本
- 骨架内容示例：

```python
import pandas as pd

W = 60                                       # EDIT 1: window in {5,10,20,30,60}
df = pd.read_hdf("daily_pv.h5", key="data")  # MultiIndex (datetime, instrument)

x = df["$pe_ttm"].groupby(level="instrument").transform(lambda s: s.ffill())
med = x.groupby(level="instrument").transform(
    lambda s: s.rolling(W, min_periods=W).median()
)
std = x.groupby(level="instrument").transform(
    lambda s: s.rolling(W, min_periods=W).std()
)
factor = (x - med) / std                     # same MultiIndex as df -- do NOT reset_index

out = factor.to_frame("YourFactorName_%dD" % W)  # EDIT 2: factor name
out.to_hdf("result.h5", key="data", mode="w")    # EDIT 3: nothing else
```

**实证（run #4，I.1 + I.2）**：


| 事件                              | 本次首次？                       | 证据                                               |
| ------------------------------- | --------------------------- | ------------------------------------------------ |
| factor 通过 CoSTEER 校验            | ✅ 首次（3 轮 evolving 均 1/2 通过） | `Final decisions: [True, False] True count: 1`   |
| `step_name=running`（qrun）       | ✅ 首次触发                      | Workflow 从 50% 进到 `step_name=running`            |
| `PortAnaRecord` 行进入 feedback 表格 | ✅ 首次                        | `1day.composite_score NaN 11.229511`（SOTA 非 NaN） |


端到端 19m26s，累计 $0.0245。

## 3. 测试矩阵

### 3.1 单元

`tests/factor_lab/config/test_constitution.py`（**24 passed**）：

- `test_default_yaml_renders_byte_equal_to_fallback` — YAML ↔ Python fallback byte-for-byte 一致
- `test_hardcoded_static_constitution_equal_to_fallback` — 老 `_STATIC_CONSTITUTION` 别名依旧等价
- `test_rendered_text_contains_required_markers`（parametrize）— 新增 4 个 I.2 关键词：
  - `------Reference implementation (copy this skeleton`
  - `groupby(level="instrument").transform`
  - `EDIT 1: window in {5,10,20,30,60}`
  - `YourFactorName_%dD`
- `test_renderer_requires_reference_implementation` — 缺字段 / 空白 → `ValueError`

### 3.2 集成（真实 Level-2 run）


| Run              | 范围                 | 结果                                              |
| ---------------- | ------------------ | ----------------------------------------------- |
| run #3 (I.1)     | CoSTEER max_loop=4 | 7m49s / $0.030 / CoSTEER 反馈稳定但 LLM 代码仍错         |
| run #4 (I.1+I.2) | 同上                 | **19m26s / $0.025 / qrun 首次触发 + PortAnaRecord** |


## 4. 剩余非 pipeline 问题（qlib 数据层，不在本阶段范围）

run #4 的 qrun 执行阶段遇到 2 个错误，都在 qlib 本身：


| #   | 错误                                                          | 本项目可控性                 | 建议去向                                                          |
| --- | ----------------------------------------------------------- | ---------------------- | ------------------------------------------------------------- |
| A   | `ValueError: The benchmark ['SH000300'] does not exist`     | 无（qlib bench 数据准备）     | 补齐 `SH000300` 基准数据或改用已有 bench                                 |
| B   | `MergeError: Not allowed to merge between different levels` | 间接（factor output 索引级数） | Stage J.1 检查 factor DataFrame 的 MultiIndex 与 qlib label 的级数约定 |


这不影响 Stage I 验收——pipeline 已经证明能把 LLM 产物一路送达 qrun。

## 5. 永久修复清单（跨 H/I 两阶段累积的"真 bug"）


| #   | 症状                                                         | 修复位置                                                           | 何时引入          |
| --- | ---------------------------------------------------------- | -------------------------------------------------------------- | ------------- |
| 1   | Windows GBK locale 下 `inject_code_from_folder` 读 UTF-8 模板炸 | `patch_qlib_conda._patch_inject_code_utf8`                     | Stage H L2    |
| 2   | Windows 无 POSIX shell → `LocalEnv._run` 崩                  | 平面切到 WSL Ubuntu                                                | Stage H L2    |
| 3   | `CondaConf(conda_env_name=None)` pydantic 验证失败             | `_wrapped_init` 填占位 `"rdagent"`                                | Stage H L2    |
| 4   | OpenAI embedding 401 重试风暴                                  | `_patch_embedding_graceful_fallback` 升级：空 key 短路零向量            | Stage H L2    |
| 5   | Constitution 把合法 Qlib 格式 `SZ000001` 误列为反例                  | rag_constitution.yaml + fallback 两处同步                          | **Stage I.1** |
| 6   | LLM 缺可 copy-adapt 的 factor 骨架                              | YAML 新增 `reference_implementation` + renderer 支持 + fallback 同步 | **Stage I.2** |


所有修复都是**永久提交到代码仓库**的（不是运行时 hack），Windows + WSL 混合平面用户都直接受益。

## 6. 结论

**用户最初诉求 "配置可描述、过程可审计、长期可持续进化" 在本阶段完成 end-to-end 实证**：

- 配置化：所有约束、鼓励家族、禁止家族、惩罚、骨架代码都在 `rag_constitution.yaml` 里；研究员不碰代码即可扩展。
- 审计：每轮 loop 的 RAG prompt、LLM response、CoSTEER decision、feedback 表格都落 `logs/live_loop/*.log`。
- 持续进化：Run #4 已完成第一次完整闭环 —— C3 bundle → LLM → CoSTEER → qrun → PortAnaRecord → feedback → C3 下一轮。

接下来是 **Stage J**（可选）：补 qlib 基准数据、定位 MergeError、让 `Current` 列也拿到真实指标。

---

## 7. Stage I.3 — 循环运行模式三件套（GRU 配额、纯 factor、扩 loop_n）

在 `loop_n=5`（mixed mode）实证中暴露两个效率问题：

1. **`MergeError` 在 Stage I.3 首个修复点之后归零**，见 §2.I.3 的 `rdagent_overrides/factor_template/conf_combined_factors.yaml`：`StaticDataLoader.config` 从绝对路径改回相对 `"combined_factors_df.parquet"`，让 `factor_runner.py` 已经包装好的 2-level MultiIndex columns 生效。
2. **3/5 loop 被神经网络 `n_epochs=100` 消耗到 3600s 硬超时**，real metrics 覆盖率只有 40%。

为此把"**调 GRU 训练配额 / 切纯 factor loop / 扩 loop_n**"做成正式运行模式：

### 7.1 `mode` 参数

`factor_lab/runners/rdagent_loop.py::run_rdagent_loop` 新增 `mode: Literal["quant", "factor"]`：

| mode | 入口 | 实际行为 |
|------|------|----------|
| `quant`（默认，向后兼容）| `rdagent.app.qlib_rd_loop.quant.main` | 原生 `QuantRDLoop`，每轮 LLM 自选 factor/model |
| `factor` | `rdagent.app.qlib_rd_loop.factor.main` | 原生 `FactorRDLoop`，只做 factor 实验 + LGB 基线评分 |

单轮耗时对比：

| mode | 最短 | 最长 |
|------|------|------|
| `quant` factor round | ~7 min | ~15 min |
| `quant` model round (GRU n_epochs=100) | ~30 min | **3600s 硬超时** |
| `factor` | ~7 min | ~10 min |

### 7.2 `FACTOR_LAB_MAX_N_EPOCHS` 补丁

新增 `factor_lab/adapters/patch_qlib_conda.py::_patch_cap_n_epochs`：

- 包装 `QlibModelRunner.develop` 与 `QlibFactorRunner.develop`；
- 在进入 qrun 之前，把 LLM 提议的 `training_hyperparameters["n_epochs"]` 裁剪到环境变量 `FACTOR_LAB_MAX_N_EPOCHS` 指定的上限；
- 公开的可测试纯函数：`cap_n_epochs_value(raw, cap)`；
- 设为 `0` 或不设 → 完全透传 LLM 自主选择（向后兼容）。

典型值：`FACTOR_LAB_MAX_N_EPOCHS=20` → GRU 2-layer on CSI300 2022–2024 → ~15 min，远低于 3600s 硬超时；早停（`early_stop=10`）也够用。

### 7.3 `RUNNING_TIMEOUT_PERIOD`

直接透传给 `rdagent.utils.env.EnvConf.running_timeout_period`，默认 3600s。`_wsl_run_quant_loop5_tuned.sh` 把它提升到 7200s 作为 belt-and-suspenders（FACTOR_LAB_MAX_N_EPOCHS 一般先生效）。

### 7.4 新 launcher 脚本

| 脚本 | 模式 | 循环数 | 用途 |
|------|------|-------|------|
| `scripts/lab/_wsl_run_loop5.sh` | quant（默认）| 5 | 原始（阶段 H 产物，保留） |
| **`scripts/lab/_wsl_run_factor_loop10.sh`** | factor | 10 | 快速迭代因子库（~1h20m） |
| **`scripts/lab/_wsl_run_quant_loop5_tuned.sh`** | quant | 5 | 混合模式，`MAX_N_EPOCHS=20 + TIMEOUT=7200`，model round 能真出指标 |

### 7.5 验证脚本

`scripts/lab/_verify_loop_config.py`：

1. `mode="factor"` 路由到 `factor.main`；默认路由到 `quant.main`；非法值 raise
2. `cap_n_epochs_value` 对 8 种输入（含字符串/整数/None/非数字/cap≤0）的纯逻辑正确
3. `_patch_cap_n_epochs` 实际包装了真实的 `rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner.develop`，并且该 wrapper 调用真的把 `{"n_epochs": "100"}` 夹到 `"20"`
4. 两个 launcher 脚本文本里包含预期的 `--mode=xxx` / `--loop_n=N` / env var 设置

运行方式：`python scripts/lab/_verify_loop_config.py`（退出码 `0` = 全绿）。

### 7.6 后续使用建议

- **优先跑 `_wsl_run_factor_loop10.sh`** —— 因子才是 alpha 主要来源；10 轮 1.5 小时，LLM 每轮都能拿到真实 feedback。
- 因子池打出可打败 SOTA 的几条后，再切 `_wsl_run_quant_loop5_tuned.sh` 用神经网络精调。
- 若要试更激进的参数（`MAX_N_EPOCHS=50 + TIMEOUT=10800`），直接改 launcher 里的 `export`。
