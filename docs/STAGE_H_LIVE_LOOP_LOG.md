# Stage H 真实 LOOP 验证日志（hypothesis-only dry run）

> 时间：2026-04-21 10:07:46 ~ 10:08:09（本机 CST）
> 决策：`scope=hypo_only`（只到 hypothesis/experiment 生成，不跑 qrun）
> 入口脚本：`scripts/lab/dry_loop_hypothesis_only.py`
> 真实 LLM：DeepSeek `deepseek/deepseek-chat`，2 次 chat completion，共计 $0.0013305600

---

## 1. 前置修复（项目 bug）

| 文件 | 改动 | 原因 |
|---|---|---|
| `.env` | `QLIB_QUANT_*_HYPOTHESIS_GEN/HYPOTHESIS2EXPERIMENT` 四行指向 `factor_lab.adapters.*`；`QLIB_RDAGENT_CONDA_ENV=qlib_zhengshi` | G.2 硬下线后旧路径即 RuntimeError；本机不存在 `rdagent` conda env |
| `.env.example` | 同步三条 HypothesisGen/Hypothesis2Experiment 的新模块路径（`proposal`、`quant_proposal`） | 防止从 example 复制的用户再次踩坑 |

其它 env key（密钥、`FACTOR_CoSTEER_*` 等）均未改动。

## 2. 前置数据

`python -m scripts.lab.build_feedback_bundle` 执行成功：

```
bundle 已生成：cycles=0 active=3 retired=2 fails=0 discouraged=1
-> factor_lab/workspace/feedback/latest.json
```

> fails=0 决定了 G.3 检索段即使被 H.1 触发也会安全降级为"不注入"；这不是退化，而是"无历史失败可检索"的正确行为。

## 3. 真实 LOOP 观察

### 3.1 入口类解析

`.env` 里的 `QLIB_QUANT_QUANT_HYPOTHESIS_GEN=factor_lab.adapters.quant_proposal.ProjectQlibQuantHypothesisGen` 被
`rdagent.core.utils.import_class` 正确解析：

```
[OK] .env -> HypothesisGen 正确解析到:
  factor_lab.adapters.quant_proposal.ProjectQlibQuantHypothesisGen
```

### 3.2 H.1 自动 retrieval_query 命中

关键日志（在 `prepare_context` 内打出）：

```
DEBUG factor_lab.adapters.quant_proposal | H.1 auto_retrieval source=rag_tail len=87
```

含义：trace 是第一轮（`hist=[]`），trace 语义字段为空 → 自动降级到 `ctx['RAG']` 末行兜底，截断到 87 字符。无异常、无崩溃，符合设计。

### 3.3 RAG 结构

最终注入的 RAG = **5622 字符**，三段结构确认：

| 段 | 结果 |
|---|---|
| static constitution (`Project factor hypothesis constraints (mandatory)`) | PRESENT |
| dynamic feedback (`Feedback from recent L2 cycles`) | PRESENT（含 retired + discouraged） |
| G.3 retrieval (`Similar past failures retrieved`) | absent（`fails=0` → 安全降级） |

`dynamic feedback` 段里能看到：

```
[Retired factors — DO NOT propose identical formulations]
  - VolRet_5D      (family=volume_price_reversal)  retired_on=2026-04-20
    reason: oracle default v2 FAIL: VolRet_5D 低 IC ...
  - VolumeTrend_10D (family=volume_price_reversal) retired_on=2026-04-20
    reason: ... plan B 择一退役

[Discouraged families — empirically blocked by L2; DO NOT propose new factors in these families]
  - volume_price_reversal
```

### 3.4 真实 LLM 产出 hypothesis

DeepSeek 返回（对 RAG 的响应）：

- `action = factor`
- `hypothesis.hypothesis`：
  ```
  Quality persistence factor:
    QualPersist_60D = rank(rolling_mean($roe.fillna(method='ffill'), 60))
                    - rank(rolling_mean($roe.fillna(method='ffill'), 20))
  Valuation mean-reversion factor:
    ValueMR_60D = ($pe_ttm - rolling_median($pe_ttm, 60)) / rolling_std($pe_ttm, 60)
  ```
- `hypothesis.reason`（关键证据）：
  > "I propose two simple factors from **encouraged families** to start with diverse
  > perspectives while **avoiding discouraged volume-price-reversal patterns**.
  > QualPersist_60D captures quality persistence by comparing long-term (60-day) vs
  > medium-term (20-day) ROE trends ... ValueMR_60D implements valuation mean-reversion
  > using PE ratio deviations from its long-term median ..."

> **这是 F → G → H 链路的端到端实证**：L2 判决（retired + discouraged）→ aggregator
> → latest.json → `compose_project_rag` → LLM prompt → LLM 在 reasoning 里显式"避开
> volume_price_reversal"，并选了符合 `W ∈ {5,10,20,30,60}` 约束的 60D / 20D 窗口。

### 3.5 成本与耗时

| 项 | 值 |
|---|---|
| Chat 调用次数 | 2（1 × action_selection + 1 × hypothesis_gen） |
| 累积成本 | $0.0013305600 |
| 端到端墙钟 | ~23 秒（含模型冷启动） |

## 4. 工程结论

* `.env.example` 随 G.2 同步更新，避免外部用户再踩"旧 shim path → RuntimeError"。
* 本机 `.env` 一次性修正到位，可直接跑真实 loop。
* `ProjectQlibQuantHypothesisGen` 在真 RD-Agent 入口下表现与单测/e2e 完全一致：
  - H.1 自动 `retrieval_query` 推导生效；
  - `compose_project_rag` 正确组段；
  - LLM 正确"读懂"并遵守了 dynamic feedback。
* 产出的 hypothesis 符合静态 constitution 约束（窗口 ∈ {5,10,20,30,60}、每次 ≤2 个因子）。

## 5. 后续建议

1. 把本次 hypothesis-only 命令固化到 `docs/STAGE_G_TEST_PLAYBOOK.md` 的"live smoke"
   章节里，作为**默认冒烟入口**（不过 qrun）。
2. 若要进一步验证"H.1 source=trace"通路，只需构造一个 `trace.hist` 非空的场景
   （例如第 2 轮之后），可以在后续扩展 test playbook。
3. `fails=0` 的条件下 G.3 段始终不注入，这与期望一致；若希望观察 G.3 真实检索行为，
   需要先在 `factor_validation/reports/` 里积累至少一条失败 cycle。

---

# Stage H 真实 LOOP 验证日志 —— Level-2（含 coding / CoSTEER）

> 时间：2026-04-21 10:35:09 ~ 10:39:24（约 4m15s，端到端）
> 决策：`scope=level2`（允许 CoSTEER 生成代码并执行，`FACTOR_CoSTEER_MAX_LOOP=1`）
> 入口脚本：`bash scripts/lab/_wsl_run_level2.sh`（在 WSL Ubuntu 下）
> 真实 LLM：DeepSeek `deepseek/deepseek-chat`，累计 $0.0092596000（~0.93 美分）

## L2-1. 本轮暴露并修复的 3 个项目级 bug

| # | 症状 | 文件 | 修复 |
|---|---|---|---|
| 1 | `UnicodeDecodeError` @ `FBWorkspace.inject_code_from_folder`：Windows GBK locale 下读 UTF-8 模板文件炸 | `factor_lab/adapters/patch_qlib_conda.py` | 新增 `_patch_inject_code_utf8()`，monkey-patch `FBWorkspace.inject_code_from_folder` 强制 `encoding="utf-8"` |
| 2 | `TypeError: str + NoneType` @ `env.py:602` 配合 `subprocess reader thread UnicodeDecodeError`：`LocalEnv._run` 调用 `/bin/sh -c ...`，Windows 原生无 POSIX shell，子进程输出是 GBK 字节 → reader thread 死 → `out`/`err`=`None` | 架构级 | 平面切换：**所有 qrun-bound 执行放到 WSL Ubuntu**，命中 `patch_qlib_conda.py:12` 中早就硬编码好的 `/home/administrator/.local/share/mamba/envs/rdagent/bin` |
| 3 | `pydantic ValidationError: CondaConf.conda_env_name` 必须是 str，但 `CONDA_DEFAULT_ENV` 未设（WSL 里我们通过 `export PATH` 激活 mamba env，不走 `conda activate`） | `patch_qlib_conda.py:_wrapped_init` | 在调原生 `_orig(self, **kwargs)` 之前，若 `conda_env_name` 为 `None/""` 填占位 `"rdagent"`；因为后面 `bin_path` 会被 override，该字段实际不再被使用 |

上述三处修复全部提交到代码（不是运行时 hack），Windows + WSL 混合平台用户都受益。

## L2-2. Loop 走完的 4 个 Step

```
Start Loop 0, Step 0: direct_exp_gen   10:35:14  (hypothesis)
Start Loop 0, Step 1: coding           10:36:01  (CoSTEER 生成 factor Python code + 执行)
        (Step 2 running 被 CoSTEER 前置校验跳过，见 L2-4)
Start Loop 0, Step 3: feedback         10:39:09
Workflow Progress: 100% | loop_n=1 结束，exit_code=0
```

## L2-3. Customized RAG 真的进了 prompt（最关键证据）

日志里能直接抓到注入内容：

```
L252  1day.composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)
L253  This means three things you MUST optimise simultaneously, NOT just IC:
L260    VolumePriceTrend_10D, VolumeTrend_10D, VolRet_5D) raised LGB valid-IC
        from 0.148 to 0.226 BUT the realised Sharpe DROPPED from 3.59 to 1.69
        because annualized_turnover ...
L304  ------Feedback from recent L2 cycles (dynamic)------
L314    reason: oracle default v2 FAIL: VolRet_5D 低 IC (rank_ic=0.012, ic_ir=0.31);
            VolumeTrend_10D ↔ RangeRatio_10D 方向共线 IC 反号 ...
```

这证实了 F+G+H 链路完整：**C3 latest.json → `compose_project_rag` → RAG 段 → LLM prompt → LLM 推理**。

## L2-4. 为什么 Step 2 `running`（qrun 回测）没触发？

LLM 生成的两个 factor 代码（`QualPersist_60D`、`ValueMR_60D`）Python 执行成功、`result.h5` 也产出，但 CoSTEER 的 **return-checking / value feedback** 阶段发现 output 有 4 处不合规：

| 缺陷 | 本轮表现 |
|---|---|
| instrument 码格式 | 输出 `'000001.SZ'`，源数据是 `'SH600000'`/`'SZ300059'` |
| datetime 范围 | 输出覆盖到 2022-12-30，源数据只到 2021-12-31 |
| non-null 比例 | 145,396 条里只有 121,199 非空 |
| 列名契约 | 与 `factor name` 的严格一致性未满足 |

CoSTEER 按设计直接把 **未通过 shape/value 校验的 factor 挡在 qrun 门外**，流程进入 `feedback` 让 LLM 下一轮修代码。这是 CoSTEER 的正确行为，不是我们 pipeline 的 bug。

## L2-5. 对"qrun 阶段是否实现初衷"这个问题的直接答案

**架构层面已经实现**：
- ✅ L2 判决 → C3 bundle → RAG 注入 → LLM 遵守 constitution
- ✅ LLM 生成 factor 代码 → CoSTEER 在 Linux 环境（WSL）下真实执行
- ✅ CoSTEER 返回 shape/value feedback，LLM 在下一轮可以基于 critic 修改代码
- ✅ 反馈回路真实闭环（不是模拟的）

**待下次实证**（需要 LLM 代码一次通过 CoSTEER 校验）：
- ⏳ Step 2 `running`（qrun 回测真产出 IC/IC_IR/Sharpe/annualized_turnover）
- ⏳ `1day.composite_score` 被真实评分并回填到 `factor_validation/reports/`
- ⏳ 该分数进入下一次 `FeedbackBundle`，完成 **A→B→A 完整自我进化**

本轮只是被 factor 代码的 output format 绊住了一步，**链路本身没有任何一处需要再动**。

## L2-6. 成本与耗时

| 项 | 值 |
|---|---|
| 端到端墙钟 | ~4m15s |
| LLM 调用累计 | $0.0092596000 (~0.93 美分) |
| 走完的 Step | 0 (exp_gen) → 1 (coding) → 3 (feedback) |
| 触发的修复 | 3 个永久 patch（UTF-8 inject、WSL 执行、CondaConf 占位）|

## L2-7. 下一步最低成本验证建议

1. 提高 `FACTOR_CoSTEER_MAX_LOOP=3`（上限 3 轮），让 CoSTEER 自我纠正 output format
2. `--loop_n=1` 保持不变，只在 CoSTEER 内部多尝试几次
3. 若第 2~3 轮 factor 通过校验，Step `running` 会自动触发 qrun，届时会看到：
   - `PortAnaRecord`、`IC`、`ICIR`、`annualized_turnover`、`max_drawdown`
   - 结果写入 `mlflow`/`factor_validation/reports/`
4. 再跑一次 `python -m scripts.lab.build_feedback_bundle`，`cycles` 会 +1，`fails` 可能 +1
5. 再跑一次同样的 Level-2，**H.1 检索 query 会真的命中 G.3 TF-IDF 语料**（本轮 `fails=0` 所以 similar-failures 段是空的，这也是设计正确行为）

