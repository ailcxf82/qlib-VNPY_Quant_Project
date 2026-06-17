# STAGE G 测试与运维指南

本指南面向研究员 / 运维，覆盖阶段 G 四个子项的日常验证、问题定位与扩展要点。阶段 F 的
`STAGE_F_TEST_GUIDE.md` 仍然有效，本文只补充 G 阶段的新增部分。

## 1. 目录速查

- [2. by_universe（G.1）](#2-by_universe-g1)
- [3. rdagent_integration shim 硬下线（G.2）](#3-rdagent_integration-shim-硬下线-g2)
- [4. embedding-based RAG 检索（G.3）](#4-embedding-based-rag-检索-g3)
- [5. discouraged penalty 软降权（G.4）](#5-discouraged-penalty-软降权-g4)
- [6. 测试分层 & 命令清单](#6-测试分层--命令清单)
- [7. 常见故障排查](#7-常见故障排查)

## 2. by_universe（G.1）

### 数据结构

`FeedbackBundle.by_universe: dict[str, UniverseSubBundle]`

每个子桶 `UniverseSubBundle` 字段：

- `universe: str` —— 与字典 key 必须相等。
- `active_factors`, `retired_factors`, `recent_fails` —— 仅包含该 universe 下的条目。
- `failure_family_counts: dict[str, int]` —— 本桶范围内的家族失败计数。
- `discouraged_families: tuple[str, ...]` —— 本桶内 `fails≥阈值` 或在 retired 出现过的 family（字典序）。

### 产出位置

```
factor_lab/workspace/feedback/
├── latest.json            # 全局视图（向后兼容）
├── latest.md              # 人工阅读，含 [Per-universe view] 段
├── latest_csi300.json     # G.1 新增：每个 universe 一份 UniverseSubBundle
├── latest_csi500.json
└── history/...
```

### 使用建议

- 对外消费一律读 `latest.json`（包含 by_universe 字段），保证 diff 最小。
- 如果你只关心某个 universe 的局部状态，用 `UniverseSubBundle.model_validate_json((workspace / f"latest_{u}.json").read_text())` 直接反序列化即可。
- universe 标签缺失（老数据 / 上游忘记填）的条目**不会**进任何桶，只会出现在父 bundle 的平铺段。遇到 `latest_xxx.json` 缺口时，先查 manifest / C1.universe 是否填全。

### 扩展点

- 如果要让 aggregator 用不同的阈值给不同 universe 做 discouraged 推导：改 `_derive_by_universe` 的 `min_fail_count` 参数（目前与父级共用同一个阈值）。
- 如果要按 universe 导出 markdown：扩 `write_feedback_bundle` 在 `also_write_per_universe=True` 时额外写 `latest_<universe>.md`（当前只写 json，避免重复渲染）。

## 3. rdagent_integration shim 硬下线（G.2）

### 行为契约

以下模块 import 时**立即** `RuntimeError`，异常信息包含新 import 路径：


| 老路径                                          | 新路径                                      |
| -------------------------------------------- | ---------------------------------------- |
| `rdagent_integration.project_experiments`    | `factor_lab.adapters.experiments`        |
| `rdagent_integration.project_proposal`       | `factor_lab.adapters.proposal`           |
| `rdagent_integration.project_quant_proposal` | `factor_lab.adapters.quant_proposal`     |
| `rdagent_integration.patch_qlib_conda`       | `factor_lab.adapters.patch_qlib_conda`   |
| `scripts/run_fin_quant.py`                   | `python -m scripts.lab.run_rdagent_loop` |


### 迁移步骤

1. `grep -R "rdagent_integration" your_code/` 找出所有引用。
2. 按上表替换 import 语句；符号名都对齐了，无需改调用方。
3. 如果 CI 里调用 `python scripts/run_fin_quant.py ...`，改为 `python -m scripts.lab.run_rdagent_loop ...`（参数不变）。
4. 跑 `python -m pytest tests/factor_lab/adapters/test_adapters_refactor.py`，12 个用例应全绿。

### 故障

- 运行时异常里只要出现 `已在阶段 G.2 下线` 或 `factor_lab.adapters` 字样，就是直接按迁移表改 import 即可。
- 如果要回退到 F 阶段的 DeprecationWarning 行为（不推荐），删掉 `rdagent_integration/__init__.py` 和各子模块里的 `raise RuntimeError`，恢复原先的 `warnings.warn(..., DeprecationWarning)` 写法。但这样会违背 G.2 的约束。

## 4. embedding-based RAG 检索（G.3）

### 什么时候启用

默认**不启用**：`compose_project_rag` 的 `retrieval_query` 参数默认 `None`，行为与 F 阶段一致。

启用方式（研究员手动 / 脚本调用）：

```python
from factor_lab.adapters.quant_proposal import compose_project_rag

rag_prompt = compose_project_rag(
    base_rag="",
    feedback_dir=Path("factor_lab/workspace/feedback"),
    include_dynamic=True,
    retrieval_query="short cycle volume price reversal",
    retrieval_top_k=3,
)
```

输出在已有 RAG 段之后追加：

```
------Similar past failures retrieved for query="short cycle volume price reversal" (G.3 top-3)------
  - sim=0.812  VolRev_5d   (family=volume_price_reversal)  stage=default  decision=FAIL  modes=[ic] [universe=csi300]
  - sim=0.731  VolRev_10d  ...
Interpretation: these are the closest past failures to the current direction; ...
```

### 可替换 backend

`EmbeddingBackend` Protocol 定义了 `fit(corpus)` + `similarity(query, doc)` 两个方法。实现自己的 backend（例如调 litellm embedding API），然后：

```python
from factor_lab.feedback.embedding import retrieve_similar_failures
hits = retrieve_similar_failures(bundle, query="...", backend=MyCustomBackend())
```

### 阈值调参

- `top_k`：默认 3，LLM prompt 不适合塞太多条。常见调到 5。
- `min_score`：默认 0.05，避免完全无关的条目。如果召回太稀，降到 0.02；如果噪声太多，升到 0.1。
- **不要**把 `top_k` 设成很大再靠 LLM 过滤，prompt token 成本会被浪费。

## 5. discouraged penalty 软降权（G.4）

### YAML 配置

`factor_lab/config/rag_constitution.yaml` 的 `discouraged_families`：

```yaml
discouraged_families:
  - id: x
    text: "Short-cycle volume-price reversals on W in {5, 10}"
    # penalty: .inf         # （默认）硬黑；不写就等同这个
  - id: y
    text: "Same-day volume spike + price reversal patterns"
    penalty: -0.5             # 软降权：仍可尝试，需附加理由
  - id: z
    text: "Anything that ranks the universe with >50% weekly turnover"
    penalty: -1.0
```

### 渲染效果

软降权条目追加：

```
   (y) Same-day volume spike + price reversal patterns   [penalty=-0.5; soft — new attempts allowed only with explicit justification of how this proposal differs]
```

硬黑条目渲染不变（byte-for-byte 兼容 F 阶段 `_FALLBACK_CONSTITUTION_TEXT`）。

### 语义规则


| penalty 值                  | 含义   | 渲染行为                                                    |
| -------------------------- | ---- | ------------------------------------------------------- |
| 缺省 / `.inf`                | 硬黑名单 | 与 F 阶段一致，无额外注解                                          |
| 具体 float（`-0.5`, `-1.0` 等） | 软降权  | 追加 `[penalty=X; soft — ...]`                            |
| NaN / 非数值                  | 非法   | `render_constitution` 抛 ValueError；`load` 层回退到 fallback |


### 升级步骤

1. 研究员编辑 `rag_constitution.yaml`，给想软降的条目加 `penalty: -0.5`。
2. 跑 `python -m pytest tests/factor_lab/config/ -q`，9 个 G.4 用例 + 原 F.3 用例全绿。
3. 运行一次 `compose_project_rag` 查看渲染（见 STAGE_G_TEST_PLAYBOOK §4）。
4. 跑一轮线上 RD-Agent loop，人工核对 prompt dump 里是否出现新注解。

## 6. 测试分层 & 命令清单


| 层级            | 命令                                                                           | 预期                     |
| ------------- | ---------------------------------------------------------------------------- | ---------------------- |
| G.1 单测        | `python -m pytest tests/factor_lab/feedback/test_feedback_by_universe.py -v` | 13 passed              |
| G.2 契约测       | `python -m pytest tests/factor_lab/adapters/test_adapters_refactor.py -v`    | 12 passed              |
| G.3 检索测       | `python -m pytest tests/factor_lab/feedback/test_feedback_embedding.py -v`   | 21 passed              |
| G.4 penalty 测 | `python -m pytest tests/factor_lab/config/test_constitution_penalty.py -v`   | 9 passed               |
| F + G 综合      | `python -m pytest tests/factor_lab tests/e2e -q`                             | 全绿                     |
| 全量回归          | `python -m pytest tests -q --ignore=tests/test_unified_strategy.py`          | 421 passed, 3 skipped  |
| 仅 E2E marker  | `python -m pytest tests/e2e -m e2e_rdagent -v`                               | 3 passed, 2 deselected |


## 7. 常见故障排查

### 7.1 `ImportError: cannot import name ... from rdagent_integration`

实际是 `RuntimeError`（G.2 硬下线）。按 §3 迁移表改 import 即可。

### 7.2 `latest_<universe>.json` 没生成

检查：

1. `write_feedback_bundle(..., also_write_per_universe=True)`（默认 True）。
2. bundle.by_universe 是否为空（说明所有因子都没有 universe 标签）。查 manifest.json 和 cert.candidate 里 universe 字段。

### 7.3 G.3 检索返回空

- 先看 query 是否是空字符串或纯空白 —— 这会被有意拒绝。
- 看 `bundle.recent_fails` 是否非空。
- 看 `min_score`（默认 0.05）是否被某场景调得太高。
- 打日志：`factor_lab.feedback.embedding` 会 warning 任何 fit / similarity 异常。

### 7.4 G.4 penalty 不生效

- 检查 YAML 是否真的被 load（`reload_constitution_text()` 后再渲染）。
- 确认 penalty 是数值而非字符串（YAML 里写 `penalty: -0.5` 而非 `penalty: "-0.5"`）。
- 若 penalty 非法，会整体回退到 fallback；查 `logger` 里的 warning 日志。

### 7.5 F.3 byte-for-byte 测试红

G.4 的默认 YAML 没带 penalty 字段，因此渲染与 fallback 字节相等。若你编辑了默认 YAML，并引入了新的文本行/缩进差异，这条契约会红。恢复方法：

- 最小化 YAML 改动：只给研究员自定义的 `discouraged` 加 `penalty`，不修改其他文本。
- 或者同步重算 `_FALLBACK_CONSTITUTION_TEXT`（见 `factor_lab/config/constitution.py` 末尾注释）。

