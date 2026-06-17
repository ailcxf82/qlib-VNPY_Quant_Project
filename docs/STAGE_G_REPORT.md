# 阶段 G — 回路增强、shim 下线、检索注入、软降权

## 1. 目标

阶段 G 在阶段 F（YAML 化 + universe 基础 + E2E 测试）之上，完成四个方向的能力增强与清理：


| 子项  | 目标                                                                                                                                        |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| G.1 | 在 `FeedbackBundle` 上加一个真正按 universe 分桶的子结构 `by_universe`；保留 F.2 的平铺字段作"全局视图"。                                                             |
| G.2 | `rdagent_integration.*` 与 `scripts/run_fin_quant.py` 的 DeprecationWarning shim 升级为 **import 时立即 `raise RuntimeError`**，附完整迁移指引。           |
| G.3 | 在 `factor_lab.feedback` 里引入 embedding-based 检索，对最近 N cycle 的失败候选做 TF-IDF 相似度召回，作为 RAG 的"针对性历史教训"段（opt-in）。                                |
| G.4 | `rag_constitution.yaml` 的 `discouraged_families` 增加可选 `penalty` 字段。`penalty=inf / 不设置` 保持 F 阶段硬黑语义；具体数值（如 -0.5）渲染为"软降权"注解，LLM 仍可尝试但须附加理由。 |


四个决策由用户在 Stage G 启动前确认：

- 范围：G.1 + G.2 + G.3 + G.4（full4）
- universe 分桶：additive（平铺段保留，by_universe 以增量字段挂上）
- shim 策略：err_shim（硬下线为 RuntimeError，但保留模块文件承载迁移指引）
- discouraged 语义：soft_score（penalty 打分，inf = 硬黑）

## 2. 关键交付物

### 2.1 G.1：by_universe 子结构

新增类 `UniverseSubBundle`（`factor_lab/feedback/schema.py`）：

```python
class UniverseSubBundle(BaseModel):
    universe: str
    active_factors: tuple[ActiveFactorSummary, ...]
    retired_factors: tuple[RetiredFactorSummary, ...]
    recent_fails: tuple[FailedCandidateSummary, ...]
    failure_family_counts: dict[str, int]
    discouraged_families: tuple[str, ...]
```

`FeedbackBundle` 增字段 `by_universe: dict[str, UniverseSubBundle] = {}`，默认空 dict 保证老 JSON 向后兼容。校验器强制 `by_universe[key].universe == key`，且 `cycle_id ⊆ cycles_included`。

聚合器 `_derive_by_universe`（`factor_lab/feedback/aggregator.py`）按规则：

- universe=None 的条目**不**进任何桶（保留在父 bundle 平铺段）；
- 每桶独立统计 `failure_family_counts` 与 `discouraged_families`；
- 按 universe 字典序排序，diff 友好。

渲染：`FeedbackBundle.to_markdown()` 在结尾追加 `[Per-universe view (G.1) ...]` 段，每个 universe 列一行 active/retired/fails 计数 + 本地失败家族分布 + 本地 discouraged。

磁盘：`write_feedback_bundle` 多写 `latest_<universe>.json`（`UniverseSubBundle.model_dump`），下游按需消费。

### 2.2 G.2：旧 shim 硬下线

以下模块 import 时**立即** `raise RuntimeError`：

- `rdagent_integration/__init__.py`
- `rdagent_integration/project_experiments.py`
- `rdagent_integration/project_proposal.py`
- `rdagent_integration/project_quant_proposal.py`
- `rdagent_integration/patch_qlib_conda.py`
- `scripts/run_fin_quant.py`

每条异常 message 均明确指出新 import 路径 / 新 CLI 命令，便于研究员/外部 CI 自助迁移。

一并修正了 `scripts/smoke_p3a_reward.py` 的两处旧路径引用。

### 2.3 G.3：embedding-based 检索

新增 `factor_lab/feedback/embedding.py`，包含：

- `_tokenize` —— snake_case + CamelCase + 全大写缩写 + 数字边界切词（纯 stdlib）。
- `TfidfBackend` —— 经典 TF-IDF，IDF 带 `log((N+1)/(df+1))+1` smoothing；cosine 相似度归一，无外部依赖。
- `EmbeddingBackend` Protocol —— 预留接口，未来可替换 sentence-transformers / litellm embeddings。
- `RetrievedFailure` —— 召回结果 dataclass（score + doc_text + 原 summary）。
- `build_failure_corpus` / `retrieve_similar_failures` —— 一站式 API。
- `render_similar_failures_section` —— 渲染成 RAG 段。

集成点：`factor_lab/adapters/quant_proposal.compose_project_rag` 新增两个**可选**参数：

```python
compose_project_rag(
    base_rag,
    feedback_dir=None,
    include_dynamic=True,
    retrieval_query=None,      # G.3：None/空 → 完全不启用检索，行为与 F 阶段一致
    retrieval_top_k=3,
)
```

分段固定顺序：

```
<base_rag>
→ static constitution
→ dynamic feedback
→ similar past failures retrieved for query=...  (G.3 段，仅当 query 非空且召回非空时出现)
```

### 2.4 G.4：discouraged 软降权

`rag_constitution.yaml.discouraged_families` 条目可选带 `penalty: <number>`：

- 缺省 / `penalty=.inf` → 硬黑名单，渲染完全保留 F 阶段原样（默认 YAML 与 `_FALLBACK_CONSTITUTION_TEXT` 的 byte-for-byte 契约保持不变）。
- 具体数值（如 `penalty=-0.5`）→ 渲染追加 `[penalty=-0.5; soft — new attempts allowed only with explicit justification of how this proposal differs]`。
- 非数值 / NaN → `render_constitution` 抛 `ValueError`；`load_constitution_text` 捕获后整体回退到 fallback，保证系统零停机。

## 3. 技术说明

### 3.1 by_universe 设计权衡

**为什么不替换平铺段？** 用户选择的 additive 策略把 by_universe 作为"分桶视图"。父 bundle 的平铺段（active_factors / retired_factors / recent_fails / counts / discouraged）仍然是完整的全局事实源；by_universe 只是按 universe 重新分桶的投影。这样：

1. 老消费者（RAG 注入器、promote 脚本、smoke test）完全无感知；
2. LLM 既能看到全局 discouraged，也能顺着 `[Per-universe view]` 看到分 universe 的局部趋势；
3. 如果未来要让 RAG 完全按 universe 切分，只需在 compose 层加参数选择"平铺 vs 分桶视图"，数据结构不用再动。

### 3.2 硬下线 shim 的影响半径

G.2 的下线会让任何继续 `import rdagent_integration.`* 或 `python scripts/run_fin_quant.py` 的代码立即红。这是有意为之：F 阶段的 DeprecationWarning 在真实运行中常被 `-W ignore` 过滤掉，导致技术债长期蒙混。异常里必带 new-path hint，所以修复路径明确。唯一需注意：如果项目外部（如部署脚本、CI workflow）还在用老命令，需要同步更新到 `python -m scripts.lab.run_rdagent_loop`。

### 3.3 TF-IDF 为什么够用

RAG 里真正想捕捉的是"这个候选是否接近某个历史失败的关键词骨架"。对因子名 `VolRev_5d` / `VolRev_10d` / `VolumePriceTrend_10D` 来说，token 层重叠已经足够区分度。经典 TF-IDF + cosine：

- 纯 stdlib，不引 scikit-learn / sentence-transformers；
- 单次 fit+retrieve 数量级 O(V×N)，N≤100 时毫秒量级；
- Protocol 开放，未来换成 litellm embedding API 只需新实现一个 `EmbeddingBackend`。

检索默认 `top_k=3, min_score=0.05`：太强的阈值会漏召回，太弱会灌噪声。`min_score` 也保证与 query 无关的旧失败不会污染 RAG。

### 3.4 为什么 penalty 默认对齐"硬黑"

F 阶段已经通过 byte-for-byte 测试把原始宪法文本锁死；G.4 的 penalty 机制如果默认就改写渲染，所有 F 阶段下游（包括 `_FALLBACK_CONSTITUTION_TEXT`）都得重算。因此：

- 默认 YAML 里**不**带 `penalty` 字段；
- 即使带了 `penalty=.inf`，渲染也等同于没带（我们有专门测试）；
- 只有研究员主动改 YAML 给某条加具体数值时，才会出现软降权注解。

这保证 F 的契约兼容，也让 G.4 成为"研究员可选启用"的新维度。

## 4. 测试覆盖


| 文件                                                       | 用例数 | 覆盖点                                                                                                                                             |
| -------------------------------------------------------- | --- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/factor_lab/feedback/test_feedback_by_universe.py` | 13  | UniverseSubBundle 校验、by_universe 默认/向后兼容/key 一致性/cycle_id 约束、aggregator 构造、write 落 latest_.json、markdown per-universe 段。                        |
| `tests/factor_lab/adapters/test_adapters_refactor.py`    | 12  | 新家 4 处 import、老 shim 5 个模块 import 立即 RuntimeError + 迁移提示、`scripts/run_fin_quant.py` 同样 raise、compose_project_rag 可用、`_PROJECT_FACTOR_RAG` 别名保留。 |
| `tests/factor_lab/feedback/test_feedback_embedding.py`   | 21  | 分词规则（含全大写缩写）、TF-IDF 相似度排序、召回安全降级（空 query/empty bundle/top_k=0）、top_k 上限、min_score 过滤、compose_project_rag 集成（opt-in、section 顺序、bundle 缺失安全）。     |
| `tests/factor_lab/config/test_constitution_penalty.py`   | 9   | 默认 YAML 不带 penalty、inf 与无 penalty 等价、finite penalty 注解、非法/NaN 抛错、load 层回退、多条混合硬/软。                                                              |


**全量回归**：`python -m pytest tests -q --ignore=tests/test_unified_strategy.py` → **421 passed, 3 skipped**（F 阶段为 389；净增 32 个用例；`test_unified_strategy.py` 的两条失败是与 G 无关的预存问题，同 F 阶段豁免）。

标记测试 `@pytest.mark.e2e_rdagent` 默认 skip 行为不变（F.4 契约）。

## 5. 非回归保证

- **F.3 byte-for-byte 契约**：`test_default_yaml_renders_byte_equal_to_fallback` 和 `test_hardcoded_static_constitution_equal_to_fallback` 继续绿。
- **F.2 universe 字段**：所有 `test_feedback_universe.py` 用例仍绿。
- **E.3 RAG 集成 / F.4 E2E**：所有 `tests/e2e/test_rdagent_feedback_e2e.py` 用例仍绿。
- **老 import 路径**：G.2 后**必定**红，且异常信息明确指向新路径；这是功能变化而非回归。

## 6. 产物路径索引

- 代码：
  - `factor_lab/feedback/schema.py` —— `UniverseSubBundle`、`FeedbackBundle.by_universe`
  - `factor_lab/feedback/aggregator.py` —— `_derive_by_universe`、`write_feedback_bundle` per-universe 落盘
  - `factor_lab/feedback/embedding.py` —— 新增整模块
  - `factor_lab/feedback/__init__.py` —— 导出 `UniverseSubBundle`
  - `factor_lab/adapters/quant_proposal.py` —— `_render_similar_failures` + `compose_project_rag` 扩参
  - `factor_lab/config/constitution.py` —— penalty 支持
  - `factor_lab/config/rag_constitution.yaml` —— penalty 字段文档注释
  - `rdagent_integration/`* —— 硬下线 shim
  - `scripts/run_fin_quant.py` —— 硬下线
  - `scripts/smoke_p3a_reward.py` —— 旧 import 清理
- 测试：上表 4 个文件。
- 文档：`docs/STAGE_G_REPORT.md`（本文件）、`docs/STAGE_G_TEST_GUIDE.md`、`docs/STAGE_G_TEST_PLAYBOOK.md`。

## 7. 已知限制

- G.3 检索 backend 只有 TF-IDF；对真正语义（例如"均值回归"和"reversion"）理解有限。未来若需要，替换 `EmbeddingBackend` 即可。
- G.3 的 query 目前需要调用方显式传入；与 RD-Agent trace 的自动对接放到阶段 H 再做。
- G.4 的 penalty 只体现在 RAG 文本里，LLM 是否真的执行软降权需要线上观察；短期内仍以 F 阶段的硬黑语义为主。
- by_universe 的落盘文件 `latest_<universe>.json` 当前未接入 `latest.md`（单独的子视图 md 不写），理由是 `latest.md` 里的 Per-universe view 段已足够；如果需要独立 md，可在 write 层再扩一行，代价极小。

## 8. 阶段 H 预案（供参考）

- **H.1**：把 G.3 的 retrieval_query 与 RD-Agent trace/hypothesis 自动对接，研究员无需传 query。
- **H.2**：penalty 从"注解"升级为"打分"：在 compose 层引入一个轻量计分器，把 discouraged 软降权分数累加到 trace summary 里，LLM prompt 看到具体数字。
- **H.3**：把 `UniverseSubBundle` 的 `by_universe` 文件纳入 factor_registry 的审计清单，给 L3 报表用。
- **H.4**：embeddings backend 替换（sentence-transformers / litellm），做 online A/B 对照现行 TF-IDF。

（以上仅为预案；正式开始前与用户再对齐范围 / 策略 / 模式。）