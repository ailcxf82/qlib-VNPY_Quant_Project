# STAGE H 测试深度方案（H.1 自动 retrieval_query）

本文是阶段 H 的测试设计文档，目标是把"能跑过单测"升级为"可发布、可观测、可回滚"的工程门槛。

---

## 1. H.1 的核心变化与测试风险

### 1.1 变化点

阶段 H.1 把 G.3 的 `retrieval_query` 从"调用方手工传参"升级为"`prepare_context` 自动推导"：

- 新增自动推导函数：
  - `_extract_candidate_query_parts(trace)`
  - `_infer_retrieval_query(trace, ctx)`
  - `_truncate_query(text, max_chars=220)`
- `ProjectQlibQuantHypothesisGen.prepare_context(...)` 改为：
  1) 先读 `super().prepare_context` 的 `ctx["RAG"]`
  2) 自动推导 query
  3) 调 `compose_project_rag(..., retrieval_query=<auto>)`

### 1.2 主要风险

| # | 风险 | 触发条件 | 影响面 |
|---|---|---|---|
| R1 | 误提取 | trace 结构变化 / 字段名不一致 | 检索为空或漂移 |
| R2 | query 过长 | trace/RAG 尾巴是大段文本 | prompt token 爆炸 |
| R3 | 隐式行为变化 | 原不传 query 的路径现在传 | G.3 段意外出现 |
| R4 | 兼容性 | dict / object / None trace | 路径裂变、难复现 |
| R5 | 故障传播 | 自动推导抛异常 | 拖垮 `prepare_context` 主流程 |

---

## 2. 测试分层（由内到外）

采用 5 层测试金字塔，避免只靠 e2e：

### L0 — 纯函数单测（毫秒级）

覆盖对象：`_truncate_query` / `_extract_candidate_query_parts` / `_infer_retrieval_query`

验证点：
- 优先级：`ctx["RETRIEVAL_QUERY"]` > trace 字段 > `ctx["RAG"]` 末行 > `None`
- 输入形态：dict trace / object trace / hist 混合 tuple+dict / trace=None
- 限长：超过上限必须截断并带 `...`
- 空输入：返回 `None`，不抛异常

对应测试文件：`tests/factor_lab/adapters/test_quant_proposal_h1.py`

### L1 — 适配器接线单测（无 I/O）

覆盖对象：`ProjectQlibQuantHypothesisGen.prepare_context`

验证点：
- 通过 monkeypatch 捕获 `compose_project_rag` 参数，断言 `retrieval_query` 确实传递
- `super().prepare_context` 返回 `ok=True/False` 时行为正确
- 在 `trace=None` 情况下仍不抛异常
- 自动推导出错时 `prepare_context` 不崩

对应测试文件：`tests/factor_lab/adapters/test_quant_proposal_h1.py`

### L2 — 组合函数回归（有轻量 I/O）

覆盖对象：`compose_project_rag` + feedback latest.json

验证点：
- 无 query 时行为与 G 阶段一致（不出现 G.3 段）
- 自动 query 生效时，RAG 按 `static → dynamic → retrieval` 顺序追加
- bundle 缺失 / 损坏时安全降级

对应测试文件：`tests/factor_lab/feedback/test_feedback_embedding.py`

### L3 — Offline E2E（不调 LLM）

覆盖对象：`aggregator → write_feedback_bundle → prepare_context → 最终 RAG`

验证点：
- 真实 artifact 输入下，`prepare_context` 不需要手工 query 也能得到带 G.3 段的最终 prompt
- `e2e_rdagent` marker 逻辑不变（默认 skip，显式才跑）

对应测试文件：`tests/e2e/test_rdagent_feedback_e2e.py`

### L4 — 手工在线冒烟（调真实模型，可选）

验证点：
- 自动 retrieval 是否改善"重复踩坑"行为（LLM 产出是否主动规避相似失败家族）
- prompt token 增量是否可接受（相比 G 阶段）

执行方式：沿用 `STAGE_G_TEST_PLAYBOOK` live smoke 流程，仅把"query 手输"步骤去掉，观察自动行为。

---

## 3. 覆盖矩阵（风险 × 测试）


| 风险                   | L0 | L1 | L2 | L3 | L4 |
| -------------------- | -- | -- | -- | -- | -- |
| R1 误提取 / 取不到          | ✅  | ✅  | ✅  | ✅  | ✅  |
| R2 query 过长          | ✅  | ✅  | ⚪  | ⚪  | ✅  |
| R3 行为突变（无 query 也多段） | ⚪  | ✅  | ✅  | ✅  | ✅  |
| R4 兼容性（trace 结构）     | ✅  | ✅  | ⚪  | ✅  | ⚪  |
| R5 故障传播（不能拖垮主流程）     | ✅  | ✅  | ✅  | ✅  | ✅  |

> 说明：⚪ 表示间接覆盖。

---

## 4. 质量门槛（发布前必须满足）

### 4.1 必过命令

```powershell
python -m pytest tests/factor_lab/adapters/test_quant_proposal_h1.py -q
python -m pytest tests/factor_lab/feedback/test_feedback_embedding.py -q
python -m pytest tests/e2e/test_rdagent_feedback_e2e.py -q
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

### 4.2 通过标准

- H.1 新增测试全绿；
- F/G 既有主干测试不降级；
- `tests/test_unified_strategy.py` 之外无新增失败；
- `e2e_rdagent` 显式运行保持通过。

---

## 5. 观测与回滚策略

### 5.1 线上观测指标（已落地）

`ProjectQlibQuantHypothesisGen.prepare_context` 内置 debug 日志（H.1）：

- `auto_query_source`：`override` / `trace` / `rag_tail` / `none`
- `auto_query_len`：截断后的 query 长度
- 日志 logger：`factor_lab.adapters.quant_proposal`

排查方式：
```python
import logging
logging.getLogger("factor_lab.adapters.quant_proposal").setLevel(logging.DEBUG)
```

### 5.2 异常回滚开关（已预留）

`ProjectQlibQuantHypothesisGen.auto_retrieval_enabled` 类属性：

- 默认 `True`，启用自动推导；
- 置 `False` 等同退回 G 阶段"不自动检索"的模式，`compose_project_rag` 不会被传 query。

示例：
```python
ProjectQlibQuantHypothesisGen.auto_retrieval_enabled = False
```

---

## 6. 下一步测试增强（H.2+，可选）

1. **属性测试（property-based）**：随机 trace 结构，保证 `_infer_retrieval_query` 永不抛异常、输出长度有界。
2. **token 预算回归**：对固定输入快照记录最终 RAG 长度，防止隐式膨胀。
3. **A/B 行为测试**：同一批 hypothesis，比较自动 retrieval 开/关时 LLM 对 discouraged 家族的命中率。
4. **噪声鲁棒性**：在 query 注入无关术语（例如日志文本）时，检索 top-k 不应明显劣化。

---

## 7. 当前结论

H.1 的测试策略已达到"可发布"粒度：

- 有纯函数级保护（定位快）；
- 有接线级断言（防 silent break）；
- 有组合回归和 e2e（防系统级回归）；
- 有线上观测 + 类级回滚开关（防上线黑盒）。

阶段 H 可以在这里收口（作为"第一版真正可用"的里程碑），或按 STAGE_G_REPORT §8 的预案继续推进 H.2（penalty 打分）/ H.3（registry 审计）/ H.4（embedding backend AB）。
