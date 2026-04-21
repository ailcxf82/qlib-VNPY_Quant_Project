# Stage H 收口报告（H.1 自动 retrieval_query + 测试深度方案）

> 对应用户诉求："进入 H 阶段，然后深度考虑测试怎么做"。
> 阶段定位：把 G.3 的"带 query 的相似失败检索"从手工传参升级为自动推导，
> 并为之建立可发布级的测试/观测/回滚基座。

---

## 1. 目标回顾

F 阶段完成了"YAML 化 + universe 维度 + e2e 冒烟"；
G 阶段完成了"by_universe 子包 + shim 硬弃用 + embedding 检索 + soft penalty"。
进入 H 阶段时还有一个关键痛点：

> G.3 引入的 `retrieval_query` 需要外部显式传参，否则拿不到"针对性历史教训"。
> 这不符合"RD-Agent loop 自足闭环"的初衷。

H 阶段的本次交付聚焦 **H.1 + 测试方法论**，把 G.3 从"可选开关"升级为"默认在线的默认行为"，并给出一套后续可复用的测试分层策略。

---

## 2. 交付清单

### 2.1 代码改动


| 文件                                                    | 状态  | 说明                                                                                                                                                                                                                   |
| ----------------------------------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factor_lab/adapters/quant_proposal.py`               | 修改  | 新增 `_truncate_query` / `_extract_candidate_query_parts` / `_infer_retrieval_query[_with_source]`；`ProjectQlibQuantHypothesisGen.prepare_context` 自动推导并传入 `retrieval_query`；加入观测日志、回滚开关 `auto_retrieval_enabled`、异常兜底 |
| `tests/factor_lab/adapters/test_quant_proposal_h1.py` | 新建  | 12 个针对 H.1 的单测（含回滚开关、异常兜底、source 标签）                                                                                                                                                                                 |
| `docs/STAGE_H_TEST_STRATEGY.md`                       | 新建  | H 阶段 5 层测试策略 / 覆盖矩阵 / 质量门槛 / 观测回滚说明                                                                                                                                                                                  |
| `docs/STAGE_H_REPORT.md`                              | 新建  | 本报告                                                                                                                                                                                                                  |


### 2.2 功能语义

自动推导优先级（从高到低）：

1. `ctx['RETRIEVAL_QUERY']`（显式 override，给热修/实验预留通道）—— `source=override`
2. `trace` 语义字段：`hypothesis_text` / `description` / `latest_hypothesis` /
  `current_hypothesis` / `current_task` / `trace.hist` 最后 3 条 —— `source=trace`
3. 兜底：`ctx['RAG']` 末行文本 —— `source=rag_tail`
4. 全部为空 —— `source=none`，不注入 G.3 段

长度统一通过 `_truncate_query(max_chars=220)` 截断，防止 prompt token 爆炸。

### 2.3 可观测与回滚（工程基线）

- DEBUG 日志：`factor_lab.adapters.quant_proposal` logger 输出
`H.1 auto_retrieval source=<...> len=<...>`。
- 类级回滚开关：`ProjectQlibQuantHypothesisGen.auto_retrieval_enabled = False`
可完全退回 G 阶段"不自动检索"行为。
- 异常兜底：`_infer_retrieval_query_with_source` 抛异常 → 降级为 `None` +
`source=error`；`compose_project_rag` 抛异常 → 保留 super 生成的 RAG，不覆盖。
- 任何失败路径均不会让 `prepare_context` 向调用方抛异常。

---

## 3. 测试结果

### 3.1 命令

```powershell
conda run -n qlib_zhengshi python -m pytest `
  tests/factor_lab/adapters/test_quant_proposal_h1.py `
  tests/factor_lab/feedback/test_feedback_embedding.py `
  tests/factor_lab/feedback/test_feedback_by_universe.py `
  tests/factor_lab/config/test_constitution_penalty.py `
  tests/factor_lab/adapters/test_adapters_refactor.py `
  tests/e2e/test_rdagent_feedback_e2e.py -q

conda run -n qlib_zhengshi python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

### 3.2 结果


| 范围                                   | 结果                        |
| ------------------------------------ | ------------------------- |
| H.1 + G 针对性集                         | **71 passed, 3 skipped**  |
| 全量回归（排除历史已知 `test_unified_strategy`） | **434 passed, 3 skipped** |
| e2e_rdagent（默认 skip，显式 marker 通过）    | 行为与 F/G 一致                |


> 3 个 skipped 对应 `@pytest.mark.e2e_rdagent`，沿用 F 阶段设定"默认不跑、显式 marker 才跑"。

---

## 4. 测试深度方案（一段落总结）

详细内容见 `docs/STAGE_H_TEST_STRATEGY.md`。核心观点：

- **L0 纯函数单测**：`_truncate_query` / `_extract_candidate_query_parts` / `_infer_retrieval_query[_with_source]`。
- **L1 适配器接线单测**：`prepare_context` 透过 monkeypatch 捕获 `compose_project_rag` 入参，覆盖自动 / 显式 override / 回滚 / 异常四条路径。
- **L2 组合回归**：`compose_project_rag + feedback latest.json`，保证段序与安全降级。
- **L3 offline e2e**：不调 LLM，跑 `aggregator → write_feedback_bundle → prepare_context`。
- **L4 可选 on-line 冒烟**：真实模型验收"自动 retrieval 改善踩坑率"。

覆盖矩阵把 5 个主要风险（误提取 / 过长 / 行为突变 / trace 结构兼容 / 故障传播）全部对齐到 L0–L4 某一层，且都至少被 2 层覆盖。

---

## 5. 非回归承诺

- 旧 `compose_project_rag(..., retrieval_query=None)` 调用签名/行为保持不变；
- 旧的 `_infer_retrieval_query` 对外仍可用（保留为薄 wrapper）；
- F 阶段字节级对齐契约（static constitution 不含 penalty 时的渲染）未被改动；
- G 阶段 by_universe / shim 硬弃用 / soft penalty 行为完全保留；
- 既有 e2e marker 策略未变。

---

## 6. 已知限制与后续建议

- `_extract_candidate_query_parts` 当前是"白名单字段 + hist tail" 的启发式方案，
若 RD-Agent 升级后改字段名，仍需在此处补字段。建议下一个阶段引入 schema 化的
trace 抽取适配器。
- `TfidfBackend` 仍是 G.3 的零依赖实现，未做 SBERT / OpenAI embedding AB。
观察一段线上行为后再决定是否升级。
- 未做"token 预算快照测试"——prompt 膨胀需要继续监控，后续可在 L2 增加固定输入
的最终 RAG 长度快照。

---

## 7. 是否继续 H.2 / H.3 / H.4？

STAGE_G_REPORT §8 预案里列过：

- **H.2**：`penalty` 真正参与评分（目前只是文本注记，尚未回灌到 scorer）；
- **H.3**：registry 决策（active / retired）的审计沿革视图；
- **H.4**：embedding backend AB（TF-IDF vs 预训练向量）。

三者都是"锦上添花"，当前 H.1 + 测试方法论本身已经让 RD-Agent loop 自足闭环，
也给出了"可发布、可观测、可回滚"的标准。因此建议把 H 阶段分为两个里程碑：

- **H（当前）**：本报告所述 H.1 + 测试基座——**可以作为第一版真正可用的版本收口**。
- **H+（可选延伸）**：按需推进 H.2–H.4，每个子项独立评估 ROI。

是否继续 H+ 由使用方决定；本报告视作 H 主线收口。