# 阶段 F 交付报告（配置化 + universe 维度 + 端到端闭环验证）

> 阶段 F 在阶段 E 的基础上把**固化在 Python 字符串与 regex 列表里的领域知识**
> 迁到 YAML，把 `universe` 一等地位带进 FeedbackBundle，并给 RD-Agent feedback
> 闭环添加一条不依赖 LLM 的端到端验证路径。整个阶段继续坚持"失败永远降级 +
> 零回归"这条运维准则。
>
> 先读：
>
> - `docs/STAGE_E_REPORT.md` —— 阶段 E 交付的反馈闭环 / shim 目录 / RAG 注入基础
> - `docs/STAGE_F_TEST_GUIDE.md` —— 架构与接口级测试指南
> - `docs/STAGE_F_TEST_PLAYBOOK.md` —— 复制即跑 playbook

---

## 1. 目标与决策

### 1.1 目标

承接阶段 E 对 "RD-Agent ↔ L2 判决闭环" 的搭建，阶段 F 把剩余的 4 处手工维护
风险一并收掉，让研究员能在**不动 Python 代码**的前提下：

- 扩展家族分类规则（F.1）
- 增减 encouraged / discouraged 家族、可用列、允许窗口（F.3）
- 区分不同股票池（csi300 / csi500 / all）的 L2 结论（F.2）
- 把"我改了 RAG，下一轮 LLM 真的看到了吗"这类问题变成 pytest 断言（F.4）

### 1.2 用户决策快照

阶段 F 启动前的 4 项关键决策（收到用户明确答复）：


| 项目       | 决策                                                           | 代码落地                                                              |
| -------- | ------------------------------------------------------------ | ----------------------------------------------------------------- |
| 范围       | 全量 F.1 + F.2 + F.3 + F.4                                     | 见下方交付列表                                                           |
| YAML 策略  | **YAML 主源 + Python fallback**（`warning` 提示研究员修复）             | `factor_lab/config/`*                                             |
| universe | 仅在 summary 层加 `universe` 字段，markdown 末尾 `[universe=...]`；不分桶 | `FailedCandidateSummary/ActiveFactorSummary/RetiredFactorSummary` |
| 端到端模式    | `@pytest.mark.e2e_rdagent`，默认 skip                           | `tests/conftest.py` + `tests/e2e/`                                |


---

## 2. 交付清单

### 2.1 F.1 家族分类 YAML 化


| 新增 / 修改 | 路径                                                | 说明                                                                                   |
| ------- | ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 新增      | `factor_lab/config/__init__.py`                   | 包入口，re-export 家族 API                                                                 |
| 新增      | `factor_lab/config/factor_families.yaml`          | 事实源：9 条 `label + regex pattern` 规则                                                   |
| 新增      | `factor_lab/config/families.py`                   | Loader / 校验 / 缓存 / fallback；`DEFAULT_FAMILY_RULES` 与 YAML 默认值同步                      |
| 修改      | `factor_lab/feedback/aggregator.py`               | `classify_family` 从硬编码改为调用 `factor_lab.config.classify_family`；老 symbol 保留 re-export |
| 新增      | `tests/factor_lab/config/test_factor_families.py` | 16 条单测：YAML 解析、schema 校验、所有降级路径、classifier 行为、reload                                 |


### 2.2 F.2 universe 维度


| 新增 / 修改 | 路径                                                    | 说明                                                                                                                                                                                                   |
| ------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 修改      | `factor_lab/feedback/schema.py`                       | `ActiveFactorSummary / RetiredFactorSummary / FailedCandidateSummary` 各加 `universe: str|None`；新增 `_validate_universe` + `_UNIVERSE_PATTERN`；`to_markdown()` 每条末尾渲染 `[universe=...]`（None 时省略，保留向后兼容） |
| 修改      | `factor_lab/feedback/aggregator.py`                   | `_build_failed_candidates` 从 `candidate.universe` 读；`_build_active_summaries` / `_build_retired_summaries` 从 manifest 读（缺失保持 None）                                                                   |
| 新增      | `tests/factor_lab/feedback/test_feedback_universe.py` | 18 条单测：schema 校验、backward-compat JSON 加载、markdown 渲染、aggregator 端到端                                                                                                                                  |


### 2.3 F.3 静态宪法 YAML 化


| 新增 / 修改 | 路径                                             | 说明                                                                                                                                                       |
| ------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 新增      | `factor_lab/config/rag_constitution.yaml`      | 事实源：allowed_windows / primitives / columns / scoring / feature_universe / orthogonality / encouraged_families / discouraged_families / naming_convention |
| 新增      | `factor_lab/config/constitution.py`            | Loader + renderer + cache + fallback；`_FALLBACK_CONSTITUTION_TEXT` 与 YAML 默认渲染 **byte-for-byte** 一致                                                      |
| 修改      | `factor_lab/adapters/quant_proposal.py`        | `_STATIC_CONSTITUTION` 由 `get_constitution_text()` 懒求值得到；保留硬编码 fallback 兜底；`_PROJECT_FACTOR_RAG` 别名继续可用                                                  |
| 新增      | `tests/factor_lab/config/test_constitution.py` | 19 条单测：byte-parity、降级路径、renderer 校验、缓存、RAG 集成路径、关键词 snapshot                                                                                             |


### 2.4 F.4 端到端闭环验证


| 新增 / 修改 | 路径                                       | 说明                                                                                                                |
| ------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 修改      | `tests/conftest.py`                      | 注册 `e2e_rdagent` marker；`pytest_collection_modifyitems` 默认 skip，显式 `-m e2e_rdagent` 时放行                           |
| 新增      | `tests/e2e/test_rdagent_feedback_e2e.py` | 5 条用例：2 条非 marker 的 aggregator→bundle→RAG 集成；3 条 marker-gated 的**真实 RD-Agent 子类**链路验证（MRO / prepare_context / 降级） |


### 2.5 其他

- 阶段 E 报告 / guide / playbook 三件套**不改**（F 是增量，零破坏）；
- `docs/STAGE_F_TEST_GUIDE.md` + `docs/STAGE_F_TEST_PLAYBOOK.md` 新写，延续阶段 D/E 的风格。

---

## 3. 核心技术说明

### 3.1 YAML 主源 + Python fallback 的可靠性模型

所有配置 YAML（`factor_families.yaml`、`rag_constitution.yaml`）都遵循同一套
降级协议：

1. 文件不存在 / `pyyaml` 不可用 / YAML 解析失败 / schema 校验失败 → `logger.warning`，
  返回 Python 内置默认值；
2. 默认值在 Python 侧**有完整副本**（`DEFAULT_FAMILY_RULES` /
  `_FALLBACK_CONSTITUTION_TEXT`），修改 YAML 时必须同步 Python 副本（`test_default_yaml_loads_and_matches_python_default`
   / `test_default_yaml_renders_byte_equal_to_fallback` 这两条测试会抓一致性）；
3. 进程级 cache（`_cached_rules` / `_cached_text`）避免重复 IO；单测用
  `reload_`* 显式刷新。

优点：RD-Agent 或 aggregator 在任何异常路径下都能启动；缺点：研究员改 YAML
后如果没跑 byte-parity 测试，可能在实际 deploy 时触发默认降级导致改动"看起来
没生效"。PLAYBOOK §3 给出了快速验证命令。

### 3.2 universe 的轻量嵌入

选择最小侵入（per-field）而非分桶（bucket）：

- 优点：序列化 / 加载都是零破坏（`universe` 默认 None，老 JSON 仍可解析），
aggregator / RAG 注入器的逻辑几乎不动；
- 代价：跨 universe 的"同一 family 在 csi300 失败但在 csi500 有效"这类精细洞察
当前不会被聚合器主动高亮，需要 LLM 通过 `[universe=...]` 标签自己读懂；
- 未来升级路径（阶段 G 预留）：若需要真正按 universe 分桶，只需在 FeedbackBundle
增一个 `by_universe: dict[str, SubBundle]` 字段，本阶段的 per-field 写法对此
正交。

### 3.3 F.4 的"端到端"层级

真正触碰真实 LLM / 真实市场数据的测试成本太高、噪声太大；F.4 选择**两层**：

1. **离线集成（非 marker）**：从手工撒进 tmp_path 的 cycle artifacts 开始，跑完
  `build_feedback_bundle → write_feedback_bundle → compose_project_rag`，断言
   `volume_price_reversal` / `[universe=csi300]` / L3 active+retired 名字真的
   全部汇入最终 prompt。这是前向的"写进去就一定看到"证明。
2. **RD-Agent 子类级（marker `e2e_rdagent`）**：导入**真实**的
  `rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen`
   → 验证 MRO、`prepare_context` 确实被覆盖 → monkeypatch 掉父类 `prepare_context`
   （避免 Scenario + LLM 依赖）→ 子类 `prepare_context` 把合成后的 RAG 放到
   `ctx["RAG"]`。这是"外壳装对位置"的证明，足够排除 import 路径 / 类继承被意外
   破坏的故障。

真正 call 真实 LLM 的 live smoke 放在 **playbook §5 手工步骤**，不入 CI。

---

## 4. 测试覆盖


| 层级                          | 数量     | 关键覆盖点                                                                                                          |
| --------------------------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| 家族 YAML loader + classifier | 16     | 文件缺失 / 坏 YAML / schema mismatch / 重复 label / 非法 regex / whitespace / 缓存与 reload / 三份默认值一致                      |
| FeedbackBundle universe     | 18     | 三类 summary 接受 universe / 默认 None / 非法值拒绝；老 JSON 加载；markdown 条目末尾渲染；aggregator 对 manifest / candidate 缺字段容忍；端到端 |
| 静态宪法 YAML loader + renderer | 19     | byte-parity 三路径（YAML ↔ fallback ↔ 原硬编码）、所有降级路径、renderer 字段校验、关键词 snapshot、RAG 集成                               |
| E2E RD-Agent feedback       | 5      | 2 非 marker：discouraged 家族全路径进 prompt、阈值不达标时不污染；3 marker：MRO / prepare_context 链路 / 降级                          |
| **阶段 F 合计（新增）**             | **58** | 含 3 marker-skip；非 marker 默认 55 全部参与 CI                                                                         |
| 阶段 D/E 已有                   | 334 通过 | 零回归                                                                                                            |


**全量回归**：`python -m pytest tests -q --ignore=tests/test_unified_strategy.py` →
`389 passed, 3 skipped, 7 warnings in ~10s`（3 skipped 即 e2e_rdagent marker）。

显式 marker 运行：`python -m pytest tests/e2e -m e2e_rdagent -v` →
`3 passed`。

> 注：`tests/test_unified_strategy.py` 里 2 条 pre-existing 失败仍与 F 无关，
> 保持 ignore；`STAGE_E_REPORT.md` 已登记。

---

## 5. 非回归保证

1. **阶段 E 契约不动**：`FeedbackBundle.schema_version="1.0"` 保持；新增字段全部
  可选（默认 None）且 `extra="forbid"` 仍生效（只新增已知字段）。老 bundle JSON
   仍能 `model_validate_json` 通过（见 `test_old_bundle_json_loads_without_universe_fields`）。
2. **静态宪法文本内容不变**：YAML 渲染出来的文本 = 阶段 E 的硬编码文本 byte-for-byte
  相同；RD-Agent prompt 行为零变化（`test_hardcoded_static_constitution_equal_to_fallback`
   会抓任何无意中 drift）。
3. **家族分类结果不变**：YAML 里 9 条规则与 `DEFAULT_FAMILY_RULES` 等价，所以
  对于所有已有因子名，`classify_family` 返回值与阶段 E 完全相同。
4. **shim 路径继续可用**：`rdagent_integration.`*、`scripts/run_fin_quant.py` 在
  阶段 F 未触，继续 DeprecationWarning。

---

## 6. 产物路径速查


| 用途                       | 路径                                        |
| ------------------------ | ----------------------------------------- |
| 家族分类 YAML                | `factor_lab/config/factor_families.yaml`  |
| 家族分类 Python loader / 默认值 | `factor_lab/config/families.py`           |
| RAG 静态宪法 YAML            | `factor_lab/config/rag_constitution.yaml` |
| 静态宪法 Python loader / 渲染器 | `factor_lab/config/constitution.py`       |
| C3 契约（含 universe）        | `factor_lab/feedback/schema.py`           |
| E2E 测试（默认 skip）          | `tests/e2e/test_rdagent_feedback_e2e.py`  |
| pytest marker 注册         | `tests/conftest.py`                       |


---

## 7. 已知限制与后续

- **universe 聚合**：F.2 是"记录级"的，不做跨 universe 统计；如果未来要给 LLM
输出"family X 在 csi300 3 次 FAIL，但在 csi500 2 次 PASS"这种对比，仍需阶段
G 引入 `by_universe` 二级索引。
- **家族规则冲突**：同一因子名匹配多个 pattern 时，按 YAML 顺序第一命中胜出；
目前靠研究员手排顺序。若规则增多，可能需要引入 priority 数值。
- **静态宪法渲染格式**：renderer 为了 byte-parity 做了 padding / 前缀的硬约定。
改 YAML 结构（比如加新章节）必须同步修改 renderer 与 fallback，否则 byte-parity
测试会抓。这是特意留的"两把钥匙一起开锁"门禁。
- **真实 LLM live 验证**：F.4 只做到 "子类挂对位置 + prompt 内容被塞进去"；真的
"LLM 收到 discouraged 家族后会不会真的避开"不是算法层能自动验证的，请走
`STAGE_F_TEST_PLAYBOOK.md` §5 手工流程，人工核对一次后长期生效。

### 下一步（阶段 G 预案）

- **G.1**：`by_universe` 子结构引入 FeedbackBundle；cycle 钩子多写
`latest_<universe>.json`；RAG 注入器可选按 universe 切分。
- **G.2**：shim 下线（`rdagent_integration/`*、`scripts/run_fin_quant.py`）；
纯一步升级 / 迁移文档。
- **G.3**：`embedding-based RAG`：把 C3 的失败家族做 embedding，给 RD-Agent
similarity retrieval 使用。
- **G.4**：family 规则跨阶段迭代：把 `discouraged` 做成"降格分数"（-0.5 / -1.0）
而不是硬黑名单，让 LLM 在极端情况下仍可尝试但需附加理由。