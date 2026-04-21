# 阶段 F 测试与运维指南（YAML 配置化 + universe + E2E）

> 本指南覆盖阶段 F 交付的运维级产物：
>
> 1. F.1 家族分类 YAML（`factor_lab/config/factor_families.yaml` + loader）
> 2. F.2 FeedbackBundle 的 `universe` 字段（aggregator + markdown）
> 3. F.3 RAG 静态宪法 YAML（`factor_lab/config/rag_constitution.yaml` + renderer）
> 4. F.4 端到端 feedback 闭环测试（`tests/e2e/` + `@pytest.mark.e2e_rdagent`）
>
> 配套阅读：
>
> - `docs/STAGE_E_TEST_GUIDE.md` —— 前序反馈回路基础
> - `docs/STAGE_F_TEST_PLAYBOOK.md` —— 阶段 F 的复制即跑测试
> - `docs/STAGE_F_REPORT.md` —— 阶段 F 交付清单与结论

---

## 1. 测试分层（pytest 快速索引）


| 层                       | 目录                                                    | 说明                                       | 典型命令                                                                      |
| ----------------------- | ----------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| F.1 家族 YAML             | `tests/factor_lab/config/test_factor_families.py`     | 加载器、降级、classifier、reload                 | `pytest tests/factor_lab/config/test_factor_families.py -v`               |
| F.2 universe            | `tests/factor_lab/feedback/test_feedback_universe.py` | schema + aggregator + markdown           | `pytest tests/factor_lab/feedback/test_feedback_universe.py -v`           |
| F.3 宪法 YAML             | `tests/factor_lab/config/test_constitution.py`        | byte-parity + renderer + reload + RAG 集成 | `pytest tests/factor_lab/config/test_constitution.py -v`                  |
| F.4 E2E（默认 skip marker） | `tests/e2e/test_rdagent_feedback_e2e.py`              | 2 非 marker + 3 marker `e2e_rdagent`      | `pytest tests/e2e -v`（默认） `pytest tests/e2e -m e2e_rdagent -v`（触发 marker） |


### 一键全量

```powershell
conda activate qlib_zhengshi
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

阶段 F 基准：**389 passed, 3 skipped, 7 warnings in ~10 s**
（3 skipped 是默认跳过的 `e2e_rdagent` marker 测试）。

显式 marker：

```powershell
python -m pytest tests/e2e -m e2e_rdagent -v
# → 3 passed
```

---

## 2. F.1 家族分类 YAML

### 2.1 事实源

`factor_lab/config/factor_families.yaml`：

```yaml
version: "1.0"
rules:
  - label: volume_price_reversal
    pattern: "(VolRev|VolRet|VolTwist|RangeRatio|VolumeTrend|VolRatio)"
  # ...共 9 条，顺序决定优先级
```

### 2.2 API

```python
from factor_lab.config import classify_family, get_family_rules, reload_family_rules

classify_family("VolRev_5d")         # -> "volume_price_reversal"
rules = get_family_rules()            # 进程级缓存
reload_family_rules()                 # 强制重载（测试/热更新）
reload_family_rules(Path("..."))      # 指定 YAML 文件
```

### 2.3 校验规则


| 规则                             | 违反时行为                                |
| ------------------------------ | ------------------------------------ |
| `version` 必须 == `"1.0"`        | 回退到 `DEFAULT_FAMILY_RULES` + warning |
| `rules` 必须为非空 list             | 回退                                   |
| 每条规则 `label` 必须非空、不含空白、不含路径分隔符 | 回退                                   |
| `label` 必须唯一                   | 回退                                   |
| `pattern` 必须能 `re.compile`     | 回退                                   |


### 2.4 扩展新家族（研究员操作）

1. 在 `factor_lab/config/factor_families.yaml` 的 `rules:` 末尾**追加**新条目（保证不打乱已有顺序，避免历史 bundle 的 family 标签漂移）；
2. 在 `factor_lab/config/families.py` 的 `DEFAULT_FAMILY_RULES` 末尾**同步追加**相同规则；
3. 跑 `pytest tests/factor_lab/config/test_factor_families.py::test_default_yaml_loads_and_matches_python_default`
  确认两份默认值对齐；
4. 如果担心缓存影响，在研究员脚本里显式 `reload_family_rules()`。

---

## 3. F.2 universe 字段

### 3.1 schema 变化

```python
from factor_lab.feedback import (
    ActiveFactorSummary, RetiredFactorSummary, FailedCandidateSummary,
)
# 三者各加 universe: str | None = None
```

允许值：字母数字 `/_/-`，长度 1~64（与 `CandidateFactorPackage.universe` 同步）。
None 表示"未知 / 老数据"。

### 3.2 aggregator 行为


| 来源                                | 读取路径                                      |
| --------------------------------- | ----------------------------------------- |
| `FailedCandidateSummary.universe` | `cert["candidate"]["universe"]`（C2 内嵌 C1） |
| `ActiveFactorSummary.universe`    | `manifest.json` 的 factor 记录 `.universe`   |
| `RetiredFactorSummary.universe`   | 同上                                        |


任何一层缺字段 → 对应 summary 的 `universe=None`，markdown 省略标签。

### 3.3 markdown 渲染

每条 summary 行末尾追加 `[universe=<value>]`（None 时不渲染）。示例：

```text
[L3 active factors — DO NOT propose duplicates]
  - Quality_ROE_5d  (family=quality_persist) tags=['liqud'] [universe=csi300]

[L3 retired factors — these failed in production, avoid similar shape]
  - Old_VolRev  (family=volume_price_reversal)  retired_on=2026-03-10 [universe=csi500]
    reason: rolling IC degraded

[Recent L2 failures across last 1 cycle(s)]
  - VolRev_5d  (family=volume_price_reversal)  stage=default  decision=FAIL  modes=[ic] [universe=csi300]
```

### 3.4 向后兼容

老 bundle JSON（没有 `universe` 字段）仍能 `FeedbackBundle.model_validate_json`
通过，加载后 `universe=None`。

---

## 4. F.3 RAG 静态宪法 YAML

### 4.1 事实源 + renderer

`factor_lab/config/rag_constitution.yaml` 持有 9 组结构化字段；
`factor_lab/config/constitution.py::render_constitution(cfg)` 负责把它们拼回成
与阶段 E 硬编码字符串 **byte-for-byte 相同** 的多段 markdown。

### 4.2 API

```python
from factor_lab.config.constitution import (
    get_constitution_text,      # 进程级缓存读取
    load_constitution_text,     # 指定 YAML 路径加载
    reload_constitution_text,   # 强制重载（单测 / 热更新）
    render_constitution,        # 纯函数：YAML mapping → 文本
    _FALLBACK_CONSTITUTION_TEXT,# Python 兜底副本
)
```

`factor_lab/adapters/quant_proposal._STATIC_CONSTITUTION` 在**模块加载时**通过
`get_constitution_text()` 懒求值得到；YAML 出错则退到模块内硬编码 fallback，
保证 `from factor_lab.adapters.quant_proposal import _STATIC_CONSTITUTION` 永不
抛异常。

### 4.3 修改 YAML 后必须跑的测试

```powershell
pytest tests/factor_lab/config/test_constitution.py::test_default_yaml_renders_byte_equal_to_fallback
pytest tests/factor_lab/config/test_constitution.py::test_hardcoded_static_constitution_equal_to_fallback
pytest tests/factor_lab/config/test_constitution.py -k snapshot
```

三条全绿才代表你既改了 YAML、也同步了 Python fallback。如果只改了 YAML，
deploy 时在无 `pyyaml` 环境会触发默认降级，改动**不生效**。

### 4.4 扩展 encouraged / discouraged 家族（研究员操作）

1. 编辑 `factor_lab/config/rag_constitution.yaml` 的 `encouraged_families` / `discouraged_families`
  列表（`id` 保持小写字母，不要和已有 id 重复）；
2. 手工把等价文本同步到 `factor_lab/config/constitution.py` 的
  `_FALLBACK_CONSTITUTION_TEXT`（或在你的 PR 里请 reviewer 帮跑一次
   byte-parity 测试让 diff 自动生成）；
3. 跑 §4.3 的三条测试；
4. 可选：手工 `python -c "from factor_lab.config.constitution import get_constitution_text; print(get_constitution_text())"`
  肉眼检查一遍输出。

---

## 5. F.4 端到端闭环测试

### 5.1 marker 配置

在 `tests/conftest.py` 里已经注册：

```python
config.addinivalue_line(
    "markers",
    "e2e_rdagent: 阶段 F.4 端到端冒烟测试（需要 RD-Agent + LLM，默认 skip；"
    "用 `pytest -m e2e_rdagent` 显式触发）",
)
```

并且 `pytest_collection_modifyitems` 默认给所有带 `e2e_rdagent` 的 item 打上
`pytest.mark.skip`。仅当命令行 `-m` 表达式里包含 `e2e_rdagent` 字样才放行。

### 5.2 测试矩阵


| 用例                                                    | marker?       | 验证点                                                                                             |
| ----------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `test_feedback_pipeline_end_to_end_into_final_rag`    | 否             | aggregator → bundle → compose_project_rag，discouraged + universe + active + retired 全路径进 prompt |
| `test_pipeline_discouraged_threshold_respected`       | 否             | 单条 FAIL 不够阈值 → `Discouraged families` 段不出现                                                      |
| `test_real_rdagent_base_class_in_mro`                 | `e2e_rdagent` | ProjectQlibQuantHypothesisGen 真的继承 RD-Agent 原厂类                                                 |
| `test_prepare_context_chains_compose_project_rag`     | `e2e_rdagent` | 真实类实例 + 真实 feedback bundle + monkeypatch 父类 → RAG 含 discouraged / universe                      |
| `test_prepare_context_degrades_when_feedback_missing` | `e2e_rdagent` | feedback 盘不存在 → 只注入静态宪法，不抛                                                                      |


### 5.3 CI 建议

- **快 CI**（每提交）：默认跑，`e2e_rdagent` skip。
- **夜 CI / 人工 PR gate**：`python -m pytest tests/e2e -m e2e_rdagent -v`；耗时 <5 s，
不调 LLM，零 token 成本。
- **手工 live smoke**：见 `STAGE_F_TEST_PLAYBOOK.md` §5；需要真实 API key。

---

## 6. 常见问题 / 排错

### 6.1 我改了 YAML，但 RAG prompt 没变

依次检查：

1. `python -c "from factor_lab.config.constitution import get_constitution_text; print(get_constitution_text()[:200])"`
  —— 看看 YAML 是不是真被读到；
2. 如果开了 cache，跑 `reload_constitution_text()`；
3. 如果 log 里能看到 `[WARNING] 静态宪法 YAML ... 回退到兜底`，说明 YAML 坏了或
  schema 校验失败，仔细读报错 message 修复。

### 6.2 `test_default_yaml_loads_and_matches_python_default` 挂了

说明你动了 YAML 但没同步更新 `DEFAULT_FAMILY_RULES`（或反之）。两份**必须**严格
相等。把 YAML 里的每条 `label` / `pattern` 照抄到 Python 元组即可。

### 6.3 `test_default_yaml_renders_byte_equal_to_fallback` 挂了

说明你动了 `rag_constitution.yaml` 或 `render_constitution`，但没更新
`_FALLBACK_CONSTITUTION_TEXT`。操作步骤：

```powershell
python -c "from factor_lab.config.constitution import load_constitution_text; import pathlib; pathlib.Path('D:/tmp/new_fallback.txt').write_text(load_constitution_text(), encoding='utf-8')"
```

然后手工把 `D:/tmp/new_fallback.txt` 的内容拷贝到
`factor_lab/config/constitution.py` 的 `_FALLBACK_CONSTITUTION_TEXT`（注意保持
三引号 `"""..."""` 包裹、保留 Unicode 点 `\u00b7` / `\u2014` 转义）。

### 6.4 E2E marker 测试在我机器上失败

通常是 RD-Agent / pyyaml 安装异常；先在 conda 环境 `qlib_zhengshi` 里
`pip list | grep -iE "rdagent|pyyaml"` 检查。若环境健康但测试仍失败，reading
`tests/e2e/test_rdagent_feedback_e2e.py` 里 `_seed_cycle_artifacts` 造数据的
方式，确认 C2 证书 schema 没被上游改动。

---

## 7. 回归 checklist（阶段 F 完工口径）

```powershell
conda activate qlib_zhengshi

# 1. 仅配置化部分
python -m pytest tests/factor_lab/config -v

# 2. universe / aggregator
python -m pytest tests/factor_lab/feedback -v

# 3. E2E 非 marker
python -m pytest tests/e2e -v

# 4. E2E marker
python -m pytest tests/e2e -m e2e_rdagent -v

# 5. 全量回归
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

阶段 F 通过标准：

- §1 全绿
- §2 全绿（新 18 条 + 老 feedback 原有 28 条 = 46 条）
- §3：2 passed, 3 skipped
- §4：3 passed, 2 deselected
- §5：389 passed, 3 skipped

