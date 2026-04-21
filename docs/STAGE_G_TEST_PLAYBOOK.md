# STAGE G 测试速查 Playbook（复制粘贴即可跑）

面向"本周跑一次，确认 Stage G 四个子项都活着"的研究员 / 运维场景。默认在仓库根
`d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project` 下，conda 环境
`qlib_zhengshi` 已激活。

## 0. 一分钟健康检查

```powershell
# F + G 全量回归（跳过与 G 无关的 unified_strategy 预存红）
python -m pytest tests -q --ignore=tests/test_unified_strategy.py
```

预期：`421 passed, 3 skipped`。不符合请先看 §6 故障速查，再翻 STAGE_G_TEST_GUIDE。

## 1. G.1 by_universe

### 1.1 单测

```powershell
python -m pytest tests/factor_lab/feedback/test_feedback_by_universe.py -v
```

预期 13 个绿。

### 1.2 手工验证 per-universe 落盘

```powershell
python - << 'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, tempfile

from factor_lab.feedback.aggregator import build_feedback_bundle, write_feedback_bundle

# 构造最小 cycle + manifest + cert 目录
root = Path(tempfile.mkdtemp())
reports = root / "r"; certs = root / "c"; reg = root / "reg"
for d in (reports, certs, reg):
    d.mkdir()

(reports / "lab_cycle_x.json").write_text(json.dumps({"cycle_id": "x"}), encoding="utf-8")
cyc = certs / "x"; cyc.mkdir()
(cyc / "fid.default.json").write_text(json.dumps({
    "decision": "FAIL",
    "candidate": {"name": "VolRev_5d", "universe": "csi300"},
    "check_results": [{"name": "ic", "passed": False, "detail": {}}],
}), encoding="utf-8")
(reg / "manifest.json").write_text(json.dumps({
    "factors": [{"factor_id": "prod_alpha_abcdef01", "name": "Quality_ROE_5d",
                 "status": "active", "parquet_version": 1, "universe": "csi300"}]
}), encoding="utf-8")

bundle = build_feedback_bundle(
    reports_dir=reports, cert_dir=certs, registry_data_dir=reg, max_cycles=4
)
out_dir = root / "fb"
write_feedback_bundle(bundle, workspace_feedback_dir=out_dir)
print("files in workspace:", sorted(p.name for p in out_dir.iterdir()))
print("by_universe keys:", list(bundle.by_universe.keys()))
PY
```

预期：

```
files in workspace: ['history', 'latest.json', 'latest.md', 'latest_csi300.json']
by_universe keys: ['csi300']
```

### 1.3 查看 markdown 的 Per-universe 段

```powershell
# 用任意现有 feedback/latest.md 或 §1.2 脚本的输出
Get-Content .\factor_lab\workspace\feedback\latest.md | Select-String -Context 0,10 "Per-universe"
```

## 2. G.2 老 shim 硬下线

### 2.1 单测

```powershell
python -m pytest tests/factor_lab/adapters/test_adapters_refactor.py -v
```

预期 12 个绿。

### 2.2 手工确认老路径真的红

```powershell
python -c "import rdagent_integration.project_quant_proposal"
```

预期：

```
RuntimeError: rdagent_integration.project_quant_proposal 已在阶段 G.2 下线；
请 import factor_lab.adapters.quant_proposal
```

```powershell
python scripts/run_fin_quant.py
```

预期：

```
RuntimeError: scripts/run_fin_quant.py 已在阶段 G.2 下线；
请改用 `python -m scripts.lab.run_rdagent_loop`
```

### 2.3 确认新命令可正常 import

```powershell
python -c "from factor_lab.adapters.quant_proposal import compose_project_rag; print('ok')"
```

## 3. G.3 embedding-based 检索

### 3.1 单测

```powershell
python -m pytest tests/factor_lab/feedback/test_feedback_embedding.py -v
```

预期 21 个绿。

### 3.2 手工验证分词 + 相似度

```powershell
python - << 'PY'
from factor_lab.feedback.embedding import TfidfBackend, _tokenize

print(_tokenize("VolRev_5d"), _tokenize("QualPersist_60D"), _tokenize("MACD_dif"))

corpus = [
    "VolRev_5d volume_price_reversal ic",
    "QualPersist_60D quality_persist ic",
    "ResidMom_60D long_horizon_residual_momentum",
]
be = TfidfBackend()
be.fit(corpus)
q = "short-cycle volume price reversal"
for doc in corpus:
    print(f"{be.similarity(q, doc):.3f}  <- {doc}")
PY
```

预期：

```
['vol', 'rev', '5', 'd'] ['qual', 'persist', '60', 'd'] ['macd', 'dif']
0.???  <- VolRev_5d ...
0.???  <- QualPersist_60D ...（低于上面）
0.???  <- ResidMom_60D ...（最低或 0）
```

### 3.3 观察 RAG 段是否注入

```powershell
python - << 'PY'
from factor_lab.adapters.quant_proposal import compose_project_rag

out = compose_project_rag(
    "BASE",
    feedback_dir=None,                     # 读默认 factor_lab/workspace/feedback
    retrieval_query="volume price reversal short cycle",
    retrieval_top_k=3,
)
for line in out.splitlines():
    if "Similar past failures" in line or "sim=" in line:
        print(line)
PY
```

预期（若 workspace 里有 failed 候选）：

```
------Similar past failures retrieved for query="volume price reversal short cycle" (G.3 top-3)------
  - sim=0.812  VolRev_5d  (family=volume_price_reversal)  ...
  ...
```

若输出为空，说明默认 workspace 的 `latest.json` 里没有 recent_fails，或它们与 query 相似度都低于 0.05 —— 这本身就是"安全降级"契约，属于正常行为。

## 4. G.4 discouraged 软降权

### 4.1 单测

```powershell
python -m pytest tests/factor_lab/config/test_constitution_penalty.py tests/factor_lab/config/test_constitution.py -v
```

预期 29 个绿（21 条 F.3 + 9 条 G.4，含 9 条 invariant 参数化）。

### 4.2 手工启用某条软降权

编辑 `factor_lab/config/rag_constitution.yaml` 暂时给某条加 penalty：

```yaml
discouraged_families:
  - id: x
    text: "Short-cycle volume-price reversals on W in {5, 10}"
    penalty: -0.5
  - id: y
    text: "Same-day volume spike + price reversal patterns"
  - id: z
    text: "Anything that ranks the universe with >50% weekly turnover"
```

然后 reload + 渲染：

```powershell
python - << 'PY'
from factor_lab.config.constitution import reload_constitution_text

text = reload_constitution_text()
for line in text.splitlines():
    if "(x)" in line or "(y)" in line or "(z)" in line:
        print(line)
PY
```

预期：

```
   (x) Short-cycle volume-price reversals on W in {5, 10}   [penalty=-0.5; soft — new attempts allowed only with explicit justification of how this proposal differs]
   (y) Same-day volume spike + price reversal patterns
   (z) Anything that ranks the universe with >50% weekly turnover
```

验证后记得把 YAML 改回去（`git checkout factor_lab/config/rag_constitution.yaml`），否则 F.3 的 byte-for-byte 测试会红。

### 4.3 注意事项

- 一旦 YAML 被改动，请跑 `python -m pytest tests/factor_lab/config/ -q` 确认 byte-for-byte 契约仍保。
- penalty 只体现为 RAG 文本注解，LLM 是否真的"软降权"需要线上观察（参考 STAGE_F_TEST_PLAYBOOK §5 的 live smoke 流程）。

## 5. 综合 + E2E 标记

```powershell
# F + G 单元 + offline e2e（默认 skip 掉 e2e_rdagent 标记）
python -m pytest tests/factor_lab tests/e2e -q

# 只跑 e2e_rdagent 标记（会真的实例化 RD-Agent 基类，但不调 LLM）
python -m pytest tests/e2e -m e2e_rdagent -v
```

## 6. 故障速查

| 现象 | 排查第一步 |
|---|---|
| 一大片 `ImportError: No module named 'rdagent_integration'` | 多半是环境没装新包。先确认 `python -c "import factor_lab.adapters.quant_proposal"` 成功，然后按 STAGE_G_TEST_GUIDE §3 的迁移表替换 import。|
| `RuntimeError: ... 已在阶段 G.2 下线` | 是意料之中。按异常里提示的新 import 路径改。|
| `pydantic.ValidationError: by_universe[xxx].universe=...` | 手搓 UniverseSubBundle 时 key/value 不一致。检查 dict key 必须等于 sub.universe。|
| `ValueError: discouraged_families[...].penalty 必须是数值` | YAML 里把 penalty 写成了字符串或非法值。改成数字或直接删掉这一行（等同硬黑）。|
| `test_default_yaml_renders_byte_equal_to_fallback` 红 | 有人改了默认 YAML 的文本 / 顺序 / 缩进。要么回退改动，要么同步重算 `_FALLBACK_CONSTITUTION_TEXT`。|
| G.3 段不出现在 RAG 输出里 | 检查 `retrieval_query` 是否为 None/空串；检查 `bundle.recent_fails` 是否为空；调低 `min_score`。|
