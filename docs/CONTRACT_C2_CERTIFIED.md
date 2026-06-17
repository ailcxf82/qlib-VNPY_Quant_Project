# 契约 C2 ── CertifiedFactorRecord（L2 → L3）

> 本文件为契约规范，受 `[docs/ARCHITECTURE_FACTOR_LAB.md](ARCHITECTURE_FACTOR_LAB.md)` 约束。
>
> **Schema 实现位置**：`factor_validation/schema.py`
> **版本**：v1.0 — 2026-04-19

---

## 1. 用途

`CertifiedFactorRecord` 是 L2 验证环境向 L3 正式注册表（factor_registry）唯一的数据交换格式。
任何因子要进入 production，**必须**附带一张 `decision == "PASS"` 的 `CertifiedFactorRecord`。

L3 不接受：

- 没有 `CertifiedFactorRecord` 的因子
- `decision != "PASS"` 的因子
- `profile_hash` 与当前 profile 文件不匹配的旧证书（可能是 profile 改过没重测）

---

## 2. 嵌套结构

```
CertifiedFactorRecord
├── factor_id
├── candidate: CandidateFactorPackage    ← C1 全文 snapshot
├── profile_name + profile_hash
├── decision: PASS | FAIL | HOLD
├── overall_score
├── check_results: list[CheckResult]
│   └── (name, passed, score, threshold, detail, elapsed_ms)
├── backtest_metrics
├── validated_at
├── validator_version
└── notes
```

> 把整个 `candidate` snapshot 嵌进证书是有意为之 —— 证书必须**自包含**、可独立审计，
> 不能依赖外部 lab workspace 仍然存在。

---

## 3. CheckResult 字段


| 字段           | 类型     | 必填    | 说明                             |
| ------------ | ------ | ----- | ------------------------------ |
| `name`       | `str`  | ✅     | check 名，例如 `'ic'`、`'turnover'` |
| `passed`     | `bool` | ✅     | 本项是否通过                         |
| `score`      | `float | None` | ⬜                              |
| `threshold`  | `float | None` | ⬜                              |
| `detail`     | `dict` | ⬜     | 任意附加信息（IC 序列摘要、p 值、子时段表现等）     |
| `elapsed_ms` | `int`  | ✅     | 本项耗时（毫秒）≥0                     |


---

## 4. CertifiedFactorRecord 字段


| 字段                  | 类型                       | 必填    | 说明                                       |
| ------------------- | ------------------------ | ----- | ---------------------------------------- |
| `factor_id`         | `str`                    | ✅     | 必须等于 `candidate.factor_id`               |
| `candidate`         | `CandidateFactorPackage` | ✅     | C1 全文                                    |
| `profile_name`      | `str`                    | ✅     | 使用的 profile 名                            |
| `profile_hash`      | `str`                    | ✅     | profile YAML 的 sha256（64 位小写十六进制）        |
| `decision`          | `Decision`               | ✅     | `PASS` / `FAIL` / `HOLD`                 |
| `overall_score`     | `float`                  | ✅     | [0, 1]                                   |
| `check_results`     | `list[CheckResult]`      | ✅     | ≥ 1 项；name 不可重复                          |
| `backtest_metrics`  | `dict[str, float]`       | ⬜     | RQAlpha 真实回测的核心指标                        |
| `validated_at`      | `datetime`               | ✅     | UTC                                      |
| `validator_version` | `str`                    | ✅     | factor_validation 包版本（PEP440 风格 'x.y.z'） |
| `notes`             | `str                     | None` | ⬜                                        |


---

## 5. Decision 语义


| Decision | 含义                                                      | L3 行为                                           |
| -------- | ------------------------------------------------------- | ----------------------------------------------- |
| `PASS`   | 所有 check 通过 + overall_score ≥ profile.pass_threshold    | 自动接收（如 `auto_promote_on_pass=true`）或人工 promote  |
| `FAIL`   | 至少一个 check 未通过，或 overall_score < profile.fail_threshold | L3 拒绝，丢弃                                        |
| `HOLD`   | 介于 PASS / FAIL 之间，需人工复核                                 | L3 不自动接收；写入 `factor_registry/data/pending/`，等审批 |


**强一致性规则**（schema `model_validator` 自动校验）：

- `decision == PASS` ⇒ 所有 `check_results` 的 `passed == True`
- `decision == FAIL` ⇒ 至少一个 `check.passed == False`

---

## 6. profile_hash 规则

```python
import hashlib, pathlib
profile_hash = hashlib.sha256(
    pathlib.Path("factor_validation/profiles/strict.yaml").read_bytes()
).hexdigest()
```

- 必须是 64 位**小写**十六进制
- 任何 profile YAML 改动 → hash 变 → 旧证书自动失效（重新验证）

---

## 7. 完整 JSON 范例

```json
{
  "factor_id": "rdagent_QualPersist_60D_a1b2c3d4",
  "candidate": { "...": "见 CONTRACT_C1_CANDIDATE.md §5" },
  "profile_name": "strict",
  "profile_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "decision": "PASS",
  "overall_score": 0.78,
  "check_results": [
    {
      "name": "coverage",
      "passed": true,
      "score": 0.95,
      "threshold": 0.85,
      "detail": {"non_null_ratio": 0.95, "instrument_coverage": 0.92},
      "elapsed_ms": 320
    },
    {
      "name": "ic",
      "passed": true,
      "score": 0.62,
      "threshold": 0.30,
      "detail": {"rank_ic": 0.022, "ic_ir": 0.48},
      "elapsed_ms": 1850
    },
    {
      "name": "backtest",
      "passed": true,
      "score": 0.71,
      "threshold": 1.5,
      "detail": {"sharpe": 2.1, "max_dd": 0.12, "ann_turnover": 3.8},
      "elapsed_ms": 245000
    }
  ],
  "backtest_metrics": {
    "annualized_return": 0.18,
    "sharpe": 2.1,
    "max_drawdown": 0.12,
    "annualized_turnover": 3.8
  },
  "validated_at": "2026-04-19T13:00:00Z",
  "validator_version": "1.0.0",
  "notes": null
}
```

---

## 8. 演进规则

参照 `CONTRACT_C1_CANDIDATE.md §7`。任何字段变更必须同步更新本文件 + L2 / L3 fixture。