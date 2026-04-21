# 契约 C1 ── CandidateFactorPackage（L1 → L2）

> 本文件为契约规范，受 `[docs/ARCHITECTURE_FACTOR_LAB.md](ARCHITECTURE_FACTOR_LAB.md)` 约束。
> 任何 L1 / L2 代码变更若涉及 C1 字段，必须先更新本文件并通过评审。
>
> **Schema 实现位置**：`factor_lab/exporters/schema.py`
> **版本**：v1.0 — 2026-04-19

---

## 1. 用途

`CandidateFactorPackage` 是 L1 试验场（factor_lab）向 L2 验证环境（factor_validation）
**唯一**的数据交换格式。任何候选因子（无论来自 RD-Agent / 手工 / 外部导入）都必须通过
本契约打包后才能进入验证流程。

L1 / L2 之间禁止：

- 直接传递 pandas DataFrame
- 直接传递 callable
- 跨层访问对方内部目录

---

## 2. 字段速查


| 字段            | 类型                                       | 是否必填  | 说明                                      |
| ------------- | ---------------------------------------- | ----- | --------------------------------------- |
| `factor_id`   | `str`                                    | ✅     | 唯一 ID，格式 `<source>_<name>_<short_hash>` |
| `name`        | `str`                                    | ✅     | 人类可读名，必须与 `values.parquet` 单列列名一致       |
| `source`      | `Literal["rdagent","manual","external"]` | ✅     | 候选来源                                    |
| `hypothesis`  | `str`                                    | ✅     | LLM 假设原文 / 设计动机；非空，≤4000 字              |
| `formulation` | `str`                                    | ✅     | 数学公式 / pseudocode；非空，≤4000 字            |
| `code_path`   | `Path`                                   | ✅     | `factor.py` 路径                          |
| `values_path` | `Path`                                   | ✅     | `values.parquet` 路径                     |
| `universe`    | `str`                                    | ✅     | 适用股票池：`csi300` / `csi500` / `all` 等     |
| `date_range`  | `tuple[date, date]`                      | ✅     | 因子值覆盖的闭区间                               |
| `lab_metrics` | `dict[str, float]`                       | ⬜     | L1 自报指标，**仅参考**；L2 必须重新计算               |
| `parent_loop` | `int                                     | None` | 条件                                      |
| `created_at`  | `datetime`                               | ✅     | UTC 时间戳                                 |
| `lab_run_id`  | `str`                                    | ✅     | 一次 lab run 的 UUID / 短串，便于审计             |


---

## 3. factor_id 命名规则

格式：`<source>_<name>_<short_hash>`

- `source`：仅小写字母数字（与字段 `source` 取值一致）
- `name`：字母数字 + 下划线，长度 1~64，必须以字母开头
- `short_hash`：8~16 位十六进制（建议为因子代码 + 配置的 sha256 截断）

**正则**（schema 内已实现）：

```regex
^[a-z0-9]+_[A-Za-z0-9_]{1,64}_[0-9a-f]{8,16}$
```

**示例**：


| 合法                                 | 非法                                       |
| ---------------------------------- | ---------------------------------------- |
| `rdagent_QualPersist_60D_a1b2c3d4` | `RDAgent_QualPersist_60D_xxx`（source 大写） |
| `manual_MyFactor_v1_0123456789ab`  | `rdagent_QualPersist 60D_a1b2c3d4`（含空格）  |
| `external_MomentumX_deadbeef`      | `rdagent_QualPersist_60D`（缺 hash）        |


---

## 4. values.parquet 物料约束

通过 `CandidateFactorPackage.validate_artifacts(check_parquet_schema=True)` 校验，必须满足：

- 两级 `MultiIndex`，名为 `["datetime", "instrument"]`，顺序固定
- 仅一列，列名 == `name`
- dtype 为 `float64`
- 时间索引落在 `date_range` 内（允许实际窗口比 `date_range` 短，不允许超出）

**示例**：

```python
import pandas as pd
df = pd.DataFrame(
    {"QualPersist_60D": [...]},
    index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "SH600000"), ...],
        names=["datetime", "instrument"],
    ),
)
df.to_parquet("workspace/candidates/<factor_id>/values.parquet")
```

---

## 5. 完整 JSON 范例

```json
{
  "factor_id": "rdagent_QualPersist_60D_a1b2c3d4",
  "name": "QualPersist_60D",
  "source": "rdagent",
  "hypothesis": "Slow-moving quality factor: 60-day rolling mean of ROE captures earnings persistence and reduces turnover.",
  "formulation": "rank(rolling_mean($roe, 60)) cross-sectionally each day",
  "code_path": "factor_lab/workspace/candidates/rdagent_QualPersist_60D_a1b2c3d4/factor.py",
  "values_path": "factor_lab/workspace/candidates/rdagent_QualPersist_60D_a1b2c3d4/values.parquet",
  "universe": "csi300",
  "date_range": ["2022-01-01", "2025-10-31"],
  "lab_metrics": {"lab_rank_ic": 0.018, "lab_ic_ir": 0.42},
  "parent_loop": 0,
  "created_at": "2026-04-19T12:34:56Z",
  "lab_run_id": "lab-2026-04-19-deadbeef"
}
```

---

## 6. 不变量与一致性规则

Schema `model_validator` 自动校验：

1. `factor_id` 中段（去掉首段 source 与末段 hash 后）必须等于 `name`
2. `date_range[0] <= date_range[1]`
3. `source == 'manual'` ⇒ `parent_loop is None`
4. `source == 'rdagent'` ⇒ `parent_loop is not None`
5. `lab_metrics` 中所有值必须是有限浮点数

`extra='forbid'` ⇒ 出现未知字段直接拒绝（防止字段污染）。
`frozen=True` ⇒ 实例不可修改（任何 setter 都会抛错）。

---

## 7. 演进规则

- 新增**可选**字段：minor 版本，向后兼容
- 新增**必填**字段或修改字段语义：major 版本，必须同步更新所有 L1 / L2 代码 + 全部 fixture
- 删除字段：major 版本，需提供数据迁移脚本

