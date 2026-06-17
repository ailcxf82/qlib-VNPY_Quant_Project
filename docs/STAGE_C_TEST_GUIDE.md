# 阶段 C 测试与运维指南（L2 因子验证 + Promote / Retire 流水线）

> 本指南覆盖阶段 C 交付的四项可执行产物：
>
> 1. 离线 L2 验证（`factor_validation/`，含 coverage/ic/orthogonality/turnover 四个 check）
> 2. Candidate 验证 CLI（`scripts.promote.validate_candidate`）
> 3. 证书 → L3 注册 CLI（`scripts.promote.promote_certified`）
> 4. L3 因子退役 CLI（`scripts.promote.retire_factor`）
>
> 以及 C.6 版本的 **turnover_check 算法升级**与三套 profile 的阈值对齐。
>
> 配合阅读：
>
> - `docs/ARCHITECTURE_FACTOR_LAB.md` —— 三层架构总览
> - `docs/CONTRACT_C1_CANDIDATE.md` —— L1→L2 候选包 schema
> - `docs/CONTRACT_C2_CERTIFIED.md` —— L2→L3 证书 schema
> - `docs/VALIDATION_PROFILE_GUIDE.md` —— profile 字段语义

---

## 1. 测试分层（pytest 快速索引）


| 层                   | 目录                                                                                                             | 说明                                                           | 典型命令                                                           |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| 契约层                 | `tests/factor_lab/` · `tests/factor_validation/` 中 schema 相关                                                   | Pydantic schema 强校验                                          | `pytest tests/factor_lab tests/factor_validation -q -k schema` |
| L2 check 单元         | `tests/factor_validation/test_{coverage,ic,orthogonality,turnover}_check.py`                                   | 每个 check 独立测用例                                               | `pytest tests/factor_validation -q`                            |
| L2 编排               | `tests/factor_validation/test_orchestrator.py`                                                                 | profile 加载 / aggregate / decide                              | 同上                                                             |
| L3 存储               | `tests/factor_registry/`                                                                                       | `ParquetStore` · `FactorRegistry` · `ProductionFactorLoader` | `pytest tests/factor_registry -q`                              |
| Feature pipeline 集成 | `tests/feature/test_pipeline_registry_integration.py`                                                          | L3 loader + legacy fallback 等价性                              | `pytest tests/feature -q`                                      |
| Promote CLI         | `tests/promote/test_validate_candidate_cli.py` · `test_promote_certified_cli.py` · `test_retire_factor_cli.py` | validate / promote / retire CLI                              | `pytest tests/promote -q`                                      |
| Legacy migration    | `tests/promote/test_migrate_legacy_factors.py`                                                                 | 一次性迁移脚本                                                      | 同上                                                             |


### 一键全量

```bash
# Conda 环境：qlib_zhengshi（已安装 qlib + pydantic v2 + pandas + pyarrow）
python -m pytest tests/factor_validation tests/factor_registry tests/feature tests/promote -q
```

阶段 C 交付后的基准：**174 项全 PASS，整体 <10s**（纯离线单测，未跑 qlib.D）。

---

## 2. 核心算法：`turnover_check`（C.6 升级点）

### 2.1 旧语义（被淘汰）

```
daily_rank_turnover = 每日 |Δrank|>0 的股票占比的多日平均
```

问题：`rank(method='average')` 下任何一点点因子值扰动都会在同值区间产生 ±0.5 的 rank
漂移；对稠密连续因子，"有变化"的股票占比几乎恒为 1.0，完全丧失区分度（阶段 C oracle
复测实证过，5 个 legacy 因子全部 ≈1.0）。

### 2.2 新语义（C.6 起）

```
turnover_day_d       = (1 − spearman(rank_day_d, rank_day_{d-1})) / 2
daily_rank_turnover  = mean_over_valid_days(turnover_day_d)   ∈ [0, 1]
```

解释：


| ρ    | 含义               | turnover |
| ---- | ---------------- | -------- |
| +1.0 | rank 完全保持        | 0        |
| 0    | random reshuffle | ~0.5     |
| −1.0 | rank 完全反转        | 1        |


### 2.3 同步更新的阈值


| profile                | `max_daily_rank_turnover` | `min_rank_autocorr` | 动机            |
| ---------------------- | ------------------------- | ------------------- | ------------- |
| `strict`               | 0.30                      | 0.60                | 必须显著慢于 random |
| `default`              | 0.35                      | 0.50                | 宽一档，允许中速因子    |
| `exploratory`          | 0.55                      | 0.10                | 只排除反转型        |
| `manual_grandfathered` | —（未启用 turnover）           | —                   | 仅 coverage    |


### 2.4 新算法验证

`tests/factor_validation/test_turnover_check.py` 新增 `test_reversed_factor_has_turnover_near_one`，
构造日间 rank 反转数据（基础单调漂移 × ±1 交替）→ 断言 `daily_rank_turnover > 0.90`。
加上原有慢因子 / 随机因子测例（`< 0.15` 与 `~0.5`），语义端到端校验。

---

## 3. Promote 流水线：三步走

**物理 parquet 与 manifest 元信息分离**是 L3 的基本承诺。Promote 流水线被拆成三个
幂等 CLI，方便 oracle 复测、CI/CD 分阶段触发、人工干预：

```
           ┌────────────────────────────┐
  C1 JSON  │ validate_candidate         │
──────────►│ (L1 → L2 离线验证)          │───► CertifiedFactorRecord JSON
           │  仅写 --out-cert            │
           └────────────────────────────┘
                         │
                         ▼  (只有 PASS 才进入下一步)
           ┌────────────────────────────┐
  cert.json│ promote_certified          │
──────────►│ 校验 parquet 列存在 + 注册  │───► manifest.json update
           │ (L2 → L3 原子 manifest 更新)│     data/certified/<fid>.json
           └────────────────────────────┘
                         │
                         ▼
           ┌────────────────────────────┐
  factor_id│ retire_factor              │
──────────►│ 退役（manifest 层 soft del）│───► data/retired/<fid>.json
           │ (不改 parquet 物理文件)      │
           └────────────────────────────┘
```

### 3.1 `validate_candidate`

作用：单个候选因子跑 profile 验证，输出 C2 证书（JSON）。**不会改 L3 registry。**

```bash
# 模式一：完整 C1 JSON 输入（L1 exporter 产物）
python -m scripts.promote.validate_candidate \
    --candidate-json factor_lab/workspace/candidates/<fid>/candidate.json \
    --profile factor_validation/profiles/default.yaml \
    --out-cert factor_validation/certificates/<fid>.json

# 模式二：即席（dev 调试，无 C1 JSON 时）
python -m scripts.promote.validate_candidate \
    --values-path path/to/values.parquet \
    --code-path   path/to/factor.py \
    --factor-id   manual_MyFactor_01234567 \
    --name        MyFactor \
    --source      manual \
    --hypothesis  "..." \
    --formulation "..." \
    --universe    csi300 \
    --date-range  2024-07-01 2026-04-07 \
    --profile     factor_validation/profiles/default.yaml
```

返回码：

- `0` PASS
- `2` HOLD（所有 check 通过但综合分低于 pass_threshold）
- `3` FAIL
- `1` 参数 / IO 错误

### 3.2 `promote_certified`

作用：吃一份 **PASS** 证书，原子更新 `factor_registry/data/manifest.json`。

前置条件：`factor_registry/parquet/factors_v<N>.parquet` **已存在**且包含候选 `name`
列。本脚本不写 parquet —— 物理数据文件只能通过：

1. `migrate_legacy_factors`（阶段 B，一次性 legacy 迁入）
2. 未来的 L1→L3 "新因子批量合入" 流程（阶段 D 交付）
3. 手动调用 `ParquetStore.write_version`

```bash
python -m scripts.promote.promote_certified \
    --cert factor_validation/certificates/<fid>.json \
    --parquet-version 1 \
    --tags production,rdagent \
    [--allow-overwrite]
```

返回码：`0` 注册成功；`2` 决议不是 PASS；`3` parquet 校验失败；`4` registry 层异常。

### 3.3 `retire_factor`

作用：把一个或多个 active 因子的 manifest status 设为 `retired`；证书副本从
`data/certified/` 搬到 `data/retired/`。**完全不改 parquet 物理文件。**

```bash
# 单个
python -m scripts.promote.retire_factor \
    --factor-id manual_VolRet_5D_11d70180fb \
    --reason "oracle default FAIL: rank_ic=0.006, ic_ir=0.024"

# 批量（每行一个 factor_id；# 开头的行忽略）
python -m scripts.promote.retire_factor \
    --ids-file tools/to_retire_v1.txt \
    --reason "orthogonality cluster pruning"

# 演练不落盘
python -m scripts.promote.retire_factor \
    --ids-file tools/to_retire_v1.txt \
    --reason "trial" \
    --dry-run
```

下游影响：`ProductionFactorLoader(only_active=True)`（默认）在下次 load 时自动跳过
retired 因子；历史回测仍可用 `only_active=False` 完整读取。

---

## 4. Oracle 复测（决策支持）

阶段 C 专用工具 `scripts.promote.oracle_revalidate_legacy`：把当前 L3 active 的全部
因子在**更严格**的 profile 下重跑，不修改 registry，输出 Markdown 报告供决策。

```bash
python -m scripts.promote.oracle_revalidate_legacy \
    --profile factor_validation/profiles/default.yaml \
    --write-report factor_validation/reports/legacy_oracle_default_v2.md
```

报告结构：总览表 + 逐因子四 check 详情（含 orthogonality 的 top-3 相关因子列表）。

**Windows / GBK 控制台兼容性**：

- 报告始终以 UTF-8 落盘（文件内容正常）；
- 控制台输出遇 `UnicodeEncodeError` 时仅记一条 warning，请直接看 `--write-report` 文件。

---

## 5. 回归测试清单（新增因子 / 改 check 时必跑）

1. **动算法类**（改 check 算法后必跑）
  ```
   pytest tests/factor_validation/ -q
  ```
2. **动存储类**（改 `ParquetStore` / `FactorRegistry` 后必跑）
  ```
   pytest tests/factor_registry/ tests/feature/ -q
  ```
3. **动 CLI 类**（改 `scripts/promote/`* 后必跑）
  ```
   pytest tests/promote/ -q
  ```
4. **动 profile 阈值类**（改任一 `factor_validation/profiles/*.yaml` 后）
  - 跑 1 + oracle 复测：
5. **动 feature pipeline / run_train 类**（集成回归）
  - 1 + 2 之外，额外跑 `tests/feature/test_pipeline_registry_integration.py -v` 确认
   L3 loader / legacy fallback 分支等价。

---

## 6. 已知限制与下一步


| 限制                         | 原因                                      | 计划                        |
| -------------------------- | --------------------------------------- | ------------------------- |
| L2 validator 不跑回测          | 阶段 C 仅离线 4 check，backtest/marginal 留给 D | 阶段 D 接 RQAlpha            |
| `validate_candidate` 不自动注册 | 有意为之：幂等 + 可重跑                           | 通过 `promote_certified` 串联 |
| 新因子写 parquet 仍需手动          | 阶段 D 会给出 L1→L3 "批量合入" 流程                | 阶段 D                      |
| Windows GBK stdout 仍可能乱码   | 控制台编码不可控                                | 始终走 `--write-report`      |


