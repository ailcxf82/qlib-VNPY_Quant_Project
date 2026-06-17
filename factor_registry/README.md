# factor_registry —— L3 正式因子注册表

> 本包是三层架构的 L3 层。请先阅读 [`docs/ARCHITECTURE_FACTOR_LAB.md`](../docs/ARCHITECTURE_FACTOR_LAB.md)。

## 子目录

| 路径 | 职责 |
|---|---|
| `registry.py` | `FactorRegistry`：register / list / retire / query（阶段 B 实现） |
| `store.py` | 物料存储：把因子值落到 production parquet，按版本号管理（阶段 B 实现） |
| `schema.py` | `ProductionFactorRecord` 等 schema |
| `data/` | **入版本控制**；manifest.json + 证书副本 + 退役档案 |
| `data/manifest.json` | 当前 active 因子清单 |
| `data/certified/` | 每个 active 因子一份证书副本（来自 L2） |
| `data/retired/` | 已退役因子证书归档 |
| `parquet/` | gitignore；production 因子值仓库（按版本号），生产管线唯一数据源 |

## 对外唯一接口

任何 production 模块要读因子，**必须**走 `feature.production_factor_loader`，禁止直接 import
`factor_registry.store` / 直接读 parquet。

## 不变量

- `data/manifest.json` 中的 `factor_id` 与 `data/certified/<factor_id>.json` 一一对应
- 任何写入 `parquet/` 的因子必须先通过 `FactorRegistry.register()` 落 manifest
- 退役动作必须 `FactorRegistry.retire(factor_id)`，证书移到 `data/retired/`

## 当前状态

- 阶段 A：仅有 schema 和目录骨架
- 阶段 B：实现 `registry.py` + `store.py` + `production_factor_loader`，并把现有
  `combined_factors_df.parquet` 一次性迁移进来
