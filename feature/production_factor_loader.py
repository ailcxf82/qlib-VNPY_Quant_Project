"""
``ProductionFactorLoader`` —— 读取 L3 production 因子的**唯一**对外接口。

此模块是 production 训练 / 预测 / 回测路径访问因子数据的窄接口（见
``docs/ARCHITECTURE_FACTOR_LAB.md`` §5）。任何 production 代码都应该通过这里拿因子值，
**禁止**直接读 ``git_ignore_folder/combined_factors_df.parquet`` 或 ``factor_registry/parquet/*``。

工作流：

1. 读 ``factor_registry.registry.FactorRegistry`` 的 ``list_active()`` 得到 active 因子清单；
2. 按 ``parquet_version`` 分组，从 ``factor_registry.store.ParquetStore`` 读取对应的 parquet
   列子集；
3. 把多版本的结果按 column 维度合并（目前仅支持同一 version，多版本预留扩展点）；
4. 若传入 ``align_index``，按该 MultiIndex reindex 对齐（供特征 pipeline 合并）。

设计原则：

* 读优先、不改写；所有写入走 ``FactorRegistry`` / ``ParquetStore``。
* 空 registry 不报错，返回空 DataFrame + 空列表（供 B 阶段迁移脚本未跑时的过渡）。
* 完全等价性：给定相同 registry + parquet 文件，``load_active()`` 的输出**按值**确定。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from factor_registry.registry import FactorRegistry
from factor_registry.schema import ProductionFactorRecord
from factor_registry.store import ParquetStore

logger = logging.getLogger(__name__)

# 项目根：feature/ 目录位于 <root>/feature/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "factor_lab.yaml"


class ProductionFactorLoader:
    """Production 因子值的唯一读取入口。

    Parameters
    ----------
    registry : FactorRegistry
        已加载的 manifest 注册表。
    store : ParquetStore
        与 registry 对应的 parquet 物理存储。
    parquet_version : int | None
        指定消费的 parquet 版本。None → 使用 registry 中 active 记录的最大版本号；
        若 registry 为空，则 None 时返回空数据不报错。
    """

    def __init__(
        self,
        *,
        registry: FactorRegistry,
        store: ParquetStore,
        parquet_version: int | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self._explicit_version = parquet_version

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_default_paths(
        cls,
        *,
        project_root: Path | None = None,
        parquet_version: int | None = None,
        config_path: Path | None = None,
    ) -> "ProductionFactorLoader":
        """按 ``config/factor_lab.yaml`` 和默认目录布局构造 loader。"""
        root = Path(project_root).resolve() if project_root else _PROJECT_ROOT
        cfg_path = Path(config_path).resolve() if config_path else _DEFAULT_CONFIG

        reg_dir, parquet_dir, cfg_version = _parse_factor_lab_config(cfg_path, root)
        registry = FactorRegistry(reg_dir)
        store = ParquetStore(parquet_dir)
        version = parquet_version if parquet_version is not None else cfg_version
        return cls(registry=registry, store=store, parquet_version=version)

    # ----------------------------------------------------------------- version

    def resolve_version(self) -> int | None:
        """决定当前应读取的 parquet 版本。

        优先级：
        1. 构造时显式传入的 ``parquet_version``（若对应文件存在于 store）；
        2. registry 内 active 记录中最大的 ``parquet_version``；
        3. 若都没有，返回 None（调用方应按"空因子集"语义处理）。
        """
        if self._explicit_version is not None:
            if self.store.exists(self._explicit_version):
                return self._explicit_version
            logger.warning(
                "loader: 指定 parquet_version=%d 在 store 中不存在（%s）",
                self._explicit_version,
                self.store.root,
            )
            return None
        active = self.registry.list_active()
        if not active:
            return None
        return max(r.parquet_version for r in active)

    # ----------------------------------------------------------------- records

    def _active_records_for(
        self, version: int
    ) -> list[ProductionFactorRecord]:
        """取该版本下 active 记录，按 (registered_at, factor_id) 排序（输出列顺序稳定）。"""
        items = self.registry.list_by_version(version, only_active=True)
        return sorted(items, key=lambda r: (r.registered_at, r.factor_id))

    def list_active_columns(self) -> list[str]:
        """返回当前应该被 production 消费的因子列名。"""
        version = self.resolve_version()
        if version is None:
            return []
        return [r.parquet_column for r in self._active_records_for(version)]

    # -------------------------------------------------------------------- load

    def load_active(
        self,
        *,
        align_index: pd.MultiIndex | None = None,
    ) -> pd.DataFrame:
        """读取当前 active 因子值。

        Parameters
        ----------
        align_index : pd.MultiIndex | None
            若给出，输出 DataFrame 的 index 会 reindex 对齐到它（填充 NaN）；
            否则返回 store 中的原始 index。

        Returns
        -------
        pd.DataFrame
            MultiIndex(datetime, instrument)，列 == ``list_active_columns()``。
            若没有任何 active 因子，返回**形状合法**的空 DataFrame：
            * ``align_index is None`` → 0 行 0 列
            * 否则 → len(align_index) 行 0 列
        """
        version = self.resolve_version()
        cols = self.list_active_columns()
        if version is None or not cols:
            if align_index is not None:
                return pd.DataFrame(index=align_index)
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [], names=["datetime", "instrument"]
                )
            )

        df = self.store.read_version(version, columns=cols)
        # 强制按 cols 顺序输出，避免 parquet 内列序漂移影响下游特征顺序
        df = df.loc[:, cols]
        if align_index is not None:
            df = df.reindex(align_index)
        return df

    def load_columns(
        self,
        columns: list[str],
        *,
        align_index: pd.MultiIndex | None = None,
    ) -> pd.DataFrame:
        """按用户显式传入的列名列表读取；不做 active 过滤。

        适用场景：预测 / 回测时为了等价复现 **某次训练** 的因子集，需要读取历史快照。
        """
        if not columns:
            raise ValueError("columns 不能为空；如要读全部 active 请调 load_active()")
        version = self.resolve_version()
        if version is None:
            raise RuntimeError(
                "no production parquet version available；请先跑 migrate_legacy_factors "
                "或确认 factor_lab.yaml.registry.current_parquet_version"
            )
        df = self.store.read_version(version, columns=columns)
        df = df.loc[:, columns]
        if align_index is not None:
            df = df.reindex(align_index)
        return df


# ====================================================================== helpers


def _parse_factor_lab_config(
    cfg_path: Path, project_root: Path
) -> tuple[Path, Path, int | None]:
    """解析 ``config/factor_lab.yaml`` 得到 (data_dir, parquet_dir, current_version)。"""
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"factor_lab 配置不存在：{cfg_path}（如需自定义请传 config_path）"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    reg_cfg = cfg.get("registry") or {}
    data_dir = _abs(reg_cfg.get("data_dir", "factor_registry/data"), project_root)
    parquet_dir = _abs(
        reg_cfg.get("parquet_dir", "factor_registry/parquet"), project_root
    )
    version = reg_cfg.get("current_parquet_version")
    if version is not None:
        try:
            version = int(version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"factor_lab.yaml registry.current_parquet_version 必须为整数: {version!r}"
            ) from exc
    return data_dir, parquet_dir, version


def _abs(path_like: str | Path, project_root: Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


# ----------------------------------------------------------------------- 便捷函数


def load_active_factors(
    *,
    project_root: Path | None = None,
    parquet_version: int | None = None,
    align_index: pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """给 pipeline 用的极简函数式封装，返回 ``(df, column_names)``。

    Returns
    -------
    df : pd.DataFrame
        参见 :meth:`ProductionFactorLoader.load_active`。
    column_names : list[str]
        与 df.columns 顺序一致的列名列表（方便调用方注入 feature_sets）。
    """
    loader = ProductionFactorLoader.from_default_paths(
        project_root=project_root, parquet_version=parquet_version
    )
    cols = loader.list_active_columns()
    df = loader.load_active(align_index=align_index)
    return df, cols
