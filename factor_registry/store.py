"""
``ParquetStore`` —— L3 production 因子物理存储的版本化管理。

职责边界（见 ``docs/ARCHITECTURE_FACTOR_LAB.md`` §3）：

* 仅管 **物理文件** ``factors_v<N>.parquet``；不涉及 manifest / 证书等元信息（那是
  ``FactorRegistry`` 的事）。
* 写入为 **原子写**：先写 ``*.tmp`` 再 ``os.replace``，避免半截文件污染 production。
* 读取支持 **列裁剪**（pyarrow 原生），配合 L3 只取 active 列的消费模式。
* 写入前会强校验 MultiIndex、列 dtype、列名合法性——L3 数据质量红线。

本模块 **不引入 qlib / pydantic schema**，保持纯数据层，方便单测不落库。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_VERSION_FILE_PATTERN = re.compile(r"^factors_v(\d+)\.parquet$")
_COLUMN_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ParquetStoreError(RuntimeError):
    """``ParquetStore`` 抛出的所有业务异常基类。"""


class ParquetStore:
    """版本化 production parquet 存储。

    Parameters
    ----------
    root : Path
        parquet 根目录，通常是 ``factor_registry/parquet/``。若不存在会自动创建。
    """

    #: 文件名模板；``version`` 为整数 >= 1。
    FILE_TEMPLATE = "factors_v{version}.parquet"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ path

    def path_for(self, version: int) -> Path:
        """返回指定版本号对应的绝对路径（文件不一定存在）。"""
        if not isinstance(version, int) or version < 1:
            raise ParquetStoreError(
                f"version 必须为正整数，当前: {version!r}"
            )
        return self.root / self.FILE_TEMPLATE.format(version=version)

    def exists(self, version: int) -> bool:
        """版本对应的 parquet 是否已存在。"""
        return self.path_for(version).is_file()

    # ----------------------------------------------------------------- write

    def write_version(
        self,
        df: pd.DataFrame,
        version: int,
        *,
        overwrite: bool = False,
    ) -> Path:
        """把 ``df`` 原子地写入 ``factors_v<version>.parquet``。

        Parameters
        ----------
        df : pd.DataFrame
            必须为 MultiIndex ``(datetime, instrument)``；所有列 dtype 均为 float64；
            列名符合 ``^[A-Za-z][A-Za-z0-9_]{0,63}$``。
        version : int
            正整数版本号。
        overwrite : bool
            目标文件已存在时：默认报错；``True`` 时覆盖。

        Returns
        -------
        Path
            最终落盘的文件绝对路径。
        """
        target = self.path_for(version)
        if target.exists() and not overwrite:
            raise ParquetStoreError(
                f"目标版本已存在（如需覆盖请传 overwrite=True）: {target}"
            )
        self._validate_df(df)

        # 原子写：在同一目录下创建 NamedTemporaryFile，再 os.replace 到目标。
        # 跨文件系统 os.replace 可能失败；保持在同目录最稳。
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=".tmp.parquet",
            dir=str(self.root),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        try:
            df.to_parquet(tmp_path, engine="pyarrow")
            os.replace(tmp_path, target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        logger.info(
            "ParquetStore 写入 v%d 成功：%s（rows=%d, cols=%d）",
            version,
            target,
            len(df),
            df.shape[1],
        )
        return target

    # ------------------------------------------------------------------ read

    def read_version(
        self,
        version: int,
        *,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """读取指定版本的 parquet，可按列裁剪。

        Parameters
        ----------
        version : int
            目标版本号。
        columns : list[str] | None
            只读这些列；``None`` 读全部。任一列不存在会抛错。

        Returns
        -------
        pd.DataFrame
            索引按 ``(datetime, instrument)`` 排序的 DataFrame。
        """
        target = self.path_for(version)
        if not target.is_file():
            raise ParquetStoreError(f"版本不存在: {target}")
        if columns is not None:
            if not columns:
                raise ParquetStoreError("columns 不能是空列表；传 None 读全部")
            bad = [c for c in columns if not isinstance(c, str) or not c.strip()]
            if bad:
                raise ParquetStoreError(f"columns 含无效项: {bad}")
            # 先只读 schema 校验列存在，避免 pyarrow 偏移读到一半报难懂的错
            available = self._peek_columns(target)
            missing = [c for c in columns if c not in available]
            if missing:
                raise ParquetStoreError(
                    f"请求的列不存在于 v{version}: {missing}；"
                    f"可用列: {available}"
                )
        df = pd.read_parquet(target, engine="pyarrow", columns=columns)
        # pyarrow 读取后索引顺序不保证，这里强制稳定
        if isinstance(df.index, pd.MultiIndex):
            df = df.sort_index()
        return df

    def _peek_columns(self, path: Path) -> list[str]:
        """仅读 parquet schema 得到列清单（不触发数据读取）。"""
        import pyarrow.parquet as pq  # noqa: WPS433 延迟导入

        schema = pq.ParquetFile(str(path)).schema_arrow
        # 数据列 = schema 列 - 索引列
        idx_names = set(schema.pandas_metadata.get("index_columns", []))  # type: ignore[union-attr]
        return [n for n in schema.names if n not in idx_names]

    # -------------------------------------------------------------- versions

    def list_versions(self) -> list[int]:
        """扫描 root 下所有 ``factors_v<N>.parquet``，返回升序版本号。"""
        versions: list[int] = []
        for entry in self.root.iterdir():
            if not entry.is_file():
                continue
            m = _VERSION_FILE_PATTERN.match(entry.name)
            if m:
                versions.append(int(m.group(1)))
        return sorted(versions)

    def latest_version(self) -> int | None:
        """最新（最大）版本号；若无任何版本返回 ``None``。"""
        versions = self.list_versions()
        return versions[-1] if versions else None

    # -------------------------------------------------------------- validate

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        """写入前的硬校验——违反则拒绝落盘。"""
        if not isinstance(df, pd.DataFrame):
            raise ParquetStoreError(f"df 必须是 DataFrame，当前: {type(df)!r}")
        if df.empty:
            raise ParquetStoreError("df 为空，拒绝写入")

        idx = df.index
        if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
            raise ParquetStoreError(
                f"index 必须为两级 MultiIndex，当前: {idx!r}"
            )
        if list(idx.names) != ["datetime", "instrument"]:
            raise ParquetStoreError(
                f"MultiIndex 名必须为 ['datetime','instrument']，当前: {list(idx.names)}"
            )

        if df.columns.empty:
            raise ParquetStoreError("df 必须至少含一列因子值")
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            raise ParquetStoreError(f"df 列名重复: {dup_cols}")
        bad_names = [
            c for c in df.columns
            if not isinstance(c, str) or not _COLUMN_NAME_PATTERN.match(c)
        ]
        if bad_names:
            raise ParquetStoreError(
                f"列名必须以字母开头、仅含字母数字下划线、长度<=64: {bad_names}"
            )

        bad_dtype = {
            col: str(dt) for col, dt in df.dtypes.items() if str(dt) != "float64"
        }
        if bad_dtype:
            raise ParquetStoreError(
                f"所有列 dtype 必须为 float64，违规: {bad_dtype}"
            )
