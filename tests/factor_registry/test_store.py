"""单元测试：``factor_registry.store.ParquetStore``。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_registry.store import ParquetStore, ParquetStoreError


def _mk_df(cols=("FactorA", "FactorB")) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-03"), "SH600000"),
            (pd.Timestamp("2025-01-03"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {c: [0.1, 0.2, 0.3, 0.4] for c in cols}, index=idx, dtype="float64"
    )


# --------------------------------------------------------------------- path_for


def test_path_for_uses_template(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    assert st.path_for(1) == (tmp_path / "factors_v1.parquet").resolve()
    assert st.path_for(12) == (tmp_path / "factors_v12.parquet").resolve()


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1", None])
def test_path_for_rejects_bad_version(tmp_path: Path, bad) -> None:
    st = ParquetStore(tmp_path)
    with pytest.raises(ParquetStoreError):
        st.path_for(bad)  # type: ignore[arg-type]


def test_exists_false_initially(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    assert st.exists(1) is False


# ---------------------------------------------------------------- write/read RT


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    target = st.write_version(df, 1)
    assert target.is_file()
    assert st.exists(1)
    got = st.read_version(1)
    pd.testing.assert_frame_equal(got, df)


def test_write_overwrite_flag(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(), 1)
    with pytest.raises(ParquetStoreError, match="overwrite"):
        st.write_version(_mk_df(), 1)
    st.write_version(_mk_df(cols=("Alpha1",)), 1, overwrite=True)
    got = st.read_version(1)
    assert list(got.columns) == ["Alpha1"]


def test_read_version_missing(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    with pytest.raises(ParquetStoreError, match="不存在"):
        st.read_version(5)


def test_read_columns_subset(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(cols=("FactorA", "FactorB", "FactorC")), 1)
    got = st.read_version(1, columns=["FactorA", "FactorC"])
    assert list(got.columns) == ["FactorA", "FactorC"]
    assert len(got) == 4


def test_read_columns_missing_raises(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(), 1)
    with pytest.raises(ParquetStoreError, match="不存在于 v1"):
        st.read_version(1, columns=["FactorA", "NotExist"])


def test_read_columns_empty_list_raises(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(), 1)
    with pytest.raises(ParquetStoreError, match="空列表"):
        st.read_version(1, columns=[])


def test_read_sort_index(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    shuffled = df.iloc[[2, 0, 3, 1]]
    st.write_version(shuffled, 1)
    got = st.read_version(1)
    # 读回后索引排序稳定
    assert got.index.is_monotonic_increasing


# ---------------------------------------------------------------- list_versions


def test_list_and_latest_versions(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    assert st.list_versions() == []
    assert st.latest_version() is None

    st.write_version(_mk_df(), 3)
    st.write_version(_mk_df(), 1)
    st.write_version(_mk_df(), 2)
    assert st.list_versions() == [1, 2, 3]
    assert st.latest_version() == 3


def test_list_versions_ignores_unrelated(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(), 1)
    (tmp_path / "other.parquet").write_bytes(b"x")
    (tmp_path / "factors_vA.parquet").write_bytes(b"x")
    (tmp_path / "subdir").mkdir()
    assert st.list_versions() == [1]


# --------------------------------------------------------------------- validate


def test_reject_non_dataframe(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    with pytest.raises(ParquetStoreError, match="DataFrame"):
        st.write_version({"a": 1}, 1)  # type: ignore[arg-type]


def test_reject_empty(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df().iloc[0:0]
    with pytest.raises(ParquetStoreError, match="为空"):
        st.write_version(df, 1)


def test_reject_single_index(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = pd.DataFrame({"Foo": [1.0]}, dtype="float64")
    with pytest.raises(ParquetStoreError, match="MultiIndex"):
        st.write_version(df, 1)


def test_reject_wrong_index_names(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    df = df.rename_axis(index={"datetime": "date"})
    with pytest.raises(ParquetStoreError, match=r"\['datetime','instrument'\]"):
        st.write_version(df, 1)


def test_reject_bad_column_name(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    df = df.rename(columns={"FactorA": "1BadName"})
    with pytest.raises(ParquetStoreError, match="列名"):
        st.write_version(df, 1)


def test_reject_non_float64_dtype(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    df["FactorA"] = df["FactorA"].astype("float32")
    with pytest.raises(ParquetStoreError, match="float64"):
        st.write_version(df, 1)


def test_reject_duplicate_columns(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    df = _mk_df()
    df.columns = ["FactorA", "FactorA"]
    with pytest.raises(ParquetStoreError, match="重复"):
        st.write_version(df, 1)


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    st = ParquetStore(tmp_path)
    st.write_version(_mk_df(), 1)
    tmps = list(tmp_path.glob("*.tmp*"))
    assert tmps == []
