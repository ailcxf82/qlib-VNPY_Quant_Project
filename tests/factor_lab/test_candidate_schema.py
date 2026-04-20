"""C1 CandidateFactorPackage schema 单元测试。

覆盖：
* 正常构造 + 序列化往返
* factor_id 命名规则
* name 命名规则
* date_range 顺序
* source / parent_loop 一致性
* lab_metrics 非有限值
* model_config: frozen / extra=forbid
* validate_artifacts: 文件存在 / parquet 结构 / 时间窗
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from factor_lab import CandidateFactorPackage


# ====================================================================== ok
class TestConstructionOK:
    def test_minimal_candidate_constructs(self, minimal_candidate):
        assert minimal_candidate.factor_id == "rdagent_QualPersist_60D_a1b2c3d4"
        assert minimal_candidate.name == "QualPersist_60D"
        assert minimal_candidate.source == "rdagent"
        assert minimal_candidate.parent_loop == 0

    def test_serialization_roundtrip(self, minimal_candidate):
        data = minimal_candidate.model_dump(mode="json")
        restored = CandidateFactorPackage.model_validate(data)
        assert restored == minimal_candidate

    def test_manual_source_no_parent_loop(self, tmp_path):
        name = "MyFactor"
        candidate = CandidateFactorPackage(
            factor_id=f"manual_{name}_0123abcd",
            name=name,
            source="manual",
            hypothesis="manual one",
            formulation="x",
            code_path=tmp_path / "factor.py",
            values_path=tmp_path / "v.parquet",
            universe="all",
            date_range=(date(2024, 1, 1), date(2024, 12, 31)),
            parent_loop=None,
            created_at=datetime.now(timezone.utc),
            lab_run_id="lab-manual-001",
        )
        assert candidate.source == "manual"
        assert candidate.parent_loop is None


# ============================================================== factor_id
class TestFactorIdValidation:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "RDAGENT_X_a1b2c3d4",          # source 大写
            "rdagent_X_a1b2c3",            # hash <8
            "rdagent_X_a1b2c3d4e5f6g7h8",  # hash 含非 hex
            "rdagent_X 60_a1b2c3d4",       # 含空格
            "rdagent__a1b2c3d4",           # name 空
            "rdagent-X-a1b2c3d4",          # 用了 -
            "rdagent_X",                   # 缺 hash
        ],
    )
    def test_invalid_factor_id_format(self, tmp_path, bad_id):
        with pytest.raises(ValidationError):
            CandidateFactorPackage(
                factor_id=bad_id,
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "factor.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )

    def test_factor_id_name_segment_must_match_name(self, tmp_path):
        with pytest.raises(ValidationError, match="name 段"):
            CandidateFactorPackage(
                factor_id="rdagent_OtherName_a1b2c3d4",
                name="QualPersist_60D",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "factor.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )


# =================================================================== name
class TestNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "1StartsDigit",
            "has space",
            "has-dash",
            "",
            "a" * 65,
        ],
    )
    def test_invalid_name(self, tmp_path, bad_name):
        with pytest.raises(ValidationError):
            CandidateFactorPackage(
                factor_id=f"rdagent_{bad_name}_a1b2c3d4",
                name=bad_name,
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )


# ================================================================ source / parent_loop
class TestSourceParentLoopConsistency:
    def test_manual_must_have_no_parent_loop(self, tmp_path):
        with pytest.raises(ValidationError, match="manual"):
            CandidateFactorPackage(
                factor_id="manual_X_0123abcd",
                name="X",
                source="manual",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=3,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )

    def test_rdagent_must_have_parent_loop(self, tmp_path):
        with pytest.raises(ValidationError, match="rdagent"):
            CandidateFactorPackage(
                factor_id="rdagent_X_a1b2c3d4",
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=None,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )


# =================================================================== date_range
class TestDateRange:
    def test_start_after_end_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="date_range"):
            CandidateFactorPackage(
                factor_id="rdagent_X_a1b2c3d4",
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2025, 6, 1), date(2024, 1, 1)),
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )


# =================================================================== lab_metrics
class TestLabMetrics:
    def test_nan_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="lab_metrics"):
            CandidateFactorPackage(
                factor_id="rdagent_X_a1b2c3d4",
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                lab_metrics={"ic": float("nan")},
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )

    def test_inf_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            CandidateFactorPackage(
                factor_id="rdagent_X_a1b2c3d4",
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                lab_metrics={"ic": float("inf")},
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
            )


# ================================================================= model_config
class TestModelConfig:
    def test_frozen(self, minimal_candidate):
        with pytest.raises(ValidationError):
            minimal_candidate.name = "Mutated"

    def test_extra_forbidden(self, tmp_path):
        with pytest.raises(ValidationError):
            CandidateFactorPackage(
                factor_id="rdagent_X_a1b2c3d4",
                name="X",
                source="rdagent",
                hypothesis="h",
                formulation="f",
                code_path=tmp_path / "f.py",
                values_path=tmp_path / "v.parquet",
                universe="all",
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
                parent_loop=0,
                created_at=datetime.now(timezone.utc),
                lab_run_id="lab-x-001",
                unknown_field="oops",
            )


# ============================================================ validate_artifacts
class TestValidateArtifacts:
    def test_ok_minimal(self, minimal_candidate):
        minimal_candidate.validate_artifacts(check_parquet_schema=True)

    def test_missing_code_path(self, minimal_candidate):
        # 删 code 文件后再验
        minimal_candidate.code_path.unlink()
        with pytest.raises(ValueError, match="code_path 不存在"):
            minimal_candidate.validate_artifacts()

    def test_missing_values_path(self, minimal_candidate):
        minimal_candidate.values_path.unlink()
        with pytest.raises(ValueError, match="values_path 不存在"):
            minimal_candidate.validate_artifacts()

    def test_wrong_code_suffix(self, tmp_path, minimal_candidate):
        # 重新构造一个使用 .txt 的候选
        bad_code = tmp_path / "factor.txt"
        bad_code.write_text("not python")
        cand = minimal_candidate.model_copy(update={"code_path": bad_code})
        with pytest.raises(ValueError, match=".py"):
            cand.validate_artifacts()

    def test_parquet_wrong_index_levels(self, tmp_path, minimal_candidate):
        # 构造单 index parquet
        bad = tmp_path / "bad.parquet"
        pd.DataFrame({"QualPersist_60D": [0.1]}, index=pd.Index(["x"], name="x")).to_parquet(bad)
        cand = minimal_candidate.model_copy(update={"values_path": bad})
        with pytest.raises(ValueError, match="MultiIndex"):
            cand.validate_artifacts(check_parquet_schema=True)

    def test_parquet_wrong_column_name(self, tmp_path, minimal_candidate):
        bad = tmp_path / "bad.parquet"
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "SH600000")],
            names=["datetime", "instrument"],
        )
        pd.DataFrame({"WRONG_NAME": [0.1]}, index=idx, dtype="float64").to_parquet(bad)
        cand = minimal_candidate.model_copy(update={"values_path": bad})
        with pytest.raises(ValueError, match="列名"):
            cand.validate_artifacts(check_parquet_schema=True)

    def test_parquet_wrong_dtype(self, tmp_path, minimal_candidate):
        bad = tmp_path / "bad.parquet"
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "SH600000")],
            names=["datetime", "instrument"],
        )
        pd.DataFrame({"QualPersist_60D": [1]}, index=idx, dtype="int64").to_parquet(bad)
        cand = minimal_candidate.model_copy(update={"values_path": bad})
        with pytest.raises(ValueError, match="float64"):
            cand.validate_artifacts(check_parquet_schema=True)

    def test_parquet_time_window_overflow(self, tmp_path, minimal_candidate):
        bad = tmp_path / "bad.parquet"
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2030-01-02"), "SH600000")],  # 远超 date_range
            names=["datetime", "instrument"],
        )
        pd.DataFrame({"QualPersist_60D": [0.1]}, index=idx, dtype="float64").to_parquet(bad)
        cand = minimal_candidate.model_copy(update={"values_path": bad})
        with pytest.raises(ValueError, match="超出声明 date_range"):
            cand.validate_artifacts(check_parquet_schema=True)

    def test_skip_parquet_schema_check(self, tmp_path, minimal_candidate):
        # 即使 parquet 内容不规范，check_parquet_schema=False 也应通过
        bad = tmp_path / "bad.parquet"
        pd.DataFrame({"WRONG": [1]}).to_parquet(bad)
        cand = minimal_candidate.model_copy(update={"values_path": bad})
        cand.validate_artifacts(check_parquet_schema=False)


# ============================================================ Path 强制规范化
class TestPathCoercion:
    def test_string_path_coerced_to_path(self, tmp_path):
        # 给字符串，应被自动转为 Path
        code_path = tmp_path / "f.py"
        code_path.write_text("# x")
        values_path = tmp_path / "v.parquet"
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-06-01"), "SH600000")],
            names=["datetime", "instrument"],
        )
        pd.DataFrame({"X": [0.1]}, index=idx, dtype="float64").to_parquet(values_path)

        cand = CandidateFactorPackage(
            factor_id="rdagent_X_a1b2c3d4",
            name="X",
            source="rdagent",
            hypothesis="h",
            formulation="f",
            code_path=str(code_path),
            values_path=str(values_path),
            universe="all",
            date_range=(date(2024, 1, 1), date(2024, 12, 31)),
            parent_loop=0,
            created_at=datetime.now(timezone.utc),
            lab_run_id="lab-x-001",
        )
        assert isinstance(cand.code_path, Path)
        assert isinstance(cand.values_path, Path)
