"""C2 CertifiedFactorRecord & CheckResult schema 单元测试。

覆盖：
* 正常构造 + 序列化往返
* CheckResult: score 范围、threshold 有限值、frozen
* CertifiedFactorRecord: factor_id / candidate.factor_id 一致
* check_results 名字唯一
* PASS / FAIL 与 check.passed 强一致性
* HOLD 不强约束
* profile_hash 格式
* validator_version 格式
* backtest_metrics 非有限值
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from factor_lab import CandidateFactorPackage
from factor_validation import CertifiedFactorRecord, CheckResult, Decision


def _good_hash() -> str:
    return hashlib.sha256(b"some profile yaml").hexdigest()


def _passing_check(name: str = "ic", score: float = 0.7) -> CheckResult:
    return CheckResult(
        name=name,
        passed=True,
        score=score,
        threshold=0.3,
        detail={"sample": 1},
        elapsed_ms=100,
    )


def _failing_check(name: str = "turnover", score: float = 0.1) -> CheckResult:
    return CheckResult(
        name=name,
        passed=False,
        score=score,
        threshold=0.5,
        detail={"why": "too high"},
        elapsed_ms=80,
    )


# ============================================================ CheckResult
class TestCheckResult:
    def test_minimal_ok(self):
        c = CheckResult(name="ic", passed=True, elapsed_ms=10)
        assert c.score is None and c.threshold is None and c.detail == {}

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError, match="score"):
            CheckResult(name="ic", passed=True, score=1.5, elapsed_ms=10)

    def test_score_negative(self):
        with pytest.raises(ValidationError, match="score"):
            CheckResult(name="ic", passed=True, score=-0.1, elapsed_ms=10)

    def test_score_nan(self):
        with pytest.raises(ValidationError, match="score"):
            CheckResult(name="ic", passed=True, score=float("nan"), elapsed_ms=10)

    def test_threshold_inf(self):
        with pytest.raises(ValidationError, match="threshold"):
            CheckResult(
                name="ic", passed=True, score=0.5, threshold=float("inf"), elapsed_ms=10
            )

    def test_negative_elapsed(self):
        with pytest.raises(ValidationError):
            CheckResult(name="ic", passed=True, elapsed_ms=-1)

    def test_frozen(self):
        c = CheckResult(name="ic", passed=True, elapsed_ms=5)
        with pytest.raises(ValidationError):
            c.passed = False


# ====================================================== CertifiedFactorRecord
class TestCertifiedFactorRecord:
    def test_minimal_pass(self, minimal_candidate):
        rec = CertifiedFactorRecord(
            factor_id=minimal_candidate.factor_id,
            candidate=minimal_candidate,
            profile_name="strict",
            profile_hash=_good_hash(),
            decision=Decision.PASS,
            overall_score=0.78,
            check_results=[_passing_check("ic"), _passing_check("coverage", 0.9)],
            backtest_metrics={"sharpe": 2.1, "max_drawdown": 0.12},
            validated_at=datetime(2026, 4, 19, 13, 0, tzinfo=timezone.utc),
            validator_version="1.0.0",
        )
        assert rec.decision == Decision.PASS
        assert rec.overall_score == 0.78

    def test_serialization_roundtrip(self, minimal_candidate):
        rec = CertifiedFactorRecord(
            factor_id=minimal_candidate.factor_id,
            candidate=minimal_candidate,
            profile_name="strict",
            profile_hash=_good_hash(),
            decision=Decision.PASS,
            overall_score=0.6,
            check_results=[_passing_check("ic")],
            validated_at=datetime.now(timezone.utc),
            validator_version="1.0.0",
        )
        data = rec.model_dump(mode="json")
        restored = CertifiedFactorRecord.model_validate(data)
        assert restored == rec

    def test_factor_id_must_match_candidate(self, minimal_candidate):
        with pytest.raises(ValidationError, match="不一致"):
            CertifiedFactorRecord(
                factor_id="rdagent_OtherName_deadbeef",
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_pass_requires_all_passed(self, minimal_candidate):
        with pytest.raises(ValidationError, match="check 未通过"):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic"), _failing_check("turnover")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_fail_requires_some_failed(self, minimal_candidate):
        with pytest.raises(ValidationError, match="所有 check 都通过"):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.FAIL,
                overall_score=0.2,
                check_results=[_passing_check("ic"), _passing_check("coverage")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_hold_no_strict_check_constraint(self, minimal_candidate):
        # HOLD 允许 check 部分通过、部分失败
        rec = CertifiedFactorRecord(
            factor_id=minimal_candidate.factor_id,
            candidate=minimal_candidate,
            profile_name="default",
            profile_hash=_good_hash(),
            decision=Decision.HOLD,
            overall_score=0.45,
            check_results=[_passing_check("ic"), _failing_check("turnover")],
            validated_at=datetime.now(timezone.utc),
            validator_version="1.0.0",
            notes="边缘案例，等人工复核",
        )
        assert rec.decision == Decision.HOLD

    def test_check_results_unique_names(self, minimal_candidate):
        with pytest.raises(ValidationError, match="重名"):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic"), _passing_check("ic", 0.5)],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_check_results_min_length(self, minimal_candidate):
        with pytest.raises(ValidationError):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_overall_score_range(self, minimal_candidate):
        with pytest.raises(ValidationError):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=1.5,
                check_results=[_passing_check("ic")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    @pytest.mark.parametrize(
        "bad_hash",
        [
            "ABC" * 21 + "x",   # 含大写
            "0" * 63,           # 长度不对
            "0" * 65,
            "g" * 64,           # 非 hex
        ],
    )
    def test_profile_hash_format(self, minimal_candidate, bad_hash):
        with pytest.raises(ValidationError, match="profile_hash"):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=bad_hash,
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    @pytest.mark.parametrize(
        "version,ok",
        [
            ("1.0.0", True),
            ("0.9.7", True),
            ("2.10.3-rc1", True),
            ("1.0.0+build.5", True),
            ("1.0", False),
            ("v1.0.0", False),
            ("1.0.0.0", False),
        ],
    )
    def test_validator_version(self, minimal_candidate, version, ok):
        kwargs = dict(
            factor_id=minimal_candidate.factor_id,
            candidate=minimal_candidate,
            profile_name="strict",
            profile_hash=_good_hash(),
            decision=Decision.PASS,
            overall_score=0.6,
            check_results=[_passing_check("ic")],
            validated_at=datetime.now(timezone.utc),
            validator_version=version,
        )
        if ok:
            CertifiedFactorRecord(**kwargs)
        else:
            with pytest.raises(ValidationError, match="validator_version"):
                CertifiedFactorRecord(**kwargs)

    def test_backtest_metrics_nan_rejected(self, minimal_candidate):
        with pytest.raises(ValidationError, match="backtest_metrics"):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic")],
                backtest_metrics={"sharpe": float("nan")},
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
            )

    def test_frozen(self, minimal_candidate):
        rec = CertifiedFactorRecord(
            factor_id=minimal_candidate.factor_id,
            candidate=minimal_candidate,
            profile_name="strict",
            profile_hash=_good_hash(),
            decision=Decision.PASS,
            overall_score=0.6,
            check_results=[_passing_check("ic")],
            validated_at=datetime.now(timezone.utc),
            validator_version="1.0.0",
        )
        with pytest.raises(ValidationError):
            rec.decision = Decision.FAIL

    def test_extra_forbidden(self, minimal_candidate):
        with pytest.raises(ValidationError):
            CertifiedFactorRecord(
                factor_id=minimal_candidate.factor_id,
                candidate=minimal_candidate,
                profile_name="strict",
                profile_hash=_good_hash(),
                decision=Decision.PASS,
                overall_score=0.6,
                check_results=[_passing_check("ic")],
                validated_at=datetime.now(timezone.utc),
                validator_version="1.0.0",
                rogue_field="x",
            )
