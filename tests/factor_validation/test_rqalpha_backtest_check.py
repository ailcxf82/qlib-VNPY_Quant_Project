"""单测：``factor_validation.checks.rqalpha_backtest_check``。

为避免真跑 RQAlpha（每次 5~15 分钟且依赖本机 bundle），所有测试都通过
``config["_runner_override"]`` 注入一个轻量 fake runner，它只做两件事：

1. 读 check 传进来的 ``prediction_path`` 做 sanity check
2. 往 ``output_dir/report.json`` 写一份预设 summary

这样就能独立测：
    * 预测 CSV 生成 & 列对齐
    * summary 解析 & 多 key 兼容
    * 阈值判决 & 负向翻转
    * 缺字段 / 缺文件的错误分支
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.rqalpha_backtest_check import RqalphaBacktestCheck


_HEX_TAG_DEFAULT = "abcdef12"


def _stub_candidate(
    tmp_path: Path, name: str, df: pd.DataFrame, factor_id: str | None = None
) -> CandidateFactorPackage:
    tag = (factor_id or _HEX_TAG_DEFAULT).lower()
    hex_tag = "".join(c for c in tag if c in "0123456789abcdef")
    if len(hex_tag) < 8:
        hex_tag = (hex_tag + _HEX_TAG_DEFAULT)[:8]
    full_id = f"manual_{name}_{hex_tag}"
    vp = tmp_path / f"{full_id}.parquet"
    cp = tmp_path / f"{full_id}.py"
    df[[name]].astype("float64").to_parquet(vp)
    cp.write_text(
        '"""stub"""\n\ndef calculate(df):\n    return df\n', encoding="utf-8"
    )
    return CandidateFactorPackage(
        factor_id=full_id,
        name=name,
        source="manual",
        hypothesis="t",
        formulation="t",
        code_path=cp,
        values_path=vp,
        universe="csi300",
        date_range=(date(2025, 1, 2), date(2025, 12, 31)),
        lab_metrics={},
        parent_loop=None,
        created_at=pd.Timestamp("2026-04-19", tz="UTC").to_pydatetime(),
        lab_run_id="lab-rqa-run-00001",
    )


def _mk_series(arr: np.ndarray, instruments: list[str], name: str) -> pd.DataFrame:
    T, N = arr.shape
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({name: arr.reshape(-1)}, index=idx).astype("float64")


def _ctx(project_root: Path) -> CheckContext:
    return CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 6, 30)),
        benchmark="SH000300",
        project_root=project_root,
    )


def _prepare_project_root(
    tmp_path: Path, rqalpha_config_name: str = "rqalpha_config.yaml"
) -> Path:
    """在 tmp_path 下放一个 minimal config/<rqalpha_config.yaml>，模拟项目根。"""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "config").mkdir()
    (root / "config" / rqalpha_config_name).write_text(
        "base:\n  start_date: '2025-01-02'\n  end_date: '2025-06-30'\n",
        encoding="utf-8",
    )
    return root


def _fake_runner_factory(
    summary: dict,
    *,
    record: list | None = None,
    write_report: bool = True,
):
    """构造一个 fake run_rqalpha_backtest。

    record（如果提供）会被 runner 写入一条字典，方便断言 check 传进来的参数。
    """

    def _fake_runner(
        *,
        rqalpha_config_path: str,
        prediction_path: str,
        industry_path,
        strategy_path,
        full_invested: bool,
        score_col: str,
        output_dir: str,
    ):
        # 1) prediction CSV 基本 sanity：列齐全 + 非空
        df = pd.read_csv(prediction_path)
        required = {"datetime", "instrument", score_col}
        if not required.issubset(df.columns):
            raise AssertionError(
                f"fake_runner: prediction 缺列，need={required}, got={list(df.columns)}"
            )
        if df.empty:
            raise AssertionError("fake_runner: prediction 为空")

        # 2) 写 report.json
        out = Path(output_dir)
        if write_report:
            (out / "report.json").write_text(
                json.dumps({"summary": summary}, ensure_ascii=False),
                encoding="utf-8",
            )

        # 3) 记录调用参数
        if record is not None:
            record.append(
                {
                    "rqalpha_config_path": rqalpha_config_path,
                    "prediction_path": prediction_path,
                    "strategy_path": strategy_path,
                    "full_invested": full_invested,
                    "score_col": score_col,
                    "output_dir": output_dir,
                    "prediction_rows": len(df),
                }
            )
        return {"_fake": True}

    return _fake_runner


# ---------------------------------------------------------- 正常 PASS


def test_passes_with_strong_metrics(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 40, 30
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "pass_ok")

    record: list = []
    runner = _fake_runner_factory(
        summary={
            "sharpe": 2.1,
            "annualized_returns": 0.22,
            "max_drawdown": -0.08,
            "volatility": 0.12,
            "total_returns": 0.35,
        },
        record=record,
    )

    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "min_annual_return": 0.05,
        "min_total_return": 0.0,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }

    result = RqalphaBacktestCheck().run(cand, _ctx(root), cfg)
    assert result.passed, result.detail
    assert result.detail["sharpe"] == pytest.approx(2.1)
    assert result.detail["max_drawdown"] == pytest.approx(0.08)
    assert result.detail["direction"] == 1
    assert 0.0 < result.score <= 1.0
    assert len(record) == 1
    assert record[0]["prediction_rows"] == T * N


# ---------------------------------------------------------- FAIL on low sharpe


def test_fails_on_low_sharpe(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 40, 30
    rng = np.random.default_rng(1)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "low_sharpe")

    runner = _fake_runner_factory(
        summary={
            "sharpe": 0.3,
            "annualized_returns": 0.03,
            "max_drawdown": 0.10,
        }
    )

    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }

    result = RqalphaBacktestCheck().run(cand, _ctx(root), cfg)
    assert not result.passed
    assert result.detail["sharpe"] == pytest.approx(0.3)


# ---------------------------------------------------------- FAIL on high drawdown


def test_fails_on_high_drawdown(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 40, 30
    rng = np.random.default_rng(2)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "high_dd")

    runner = _fake_runner_factory(
        summary={
            "sharpe": 2.0,
            "annualized_returns": 0.20,
            "max_drawdown": 0.55,  # 大幅超过 0.30 阈值
        }
    )

    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }

    result = RqalphaBacktestCheck().run(cand, _ctx(root), cfg)
    assert not result.passed
    assert result.detail["max_drawdown"] == pytest.approx(0.55)


# ---------------------------------------------------------- allow_negative + 负 sharpe


def test_allow_negative_flips_direction(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 40, 30
    rng = np.random.default_rng(3)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "flip")

    runner = _fake_runner_factory(
        summary={
            "sharpe": -1.8,
            "annualized_returns": -0.18,
            "max_drawdown": 0.10,
            "total_returns": -0.25,
        }
    )

    cfg_strict = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "min_annual_return": 0.05,
        "allow_negative": False,
        "output_root": str(tmp_path / "reports_no_neg"),
        "_runner_override": runner,
    }
    r1 = RqalphaBacktestCheck().run(cand, _ctx(root), cfg_strict)
    assert not r1.passed
    assert r1.detail["direction"] == 1

    runner2 = _fake_runner_factory(
        summary={
            "sharpe": -1.8,
            "annualized_returns": -0.18,
            "max_drawdown": 0.10,
            "total_returns": -0.25,
        }
    )
    cfg_allow = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "min_annual_return": 0.05,
        "allow_negative": True,
        "output_root": str(tmp_path / "reports_allow_neg"),
        "_runner_override": runner2,
    }
    r2 = RqalphaBacktestCheck().run(cand, _ctx(root), cfg_allow)
    assert r2.passed, r2.detail
    assert r2.detail["direction"] == -1
    # 原始 sharpe 仍然是 -1.8
    assert r2.detail["sharpe"] == pytest.approx(-1.8)


# ---------------------------------------------------------- runner 没写 report


def test_missing_report_raises(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 30, 20
    rng = np.random.default_rng(4)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "noreport")

    runner = _fake_runner_factory(summary={}, write_report=False)

    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }
    with pytest.raises(ValueError, match="report.json"):
        RqalphaBacktestCheck().run(cand, _ctx(root), cfg)


# ---------------------------------------------------------- report 缺核心字段


def test_missing_sharpe_in_report_raises(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 30, 20
    rng = np.random.default_rng(5)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "nosharpe")

    runner = _fake_runner_factory(
        summary={
            # 故意不给 sharpe
            "annualized_returns": 0.2,
            "max_drawdown": 0.1,
        }
    )
    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }
    with pytest.raises(ValueError, match="sharpe"):
        RqalphaBacktestCheck().run(cand, _ctx(root), cfg)


# ---------------------------------------------------------- 缺必填配置


def test_missing_config_raises(tmp_path: Path) -> None:
    root = _prepare_project_root(tmp_path)
    T, N = 30, 20
    rng = np.random.default_rng(6)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "miscfg")

    runner = _fake_runner_factory(
        summary={"sharpe": 1.5, "annualized_returns": 0.1, "max_drawdown": 0.1}
    )

    # 缺 min_sharpe
    with pytest.raises(ValueError, match="min_sharpe"):
        RqalphaBacktestCheck().run(
            cand,
            _ctx(root),
            {
                "max_drawdown": 0.30,
                "output_root": str(tmp_path / "reports"),
                "_runner_override": runner,
            },
        )
    # 缺 max_drawdown
    with pytest.raises(ValueError, match="max_drawdown"):
        RqalphaBacktestCheck().run(
            cand,
            _ctx(root),
            {
                "min_sharpe": 1.0,
                "output_root": str(tmp_path / "reports"),
                "_runner_override": runner,
            },
        )


# ---------------------------------------------------------- rqalpha 配置不存在


def test_missing_rqalpha_config_raises(tmp_path: Path) -> None:
    # 不创建 config/rqalpha_config.yaml
    root = tmp_path / "proj_empty"
    root.mkdir()
    T, N = 30, 20
    rng = np.random.default_rng(7)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "nocfg")

    runner = _fake_runner_factory(
        summary={"sharpe": 1.5, "annualized_returns": 0.1, "max_drawdown": 0.1}
    )
    cfg = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.30,
        "output_root": str(tmp_path / "reports"),
        "_runner_override": runner,
    }
    with pytest.raises(ValueError, match="rqalpha 配置不存在"):
        RqalphaBacktestCheck().run(cand, _ctx(root), cfg)
