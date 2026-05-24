"""Tests for PortAna-safe backtest end index logic."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_data_window import (  # noqa: E402
    calc_window,
    safe_backtest_end_index,
)


def test_safe_backtest_end_index_requires_buffer() -> None:
    assert safe_backtest_end_index(10, 11) == 10
    with pytest.raises(ValueError):
        safe_backtest_end_index(10, 10)


@pytest.mark.skipif(
    not Path("/mnt/d/qlib_data/qlib_data").exists()
    and not Path("D:/qlib_data/qlib_data").exists(),
    reason="qlib data not available",
)
def test_calc_window_backtest_before_test_end() -> None:
    w = calc_window(date(2026, 5, 23))
    assert w.test_end <= date(2026, 5, 23)
    assert w.backtest_end < w.test_end
