"""Tests for workspace PortAna date patch."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_lab.adapters.portana_dates import patch_workspace_portana_dates  # noqa: E402

_BACKTEST_END = re.compile(r"^        end_time:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


@pytest.mark.skipif(
    not Path("/mnt/d/qlib_data/qlib_data").exists()
    and not Path("D:/qlib_data/qlib_data").exists(),
    reason="qlib data not available",
)
def test_patch_workspace_reverts_unsafe_backtest_end(tmp_path: Path) -> None:
    src = (
        ROOT
        / "rdagent_overrides"
        / "factor_template"
        / "conf_combined_factors.yaml"
    )
    conf = tmp_path / "conf_combined_factors.yaml"
    text = src.read_text(encoding="utf-8").replace(
        "        end_time: 2026-05-21", "        end_time: 2026-05-22", 1
    )
    conf.write_text(text, encoding="utf-8")
    assert _BACKTEST_END.search(text).group(1) == "2026-05-22"
    assert patch_workspace_portana_dates(tmp_path) is True
    assert _BACKTEST_END.search(conf.read_text(encoding="utf-8")).group(1) == "2026-05-21"
