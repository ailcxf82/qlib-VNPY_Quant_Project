"""E.2 CLI 冒烟：scripts.lab.build_feedback_bundle main()."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.lab import build_feedback_bundle as cli


def test_cli_end_to_end_writes_bundle(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"
    feedback = tmp_path / "feedback"

    reports.mkdir()
    # 一个最小合法的 cycle 报告（mtime 无所谓；列表只有一份）
    (reports / "lab_cycle_cli-abc.json").write_text(
        json.dumps({"cycle_id": "cli-abc", "candidates": []}),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--reports-dir",
            str(reports),
            "--cert-dir",
            str(certs),
            "--registry-data-dir",
            str(reg),
            "--feedback-dir",
            str(feedback),
            "--max-cycles",
            "4",
            "--log-level",
            "WARNING",
            "--notes",
            "cli smoke",
        ]
    )
    assert rc == 0
    assert (feedback / "latest.json").exists()
    assert (feedback / "latest.md").exists()
    payload = json.loads((feedback / "latest.json").read_text(encoding="utf-8"))
    assert payload["cycles_included"] == ["cli-abc"]
    assert payload["notes"] == "cli smoke"


def test_cli_rejects_bad_max_cycles(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "--reports-dir",
            str(tmp_path),
            "--cert-dir",
            str(tmp_path),
            "--registry-data-dir",
            str(tmp_path),
            "--feedback-dir",
            str(tmp_path / "feedback"),
            "--max-cycles",
            "0",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 1  # 走 build_feedback_bundle 内部 ValueError 分支
