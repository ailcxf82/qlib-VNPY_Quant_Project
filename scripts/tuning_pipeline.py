"""
一键执行：实验运行 -> 指标汇总 -> HTML 报告。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG_PATH = PROJECT_ROOT / "debug-f559e7.log"
DEBUG_SESSION_ID = "f559e7"


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键执行微调对比流水线")
    parser.add_argument("--spec", type=str, default="data/tuning/specs/sample_experiments.yaml")
    parser.add_argument("--runs-dir", type=str, default="data/tuning/runs")
    parser.add_argument("--summary-dir", type=str, default="data/tuning/summary")
    parser.add_argument("--python", type=str, default="", help="指定子进程使用的 Python 解释器")
    parser.add_argument("--only", type=str, choices=["lgb", "gru"], default=None)
    parser.add_argument("--max-experiments", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["manual", "smoke", "small", "full"],
        default="manual",
        help="按阶段预设运行参数: smoke/small/full",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    run_id = os.environ.get("DEBUG_RUN_ID", "pre-fix")
    print(">>>", " ".join(cmd))
    # region agent log
    _dbg(run_id, "H3", "scripts/tuning_pipeline.py:run_cmd", "subprocess_start", {"cmd": cmd, "cwd": str(PROJECT_ROOT)})
    # endregion
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    # region agent log
    _dbg(run_id, "H3", "scripts/tuning_pipeline.py:run_cmd", "subprocess_ok", {"cmd": cmd})
    # endregion


def main() -> None:
    args = parse_args()
    stage_only = args.only
    stage_max = args.max_experiments
    stage_dry_run = args.dry_run
    if args.stage == "smoke":
        if stage_max <= 0:
            stage_max = 2
        stage_dry_run = True
    elif args.stage == "small":
        if stage_max <= 0:
            stage_max = 4
    elif args.stage == "full":
        stage_max = 0

    py = ""
    if args.python:
        py = args.python
    elif os.environ.get("TUNING_PYTHON"):
        py = str(os.environ.get("TUNING_PYTHON"))
    else:
        # 优先使用已验证稳定的 Miniconda 解释器，避免系统 python 环境缺依赖导致失败
        conda_py = Path.home() / "miniconda3" / "python.exe"
        py = str(conda_py) if conda_py.exists() else sys.executable
    run_id = os.environ.get("DEBUG_RUN_ID", "pre-fix")
    # region agent log
    _dbg(
        run_id,
        "H4",
        "scripts/tuning_pipeline.py:main",
        "pipeline_start",
        {
            "python": py,
            "spec": args.spec,
            "only": stage_only,
            "max_experiments": stage_max,
            "dry_run": stage_dry_run,
            "stage": args.stage,
        },
    )
    # endregion

    cmd1 = [
        py,
        "scripts/run_tuning_experiments.py",
        "--spec",
        args.spec,
        "--runs-dir",
        args.runs_dir,
    ]
    if stage_only:
        cmd1 += ["--only", stage_only]
    if stage_max > 0:
        cmd1 += ["--max-experiments", str(stage_max)]
    if stage_dry_run:
        cmd1 += ["--dry-run"]
    run_cmd(cmd1)

    cmd2 = [
        py,
        "scripts/collect_tuning_results.py",
        "--runs-dir",
        args.runs_dir,
        "--out-dir",
        args.summary_dir,
    ]
    run_cmd(cmd2)

    cmd3 = [
        py,
        "scripts/render_tuning_report.py",
        "--summary-csv",
        f"{args.summary_dir}/experiment_summary.csv",
        "--window-csv",
        f"{args.summary_dir}/window_level_metrics.csv",
        "--out",
        f"{args.summary_dir}/tuning_report.html",
    ]
    run_cmd(cmd3)

    print(f"完成。打开报告: {PROJECT_ROOT / args.summary_dir / 'tuning_report.html'}")


if __name__ == "__main__":
    main()
