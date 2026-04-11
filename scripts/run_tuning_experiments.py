"""
批量运行微调实验（不修改原始配置文件）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG_PATH = PROJECT_ROOT / "debug-f559e7.log"
DEBUG_SESSION_ID = "f559e7"


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
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


def load_yaml(path: Path) -> Dict[str, Any]:
    run_id = os.environ.get("DEBUG_RUN_ID", "pre-fix")
    # region agent log
    _dbg(run_id, "H1", "scripts/run_tuning_experiments.py:load_yaml", "load_yaml_enter", {"path": str(path)})
    # endregion
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        # region agent log
        _dbg(
            run_id,
            "H1",
            "scripts/run_tuning_experiments.py:load_yaml",
            "yaml_import_failed",
            {"python_executable": sys.executable, "error": repr(e)},
        )
        # endregion
        raise
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_by_dotted_path(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = data
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行微调实验")
    parser.add_argument(
        "--spec",
        type=str,
        default="data/tuning/specs/sample_experiments.yaml",
        help="实验规格文件",
    )
    parser.add_argument(
        "--base-pipeline",
        type=str,
        default="config/pipeline.yaml",
        help="基础 pipeline 配置",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="data/tuning/runs",
        help="实验输出目录",
    )
    parser.add_argument("--only", type=str, choices=["lgb", "gru"], default=None, help="只跑某一类模型")
    parser.add_argument("--max-experiments", type=int, default=0, help="最多执行 N 个实验，0 表示全部")
    parser.add_argument("--dry-run", action="store_true", help="仅生成配置，不执行训练")
    parser.add_argument("--python", type=str, default=sys.executable, help="训练命令使用的 python 可执行文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    run_id = os.environ.get("DEBUG_RUN_ID", "pre-fix")
    # region agent log
    _dbg(
        run_id,
        "H2",
        "scripts/run_tuning_experiments.py:main",
        "runner_start",
        {"python_executable": sys.executable, "spec": args.spec, "runs_dir": args.runs_dir},
    )
    # endregion

    spec_path = (PROJECT_ROOT / args.spec).resolve()
    base_pipeline_path = (PROJECT_ROOT / args.base_pipeline).resolve()
    runs_dir = (PROJECT_ROOT / args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    spec = load_yaml(spec_path)
    experiments: List[Dict[str, Any]] = list(spec.get("experiments", []))
    defaults: Dict[str, Any] = spec.get("defaults", {}) or {}

    if args.only:
        experiments = [e for e in experiments if str(e.get("model_type", "")).lower() == args.only]
    if args.max_experiments and args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]

    if not experiments:
        raise ValueError("未找到可执行实验（请检查 spec / --only / --max-experiments）。")

    base_pipeline = load_yaml(base_pipeline_path)
    base_lgb_path = (PROJECT_ROOT / base_pipeline["lightgbm_config"]).resolve()
    base_gru_path = (PROJECT_ROOT / base_pipeline["gru_config"]).resolve()
    base_data_path = (PROJECT_ROOT / base_pipeline["data_config"]).resolve()

    for idx, exp in enumerate(experiments, start=1):
        exp_id = str(exp["experiment_id"]).strip()
        model_type = str(exp["model_type"]).strip().lower()
        overrides: Dict[str, Any] = exp.get("param_overrides", {}) or {}
        data_overrides: Dict[str, Any] = exp.get("data_overrides", {}) or {}
        pipeline_overrides: Dict[str, Any] = exp.get("pipeline_overrides", {}) or {}
        notes = exp.get("notes", defaults.get("notes", ""))

        run_dir = runs_dir / exp_id
        configs_dir = run_dir / "configs"
        logs_dir = run_dir / "logs"
        models_dir = run_dir / "models"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        pipeline_cfg = deepcopy(base_pipeline)
        lgb_cfg = load_yaml(base_lgb_path)
        gru_cfg = load_yaml(base_gru_path)
        data_cfg = load_yaml(base_data_path)

        if model_type == "lgb":
            target_cfg = lgb_cfg
            target_path = configs_dir / "model_lgb.yaml"
        elif model_type == "gru":
            target_cfg = gru_cfg
            target_path = configs_dir / "model_gru.yaml"
        else:
            raise ValueError(f"不支持 model_type={model_type}, 实验={exp_id}")

        for dotted_key, value in overrides.items():
            set_by_dotted_path(target_cfg, str(dotted_key), value)
        for dotted_key, value in data_overrides.items():
            set_by_dotted_path(data_cfg, str(dotted_key), value)
        for dotted_key, value in pipeline_overrides.items():
            set_by_dotted_path(pipeline_cfg, str(dotted_key), value)

        data_path = configs_dir / "data.yaml"
        pipeline_path = configs_dir / "pipeline.yaml"
        dump_yaml(data_path, data_cfg)
        dump_yaml(target_path, target_cfg)

        # 另一侧模型配置也复制一份，保证实验可复现
        if model_type == "lgb":
            dump_yaml(configs_dir / "model_gru.yaml", gru_cfg)
        else:
            dump_yaml(configs_dir / "model_lgb.yaml", lgb_cfg)

        pipeline_cfg["data_config"] = str(data_path)
        pipeline_cfg["lightgbm_config"] = str(configs_dir / "model_lgb.yaml")
        pipeline_cfg["gru_config"] = str(configs_dir / "model_gru.yaml")
        pipeline_cfg["paths"]["model_dir"] = str(models_dir)
        pipeline_cfg["paths"]["log_dir"] = str(logs_dir)
        pipeline_cfg["paths"]["prediction_dir"] = str(run_dir / "predictions")
        pipeline_cfg["paths"]["backtest_dir"] = str(run_dir / "backtest")
        pipeline_cfg["paths"]["oof_dir"] = str(run_dir / "oof")
        pipeline_cfg["paths"]["meta_dir"] = str(run_dir / "meta")
        dump_yaml(pipeline_path, pipeline_cfg)

        result = {
            "experiment_id": exp_id,
            "model_type": model_type,
            "notes": notes,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_pipeline_config": str(base_pipeline_path),
            "base_pipeline_hash": file_sha1(base_pipeline_path),
            "param_overrides": overrides,
            "data_overrides": data_overrides,
            "pipeline_overrides": pipeline_overrides,
            "data_scope": {
                "instruments": (data_cfg.get("data", {}) or {}).get("instruments"),
                "start_time": (data_cfg.get("data", {}) or {}).get("start_time"),
                "end_time": (data_cfg.get("data", {}) or {}).get("end_time"),
                "label": (data_cfg.get("data", {}) or {}).get("label"),
                "active_feature_sets": (data_cfg.get("data", {}) or {}).get("active_feature_sets"),
            },
            "run_dir": str(run_dir),
        }

        print(f"[{idx}/{len(experiments)}] {exp_id} ({model_type})")
        if args.dry_run:
            result["status"] = "dry_run"
            result["trained"] = False
            result["started_at"] = datetime.now().isoformat(timespec="seconds")
            result["finished_at"] = result["started_at"]
            result["has_fresh_metrics"] = False
            result["skip_reason"] = "dry_run_no_training"
            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            continue

        cmd = [args.python, "run_train.py", "--config", str(pipeline_path)]
        started_at = datetime.now()
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = round(time.time() - started, 2)
        with open(run_dir / "run.log", "w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
            if proc.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)

        result["elapsed_seconds"] = elapsed
        result["returncode"] = proc.returncode
        result["status"] = "ok" if proc.returncode == 0 else "failed"
        result["trained"] = bool(proc.returncode == 0)
        result["started_at"] = started_at.isoformat(timespec="seconds")
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result["metrics_csv"] = str(logs_dir / "training_metrics.csv")
        metrics_path = logs_dir / "training_metrics.csv"
        if proc.returncode == 0 and metrics_path.exists():
            metrics_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime)
            result["metrics_mtime"] = metrics_mtime.isoformat(timespec="seconds")
            result["has_fresh_metrics"] = bool(metrics_mtime >= started_at)
        else:
            result["has_fresh_metrics"] = False
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"完成，实验输出目录: {runs_dir}")


if __name__ == "__main__":
    main()
