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


def _parse_date(v: Any) -> Any:
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d")
    except Exception:
        return None


def preflight_check(
    pipeline_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """训练前轻量预检（不初始化 qlib），拦截可预测的配置错误。

    返回结构：{"status": "pass|fail", "checks": [ {"name", "ok", "detail"} ... ]}
    """
    checks: List[Dict[str, Any]] = []

    qlib_cfg = (pipeline_cfg.get("qlib") or data_cfg.get("qlib") or {}) if isinstance(pipeline_cfg, dict) else {}
    provider_uri = qlib_cfg.get("provider_uri") if isinstance(qlib_cfg, dict) else None
    if not provider_uri:
        checks.append({"name": "qlib_provider_uri", "ok": False, "detail": "provider_uri 未设置"})
    else:
        ok = Path(str(provider_uri)).exists()
        checks.append({
            "name": "qlib_provider_uri",
            "ok": bool(ok),
            "detail": f"provider_uri={provider_uri} exists={ok}",
        })

    data_section = data_cfg.get("data", {}) if isinstance(data_cfg, dict) else {}
    start_raw = data_section.get("start_time")
    end_raw = data_section.get("end_time")
    start_dt = _parse_date(start_raw)
    end_dt = _parse_date(end_raw)
    if start_dt is None or end_dt is None:
        checks.append({
            "name": "date_range_format",
            "ok": False,
            "detail": f"start_time={start_raw}, end_time={end_raw}",
        })
        span_days = 0
    else:
        ok = start_dt <= end_dt
        span_days = (end_dt - start_dt).days
        checks.append({
            "name": "date_range_order",
            "ok": bool(ok),
            "detail": f"start_time={start_raw}, end_time={end_raw}, span_days={span_days}",
        })

    feature_sets = data_section.get("feature_sets") or {}
    active = data_section.get("active_feature_sets") or []
    missing = [k for k in active if k not in feature_sets]
    checks.append({
        "name": "active_feature_sets_resolved",
        "ok": (len(missing) == 0 and len(active) > 0),
        "detail": f"active={list(active)}, missing={missing}",
    })

    # 注：trainer 以 start_time/end_time 作为"评估/验证窗口"，训练会向前回溯
    # model_train_days（从 Qlib calendar 拿历史数据）。因此 span_days 与
    # (model_train_days + valid_days + test_days) 的比较不是硬约束，仅作信息
    # 提示，避免误挡本来能跑的实验。
    rolling = pipeline_cfg.get("rolling", {}) if isinstance(pipeline_cfg, dict) else {}
    train_days = int(rolling.get("train_days", 0) or 0)
    valid_days = int(rolling.get("valid_days", 0) or 0)
    test_days = int(rolling.get("test_days", 0) or 0)
    model_train_days = rolling.get("model_train_days") or {}
    max_model_train = max(
        [int(v) for v in model_train_days.values() if v is not None] + [train_days]
    ) if model_train_days else train_days
    required_days = max_model_train + valid_days + test_days
    if span_days > 0 and required_days > 0:
        checks.append({
            "name": "time_window_info",
            "ok": True,  # 信息项，始终通过
            "detail": (
                f"span_days={span_days}, rolling_required_hint>={required_days} "
                f"(max_model_train={max_model_train}, valid={valid_days}, test={test_days}); "
                "note: trainer 会向前回溯 model_train_days，不要求 span 覆盖整段训练窗口"
            ),
        })

    label_expr = data_section.get("label")
    checks.append({
        "name": "label_defined",
        "ok": bool(label_expr),
        "detail": f"label={label_expr}",
    })

    all_ok = all(bool(c.get("ok")) for c in checks)
    return {"status": "pass" if all_ok else "fail", "checks": checks}


def tail_text(text: str, n_lines: int = 40, max_chars: int = 4000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-n_lines:]) if len(lines) > n_lines else "\n".join(lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


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

        pf = preflight_check(pipeline_cfg, data_cfg)

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
            "preflight": pf,
            "error_tail": "",
            "run_dir": str(run_dir),
        }

        print(f"[{idx}/{len(experiments)}] {exp_id} ({model_type})")

        if pf["status"] == "fail":
            failed_checks = [c for c in pf["checks"] if not c.get("ok")]
            detail = "; ".join(f"{c['name']}: {c['detail']}" for c in failed_checks)
            print(f"  [preflight] FAIL: {detail}")
            result["status"] = "preflight_failed"
            result["trained"] = False
            result["started_at"] = datetime.now().isoformat(timespec="seconds")
            result["finished_at"] = result["started_at"]
            result["has_fresh_metrics"] = False
            result["skip_reason"] = "preflight_failed"
            result["error_tail"] = detail[:4000]
            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            continue

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

        cmd = [args.python, "-u", "run_train.py", "--config", str(pipeline_path)]
        started_at = datetime.now()
        started = time.time()

        # 先写一份 "running" 状态的 result.json，便于 Ctrl+C 后仍能看到已启动
        result["status"] = "running"
        result["trained"] = False
        result["started_at"] = started_at.isoformat(timespec="seconds")
        result["finished_at"] = None
        result["has_fresh_metrics"] = False
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 实时透传 stdout+stderr：一路 print 到父进程终端（加 exp_id 前缀），
        # 一路落盘到 run_dir/run.log；父进程用 `python -u` 保证子进程无缓冲。
        # 这样用户可以看到训练进度；中断时也能通过 run.log 定位卡点。
        run_log_path = run_dir / "run.log"
        prefix = f"[{exp_id}] "
        stdout_lines: List[str] = []
        stderr_tail_buffer: List[str] = []
        popen_kwargs = dict(
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并到 stdout，保证时序
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        # region agent log（仅启动摘要，无敏感信息）
        _dbg(run_id, "H3", "scripts/run_tuning_experiments.py:main",
             "subprocess_start_stream", {"cmd": cmd, "run_dir": str(run_dir)})
        # endregion
        returncode = -1
        interrupted = False
        try:
            with open(run_log_path, "w", encoding="utf-8") as log_f, \
                 subprocess.Popen(cmd, **popen_kwargs) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    stdout_lines.append(line)
                    # stderr_tail_buffer 保留最后 80 行，用于失败时 error_tail
                    stderr_tail_buffer.append(line)
                    if len(stderr_tail_buffer) > 80:
                        stderr_tail_buffer.pop(0)
                    # 透传到父进程终端 + 落盘
                    sys.stdout.write(prefix + line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()
                returncode = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            try:
                proc.terminate()
            except Exception:
                pass
            print(f"{prefix}[interrupted by user]")

        elapsed = round(time.time() - started, 2)
        result["elapsed_seconds"] = elapsed
        result["returncode"] = returncode
        if interrupted:
            result["status"] = "interrupted"
            result["trained"] = False
        else:
            result["status"] = "ok" if returncode == 0 else "failed"
            result["trained"] = bool(returncode == 0)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result["metrics_csv"] = str(logs_dir / "training_metrics.csv")
        metrics_path = logs_dir / "training_metrics.csv"
        if returncode == 0 and metrics_path.exists():
            metrics_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime)
            result["metrics_mtime"] = metrics_mtime.isoformat(timespec="seconds")
            result["has_fresh_metrics"] = bool(metrics_mtime >= started_at)
        else:
            result["has_fresh_metrics"] = False

        if returncode != 0 or interrupted:
            result["error_tail"] = tail_text("".join(stderr_tail_buffer), n_lines=40)
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if interrupted:
            print(f"{prefix}[tuning runner] KeyboardInterrupt; 终止后续实验。")
            break

    print(f"完成，实验输出目录: {runs_dir}")


if __name__ == "__main__":
    main()
