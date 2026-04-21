"""
P3a smoke test: validate RD-Agent reward-loop changes WITHOUT calling any LLM.

Checks:
  1. All five rdagent_overrides/*.yaml files parse and contain the new
     time window / benchmark settings.
  2. rdagent_overrides/factor_template/read_exp_res.py + model_template version
     are syntactically valid Python.
  3. Importing patch_qlib_conda + apply_qlib_conda_env_patch successfully extends
     rdagent.scenarios.qlib.developer.feedback.IMPORTANT_METRICS with the new
     turnover / composite_score / information_ratio keys.
  4. project_quant_proposal._PROJECT_FACTOR_RAG contains the orthogonality
     constraint and the failure-family blacklist.

Run:  python scripts/smoke_p3a_reward.py
Exit code 0 == all checks passed.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_BENCHMARK = "SH000300"
REQUIRED_TRAIN = ["2022-01-01", "2024-06-30"]
REQUIRED_VALID = ["2024-07-01", "2024-12-31"]
REQUIRED_TEST = ["2025-01-01", "2025-10-31"]


YAMLS_STRICT = [
    "rdagent_overrides/factor_template/conf_baseline.yaml",
    "rdagent_overrides/factor_template/conf_combined_factors.yaml",
]
YAMLS_JINJA = [
    "rdagent_overrides/factor_template/conf_combined_factors_sota_model.yaml",
    "rdagent_overrides/model_template/conf_baseline_factors_model.yaml",
    "rdagent_overrides/model_template/conf_sota_factors_model.yaml",
]

PY_FILES = [
    "rdagent_overrides/factor_template/read_exp_res.py",
    "rdagent_overrides/model_template/read_exp_res.py",
]


def _check_yaml(path: Path) -> list[str]:
    errors: list[str] = []
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        errors.append("YAML root is not a mapping")
        return errors

    bench = cfg.get("benchmark")
    if bench != REQUIRED_BENCHMARK:
        errors.append(f"benchmark = {bench!r} (expected {REQUIRED_BENCHMARK!r})")

    handler = cfg.get("data_handler_config", {}) or {}
    if str(handler.get("start_time")) != "2022-01-01":
        errors.append(f"data_handler_config.start_time = {handler.get('start_time')!r}")
    if str(handler.get("end_time")) != "2025-10-31":
        errors.append(f"data_handler_config.end_time = {handler.get('end_time')!r}")

    port_cfg = cfg.get("port_analysis_config", {}) or {}
    bt = (port_cfg.get("backtest") or {})
    if str(bt.get("start_time")) != "2025-01-01":
        errors.append(f"backtest.start_time = {bt.get('start_time')!r}")
    if str(bt.get("end_time")) != "2025-10-31":
        errors.append(f"backtest.end_time = {bt.get('end_time')!r}")

    task = (cfg.get("task") or {})
    dataset = (task.get("dataset") or {}).get("kwargs") or {}
    segments = dataset.get("segments") or {}
    train = [str(x) for x in (segments.get("train") or [])]
    valid = [str(x) for x in (segments.get("valid") or [])]
    test = [str(x) for x in (segments.get("test") or [])]
    if train != REQUIRED_TRAIN:
        errors.append(f"segments.train = {train} (expected {REQUIRED_TRAIN})")
    if valid != REQUIRED_VALID:
        errors.append(f"segments.valid = {valid} (expected {REQUIRED_VALID})")
    if test != REQUIRED_TEST:
        errors.append(f"segments.test = {test} (expected {REQUIRED_TEST})")

    return errors


def _check_python(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    must_contain = [
        "annualized_turnover",
        "composite_score",
        "_latest_recorder",
        "qlib_res.csv",
        "ret.pkl",
    ]
    missing = [tok for tok in must_contain if tok not in src]
    return [f"missing token: {tok}" for tok in missing]


def _check_rag() -> list[str]:
    from factor_lab.adapters.quant_proposal import _PROJECT_FACTOR_RAG

    rag = _PROJECT_FACTOR_RAG
    must_contain = [
        "composite_score",
        "annualized_turnover",
        "Spearman",
        "0.50",
        "lgb_short_cycle",
        "Discouraged",
        "Encouraged",
    ]
    return [f"RAG missing token: {tok}" for tok in must_contain if tok not in rag]


def _check_monkey_patch() -> list[str]:
    """Confirm patch_qlib_conda extends IMPORTANT_METRICS in-place."""
    try:
        import rdagent.scenarios.qlib.developer.feedback as feedback_mod
    except Exception as exc:
        return [f"cannot import rdagent feedback (need rdagent installed): {exc}"]
    from factor_lab.adapters.patch_qlib_conda import _patch_feedback_important_metrics

    _patch_feedback_important_metrics()
    extra = [
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.annualized_turnover",
        "1day.composite_score",
    ]
    missing = [k for k in extra if k not in feedback_mod.IMPORTANT_METRICS]
    return [f"IMPORTANT_METRICS missing key: {k}" for k in missing]


def main() -> int:
    failures: list[str] = []

    print("[1a/4] checking strict-YAML configs ...")
    for rel in YAMLS_STRICT:
        p = ROOT / rel
        if not p.exists():
            failures.append(f"{rel}: file not found")
            continue
        errs = _check_yaml(p)
        if errs:
            for e in errs:
                failures.append(f"{rel}: {e}")
        else:
            print(f"  ok  {rel}")

    print("[1b/4] checking Jinja-templated YAML configs (regex) ...")
    expected_lines = [
        f"benchmark: &benchmark {REQUIRED_BENCHMARK}",
        "start_time: 2022-01-01",
        "end_time: 2025-10-31",
        "start_time: 2025-01-01",
        "fit_start_time: 2022-01-01",
        "fit_end_time: 2024-06-30",
        "train: [2022-01-01, 2024-06-30]",
        "valid: [2024-07-01, 2024-12-31]",
        "test: [2025-01-01, 2025-10-31]",
    ]
    for rel in YAMLS_JINJA:
        p = ROOT / rel
        if not p.exists():
            failures.append(f"{rel}: file not found")
            continue
        text = p.read_text(encoding="utf-8")
        local_fail = False
        for needle in expected_lines:
            if needle not in text:
                failures.append(f"{rel}: missing line {needle!r}")
                local_fail = True
        if not local_fail:
            print(f"  ok  {rel}")

    print("[2/4] checking read_exp_res.py files ...")
    for rel in PY_FILES:
        p = ROOT / rel
        if not p.exists():
            failures.append(f"{rel}: file not found")
            continue
        errs = _check_python(p)
        if errs:
            for e in errs:
                failures.append(f"{rel}: {e}")
        else:
            print(f"  ok  {rel}")

    print("[3/4] checking project_quant_proposal RAG ...")
    errs = _check_rag()
    if errs:
        failures.extend(errs)
    else:
        print("  ok  project_quant_proposal._PROJECT_FACTOR_RAG")

    print("[4/4] checking patch_qlib_conda IMPORTANT_METRICS extension ...")
    errs = _check_monkey_patch()
    if errs:
        failures.extend(errs)
    else:
        print("  ok  IMPORTANT_METRICS extended")

    print()
    if failures:
        print(f"FAIL  {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS  all P3a smoke checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
