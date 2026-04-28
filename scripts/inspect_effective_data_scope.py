"""Inspect the effective data scope used by Factor Lab/RD-Agent production factors.

Examples:
    python scripts/inspect_effective_data_scope.py
    python scripts/inspect_effective_data_scope.py --skip-qlib
    python scripts/inspect_effective_data_scope.py --pipeline config/pipeline.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _abs(path_like: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else base / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _date_bounds(index: pd.Index) -> tuple[str | None, str | None]:
    if not isinstance(index, pd.MultiIndex) or "datetime" not in index.names or len(index) == 0:
        return None, None
    dates = pd.to_datetime(index.get_level_values("datetime"))
    return str(dates.min().date()), str(dates.max().date())


def _instrument_count(index: pd.Index) -> int | None:
    if not isinstance(index, pd.MultiIndex) or "instrument" not in index.names:
        return None
    return int(pd.Index(index.get_level_values("instrument")).nunique())


def _active_manifest_records(manifest: dict[str, Any], version: int | None) -> list[dict[str, Any]]:
    records = [r for r in manifest.get("factors", []) if isinstance(r, dict) and r.get("status") == "active"]
    if version is not None:
        records = [r for r in records if int(r.get("parquet_version", -1)) == version]
    return records


def _resolve_current_version(factor_lab_cfg: dict[str, Any], manifest: dict[str, Any]) -> int | None:
    raw = (factor_lab_cfg.get("registry") or {}).get("current_parquet_version")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    versions = [
        int(r["parquet_version"])
        for r in manifest.get("factors", [])
        if isinstance(r, dict) and r.get("status") == "active" and r.get("parquet_version") is not None
    ]
    return max(versions) if versions else None


def _inspect_parquet(
    path: Path,
    columns: list[str],
    *,
    align_index: pd.MultiIndex | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": None,
        "cols": None,
        "date_min": None,
        "date_max": None,
        "n_instruments": None,
        "coverage_pct": {},
        "aligned_coverage_pct": {},
        "error": None,
    }
    if not path.exists():
        return info
    try:
        df = pd.read_parquet(path, columns=columns or None)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["rows"] = int(len(df))
    info["cols"] = int(df.shape[1])
    info["date_min"], info["date_max"] = _date_bounds(df.index)
    info["n_instruments"] = _instrument_count(df.index)
    if df.shape[1] > 0:
        info["coverage_pct"] = {
            str(k): round(float(v), 2)
            for k, v in df.notna().mean().mul(100.0).sort_values().items()
        }
    if align_index is not None and df.shape[1] > 0:
        aligned = df.reindex(align_index)
        info["aligned_coverage_pct"] = {
            str(k): round(float(v), 2)
            for k, v in aligned.notna().mean().mul(100.0).sort_values().items()
        }
    return info


def _inspect_qlib_scope(
    *,
    provider_uri: str,
    region: str,
    instruments: str,
    start: str,
    end: str,
    skip: bool,
) -> tuple[dict[str, Any], pd.MultiIndex | None]:
    info: dict[str, Any] = {
        "enabled": not skip,
        "provider_uri": provider_uri,
        "region": region,
        "instruments": instruments,
        "start_time": start,
        "end_time": end,
        "n_instruments": None,
        "feature_rows": None,
        "feature_date_min": None,
        "feature_date_max": None,
        "error": None,
    }
    if skip:
        return info, None
    try:
        import qlib  # noqa: WPS433
        from qlib.data import D  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"import qlib failed: {type(exc).__name__}: {exc}"
        return info, None

    try:
        qlib.init(provider_uri=provider_uri, region=region, expression_cache=None)
        inst_cfg = D.instruments(instruments)
        inst_list = D.list_instruments(inst_cfg, start_time=start, end_time=end, as_list=True)
        info["n_instruments"] = int(len(inst_list))
        if not inst_list:
            info["error"] = "D.list_instruments returned empty list"
            return info, None
        panel = D.features(inst_list, ["$close_qfq"], start_time=start, end_time=end, freq="day")
        info["feature_rows"] = int(len(panel))
        info["feature_date_min"], info["feature_date_max"] = _date_bounds(panel.index)
        return info, panel.index if isinstance(panel.index, pd.MultiIndex) else None
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info, None


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_path = _abs(args.pipeline)
    pipeline_cfg = _load_yaml(pipeline_path)
    data_path = _abs(pipeline_cfg.get("data_config", "config/data.yaml"))
    data_cfg = _load_yaml(data_path)
    factor_lab_path = _abs(args.factor_lab)
    factor_lab_cfg = _load_yaml(factor_lab_path)
    template_path = _abs(args.rdagent_template)
    template_cfg = _load_yaml(template_path)

    qlib_cfg = data_cfg.get("qlib") or {}
    data_section = data_cfg.get("data") or {}
    main_instruments = str(data_section.get("instruments", ""))
    main_start = str(data_section.get("start_time", ""))
    main_end = str(data_section.get("end_time", ""))

    rdagent_cfg = ((factor_lab_cfg.get("lab") or {}).get("rdagent") or {})
    registry_cfg = factor_lab_cfg.get("registry") or {}
    manifest_path = _abs(registry_cfg.get("data_dir", "factor_registry/data")) / "manifest.json"
    manifest = _load_json(manifest_path)
    version = args.parquet_version or _resolve_current_version(factor_lab_cfg, manifest)
    active_records = _active_manifest_records(manifest, version)
    active_cols = [str(r.get("parquet_column") or r.get("name") or r.get("factor_name")) for r in active_records]
    parquet_path = _abs(registry_cfg.get("parquet_dir", "factor_registry/parquet")) / f"factors_v{version}.parquet"

    qlib_info, align_index = _inspect_qlib_scope(
        provider_uri=str(qlib_cfg.get("provider_uri", "")),
        region=str(qlib_cfg.get("region", "cn")),
        instruments=main_instruments,
        start=main_start,
        end=main_end,
        skip=bool(args.skip_qlib),
    )
    parquet_info = _inspect_parquet(parquet_path, active_cols, align_index=align_index)

    return {
        "pipeline": {
            "path": str(pipeline_path),
            "data_config": str(data_path),
            "main_instruments": main_instruments,
            "main_start_time": main_start,
            "main_end_time": main_end,
        },
        "factor_lab": {
            "path": str(factor_lab_path),
            "rdagent_universe": rdagent_cfg.get("universe"),
            "rdagent_time_window": rdagent_cfg.get("time_window"),
            "benchmark": rdagent_cfg.get("benchmark"),
            "quality_gate": registry_cfg.get("quality_gate"),
        },
        "rdagent_template": {
            "path": str(template_path),
            "market": template_cfg.get("market"),
            "benchmark": template_cfg.get("benchmark"),
            "handler_start_time": (template_cfg.get("data_handler_config") or {}).get("start_time"),
            "handler_end_time": (template_cfg.get("data_handler_config") or {}).get("end_time"),
            "segments": (((template_cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments"),
        },
        "qlib_scope": qlib_info,
        "production_factors": {
            "manifest_path": str(manifest_path),
            "current_version": version,
            "active_count": len(active_records),
            "active_columns": active_cols,
            "manifest_records": [
                {
                    "factor_id": r.get("factor_id"),
                    "column": r.get("parquet_column") or r.get("name") or r.get("factor_name"),
                    "date_min": r.get("date_min"),
                    "date_max": r.get("date_max"),
                    "coverage_pct": r.get("coverage_pct"),
                    "ic": r.get("ic"),
                    "icir": r.get("icir"),
                }
                for r in active_records
            ],
            "parquet": parquet_info,
        },
    }


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print("Effective Data Scope")
    print("=" * 80)
    pipe = report["pipeline"]
    print(f"[Pipeline] {pipe['path']}")
    print(f"  data_config : {pipe['data_config']}")
    print(f"  instruments : {pipe['main_instruments']}")
    print(f"  time_window : {pipe['main_start_time']} -> {pipe['main_end_time']}")

    lab = report["factor_lab"]
    print(f"\n[Factor Lab] {lab['path']}")
    print(f"  universe    : {lab['rdagent_universe']}")
    print(f"  benchmark   : {lab['benchmark']}")
    print(f"  time_window : {lab['rdagent_time_window']}")
    print(f"  quality_gate: {lab['quality_gate']}")

    tpl = report["rdagent_template"]
    print(f"\n[RD-Agent Template] {tpl['path']}")
    print(f"  market      : {tpl['market']}")
    print(f"  benchmark   : {tpl['benchmark']}")
    print(f"  handler     : {tpl['handler_start_time']} -> {tpl['handler_end_time']}")
    print(f"  segments    : {tpl['segments']}")

    qlib_info = report["qlib_scope"]
    print("\n[Qlib Actual Scope]")
    print(f"  provider_uri: {qlib_info['provider_uri']}")
    print(f"  instruments : {qlib_info['instruments']}")
    print(f"  n_instruments: {qlib_info['n_instruments']}")
    print(f"  feature_rows : {qlib_info['feature_rows']}")
    print(f"  feature_dates: {qlib_info['feature_date_min']} -> {qlib_info['feature_date_max']}")
    if qlib_info.get("error"):
        print(f"  error       : {qlib_info['error']}")

    prod = report["production_factors"]
    pq = prod["parquet"]
    print("\n[Production Factors]")
    print(f"  manifest    : {prod['manifest_path']}")
    print(f"  version     : {prod['current_version']}")
    print(f"  active_count: {prod['active_count']}")
    print(f"  columns     : {prod['active_columns']}")
    print(f"  parquet     : {pq['path']}")
    print(f"  parquet_ok  : {pq['exists'] and not pq['error']}")
    print(f"  parquet_rows: {pq['rows']}")
    print(f"  parquet_cols: {pq['cols']}")
    print(f"  parquet_dt  : {pq['date_min']} -> {pq['date_max']}")
    print(f"  parquet_inst: {pq['n_instruments']}")
    if pq.get("error"):
        print(f"  error       : {pq['error']}")
    if pq.get("aligned_coverage_pct"):
        print("  aligned_coverage_pct:")
        for name, cov in pq["aligned_coverage_pct"].items():
            print(f"    - {name}: {cov}%")
    elif pq.get("coverage_pct"):
        print("  raw_coverage_pct:")
        for name, cov in pq["coverage_pct"].items():
            print(f"    - {name}: {cov}%")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect effective Factor Lab/RD-Agent data scope.")
    parser.add_argument("--pipeline", default="config/pipeline.yaml")
    parser.add_argument("--factor-lab", default="config/factor_lab.yaml")
    parser.add_argument("--rdagent-template", default="rdagent_overrides/factor_template/conf_combined_factors.yaml")
    parser.add_argument("--parquet-version", type=int, default=None)
    parser.add_argument("--skip-qlib", action="store_true", help="只读配置和 parquet，不初始化 qlib")
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于脚本消费")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
