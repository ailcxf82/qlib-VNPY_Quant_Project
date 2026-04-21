"""阶段 C oracle：用 ``default.yaml`` 复测当前 L3 active 的 5 个 legacy 因子。

**目的**：``manual_grandfathered.yaml`` 只查 coverage、阈值很宽，所以 5 个 legacy
因子全 PASS 不能说明它们在严格 profile 下也活得下来。本脚本走 4-check 的 default
profile 重新打分，**不写 registry**，只输出报告，供决策：
  * 全 PASS → 直接进 D 阶段
  * 部分 HOLD/FAIL → 调阈值 / 退役低质 legacy / 调整 default profile

**数据来源**：
  * factors values 直接从 ``factor_registry/parquet/factors_v<N>.parquet`` 抽列
    （migrate 时已经把每列单独写到了 workspace/candidates/<factor_id>/values.parquet，
    这里复用该 workspace，避免任何"再迁移"副作用）
  * label/参照集走 profile.data_sources（已指向 oos_labels_all.parquet + factors_v1.parquet）

CLI：
    python -m scripts.promote.oracle_revalidate_legacy \
        --profile factor_validation/profiles/default.yaml \
        [--write-report factor_validation/reports/legacy_oracle_default.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_lab.exporters.schema import CandidateFactorPackage  # noqa: E402
from factor_validation.orchestrator import validate_candidate  # noqa: E402
from factor_validation.schema import CertifiedFactorRecord, Decision  # noqa: E402

logger = logging.getLogger("oracle_revalidate_legacy")

DEFAULT_PROFILE = (
    _PROJECT_ROOT / "factor_validation" / "profiles" / "default.yaml"
)
DEFAULT_REGISTRY_DATA_DIR = _PROJECT_ROOT / "factor_registry" / "data"
WORKSPACE_ROOT = _PROJECT_ROOT / "factor_lab" / "workspace" / "candidates"


def _load_active_entries(registry_data_dir: Path) -> list[dict]:
    manifest_path = registry_data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    factors = raw.get("factors", [])
    return [e for e in factors if e.get("status") == "active"]


def _candidate_from_workspace(entry: dict, universe: str) -> CandidateFactorPackage:
    factor_id = entry["factor_id"]
    name = entry["name"]
    ws = WORKSPACE_ROOT / factor_id
    code_path = ws / "factor.py"
    values_path = ws / "values.parquet"
    if not values_path.exists():
        raise FileNotFoundError(
            f"workspace 缺 values.parquet：{values_path}（请先跑 migrate_legacy_factors）"
        )
    if not code_path.exists():
        raise FileNotFoundError(f"workspace 缺 factor.py：{code_path}")

    import pandas as pd  # local import — 减少 cold-start 噪音

    df = pd.read_parquet(values_path)
    dt_index = df.index.get_level_values("datetime")
    return CandidateFactorPackage(
        factor_id=factor_id,
        name=name,
        source="manual",
        hypothesis="Oracle re-validation of grandfathered legacy factor under stricter profile.",
        formulation=f"(legacy oracle re-validation) column={name}",
        code_path=code_path,
        values_path=values_path,
        universe=universe,
        date_range=(
            pd.Timestamp(dt_index.min()).date(),
            pd.Timestamp(dt_index.max()).date(),
        ),
        lab_metrics={"non_null_ratio": float(df[name].notna().mean())},
        parent_loop=None,
        created_at=datetime.now(timezone.utc),
        lab_run_id=f"oracle-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    )


def revalidate_all(
    *,
    profile_path: Path,
    registry_data_dir: Path = DEFAULT_REGISTRY_DATA_DIR,
    universe: str = "csi300",
) -> list[CertifiedFactorRecord]:
    active_entries = _load_active_entries(registry_data_dir)
    if not active_entries:
        raise RuntimeError(
            "manifest 没有 active 因子；请先跑 migrate_legacy_factors"
        )
    logger.info(
        "oracle: 复测 %d 个 active 因子 against profile=%s",
        len(active_entries),
        profile_path.name,
    )

    certs: list[CertifiedFactorRecord] = []
    for entry in active_entries:
        cand = _candidate_from_workspace(entry, universe=universe)
        cand.validate_artifacts(check_parquet_schema=True)
        cert = validate_candidate(
            cand,
            profile_path,
            project_root=_PROJECT_ROOT,
            notes=f"oracle revalidate against {profile_path.name}",
        )
        certs.append(cert)
        logger.info(
            "  - %-26s decision=%-4s overall=%.4f",
            cand.name,
            cert.decision.value,
            cert.overall_score,
        )
    return certs


def _format_report_md(
    certs: list[CertifiedFactorRecord], profile_path: Path
) -> str:
    profile_name = profile_path.stem
    lines: list[str] = []
    lines.append(f"# L2 Oracle 复测报告 — profile=`{profile_name}`")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    try:
        rel = profile_path.resolve().relative_to(_PROJECT_ROOT)
    except ValueError:
        rel = profile_path  # 不在 project_root 之下时按原样写
    lines.append(f"- profile 文件：`{rel}`")
    lines.append(f"- 复测因子数：{len(certs)}")
    n_pass = sum(1 for c in certs if c.decision == Decision.PASS)
    n_hold = sum(1 for c in certs if c.decision == Decision.HOLD)
    n_fail = sum(1 for c in certs if c.decision == Decision.FAIL)
    lines.append(f"- 决议汇总：PASS={n_pass}  HOLD={n_hold}  FAIL={n_fail}")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append("| factor_id | name | decision | overall_score | "
                 "coverage | ic | orthogonality | turnover |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def _check(c: CertifiedFactorRecord, name: str) -> str:
        for r in c.check_results:
            if r.name == name:
                tick = "[OK]" if r.passed else "[X]"
                s = "n/a" if r.score is None else f"{r.score:.3f}"
                return f"{tick} {s}"
        return "-"

    for c in certs:
        lines.append(
            f"| `{c.factor_id}` | {c.candidate.name} | "
            f"**{c.decision.value}** | {c.overall_score:.4f} | "
            f"{_check(c, 'coverage')} | {_check(c, 'ic')} | "
            f"{_check(c, 'orthogonality')} | {_check(c, 'turnover')} |"
        )
    lines.append("")

    lines.append("## 逐因子详情")
    for c in certs:
        lines.append("")
        lines.append(f"### {c.candidate.name}  ({c.decision.value})")
        lines.append(
            f"- factor_id: `{c.factor_id}`"
        )
        lines.append(f"- overall_score: **{c.overall_score:.4f}**")
        lines.append("")
        for r in c.check_results:
            tick = "[OK] PASS" if r.passed else "[X] FAIL"
            score_repr = "n/a" if r.score is None else f"{r.score:.4f}"
            thr_repr = (
                "—" if r.threshold is None else f"{r.threshold:.4f}"
            )
            lines.append(
                f"- **{r.name}** {tick}  score={score_repr} threshold={thr_repr}"
            )
            if r.detail:
                detail_str = ", ".join(
                    f"{k}={v}"
                    for k, v in r.detail.items()
                    if not isinstance(v, dict)
                )
                lines.append(f"    - {detail_str}")
                # per_ref_corr (orthogonality) 单独展开 top-3
                per_ref = r.detail.get("per_ref_corr")
                if isinstance(per_ref, dict) and per_ref:
                    top = list(per_ref.items())[:3]
                    pairs = ", ".join(f"{k}={v:+.3f}" for k, v in top)
                    lines.append(f"    - per_ref_corr 前三: {pairs}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle re-validation of legacy active factors against a stricter profile."
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="ValidationProfile YAML（默认 default.yaml）",
    )
    p.add_argument(
        "--registry-data-dir",
        type=Path,
        default=DEFAULT_REGISTRY_DATA_DIR,
        help="manifest.json 所在目录",
    )
    p.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="若提供，把 markdown 报告写到此路径；否则只 stdout",
    )
    p.add_argument("--universe", default="csi300")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    certs = revalidate_all(
        profile_path=args.profile,
        registry_data_dir=args.registry_data_dir,
        universe=args.universe,
    )
    md = _format_report_md(certs, args.profile)
    # 先落盘 — stdout 在 Windows GBK 控制台可能编码失败
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(md, encoding="utf-8")
        logger.info("报告写入 %s", args.write_report)
    try:
        print(md)
    except UnicodeEncodeError as exc:
        logger.warning("stdout 不支持当前字符集，跳过 stdout 输出（%s）；请直接看 --write-report 文件", exc)
    n_fail = sum(1 for c in certs if c.decision == Decision.FAIL)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
