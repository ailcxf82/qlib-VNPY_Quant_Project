"""
通用 L2 候选因子验证 CLI。

一个候选因子包 (``CandidateFactorPackage`` / 契约 C1) **进入 L2** 的标准入口，输出
一份 ``CertifiedFactorRecord`` (契约 C2) JSON。本脚本**不碰 L3 registry** —— 真正写进
production 走 ``scripts.promote.promote_certified``。这条流水线分拆的动机：

* ``validate_candidate``        ── 幂等、只读（除 ``--out-cert`` 落盘）、可重跑无副作用。
* ``promote_certified``         ── 读一份 PASS 证书，再写 L3 registry / parquet 物理文件。
* ``retire_factor``             ── 退役 active 因子。

这样 oracle 复测、人工 sanity-check、CI/CD 里的 promote 决策都能复用前两个脚本。

典型用法
--------

1. **直接从 candidate JSON 验证**（L1 exporter 产出的 C1 文件）：

       python -m scripts.promote.validate_candidate \
           --candidate-json factor_lab/workspace/candidates/<fid>/candidate.json \
           --profile factor_validation/profiles/default.yaml \
           --out-cert factor_validation/certificates/<fid>.json

2. **根据现成 parquet + 元信息即席构造候选**（Dev 调试用；rdagent/manual 都行）：

       python -m scripts.promote.validate_candidate \
           --values-path .../values.parquet \
           --code-path .../factor.py \
           --factor-id manual_MyFactor_01234567 \
           --name MyFactor \
           --source manual \
           --hypothesis "xxx" \
           --formulation "xxx" \
           --universe csi300 \
           --date-range 2024-07-01 2026-04-07 \
           --profile factor_validation/profiles/default.yaml

返回码
------

* ``0``  PASS
* ``2``  HOLD  (所有 check 通过但综合分低于 pass_threshold)
* ``3``  FAIL
* ``1``  参数或 IO 错误（argparse 异常 / 找不到文件 / profile 解析失败 等）

所有决议场景 **都会落盘证书**（如果指定了 ``--out-cert``）——便于事后追溯。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_lab.exporters.schema import CandidateFactorPackage  # noqa: E402
from factor_validation.orchestrator import validate_candidate  # noqa: E402
from factor_validation.schema import CertifiedFactorRecord, Decision  # noqa: E402

logger = logging.getLogger("validate_candidate")

EXIT_PASS = 0
EXIT_BAD_ARGS = 1
EXIT_HOLD = 2
EXIT_FAIL = 3


# --------------------------------------------------------------- candidate loading


def _load_candidate_from_json(path: Path) -> CandidateFactorPackage:
    raw = path.read_text(encoding="utf-8")
    # pydantic v2：model_validate_json 直接返回实例
    return CandidateFactorPackage.model_validate_json(raw)


def _load_candidate_from_cli(args: argparse.Namespace) -> CandidateFactorPackage:
    """从 CLI 参数拼装 candidate；仅给 dev/即席使用。"""
    if args.values_path is None or args.code_path is None:
        raise SystemExit(
            "即席模式必须同时给 --values-path 与 --code-path（或改用 --candidate-json）"
        )
    required = {
        "factor_id": args.factor_id,
        "name": args.name,
        "source": args.source,
        "hypothesis": args.hypothesis,
        "formulation": args.formulation,
        "universe": args.universe,
        "date_range": args.date_range,
    }
    missing = [k for k, v in required.items() if v in (None, "", [])]
    if missing:
        raise SystemExit(f"即席模式缺必填字段: {missing}")

    start_d, end_d = args.date_range  # argparse 已把两个字符串转成 date
    # lab_run_id / created_at 允许缺省：即席模式视为 manual，标签一次性生成
    run_id = args.lab_run_id or (
        f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        + hashlib.sha1(str(args.values_path).encode()).hexdigest()[:6]
    )
    created_at = args.created_at or datetime.now(timezone.utc)

    # parent_loop：source=rdagent 时必须，schema 层会强校验
    parent_loop: int | None = args.parent_loop

    return CandidateFactorPackage(
        factor_id=args.factor_id,
        name=args.name,
        source=args.source,
        hypothesis=args.hypothesis,
        formulation=args.formulation,
        code_path=args.code_path,
        values_path=args.values_path,
        universe=args.universe,
        date_range=(start_d, end_d),
        lab_metrics={},
        parent_loop=parent_loop,
        created_at=created_at,
        lab_run_id=run_id,
    )


def _load_candidate(args: argparse.Namespace) -> CandidateFactorPackage:
    if args.candidate_json is not None:
        return _load_candidate_from_json(args.candidate_json)
    return _load_candidate_from_cli(args)


# ------------------------------------------------------------------------ runner


def run(args: argparse.Namespace) -> tuple[CertifiedFactorRecord, int]:
    """加载 candidate → 跑 L2 → 返回 (证书, exit_code)。"""
    cand = _load_candidate(args)
    # 物料校验：文件存在 + parquet schema；失败直接抛
    cand.validate_artifacts(check_parquet_schema=True)

    cert = validate_candidate(
        cand,
        profile_path=args.profile,
        project_root=_PROJECT_ROOT,
        notes=args.notes,
    )

    if args.out_cert is not None:
        args.out_cert.parent.mkdir(parents=True, exist_ok=True)
        args.out_cert.write_text(
            cert.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("证书落盘：%s", args.out_cert)

    if cert.decision == Decision.PASS:
        return cert, EXIT_PASS
    if cert.decision == Decision.HOLD:
        return cert, EXIT_HOLD
    return cert, EXIT_FAIL


def summarize(cert: CertifiedFactorRecord) -> str:
    lines = [
        f"factor_id      : {cert.factor_id}",
        f"profile        : {cert.profile_name} (hash={cert.profile_hash[:8]}…)",
        f"decision       : {cert.decision.value}",
        f"overall_score  : {cert.overall_score:.4f}",
        "checks:",
    ]
    for r in cert.check_results:
        tick = "[OK]" if r.passed else "[X] "
        s = "n/a" if r.score is None else f"{r.score:.3f}"
        lines.append(f"  {tick} {r.name:<14s} score={s:>6s}  passed={r.passed}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CLI


def _date_pair(values: list[str]) -> tuple[date, date]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--date-range 需要 2 个 ISO 日期")
    return (date.fromisoformat(values[0]), date.fromisoformat(values[1]))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a CandidateFactorPackage against a ValidationProfile."
    )
    p.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="ValidationProfile YAML 路径",
    )
    p.add_argument(
        "--out-cert",
        type=Path,
        default=None,
        help="证书 JSON 输出路径；不填则仅打印",
    )
    p.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help="(优先)从 C1 JSON 文件读入 CandidateFactorPackage；给了本项则忽略其他 --xxx 字段",
    )

    g = p.add_argument_group("即席 candidate（--candidate-json 未给时才使用）")
    g.add_argument("--factor-id", default=None)
    g.add_argument("--name", default=None)
    g.add_argument(
        "--source",
        choices=("rdagent", "manual", "external"),
        default=None,
    )
    g.add_argument("--hypothesis", default=None)
    g.add_argument("--formulation", default=None)
    g.add_argument("--code-path", type=Path, default=None)
    g.add_argument("--values-path", type=Path, default=None)
    g.add_argument("--universe", default=None)
    g.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        type=str,
    )
    g.add_argument("--parent-loop", type=int, default=None)
    g.add_argument("--lab-run-id", default=None)
    g.add_argument(
        "--created-at",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        default=None,
        help="ISO8601；默认 now(UTC)",
    )

    p.add_argument("--notes", default=None)
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args(list(argv) if argv is not None else None)

    if ns.date_range is not None:
        ns.date_range = _date_pair(ns.date_range)
    return ns


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    try:
        cert, code = run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("validate_candidate 执行失败: %s", exc, exc_info=True)
        return EXIT_BAD_ARGS

    # 摘要落控制台
    try:
        print(summarize(cert))
    except UnicodeEncodeError:
        # Windows GBK stdout 也要容忍
        logger.warning("stdout 不支持当前字符集，跳过 stdout 摘要；请看 --out-cert JSON")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
