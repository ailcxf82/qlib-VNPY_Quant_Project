"""
Retire 一个已 active 的 L3 因子。

语义（见 ``factor_registry.registry.FactorRegistry.retire``）：

* manifest 记录 ``status -> retired``、写入 ``retired_at`` / ``retire_reason``；
* 证书副本从 ``data/certified/<fid>.json`` 搬到 ``data/retired/<fid>.json``；
* **不改动 parquet 物理文件**。旧 ``factors_v<N>.parquet`` 仍完整保留，下游
  ``ProductionFactorLoader`` 默认 ``only_active=True`` 会自动过滤掉 retired 因子。
  历史回测 / 时光旅行复盘依旧可以通过 ``only_active=False`` 读取完整列。

典型用法
--------

    python -m scripts.promote.retire_factor \
        --factor-id manual_VolRet_5D_abc1234567 \
        --reason "oracle default profile FAIL: rank_ic<0.01 + ic_ir<0"

    # 批量退役（一行一个 factor_id 的纯文本文件）：
    python -m scripts.promote.retire_factor \
        --ids-file to_retire.txt \
        --reason "legacy grandfathered factors failing new default profile"

返回码
------

* ``0``  全部退役成功（或待退役列表为空）
* ``1``  参数 / IO 错误
* ``2``  部分/全部因子退役失败（详见 stderr）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_registry.registry import FactorRegistry, RegistryError  # noqa: E402
from factor_registry.schema import ProductionStatus  # noqa: E402

logger = logging.getLogger("retire_factor")

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "factor_lab.yaml"

EXIT_OK = 0
EXIT_BAD_ARGS = 1
EXIT_PARTIAL = 2


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_data_dir(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    reg_cfg = cfg.get("registry") or {}
    return _PROJECT_ROOT / reg_cfg.get("data_dir", "factor_registry/data")


def _read_ids_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"--ids-file 不存在：{path}")
    ids: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return ids


def retire_many(
    *,
    factor_ids: list[str],
    reason: str,
    registry_data_dir: Path,
    dry_run: bool = False,
) -> tuple[int, list[dict]]:
    """
    返回 (exit_code, results)。``results`` 是纯字典列表，便于测试断言。
    """
    if not reason or not reason.strip():
        return EXIT_BAD_ARGS, [{"error": "reason 不能为空"}]
    if not factor_ids:
        logger.warning("retire: 待退役列表为空")
        return EXIT_OK, []

    registry = FactorRegistry(registry_data_dir)
    results: list[dict] = []
    any_fail = False

    for fid in factor_ids:
        entry: dict = {"factor_id": fid, "ok": False}
        try:
            rec = registry.get(fid)
            if rec.status == ProductionStatus.RETIRED:
                entry["skipped"] = "already retired"
                entry["ok"] = True
                logger.info("skip %s: 已退役", fid)
            elif dry_run:
                entry["dry_run"] = True
                entry["ok"] = True
                logger.info("dry-run %s (status=%s)", fid, rec.status.value)
            else:
                retired = registry.retire(fid, reason=reason)
                entry["ok"] = True
                entry["retired_at"] = retired.retired_at.isoformat()
                logger.info("retired %s", fid)
        except RegistryError as exc:
            entry["error"] = str(exc)
            any_fail = True
            logger.error("retire %s 失败：%s", fid, exc)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            any_fail = True
            logger.error("retire %s 未预期异常：%s", fid, exc, exc_info=True)
        results.append(entry)

    return (EXIT_PARTIAL if any_fail else EXIT_OK), results


# ------------------------------------------------------------------------- CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retire active factors in L3 registry.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--factor-id",
        action="append",
        default=None,
        help="单个 factor_id；可重复多次指定多个",
    )
    group.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="每行一个 factor_id 的纯文本文件（# 行忽略）",
    )
    p.add_argument(
        "--reason",
        required=True,
        help="退役原因（强制）；写入 ProductionFactorRecord.retire_reason",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将被退役的因子，不改 manifest",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    try:
        if args.ids_file is not None:
            ids = _read_ids_file(args.ids_file)
        else:
            ids = list(args.factor_id or [])
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return EXIT_BAD_ARGS

    data_dir = _resolve_data_dir(args.config)
    code, results = retire_many(
        factor_ids=ids,
        reason=args.reason,
        registry_data_dir=data_dir,
        dry_run=args.dry_run,
    )
    # 汇总
    ok = sum(1 for r in results if r.get("ok"))
    fail = sum(1 for r in results if not r.get("ok"))
    logger.info("retire: 完成 %d/%d，失败 %d", ok, len(results), fail)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
