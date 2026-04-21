"""
把一份 PASS 证书 (``CertifiedFactorRecord`` / 契约 C2) 推进 L3 production registry。

工作流（见 ``docs/ARCHITECTURE_FACTOR_LAB.md`` §3）：

1. 读入 ``--cert`` JSON；pydantic 强校验其为合法 ``CertifiedFactorRecord``。
2. 校验 ``decision == PASS``；否则直接拒绝（HOLD/FAIL 不得进 L3）。
3. 校验 ``factor_registry/parquet/factors_v<N>.parquet`` 已存在且包含 ``candidate.name``
   这一列（保证 L3 读路径能找到物理数据）。
4. 调 ``FactorRegistry.register`` 原子更新 manifest；证书副本写入 ``data/certified/``。

本脚本**不写 parquet**——物理数据文件假定由如下流程之一提前落盘：
* ``scripts.promote.migrate_legacy_factors`` —— 从 legacy parquet 批量迁入
* 未来的 L1 → factor_registry merger（D 阶段交付）
* 人工 ``ParquetStore.write_version`` 调用

CLI
---

    python -m scripts.promote.promote_certified \
        --cert factor_validation/certificates/rdagent_Foo_abcd1234.json \
        --parquet-version 2 \
        --tags production,new_cycle

返回码：0=成功注册；1=参数/IO 错误；2=决议不是 PASS；3=parquet 物理文件校验失败；
         4=registry 层异常（重复 active 且未 --allow-overwrite 等）。
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
from factor_validation.schema import CertifiedFactorRecord, Decision  # noqa: E402

logger = logging.getLogger("promote_certified")

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "factor_lab.yaml"

EXIT_OK = 0
EXIT_BAD_ARGS = 1
EXIT_NOT_PASS = 2
EXIT_PARQUET_MISSING = 3
EXIT_REGISTRY_ERROR = 4


# ----------------------------------------------------------------- helpers


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dirs(config_path: Path) -> tuple[Path, Path, int]:
    """返回 (registry_data_dir, parquet_store_dir, default_parquet_version)。"""
    cfg = _load_config(config_path)
    reg_cfg = cfg.get("registry") or {}
    data_dir = _PROJECT_ROOT / reg_cfg.get("data_dir", "factor_registry/data")
    parquet_dir = _PROJECT_ROOT / reg_cfg.get("parquet_dir", "factor_registry/parquet")
    version = int(reg_cfg.get("current_parquet_version", 1))
    return data_dir, parquet_dir, version


def _parquet_path_for(parquet_dir: Path, version: int) -> Path:
    return parquet_dir / f"factors_v{version}.parquet"


def _verify_parquet_has_column(parquet_path: Path, name: str) -> None:
    """只读 parquet 的列头（避免加载全量数据），确认目标列存在。"""
    import pyarrow.parquet as pq

    if not parquet_path.exists():
        raise FileNotFoundError(f"未找到 parquet 物理文件：{parquet_path}")
    schema = pq.read_schema(parquet_path)
    cols = [f.name for f in schema]
    if name not in cols:
        raise ValueError(
            f"parquet {parquet_path.name} 不含列 {name!r}；现有列 {cols[:10]}"
            f"{'…' if len(cols) > 10 else ''}"
        )


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


# ------------------------------------------------------------------ promote


def promote(
    *,
    cert_path: Path,
    parquet_version: int,
    registry_data_dir: Path,
    parquet_store_dir: Path,
    tags: list[str],
    allow_overwrite: bool = False,
) -> tuple[int, str]:
    """
    返回 (exit_code, message)。成功时 message 是注册摘要。
    """
    if not cert_path.exists():
        return EXIT_BAD_ARGS, f"证书文件不存在: {cert_path}"

    try:
        cert = CertifiedFactorRecord.model_validate_json(
            cert_path.read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        return EXIT_BAD_ARGS, f"证书 JSON 解析失败: {exc}"

    if cert.decision != Decision.PASS:
        return (
            EXIT_NOT_PASS,
            f"decision={cert.decision.value}，拒绝注册；只有 PASS 可入 L3。",
        )

    parquet_path = _parquet_path_for(parquet_store_dir, parquet_version)
    try:
        _verify_parquet_has_column(parquet_path, cert.candidate.name)
    except (FileNotFoundError, ValueError) as exc:
        return EXIT_PARQUET_MISSING, f"parquet 物理校验失败: {exc}"

    registry = FactorRegistry(registry_data_dir)
    try:
        record = registry.register(
            cert,
            parquet_version=parquet_version,
            tags=tags,
            allow_overwrite=allow_overwrite,
        )
    except RegistryError as exc:
        return EXIT_REGISTRY_ERROR, f"registry 注册失败: {exc}"

    return EXIT_OK, (
        f"[OK] 已注册 factor_id={record.factor_id} "
        f"parquet_version={record.parquet_version} "
        f"status={record.status.value} tags={record.tags}"
    )


# ------------------------------------------------------------------------- CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Promote a PASSed certificate into factor_registry (L3)."
    )
    p.add_argument("--cert", type=Path, required=True, help="证书 JSON 路径")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="factor_lab.yaml 路径（用于读 registry.data_dir / parquet_dir / 默认版本号）",
    )
    p.add_argument(
        "--parquet-version",
        type=int,
        default=None,
        help="目标 factors_v<N>.parquet 版本号；缺省从 config 里取 current_parquet_version",
    )
    p.add_argument(
        "--tags",
        default="",
        help="逗号分隔的 tag 列表，写入 ProductionFactorRecord.tags（例如 'production,rdagent'）",
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="允许覆盖已 active 的同 factor_id 记录",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    data_dir, parquet_dir, cfg_version = _resolve_dirs(args.config)
    version = args.parquet_version if args.parquet_version is not None else cfg_version

    code, msg = promote(
        cert_path=args.cert,
        parquet_version=version,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=_parse_tags(args.tags),
        allow_overwrite=args.allow_overwrite,
    )
    if code == EXIT_OK:
        logger.info(msg)
    else:
        logger.error(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        pass
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
