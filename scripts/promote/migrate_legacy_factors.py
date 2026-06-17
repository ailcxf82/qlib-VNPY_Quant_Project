"""
一次性迁移脚本：把 ``git_ignore_folder/combined_factors_df.parquet`` 的历史因子
补发证书后注册到 L3 factor_registry。

使用场景：阶段 B。阶段 C 之后新因子走 ``factor_lab → factor_validation → factor_registry``
的完整流水线，**不再使用本脚本**。

工作流：
1. 读取源 parquet（默认 ``git_ignore_folder/combined_factors_df.parquet``），逐列切片：
   - 每列 → 写一份 ``factor_lab/workspace/candidates/<factor_id>/values.parquet``（单列）
   - 同目录写一份占位 ``factor.py`` 描述迁移来源
2. 构造 ``CandidateFactorPackage(source='manual', ...)``
3. 调 ``factor_validation.orchestrator.validate_candidate`` 跑 ``manual_grandfathered.yaml``
4. 把整个源 parquet 原封不动写成 ``factor_registry/parquet/factors_v<N>.parquet``
5. 把每个 PASS 的证书 register 到 ``factor_registry/data/manifest.json``

幂等性：
* 同名 factor_id 已存在时用 ``--overwrite`` 允许覆盖
* parquet_version 已存在时用 ``--overwrite-parquet``

CLI：
    python -m scripts.promote.migrate_legacy_factors \
        --source git_ignore_folder/combined_factors_df.parquet \
        --parquet-version 1 \
        --profile factor_validation/profiles/manual_grandfathered.yaml

默认参数全部从 ``config/factor_lab.yaml`` 推断，可不传。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_lab.exporters.schema import CandidateFactorPackage  # noqa: E402
from factor_registry.registry import FactorRegistry  # noqa: E402
from factor_registry.store import ParquetStore  # noqa: E402
from factor_validation.orchestrator import validate_candidate  # noqa: E402
from factor_validation.schema import Decision  # noqa: E402


logger = logging.getLogger("migrate_legacy_factors")

DEFAULT_SOURCE = _PROJECT_ROOT / "git_ignore_folder" / "combined_factors_df.parquet"
DEFAULT_PROFILE = (
    _PROJECT_ROOT / "factor_validation" / "profiles" / "manual_grandfathered.yaml"
)
DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "factor_lab.yaml"
WORKSPACE_ROOT = _PROJECT_ROOT / "factor_lab" / "workspace" / "candidates"
FACTOR_PY_TEMPLATE = '''"""
{factor_id}

阶段 B legacy migration 占位文件——本因子来自 P0-1 阶段 RD-Agent 导出，
原始代码散落于 ``git_ignore_folder/RD-Agent_workspace/<workspace>/factor.py``。

本占位文件保留 CandidateFactorPackage 对 ``code_path`` 的硬约束，
只用于证书归档，不参与任何 production 计算。
"""

# Legacy migration marker; factor values live in values.parquet at the same path.
LEGACY = True
'''


# ---------------------------------------------------------------- core


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _factor_id_for(name: str, series: pd.Series) -> str:
    """基于列名 + 序列摘要生成稳定 factor_id（同样输入得到同样 id）。"""
    h = hashlib.sha1()
    h.update(name.encode("utf-8"))
    # 用可哈希的紧凑摘要；pandas.util.hash_pandas_object 对 NaN / 时间戳都稳定
    h.update(pd.util.hash_pandas_object(series, index=True).values.tobytes())
    return f"manual_{name}_{h.hexdigest()[:10]}"


def _write_candidate_workspace(
    name: str,
    factor_id: str,
    series: pd.Series,
    root: Path,
) -> tuple[Path, Path]:
    """把单列因子写到 workspace/candidates/<factor_id>/，返回 (code_path, values_path)。"""
    ws = root / factor_id
    ws.mkdir(parents=True, exist_ok=True)

    values_path = ws / "values.parquet"
    df = series.to_frame(name=name)
    # 必须 MultiIndex(datetime, instrument)
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != [
        "datetime",
        "instrument",
    ]:
        raise ValueError(
            f"series index 必须为 MultiIndex(datetime, instrument)：{df.index.names}"
        )
    df = df.astype("float64")
    df.to_parquet(values_path, engine="pyarrow")

    code_path = ws / "factor.py"
    code_path.write_text(
        FACTOR_PY_TEMPLATE.format(factor_id=factor_id), encoding="utf-8"
    )
    return code_path, values_path


def _build_candidate(
    name: str,
    series: pd.Series,
    *,
    universe: str,
    workspace_root: Path,
    now: datetime,
    run_id: str,
) -> CandidateFactorPackage:
    factor_id = _factor_id_for(name, series)
    code_path, values_path = _write_candidate_workspace(
        name, factor_id, series, workspace_root
    )

    dt_index = series.index.get_level_values("datetime")
    start_date = pd.Timestamp(dt_index.min()).date()
    end_date = pd.Timestamp(dt_index.max()).date()

    cand = CandidateFactorPackage(
        factor_id=factor_id,
        name=name,
        source="manual",
        hypothesis=(
            "Legacy RD-Agent exported factor migrated via "
            "scripts.promote.migrate_legacy_factors (grandfathered for L3 registry)."
        ),
        formulation=f"(legacy; see git_ignore_folder/RD-Agent_workspace for origin) column={name}",
        code_path=code_path,
        values_path=values_path,
        universe=universe,
        date_range=(start_date, end_date),
        lab_metrics={
            "non_null_ratio": float(series.notna().mean()),
        },
        parent_loop=None,
        created_at=now,
        lab_run_id=run_id,
    )
    cand.validate_artifacts(check_parquet_schema=True)
    return cand


def migrate(
    *,
    source_parquet: Path,
    profile_path: Path,
    parquet_version: int,
    registry_data_dir: Path,
    parquet_store_dir: Path,
    universe: str = "csi300",
    overwrite_registry: bool = False,
    overwrite_parquet: bool = False,
    now: datetime | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """执行迁移，返回每列的结果摘要（纯字典，方便日志 / 测试断言）。"""
    now = now or datetime.now(timezone.utc)
    run_id = run_id or f"legacy-migrate-{now.strftime('%Y%m%d-%H%M%S')}"

    logger.info("migrate: 读取源 parquet %s", source_parquet)
    src = pd.read_parquet(source_parquet)
    if not isinstance(src.index, pd.MultiIndex) or list(src.index.names) != [
        "datetime",
        "instrument",
    ]:
        raise ValueError(
            f"源 parquet 索引必须为 MultiIndex(datetime, instrument)：{src.index.names}"
        )
    if src.empty or src.columns.empty:
        raise ValueError(f"源 parquet 为空：{source_parquet}")
    # dtype 强转 float64（ParquetStore.write_version 会硬校验）
    src = src.astype("float64")

    logger.info(
        "migrate: 源 parquet 形状 shape=%s, 列=%s",
        src.shape,
        list(src.columns),
    )

    # ---- 1) 写 L3 production parquet（整块原样搬运）
    store = ParquetStore(parquet_store_dir)
    store.write_version(src, parquet_version, overwrite=overwrite_parquet)

    # ---- 2) 逐列造 candidate + 跑 L2 validate + register
    registry = FactorRegistry(registry_data_dir)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for col in src.columns:
        series = src[col]
        cand = _build_candidate(
            col,
            series,
            universe=universe,
            workspace_root=WORKSPACE_ROOT,
            now=now,
            run_id=run_id,
        )
        cert = validate_candidate(
            cand,
            profile_path,
            project_root=_PROJECT_ROOT,
            now=now,
            notes=f"legacy migration run_id={run_id}",
        )
        entry = {
            "name": col,
            "factor_id": cand.factor_id,
            "decision": cert.decision.value,
            "overall_score": cert.overall_score,
            "registered": False,
        }

        if cert.decision == Decision.PASS:
            try:
                registry.register(
                    cert,
                    parquet_version=parquet_version,
                    tags=["legacy", "grandfathered"],
                    now=now,
                    allow_overwrite=overwrite_registry,
                )
                entry["registered"] = True
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"register failed: {exc!r}"
                logger.error("register %s 失败：%s", cand.factor_id, exc)
        else:
            logger.warning(
                "migrate: %s decision=%s overall=%.4f，跳过 register",
                cand.factor_id,
                cert.decision.value,
                cert.overall_score,
            )
        results.append(entry)

    return results


# ---------------------------------------------------------------- CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migrate legacy combined_factors_df.parquet into factor_registry"
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="源 parquet 路径")
    p.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="ValidationProfile YAML；默认 manual_grandfathered.yaml",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="factor_lab.yaml 路径，用于解析 registry/parquet 目录",
    )
    p.add_argument(
        "--parquet-version",
        type=int,
        default=None,
        help="写入 factor_registry/parquet 的版本号；默认取 factor_lab.yaml.registry.current_parquet_version",
    )
    p.add_argument("--universe", default="csi300")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖 registry 内同名因子")
    p.add_argument("--overwrite-parquet", action="store_true", help="允许覆盖同版本 parquet")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_dirs(config_path: Path) -> tuple[Path, Path, int]:
    cfg = _load_config(config_path)
    reg_cfg = cfg.get("registry") or {}
    data_dir = _PROJECT_ROOT / reg_cfg.get("data_dir", "factor_registry/data")
    parquet_dir = _PROJECT_ROOT / reg_cfg.get("parquet_dir", "factor_registry/parquet")
    version = int(reg_cfg.get("current_parquet_version", 1))
    return data_dir, parquet_dir, version


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    data_dir, parquet_dir, cfg_version = _resolve_dirs(args.config)
    version = args.parquet_version if args.parquet_version is not None else cfg_version

    logger.info("migrate: source=%s version=%d profile=%s", args.source, version, args.profile)
    results = migrate(
        source_parquet=args.source,
        profile_path=args.profile,
        parquet_version=version,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        universe=args.universe,
        overwrite_registry=args.overwrite,
        overwrite_parquet=args.overwrite_parquet,
    )

    # 汇总
    total = len(results)
    ok = sum(1 for r in results if r["registered"])
    logger.info("migrate: 完成 %d/%d 注册成功", ok, total)
    for r in results:
        logger.info(
            "  - %s (%s) decision=%s registered=%s",
            r["name"],
            r["factor_id"],
            r["decision"],
            r["registered"],
        )
    return 0 if ok == total else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
