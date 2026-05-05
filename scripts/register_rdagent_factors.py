"""
将 RD-Agent workspace 产出的因子注册到 factor_registry，让 run_train_csi101.py 直接使用。

完整链路（二选一数据源）：

  A) workspace 模式（默认）
     RD-Agent_workspace/<id>/combined_factors_df.parquet → …

  B) 刷新统一产物（推荐，与 refresh_rdagent_parquet 对齐）
     config/factor_lab.yaml → registry.refresh_combined_parquet
     （默认 git_ignore_folder/combined_factors_df.parquet）
     python scripts/register_rdagent_factors.py --from-refresh

  后续：
     factor_registry/parquet/factors_v<N>.parquet
      → manifest.json / factor_lab.yaml current_parquet_version
      → qlib_feature_pipeline L3 → run_train_csi101.py rdagent_exported

用法（在项目根目录）：
    python scripts/register_rdagent_factors.py --from-refresh   # 统一入口（推荐）
    python scripts/register_rdagent_factors.py --workspace 38418fc744744b9695e592f385e69e3c
    python scripts/register_rdagent_factors.py --dry-run   # 仅预览，不写文件
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"
REGISTRY_PARQUET_DIR = ROOT / "factor_registry" / "parquet"
MANIFEST_PATH = ROOT / "factor_registry" / "data" / "manifest.json"
FACTOR_LAB_YAML = ROOT / "config" / "factor_lab.yaml"

# 默认使用 TOP-5 workspace（覆盖率 97-99%，数据到 2025-10-31）
DEFAULT_WS = "38418fc744744b9695e592f385e69e3c"

REFRESH_WORKSPACE_LABEL = "refresh_rdagent_parquet"


def _registry_section() -> dict[str, Any]:
    if not FACTOR_LAB_YAML.exists():
        return {}
    data = yaml.safe_load(FACTOR_LAB_YAML.read_text(encoding="utf-8")) or {}
    return data.get("registry") or {}


def _refresh_parquet_path() -> Path:
    reg = _registry_section()
    rel = reg.get("refresh_combined_parquet", "git_ignore_folder/combined_factors_df.parquet")
    return (ROOT / rel).resolve()


def _refresh_summary_json_path() -> Path:
    reg = _registry_section()
    rel = reg.get("refresh_summary_json", "git_ignore_folder/combined_factors_df.json")
    return (ROOT / rel).resolve()


def _metrics_from_refresh_summary(path: Path) -> tuple[float | None, float | None]:
    """从 refresh_rdagent_parquet 写的 summary json 取代表性 IC（无全局 ICIR 时返回 None）。"""
    if not path.exists():
        return None, None
    try:
        sj = json.loads(path.read_text(encoding="utf-8"))
        facs = sj.get("factors") or []
        if not facs:
            return None, None
        ics = [float(f["ic"]) for f in facs if f.get("ic") is not None]
        if not ics:
            return None, None
        ic_mean = sum(ics) / len(ics)
        # summary 无全局 ICIR；用各因子 ic_in_ir 均值作登记参考（可为 None）
        irs = []
        for f in facs:
            v = f.get("ic_in_ir")
            if v is not None:
                try:
                    irs.append(float(v))
                except (TypeError, ValueError):
                    pass
        icir_mean = sum(irs) / len(irs) if irs else None
        return icir_mean, ic_mean
    except Exception:
        return None, None


def _pick_workspace(ws_id: str | None) -> Path:
    ws_id = ws_id or DEFAULT_WS
    ws = WS_BASE / ws_id
    if not (ws / "combined_factors_df.parquet").exists():
        # fallback：扫描最优
        best, best_icir = None, -1.0
        for d in WS_BASE.iterdir():
            csv = d / "qlib_res.csv"
            pq = d / "combined_factors_df.parquet"
            if not csv.exists() or not pq.exists():
                continue
            try:
                m = pd.read_csv(csv, index_col=0)
                m.columns = ["value"]
                icir = float(m.loc["ICIR", "value"])
                if icir > best_icir:
                    best_icir, best = icir, d
            except Exception:
                pass
        if best is None:
            raise RuntimeError("找不到任何含 parquet 的 workspace")
        return best
    return ws


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"schema_version": 1, "factors": []}


def _save_manifest(mf: dict, dry: bool) -> None:
    txt = json.dumps(mf, ensure_ascii=False, indent=2)
    if dry:
        print(f"  [dry-run] manifest.json:\n{txt[:600]}...")
    else:
        MANIFEST_PATH.write_text(txt, encoding="utf-8")
        print(f"  manifest.json 已更新 → {MANIFEST_PATH}")


def _get_next_version(mf: dict) -> int:
    existing = {r.get("parquet_version", 0) for r in mf.get("factors", [])}
    return max(existing, default=0) + 1


def _update_factor_lab_yaml(version: int, dry: bool) -> None:
    """更新 factor_lab.yaml 中的 current_parquet_version。"""
    if not FACTOR_LAB_YAML.exists():
        print("  [警告] config/factor_lab.yaml 不存在，跳过版本号更新")
        return
    text = FACTOR_LAB_YAML.read_text(encoding="utf-8")
    import re
    new_text = re.sub(
        r"(current_parquet_version\s*:\s*)\d+",
        rf"\g<1>{version}",
        text,
    )
    if dry:
        print(f"  [dry-run] factor_lab.yaml: current_parquet_version → {version}")
    else:
        FACTOR_LAB_YAML.write_text(new_text, encoding="utf-8")
        print(f"  factor_lab.yaml → current_parquet_version: {version}")


def register(
    workspace_id: str | None = None,
    dry: bool = False,
    retire_old: bool = True,
    from_refresh: bool = False,
) -> None:
    if from_refresh:
        pq_src = _refresh_parquet_path()
        if not pq_src.exists():
            raise FileNotFoundError(
                f"统一刷新产物不存在：{pq_src}\n"
                "请先运行：conda run -n qlib_zhengshi python scripts/refresh_rdagent_parquet.py\n"
                "或在 config/factor_lab.yaml 中调整 registry.refresh_combined_parquet"
            )
        ws_dir_name = REFRESH_WORKSPACE_LABEL
        summary_path = _refresh_summary_json_path()
        icir, ic = _metrics_from_refresh_summary(summary_path)
    else:
        ws_dir = _pick_workspace(workspace_id)
        pq_src = ws_dir / "combined_factors_df.parquet"
        ws_dir_name = ws_dir.name
        summary_path = None  # unused

        # 读 qlib_res.csv 获取指标
        try:
            res = pd.read_csv(ws_dir / "qlib_res.csv", index_col=0)
            res.columns = ["value"]
            icir = float(res.loc["ICIR", "value"])
            ic = float(res.loc["IC", "value"])
        except Exception:
            icir = ic = None

    # 读取 parquet，扁平化 MultiIndex 列名
    df = pd.read_parquet(pq_src)
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(-1)
    factor_names = list(df.columns)

    # 统计覆盖率
    dts = df.index.get_level_values("datetime")
    date_min, date_max = dts.min().date(), dts.max().date()
    coverage = df.notna().mean().mean() * 100

    print(f"\n{'='*62}")
    print(f"  数据源     : {'统一刷新 parquet (--from-refresh)' if from_refresh else 'RD-Agent workspace'}")
    print(f"  workspace  : {ws_dir_name}")
    if from_refresh:
        print(f"  parquet 文件: {pq_src}")
        if summary_path and summary_path.exists():
            print(f"  summary json: {summary_path}")
    print(f"  因子数量   : {len(factor_names)}")
    print(f"  因子列表   : {factor_names}")
    print(f"  日期范围   : {date_min} → {date_max}")
    print(f"  平均覆盖率 : {coverage:.1f}%")
    if icir is not None and ic is not None:
        print(f"  ICIR / IC  : {icir:.3f} / {ic:.4f}")
    elif ic is not None:
        print(f"  IC (均值)  : {ic:.4f}")
    print(f"{'='*62}\n")

    # 确定新版本号
    mf = _load_manifest()
    new_version = _get_next_version(mf)
    print(f"[1/4] 新 parquet 版本号: v{new_version}")

    # Step 1：写 factors_v<N>.parquet（扁平列名，单级 index）
    dest_pq = REGISTRY_PARQUET_DIR / f"factors_v{new_version}.parquet"
    if dry:
        print(f"  [dry-run] 写入 {dest_pq}  shape={df.shape}")
    else:
        REGISTRY_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest_pq)
        size_mb = dest_pq.stat().st_size / 1024 / 1024
        print(f"  写入 {dest_pq}  ({size_mb:.1f} MB)")

    # Step 2：更新 manifest.json
    print(f"[2/4] 更新 manifest.json")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 退役旧版本的 active 记录
    if retire_old:
        for rec in mf["factors"]:
            if rec.get("status") == "active":
                rec["status"] = "retired"
                rec["retired_at"] = now_iso
                if dry:
                    print(f"  [dry-run] retire → {rec['factor_id']}")
                else:
                    print(f"  退役: {rec['factor_id']}")

    # 新增 RD-Agent 因子记录
    for col in factor_names:
        factor_id = f"rdagent_{col}_{new_version}"
        new_rec = {
            "factor_id": factor_id,
            "factor_name": col,
            "source": "rdagent_refresh" if from_refresh else "rdagent",
            "workspace_id": ws_dir_name,
            "parquet_column": col,
            "parquet_version": new_version,
            "status": "active",
            "registered_at": now_iso,
            "icir": icir,
            "ic": ic,
            "date_min": str(date_min),
            "date_max": str(date_max),
            "coverage_pct": round(coverage, 2),
        }
        mf["factors"].append(new_rec)
        if dry:
            print(f"  [dry-run] 新增 → {factor_id}")
        else:
            print(f"  注册: {factor_id}")

    _save_manifest(mf, dry)

    # Step 3：更新 factor_lab.yaml
    print(f"[3/4] 更新 config/factor_lab.yaml → current_parquet_version: {new_version}")
    _update_factor_lab_yaml(new_version, dry)

    # Step 4：验证
    print(f"[4/4] 验证注册结果")
    if not dry:
        mf_verify = _load_manifest()
        active = [r for r in mf_verify["factors"] if r.get("status") == "active"]
        print(f"  当前 active 因子数: {len(active)}")
        for r in active:
            print(f"    {r['factor_id']:50s} col={r['parquet_column']} v={r['parquet_version']}")

    print(f"\n{'='*62}")
    if dry:
        print("  [dry-run 完成] 以上为预览，未实际写入任何文件")
    else:
        print("  注册完成！现在可以直接运行：")
        print("  python run_train_csi101.py")
        print()
        print("  框架会自动：")
        print("  1. 通过 ProductionFactorLoader 读取 factors_v{} 中的 {} 个 AI 因子".format(new_version, len(factor_names)))
        print("  2. 合并到 LGB 特征集 rdagent_exported")
        print("  3. 与 lgb_short_cycle 基础特征一同训练 LightGBM 模型")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 RD-Agent 因子注册到 factor_registry")
    parser.add_argument(
        "--from-refresh",
        action="store_true",
        help="从 factor_lab.yaml 的 registry.refresh_combined_parquet 注册（与 refresh_rdagent_parquet 对齐）",
    )
    parser.add_argument("--workspace", default=None, help="workspace ID（与 --from-refresh 互斥；默认使用内置 TOP workspace）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写任何文件")
    parser.add_argument("--keep-old", action="store_true", help="保留旧因子的 active 状态（不退役）")
    args = parser.parse_args()
    if args.from_refresh and args.workspace:
        parser.error("--from-refresh 与 --workspace 不能同时使用")
    register(
        workspace_id=args.workspace,
        dry=args.dry_run,
        retire_old=not args.keep_old,
        from_refresh=args.from_refresh,
    )
