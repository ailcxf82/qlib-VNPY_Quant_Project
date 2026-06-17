"""
P0-1 因子诊断（用于解释"含 RD 回测变差"，并产出"保留/淘汰"建议名单）

输出三块：
  1) 共线性：14 列两两 |corr| + 与 lgb_short_cycle 价格族表达式 |corr|
  2) OOS 滚动 IC：在 train_cutoff 之前按 12 段滚动算 IC，给出 IC_mean / IC_std / IC_IR /
     IC_pos_ratio / OOS 段（>cutoff）IC
  3) LGB feature_importance：用 data/models/<pool>_models/<latest>_lgb.txt 算 gain importance，
     看 14 个 RD 因子真正"被 LGB 选中"的占比

最终给出"保留 / 淘汰"建议（基于以下规则，并写入 data/factor_analysis/p01_diag_report.md）：
  - OOS IC_IR ≥ 0.3
  - OOS IC 与全样本 IC 同号
  - 与已有 lgb_short_cycle 列两两 |corr| ≤ MAX_CORR_VS_LGB（默认 0.7）

环境：conda run -n qlib_zhengshi --no-capture-output python scripts/factor_diagnostic_p01.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("factor_diagnostic_p01")

PROVIDER_URI = "D:/qlib_data/qlib_data"
PARQUET = _ROOT / "git_ignore_folder" / "combined_factors_df.parquet"
OUT_DIR = _ROOT / "data" / "factor_analysis"
OUT_REPORT = OUT_DIR / "p01_diag_report.md"
OUT_CSV_OOS = OUT_DIR / "p01_oos_ic.csv"
OUT_CSV_CORR = OUT_DIR / "p01_corr_matrix.csv"
OUT_CSV_CROSS = OUT_DIR / "p01_corr_vs_lgb.csv"
OUT_CSV_KEEP = OUT_DIR / "p01_keep_drop.csv"


def _load_label(start: str, end: str, instruments: list[str]) -> pd.Series:
    """主工程 label：close[t+3]/close[t+1] - 1"""
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    df = D.features(
        instruments,
        ["Ref($close_qfq, -3)/Ref($close_qfq, 1) - 1"],
        start_time=start, end_time=end, freq="day",
    )
    df.columns = ["label"]
    return df["label"]


def _load_lgb_short_cycle(start: str, end: str, instruments: list[str]) -> pd.DataFrame:
    """从主工程读取 lgb_short_cycle 表达式集合，构造对应的 features DataFrame。"""
    import yaml
    from qlib.data import D

    data_cfg = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    fset = data_cfg["data"]["feature_sets"]["lgb_short_cycle"]
    expr_list = list(fset)
    df = D.features(instruments, expr_list, start_time=start, end_time=end, freq="day")
    df.columns = expr_list
    return df


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """对齐 (datetime, instrument) 双索引顺序。"""
    if df.index.names != ["datetime", "instrument"]:
        if set(df.index.names) == {"datetime", "instrument"}:
            df = df.swaplevel().sort_index()
    df.index.set_names(["datetime", "instrument"], inplace=True)
    return df.sort_index()


def _rolling_ic(factor: pd.Series, label: pd.Series, segments: list[tuple[str, str]]) -> list[dict]:
    """对每个 (start, end) 段算 RankIC。"""
    out = []
    for s, e in segments:
        ms = pd.Timestamp(s)
        me = pd.Timestamp(e)
        f_seg = factor.loc[(slice(ms, me), slice(None))]
        l_seg = label.loc[(slice(ms, me), slice(None))]
        df = pd.DataFrame({"f": f_seg, "l": l_seg}).dropna()
        if len(df) < 200:
            ic = np.nan
        else:
            ic = float(df["f"].rank().corr(df["l"].rank()))
        out.append({"start": s, "end": e, "n": len(df), "ic": ic})
    return out


def _make_segments(start: str, end: str, n: int) -> list[tuple[str, str]]:
    """把 [start, end] 等分为 n 段。"""
    sd = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    bounds = pd.date_range(sd, ed, periods=n + 1)
    segs = []
    for i in range(n):
        s = bounds[i].strftime("%Y-%m-%d")
        e = (bounds[i + 1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        segs.append((s, e))
    return segs


def _maybe_load_lgb_importance(model_dir: Path) -> Optional[Dict[str, float]]:
    """挑该目录最新一个 *_lgb.txt + 同名 _meta.json，算 gain importance。"""
    if not model_dir.exists():
        return None
    txts = sorted(model_dir.glob("*_lgb.txt"))
    if not txts:
        return None
    booster_path = txts[-1]
    meta_path = booster_path.with_name(booster_path.stem + "_meta.json")
    if not meta_path.exists():
        return None
    try:
        import lightgbm as lgb
    except Exception:
        return None
    booster = lgb.Booster(model_file=str(booster_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_names = meta.get("feature_names") or []
    gain = booster.feature_importance(importance_type="gain")
    if len(feat_names) != len(gain):
        return None
    total = float(gain.sum()) or 1.0
    return {n: float(g) / total for n, g in zip(feat_names, gain)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="P0-1 因子诊断")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-04-07")
    ap.add_argument("--train-cutoff", default="2024-12-31",
                    help="OOS 分界：≤ 此日期为 in-sample，> 此日期为 OOS")
    ap.add_argument("--n-segments", type=int, default=12, help="in-sample 段数")
    ap.add_argument("--ic-ir-threshold", type=float, default=0.30)
    ap.add_argument("--max-corr-vs-lgb", type=float, default=0.70)
    ap.add_argument("--lgb-model-dir", default="data/models/csi300_models",
                    help="拿来算 LGB feature_importance 的模型目录")
    args = ap.parse_args()

    if not PARQUET.exists():
        logger.error("MISSING parquet: %s", PARQUET); sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== 1) load parquet ===")
    pf = pd.read_parquet(str(PARQUET))
    pf = _normalize_index(pf)
    rd_cols = list(pf.columns)
    logger.info("RD-Agent factor columns: %d", len(rd_cols))

    instruments = sorted(set(pf.index.get_level_values("instrument")))
    logger.info("instruments(from parquet): %d", len(instruments))

    logger.info("=== 2) load qlib label & lgb_short_cycle ===")
    label = _normalize_index(_load_label(args.start, args.end, instruments).to_frame()).iloc[:, 0]
    lgb_df = _normalize_index(_load_lgb_short_cycle(args.start, args.end, instruments))
    logger.info("label shape=%d, lgb_short_cycle shape=%s", label.notna().sum(), lgb_df.shape)

    # ── (a) 共线性矩阵 ───────────────────────────────────────────────
    logger.info("=== 3) corr: rd-rd & rd-vs-lgb ===")
    rd_aligned = pf.reindex(label.index)
    corr_rd = rd_aligned.corr(method="spearman")
    corr_rd.to_csv(OUT_CSV_CORR)

    cross = pd.DataFrame(index=rd_cols, columns=lgb_df.columns, dtype=float)
    for r in rd_cols:
        s_r = rd_aligned[r]
        for q in lgb_df.columns:
            s_q = lgb_df[q].reindex(s_r.index)
            df = pd.DataFrame({"a": s_r, "b": s_q}).dropna()
            if len(df) < 500:
                cross.loc[r, q] = np.nan
            else:
                cross.loc[r, q] = df["a"].rank().corr(df["b"].rank())
    cross.to_csv(OUT_CSV_CROSS)

    max_abs_corr_vs_lgb = cross.abs().max(axis=1)
    argmax_lgb_col = cross.abs().idxmax(axis=1)

    # ── (b) 滚动 IC（in-sample 段 + 单独 OOS 段） ──────────────────────
    logger.info("=== 4) rolling OOS IC ===")
    cutoff = args.train_cutoff
    in_segs = _make_segments(args.start, cutoff, args.n_segments)
    oos_seg_end = args.end
    oos_segs = [(pd.Timestamp(cutoff) + pd.Timedelta(days=1), pd.Timestamp(oos_seg_end))]
    oos_segs = [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in oos_segs]

    rows_oos = []
    for c in rd_cols:
        s = rd_aligned[c]
        in_ics = _rolling_ic(s, label, in_segs)
        ic_in_vals = [r["ic"] for r in in_ics if not np.isnan(r["ic"])]
        ic_mean = float(np.mean(ic_in_vals)) if ic_in_vals else np.nan
        ic_std = float(np.std(ic_in_vals)) if ic_in_vals else np.nan
        ic_ir = ic_mean / ic_std if ic_std and ic_std > 1e-9 else np.nan
        ic_pos_ratio = float(np.mean([1.0 if v > 0 else 0.0 for v in ic_in_vals])) if ic_in_vals else np.nan
        oos_ic = _rolling_ic(s, label, oos_segs)[0]["ic"]
        rows_oos.append({
            "factor": c,
            "ic_in_mean": ic_mean,
            "ic_in_std": ic_std,
            "ic_in_ir": ic_ir,
            "ic_in_pos_ratio": ic_pos_ratio,
            "ic_oos": oos_ic,
            "ic_in_seg_jsons": json.dumps(in_ics, ensure_ascii=False),
        })
    df_oos = pd.DataFrame(rows_oos).set_index("factor")
    df_oos.to_csv(OUT_CSV_OOS)

    # ── (c) LGB importance（如可用） ──────────────────────────────────
    logger.info("=== 5) LGB feature_importance (best-effort) ===")
    importances = _maybe_load_lgb_importance(_ROOT / args.lgb_model_dir)
    if importances:
        rd_imp = {c: importances.get(c, 0.0) for c in rd_cols}
        rd_imp_total = float(sum(rd_imp.values()))
        logger.info("LGB total importance from RD cols = %.2f%%", rd_imp_total * 100)
    else:
        rd_imp = {c: np.nan for c in rd_cols}
        rd_imp_total = np.nan
        logger.warning("无法读取 LGB 模型 importance（缺 _meta.json 或 lightgbm 不可用），跳过")

    # ── (d) 决策：保留 / 淘汰 ──────────────────────────────────────
    logger.info("=== 6) decision: keep / drop ===")
    decision_rows = []
    for c in rd_cols:
        ir = df_oos.loc[c, "ic_in_ir"]
        oos_ic = df_oos.loc[c, "ic_oos"]
        in_mean = df_oos.loc[c, "ic_in_mean"]
        cor = float(max_abs_corr_vs_lgb.get(c, np.nan))
        cor_col = argmax_lgb_col.get(c, "")
        sign_ok = (
            not np.isnan(in_mean)
            and not np.isnan(oos_ic)
            and (in_mean * oos_ic > 0 or abs(oos_ic) < 1e-6)
        )
        ir_ok = not np.isnan(ir) and ir >= args.ic_ir_threshold
        cor_ok = np.isnan(cor) or cor <= args.max_corr_vs_lgb
        keep = bool(ir_ok and sign_ok and cor_ok)
        reason = []
        if not ir_ok: reason.append(f"IC_IR={ir:.2f}<{args.ic_ir_threshold}")
        if not sign_ok: reason.append(f"OOS sign flip(in={in_mean:.3f},oos={oos_ic:.3f})")
        if not cor_ok: reason.append(f"|corr_vs_lgb|={cor:.2f}>{args.max_corr_vs_lgb}({cor_col})")
        decision_rows.append({
            "factor": c,
            "ic_in_mean": round(in_mean, 4) if not np.isnan(in_mean) else None,
            "ic_in_ir": round(ir, 3) if not np.isnan(ir) else None,
            "ic_in_pos_ratio": round(df_oos.loc[c, "ic_in_pos_ratio"], 2) if not np.isnan(df_oos.loc[c, "ic_in_pos_ratio"]) else None,
            "ic_oos": round(oos_ic, 4) if not np.isnan(oos_ic) else None,
            "max_abs_corr_vs_lgb": round(cor, 3) if not np.isnan(cor) else None,
            "argmax_lgb_col": cor_col,
            "lgb_gain_importance": round(rd_imp.get(c, 0.0), 4) if not np.isnan(rd_imp.get(c, np.nan)) else None,
            "keep": keep,
            "drop_reason": "; ".join(reason) if reason else "",
        })
    df_dec = pd.DataFrame(decision_rows).set_index("factor")
    df_dec.to_csv(OUT_CSV_KEEP)

    keep_list = [c for c, row in df_dec.iterrows() if row["keep"]]
    drop_list = [c for c, row in df_dec.iterrows() if not row["keep"]]

    # ── (e) Markdown 报告 ────────────────────────────────────────────
    lines = [
        "# P0-1 因子诊断报告",
        "",
        f"- 数据：`{PARQUET.name}` × {len(rd_cols)} 列 × {len(instruments)} 只",
        f"- 时间：`{args.start}` ~ `{args.end}`，OOS cutoff = `{args.train_cutoff}`，in-sample 段数 = {args.n_segments}",
        f"- 阈值：`IC_IR ≥ {args.ic_ir_threshold}`，`|corr_vs_lgb| ≤ {args.max_corr_vs_lgb}`",
        f"- LGB importance 来源：`{args.lgb_model_dir}`（{'已读取' if importances else '未读取'}）",
        "",
        "## 1. 决策汇总",
        "",
        f"- ✅ 建议保留 **{len(keep_list)}** 个：`{', '.join(keep_list) or '（空）'}`",
        f"- ❌ 建议淘汰 **{len(drop_list)}** 个：`{', '.join(drop_list) or '（空）'}`",
        "",
        "## 2. 因子详表（按 IC_IR 降序）",
        "",
    ]
    df_show = df_dec.copy()
    df_show["_ir"] = df_show["ic_in_ir"].fillna(-9.99)
    df_show = df_show.sort_values("_ir", ascending=False).drop(columns="_ir")
    lines.append(df_show.to_markdown())
    lines.extend([
        "",
        "## 3. 与 lgb_short_cycle 的最大 |corr| 排行（潜在共线列）",
        "",
        max_abs_corr_vs_lgb.sort_values(ascending=False).round(3).to_frame("max_abs_corr_vs_lgb").to_markdown(),
        "",
        "## 4. 配套 CSV",
        "",
        f"- 共线性矩阵（RD ↔ RD）：`{OUT_CSV_CORR.relative_to(_ROOT).as_posix()}`",
        f"- 共线性矩阵（RD ↔ lgb_short_cycle）：`{OUT_CSV_CROSS.relative_to(_ROOT).as_posix()}`",
        f"- 滚动 OOS IC：`{OUT_CSV_OOS.relative_to(_ROOT).as_posix()}`",
        f"- 决策表：`{OUT_CSV_KEEP.relative_to(_ROOT).as_posix()}`",
        "",
        "## 5. 建议下一步",
        "",
        "把以下命令喂给 `refresh_rdagent_parquet.py`（B 模式，严格筛选）：",
        "",
        "```powershell",
        f"python scripts/refresh_rdagent_parquet.py --start {args.start} --end {args.end} `",
        "  --instruments 'csi500,csi300' `",
        f"  --oos-cutoff {args.train_cutoff} --ic-ir-threshold {args.ic_ir_threshold} `",
        f"  --max-corr-vs-lgb {args.max_corr_vs_lgb} --max-factors 8",
        "```",
    ])
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("=== DONE ===")
    logger.info("报告: %s", OUT_REPORT.relative_to(_ROOT).as_posix())
    logger.info("✅ keep=%d  ❌ drop=%d", len(keep_list), len(drop_list))


if __name__ == "__main__":
    main()
