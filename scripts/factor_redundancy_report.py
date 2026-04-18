"""
因子去冗余分析报告（只读 + 只出报告，不修改任何 YAML）。

读取 config/data.yaml 中 active_feature_sets 的所有 Qlib 表达式，
拉取原始面板与 label，计算：
  - 每因子：Rank IC（整段）、滚动 20 日 IC 的 ICIR、缺失率
  - 因子间：按日截面 Rank Spearman 相关的时间均值矩阵
  - 层次聚类：距离 = 1 - |corr|，阈值默认 0.15（等价于 |corr|>=0.85 算冗余）
  - 每簇内按 |ICIR| 排序，首位推荐保留，其余标记为 redundant

输出：
  - data/factor_analysis/redundancy_report_{ts}.csv
  - data/factor_analysis/redundancy_report_{ts}.md
  - data/factor_analysis/corr_heatmap_{ts}.png（若 matplotlib 可用）
  - data/factor_analysis/redundancy_report_LATEST.md（软拷贝，便于最新查看）

Usage:
  python scripts/factor_redundancy_report.py \
      --config config/data.yaml \
      --corr-threshold 0.85 \
      --out-dir data/factor_analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("factor_redundancy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="因子去冗余分析报告")
    p.add_argument("--config", type=str, default="config/data.yaml",
                   help="数据配置文件路径（包含 feature_sets / active_feature_sets / label 等）")
    p.add_argument("--start", type=str, default=None, help="起始日期（默认读配置）")
    p.add_argument("--end", type=str, default=None, help="结束日期（默认读配置）")
    p.add_argument("--instruments", type=str, default=None,
                   help="股票池（逗号分隔，默认读配置，例如 csi500 或 csi300,csi500）")
    p.add_argument("--corr-threshold", type=float, default=0.85,
                   help="冗余相关性阈值（|corr| >= 阈值视为同簇），默认 0.85")
    p.add_argument("--rolling-window", type=int, default=20,
                   help="滚动 IC 窗口（交易日），用于计算 ICIR")
    p.add_argument("--max-instruments", type=int, default=0,
                   help="可选：限制股票数量以加速（0 表示不限）")
    p.add_argument("--out-dir", type=str, default="data/factor_analysis")
    p.add_argument("--skip-heatmap", action="store_true", help="跳过热力图绘制")
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def init_qlib(provider_uri: str, region: str = "cn") -> None:
    try:
        import qlib
    except Exception as e:
        logger.error("qlib 未安装或不可用: %s", e)
        raise
    if hasattr(qlib, "is_initialized"):
        try:
            if qlib.is_initialized():
                return
        except Exception:
            pass
    qlib.init(provider_uri=provider_uri, region=region)


def parse_instruments(v: Any) -> Any:
    """支持 'csi500' 或 'csi500,csi300' 或 list，返回 qlib 可接受的形式。"""
    from qlib.data import D
    if v is None:
        raise ValueError("instruments 为空")
    if isinstance(v, list):
        tokens = [str(x).strip() for x in v if str(x).strip()]
    else:
        tokens = [t.strip() for t in str(v).split(",") if t.strip()]
    if len(tokens) == 1:
        return D.instruments(market=tokens[0])
    # 合并多市场
    merged: List[str] = []
    for t in tokens:
        try:
            inst = D.list_instruments(instruments=D.instruments(market=t), as_list=True)
            merged.extend(inst)
        except Exception as e:
            logger.warning("解析股票池 %s 失败: %s", t, e)
    merged = sorted(set(merged))
    return merged or tokens


def fetch_panel(
    factors: List[Tuple[str, str, str]],  # [(feature_set, expr, alias), ...]
    label_expr: str,
    instruments: Any,
    start: str,
    end: str,
    freq: str = "day",
) -> Tuple[pd.DataFrame, pd.Series]:
    from qlib.data import D
    exprs = [f[1] for f in factors]
    aliases = [f[2] for f in factors]
    logger.info("拉取 %d 个因子表达式 + 1 个 label 表达式 ...", len(exprs))
    feat = D.features(instruments=instruments, fields=exprs, start_time=start, end_time=end, freq=freq)
    feat.columns = aliases
    lab_df = D.features(instruments=instruments, fields=[label_expr], start_time=start, end_time=end, freq=freq)
    lab = lab_df.iloc[:, 0].rename("label")
    return feat, lab


def _ranked_by_date(s: pd.Series) -> pd.Series:
    """按日截面做 rank（pct）。输入 index 为 MultiIndex[datetime, instrument]。"""
    if s.empty:
        return s
    return s.groupby(level="datetime", group_keys=False).rank(pct=True)


def compute_ic_metrics(
    factor_df: pd.DataFrame,
    label: pd.Series,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """每因子：Rank IC（整段）、滚动 IC 的 ICIR、缺失率。"""
    label_rk = _ranked_by_date(label)
    rows: List[Dict[str, Any]] = []
    for col in factor_df.columns:
        s = factor_df[col]
        nan_ratio = float(s.isna().mean())
        s_rk = _ranked_by_date(s)
        aligned = pd.concat([s_rk.rename("f"), label_rk.rename("y")], axis=1).dropna()
        if aligned.empty:
            rows.append({"factor": col, "ic": np.nan, "icir": np.nan,
                         "n_valid_days": 0, "nan_ratio": nan_ratio})
            continue
        daily_corr = (
            aligned.groupby(level="datetime")
            .apply(lambda g: g["f"].corr(g["y"]) if len(g) > 1 else np.nan)
        )
        daily_corr = daily_corr.dropna()
        n_days = int(len(daily_corr))
        ic_overall = float(daily_corr.mean()) if n_days else np.nan
        if rolling_window and n_days >= rolling_window:
            rolling_ic = daily_corr.rolling(rolling_window).mean().dropna()
            if len(rolling_ic) > 1 and rolling_ic.std(ddof=0) > 1e-12:
                icir = float(rolling_ic.mean() / (rolling_ic.std(ddof=0) + 1e-12))
            else:
                icir = float("nan")
        else:
            icir = (
                float(daily_corr.mean() / (daily_corr.std(ddof=0) + 1e-12))
                if n_days > 1 else float("nan")
            )
        rows.append({
            "factor": col,
            "ic": ic_overall,
            "icir": icir,
            "n_valid_days": n_days,
            "nan_ratio": nan_ratio,
        })
    return pd.DataFrame(rows).set_index("factor")


def compute_corr_matrix(factor_df: pd.DataFrame) -> pd.DataFrame:
    """按日截面 Rank Spearman 相关的时间均值矩阵。

    实现方式：对每个因子先做按日 rank(pct)，再在全 panel 上直接 pearson 相关。
    这等价于"按日截面 Spearman 的跨期均值"（近似，且比逐日计算快很多）。
    """
    ranks = factor_df.apply(_ranked_by_date)
    ranks = ranks.dropna(how="all")
    corr = ranks.corr(method="pearson")
    return corr


def hierarchical_cluster(
    corr: pd.DataFrame,
    corr_threshold: float,
) -> pd.Series:
    """基于 (1 - |corr|) 做层次聚类，距离阈值 = 1 - corr_threshold。

    稳健性：若某因子在 corr 中整行全 NaN（通常因为原始表达式在当前
    时间段+股票池下完全无数据），则将其单独标记为 cluster_id=-1（无效因子），
    对其余有效子矩阵做正常聚类；避免 scipy.linkage 因 NaN 直接抛错。
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    labels = list(corr.index)
    if len(labels) <= 1:
        return pd.Series([1] * len(labels), index=labels, name="cluster_id")

    # 识别"除对角位外全 NaN"的因子。用非对角 NaN 计数 >= n-1 直接判定，
    # 避免在 isnan 布尔矩阵上动对角位引起的逻辑反转（见历史 bug）。
    corr_vals = corr.values
    n = corr_vals.shape[0]
    isnan_mat = np.isnan(corr_vals)
    diag_nan = np.diagonal(isnan_mat).astype(int)
    off_diag_nan_per_row = isnan_mat.sum(axis=1) - diag_nan
    off_diag_nan_per_col = isnan_mat.sum(axis=0) - diag_nan
    threshold = max(1, n - 1)
    invalid_mask = (off_diag_nan_per_row >= threshold) | (off_diag_nan_per_col >= threshold)
    invalid_labels = [labels[i] for i, f in enumerate(invalid_mask.tolist()) if f]
    valid_labels = [labels[i] for i, f in enumerate(invalid_mask.tolist()) if not f]

    result = pd.Series(index=labels, dtype="Int64", name="cluster_id")
    for lab in invalid_labels:
        result.loc[lab] = -1

    if len(valid_labels) <= 1:
        # 有效因子不够做聚类；剩下的全部并入簇 1
        for lab in valid_labels:
            result.loc[lab] = 1
        return result

    corr_valid = corr.loc[valid_labels, valid_labels]
    dist = 1.0 - corr_valid.abs().values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, None)
    try:
        condensed = squareform(dist, checks=False)
    except Exception as e:
        logger.warning("相关矩阵 squareform 失败: %s，退化为单一簇", e)
        for lab in valid_labels:
            result.loc[lab] = 1
        return result
    Z = linkage(condensed, method="average")
    thresh = max(0.0, min(1.0, 1.0 - corr_threshold))
    clusters = fcluster(Z, t=thresh, criterion="distance")
    for lab, cid in zip(valid_labels, clusters):
        result.loc[lab] = int(cid)
    return result


def build_factor_list(
    feature_sets: Dict[str, List[str]],
    active_sets: List[str],
) -> List[Tuple[str, str, str]]:
    """返回 [(feature_set, expr, alias)]，alias 为 F{idx} 以防表达式字符出现在列名。"""
    seen: Dict[str, Tuple[str, str]] = {}  # expr -> (feature_set, alias)
    out: List[Tuple[str, str, str]] = []
    idx = 0
    for fs in active_sets:
        exprs = feature_sets.get(fs, []) or []
        for e in exprs:
            expr = str(e).strip()
            if not expr or expr in seen:
                continue
            alias = f"F{idx:03d}"
            seen[expr] = (fs, alias)
            out.append((fs, expr, alias))
            idx += 1
    return out


def render_markdown_report(
    report_df: pd.DataFrame,
    corr_threshold: float,
    out_md: Path,
    meta: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# 因子去冗余分析报告\n")
    lines.append(f"- 生成时间：{meta.get('generated_at','')}")
    lines.append(f"- 配置文件：`{meta.get('config_path','')}`")
    lines.append(f"- 时间区间：{meta.get('start','')} ~ {meta.get('end','')}")
    lines.append(f"- 股票池：{meta.get('instruments','')}")
    lines.append(f"- 标签表达式：`{meta.get('label','')}`")
    lines.append(f"- 相关性阈值：|corr| >= **{corr_threshold}** 视为同簇")
    lines.append(f"- 因子数：{len(report_df)}，冗余簇数：{int(report_df['cluster_id'].nunique())}\n")
    lines.append("> 说明：本报告不修改 `config/data.yaml`。末尾给出可直接粘贴的推荐精简集 YAML 片段，采纳与否由人工决定。\n")

    lines.append("## 一、Top 20 因子（按 |ICIR| 排序）\n")
    top20 = (
        report_df.assign(abs_icir=lambda d: d["icir"].abs())
        .sort_values("abs_icir", ascending=False)
        .head(20)
    )
    lines.append("| factor | feature_set | expression | ic | icir | nan_ratio | cluster_id | recommend_keep |")
    lines.append("|---|---|---|---:|---:|---:|---:|:---:|")
    for _, r in top20.iterrows():
        lines.append(
            f"| `{r.name}` | {r['feature_set']} | `{r['expression']}` | "
            f"{r['ic']:.4f} | {r['icir']:.4f} | {r['nan_ratio']*100:.1f}% | "
            f"{int(r['cluster_id'])} | {'是' if bool(r['recommend_keep']) else ''} |"
        )
    lines.append("")

    invalid_df = report_df[report_df["cluster_id"] == -1]
    if len(invalid_df) > 0:
        lines.append("## 一点五、数据缺失/常数因子（cluster_id=-1）\n")
        lines.append("> 这些因子在当前时间段+股票池下无有效数据或常数，已被自动排除聚类，`recommend_keep=False`。")
        lines.append("> 建议：核对 Qlib 数据源是否缺失对应字段，或确认表达式本身是否退化。\n")
        lines.append("| factor | feature_set | expression | nan_ratio |")
        lines.append("|---|---|---|---:|")
        for _, r in invalid_df.iterrows():
            lines.append(
                f"| `{r.name}` | {r['feature_set']} | `{r['expression']}` | {r['nan_ratio']*100:.1f}% |"
            )
        lines.append("")

    lines.append("## 二、冗余簇（每簇 >= 2 个因子）\n")
    for cid, sub in report_df.groupby("cluster_id"):
        if cid == -1:
            continue
        if len(sub) < 2:
            continue
        sub_sorted = sub.assign(abs_icir=lambda d: d["icir"].abs()).sort_values("abs_icir", ascending=False)
        lines.append(f"### 簇 #{int(cid)} ({len(sub_sorted)} 个因子)\n")
        lines.append("| factor | feature_set | expression | ic | icir | 簇内排名 | recommend_keep |")
        lines.append("|---|---|---|---:|---:|---:|:---:|")
        for rank_idx, (_, r) in enumerate(sub_sorted.iterrows(), start=1):
            lines.append(
                f"| `{r.name}` | {r['feature_set']} | `{r['expression']}` | "
                f"{r['ic']:.4f} | {r['icir']:.4f} | {rank_idx} | "
                f"{'是' if bool(r['recommend_keep']) else ''} |"
            )
        lines.append("")

    lines.append("## 三、推荐精简集（建议粘贴到 `config/data.yaml`）\n")
    lines.append("> 仅保留每簇 |ICIR| 最高的代表因子。如需按 feature_set 维持分组，按原表达式在对应集合中保留即可。\n")
    kept = report_df[report_df["recommend_keep"]].copy()
    kept_by_set: Dict[str, List[str]] = {}
    for _, r in kept.iterrows():
        kept_by_set.setdefault(r["feature_set"], []).append(r["expression"])
    lines.append("```yaml")
    lines.append("feature_sets:")
    for fs_name, exprs in kept_by_set.items():
        lines.append(f"  {fs_name}_lean:")
        for e in exprs:
            lines.append(f"    - {e}")
    lines.append("active_feature_sets:")
    for fs_name in kept_by_set.keys():
        lines.append(f"  - {fs_name}_lean")
    lines.append("```")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib 不可用，跳过热力图: %s", e)
        return
    n = len(corr)
    if n == 0:
        return
    fig_w = max(6.0, min(24.0, 0.25 * n + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.9))
    im = ax.imshow(corr.abs().values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(f"|Rank Corr| Heatmap ({n} factors)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_yaml(cfg_path)
    qlib_cfg = cfg.get("qlib", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    provider_uri = qlib_cfg.get("provider_uri")
    if not provider_uri:
        raise ValueError("配置中 qlib.provider_uri 必须设置")
    init_qlib(provider_uri, region=qlib_cfg.get("region", "cn"))

    active_sets = list(data_cfg.get("active_feature_sets") or [])
    feature_sets = dict(data_cfg.get("feature_sets") or {})
    factors = build_factor_list(feature_sets, active_sets)
    if not factors:
        raise ValueError("未在 active_feature_sets 中找到任何因子")

    start = args.start or data_cfg.get("start_time")
    end = args.end or data_cfg.get("end_time")
    instruments_raw = args.instruments or data_cfg.get("instruments")
    label_expr = data_cfg.get("label")
    if not label_expr:
        raise ValueError("配置缺少 data.label")

    instruments = parse_instruments(instruments_raw)
    if args.max_instruments and isinstance(instruments, list) and len(instruments) > args.max_instruments:
        logger.info("instruments 超过 --max-instruments=%d，截断", args.max_instruments)
        instruments = instruments[: args.max_instruments]

    feat_panel, label = fetch_panel(factors, label_expr, instruments, start, end)
    logger.info("feature panel shape=%s, label len=%d", feat_panel.shape, len(label))

    ic_df = compute_ic_metrics(feat_panel, label, rolling_window=args.rolling_window)
    logger.info("IC 指标计算完成，%d 个因子", len(ic_df))

    logger.info("相关矩阵计算 ...")
    corr = compute_corr_matrix(feat_panel)
    logger.info("层次聚类 ...")
    cluster_ids = hierarchical_cluster(corr, corr_threshold=args.corr_threshold)

    alias_to_meta = {alias: (fs, expr) for (fs, expr, alias) in factors}
    report_df = ic_df.copy()
    report_df["feature_set"] = [alias_to_meta.get(a, ("", ""))[0] for a in report_df.index]
    report_df["expression"] = [alias_to_meta.get(a, ("", ""))[1] for a in report_df.index]
    report_df = report_df.join(cluster_ids, how="left")

    report_df["abs_icir"] = report_df["icir"].abs()
    report_df["cluster_rank"] = (
        report_df.groupby("cluster_id")["abs_icir"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    report_df["recommend_keep"] = report_df["cluster_rank"] == 1
    # 无效因子（cluster_id=-1，即数据缺失/常数）永远不推荐保留
    report_df.loc[report_df["cluster_id"] == -1, "recommend_keep"] = False
    report_df = report_df.drop(columns=["abs_icir"]).reset_index().rename(columns={"index": "factor"})

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"redundancy_report_{ts}.csv"
    md_path = out_dir / f"redundancy_report_{ts}.md"
    heatmap_path = out_dir / f"corr_heatmap_{ts}.png"
    latest_md_path = out_dir / "redundancy_report_LATEST.md"

    out_cols = ["factor", "feature_set", "expression", "ic", "icir",
                "n_valid_days", "nan_ratio", "cluster_id", "cluster_rank", "recommend_keep"]
    report_df = report_df[out_cols]
    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("CSV 输出: %s", csv_path)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(cfg_path),
        "start": start,
        "end": end,
        "instruments": instruments_raw,
        "label": label_expr,
    }
    render_markdown_report(
        report_df.set_index("factor"),
        corr_threshold=args.corr_threshold,
        out_md=md_path,
        meta=meta,
    )
    try:
        latest_md_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        logger.warning("写 LATEST md 失败: %s", e)
    logger.info("MD 输出: %s", md_path)

    if not args.skip_heatmap:
        try:
            plot_heatmap(corr, heatmap_path)
            logger.info("热力图: %s", heatmap_path)
        except Exception as e:
            logger.warning("热力图生成失败: %s", e)

    summary_json = {
        "generated_at": meta["generated_at"],
        "n_factors": int(len(report_df)),
        "n_clusters": int(report_df["cluster_id"].nunique()),
        "recommended_keep": int(report_df["recommend_keep"].sum()),
        "corr_threshold": args.corr_threshold,
        "csv": str(csv_path),
        "md": str(md_path),
    }
    (out_dir / f"redundancy_report_{ts}.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("完成。推荐保留: %d / %d 因子（簇数 %d）",
                summary_json["recommended_keep"], summary_json["n_factors"], summary_json["n_clusters"])


if __name__ == "__main__":
    main()
