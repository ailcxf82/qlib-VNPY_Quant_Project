"""
合并 TOP-N workspace 的因子数据，生成覆盖最完整时间区间的因子集。

用法：
    python scripts/merge_top_factors.py [--top_n 5] [--output git_ignore_folder/merged_factors.parquet]

逻辑：
  1. 扫描所有保留 workspace，读取 qlib_res.csv 获得 ICIR 排名
  2. 按 ICIR 从高到低逐个合并 combined_factors_df.parquet（outer join on datetime+instrument）
  3. 只保留出现在 TOP-n 中且 parquet 日期覆盖最完整的版本（同名因子取覆盖范围最大的 workspace）
  4. 输出合并后的 parquet，列格式为 ('feature', col_name) MultiIndex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"
DEFAULT_OUT = ROOT / "git_ignore_folder" / "merged_factors.parquet"


def _load_workspace_meta(ws_dir: Path) -> dict | None:
    csv_path = ws_dir / "qlib_res.csv"
    pq_path = ws_dir / "combined_factors_df.parquet"
    if not csv_path.exists() or not pq_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df.columns = ["value"]
        m = df["value"].to_dict()
        icir = m.get("ICIR")
        ic = m.get("IC")
        if icir is None or pd.isna(float(icir)):
            return None
        return {
            "ws": ws_dir.name,
            "path": ws_dir,
            "pq_path": pq_path,
            "ICIR": float(icir),
            "IC": float(ic) if ic else None,
        }
    except Exception:
        return None


def merge_top_factors(top_n: int = 5, output: Path = DEFAULT_OUT) -> Path:
    # 1. 收集所有 workspace 的指标
    metas = []
    for ws_dir in WS_BASE.iterdir():
        if not ws_dir.is_dir():
            continue
        meta = _load_workspace_meta(ws_dir)
        if meta:
            metas.append(meta)

    metas.sort(key=lambda x: x["ICIR"], reverse=True)
    selected = metas[:top_n]

    print(f"=== 选中 TOP-{top_n} 工作区 ===")
    for i, m in enumerate(selected, 1):
        print(f"  #{i} ICIR={m['ICIR']:.3f}  ws={m['ws'][:12]}...")

    # 2. 逐个加载 parquet，按因子名去重（保留日期范围最大的版本）
    factor_frames: dict[str, pd.DataFrame] = {}  # col_name -> single-col DataFrame

    for meta in selected:
        try:
            df = pd.read_parquet(meta["pq_path"])
        except Exception as e:
            print(f"  [跳过] {meta['ws'][:12]} 读取失败: {e}")
            continue

        # 确保统一为 MultiIndex ('feature', col_name)
        if df.columns.nlevels == 1:
            df.columns = pd.MultiIndex.from_product([["feature"], df.columns])
        df.index.names = ["datetime", "instrument"]

        dts = df.index.get_level_values("datetime")
        date_range = (dts.min().date(), dts.max().date())

        for col in df.columns.get_level_values(-1):
            col_df = df[[("feature", col)]].copy()
            if col not in factor_frames:
                factor_frames[col] = col_df
                print(f"    + 新增因子 [{col}]  来自 ws={meta['ws'][:12]}  日期={date_range}")
            else:
                # 比较哪个覆盖范围更大（用行数判断）
                existing_count = factor_frames[col].notna().sum().sum()
                new_count = col_df.notna().sum().sum()
                if new_count > existing_count:
                    factor_frames[col] = col_df
                    print(f"    ↑ 更新因子 [{col}]  更大覆盖 ({new_count}>{existing_count}行)  日期={date_range}")

    if not factor_frames:
        raise RuntimeError("没有可用的因子数据，请检查 workspace 是否存在。")

    # 3. 合并所有因子（outer join）
    print(f"\n合并 {len(factor_frames)} 个因子...")
    merged = None
    for col_name, df_col in factor_frames.items():
        if merged is None:
            merged = df_col
        else:
            merged = merged.join(df_col, how="outer")

    merged = merged.sort_index()

    # 统计
    dts = merged.index.get_level_values("datetime")
    print(f"\n合并结果：")
    print(f"  因子数量   : {len(factor_frames)}")
    print(f"  日期范围   : {dts.min().date()} → {dts.max().date()}")
    print(f"  总行数     : {len(merged):,}")
    print(f"  因子名称   : {list(merged.columns.get_level_values(-1))}")

    # 覆盖率统计
    coverage = merged.notna().sum() / len(merged) * 100
    print("\n  各因子覆盖率：")
    for col, pct in coverage.items():
        print(f"    {col[1]:40s}: {pct:.1f}%")

    # 4. 保存
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"\n已保存至: {output}  ({size_mb:.1f} MB)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 TOP-N 因子 workspace 的 parquet")
    parser.add_argument("--top_n", type=int, default=5, help="合并前 N 个最优 workspace（按 ICIR）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="输出 parquet 路径")
    args = parser.parse_args()
    merge_top_factors(top_n=args.top_n, output=args.output)
