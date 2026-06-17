"""离线导出 **当前 production ensemble** 在 OOS 窗的预测 parquet。

本脚本是 L2 ``marginal`` / ``marginal_training`` check 的前置步骤。它从训练端产出的
OOF / 推理 parquet 里抽出 "最终预测"（通常是 meta-stacking 之后的 blended score），
规范化 schema 后写入 ``factor_validation/data/`` 目录供 orchestrator 注入
``CheckContext.baseline_prediction_parquet``。

输入格式（支持三种 schema）：

1. **MultiIndex parquet**：``index=(datetime, instrument)``，列含目标预测列（默认候选名
   按顺序：``pred_ensemble`` / ``final`` / ``prediction`` / ``score``）。
2. **Flat columns parquet**：列里含 ``date``/``datetime`` 和 ``code``/``instrument``，
   自动重建 MultiIndex。
3. 用户用 ``--date-col`` / ``--instrument-col`` / ``--prediction-col`` 显式指定。

输出格式（强约定）：

* 路径：``--out``，默认 ``factor_validation/data/baseline_predictions_<tag>.parquet``。
* 结构：``MultiIndex=(datetime, instrument)``，单列 ``prediction: float64``。

**典型用法**::

    # 从 meta_oof 文件抽取 ensemble 预测
    python -m scripts.validate.prepare_baseline_prediction \\
        --src data/oof/ensemble_v1_20260101_20260407_meta_oof.parquet \\
        --prediction-col pred_ensemble \\
        --start 2025-01-01 --end 2026-04-07 \\
        --out factor_validation/data/baseline_predictions_ensemble_v1.parquet

**与 profile 对齐**：``factor_validation/profiles/{default,strict}.yaml`` 里
``data_sources.baseline_prediction_parquet`` 必须指向本脚本输出。

**注意事项**：

* 本脚本**只做转换**，不负责训练。"当前 production ensemble 的 OOS 预测" 需要先由
  ``run_train.py`` 或推理脚本产出对应 parquet。
* 如果训练端尚未产出 OOS ensemble 预测，可用 ``--synthetic-zero`` 模式生成一个
  **全零**的 baseline parquet 作占位（marginal 退化为 naive IC check；marginal_training
  仍有意义：LGB 会学到 ensemble 没见过的一切）。生产环境请**不要**长期使用 zero
  baseline，它会把 marginal 变成 ic 的重复打分。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


logger = logging.getLogger("prepare_baseline_prediction")

DEFAULT_OUT_DIR = _PROJECT_ROOT / "factor_validation" / "data"
_PRED_COL_CANDIDATES = [
    "pred_ensemble",
    "final",
    "prediction",
    "score",
    "y_hat",
    "pred_blend",
]
_DATE_COL_CANDIDATES = ["datetime", "date", "trade_date", "dt"]
_INST_COL_CANDIDATES = ["instrument", "code", "symbol", "stock_code", "ts_code"]


# ------------------------------------------------------------------ core


def _normalize_dataframe(
    df: pd.DataFrame,
    *,
    date_col: str | None,
    instrument_col: str | None,
    prediction_col: str | None,
) -> pd.DataFrame:
    """把输入 df 规范成 MultiIndex(datetime, instrument) + 单列 'prediction'。"""
    # 1) 确定 prediction 列
    if prediction_col is None:
        cand = [c for c in _PRED_COL_CANDIDATES if c in df.columns]
        if not cand:
            raise ValueError(
                f"未找到预测列；请 --prediction-col 显式指定；现有列: {list(df.columns)}"
            )
        prediction_col = cand[0]
        logger.info("自动选用 prediction_col=%r", prediction_col)
    elif prediction_col not in df.columns:
        raise ValueError(
            f"预测列 {prediction_col!r} 不在输入里；现有列: {list(df.columns)}"
        )

    # 2) 已是 (datetime, instrument) MultiIndex？
    if isinstance(df.index, pd.MultiIndex) and set(df.index.names) == {
        "datetime",
        "instrument",
    }:
        out = df[[prediction_col]].copy()
        if df.index.names != ["datetime", "instrument"]:
            out = out.reorder_levels(["datetime", "instrument"])
    else:
        if date_col is None:
            cand = [c for c in _DATE_COL_CANDIDATES if c in df.columns]
            if not cand:
                raise ValueError(
                    "无法自动识别 date 列；请 --date-col 指定；"
                    f"现有列: {list(df.columns)}"
                )
            date_col = cand[0]
        if instrument_col is None:
            cand = [c for c in _INST_COL_CANDIDATES if c in df.columns]
            if not cand:
                raise ValueError(
                    "无法自动识别 instrument 列；请 --instrument-col 指定；"
                    f"现有列: {list(df.columns)}"
                )
            instrument_col = cand[0]
        logger.info(
            "从 flat columns 重建 MultiIndex: date=%r instrument=%r prediction=%r",
            date_col,
            instrument_col,
            prediction_col,
        )
        flat = df[[date_col, instrument_col, prediction_col]].copy()
        flat = flat.rename(
            columns={
                date_col: "datetime",
                instrument_col: "instrument",
                prediction_col: "prediction",
            }
        )
        flat["datetime"] = pd.to_datetime(flat["datetime"])
        out = flat.set_index(["datetime", "instrument"])[["prediction"]]
        prediction_col = "prediction"  # 已重命名

    # 3) 统一列名为 'prediction' + 类型
    if prediction_col != "prediction":
        out = out.rename(columns={prediction_col: "prediction"})
    out["prediction"] = out["prediction"].astype("float64")
    out = out.sort_index()
    # dup 清理（OOF 可能跨 fold 有重复 (date, instrument)；保留最后一次覆盖）
    if out.index.duplicated().any():
        n_dup = int(out.index.duplicated().sum())
        logger.warning(
            "检测到 (datetime, instrument) 重复 %d 条，保留最后出现", n_dup
        )
        out = out[~out.index.duplicated(keep="last")]
    return out


def _slice_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    dt = df.index.get_level_values("datetime")
    st = pd.Timestamp(start) if start else dt.min()
    en = (
        pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        if end
        else dt.max()
    )
    mask = (dt >= st) & (dt <= en)
    return df.loc[mask]


def prepare_baseline(
    *,
    src: Path | None,
    out_path: Path,
    prediction_col: str | None,
    date_col: str | None,
    instrument_col: str | None,
    start: str | None,
    end: str | None,
    synthetic_zero_from: Path | None = None,
) -> Path:
    """主函数：读 src → 规范化 → 切窗 → 写 out_path。

    * 非 synthetic 模式：``src`` 必填，从中抽取预测。
    * ``synthetic_zero_from`` 模式：用 label_parquet 的 (datetime, instrument) 索引
      作全零 baseline。
    """
    if synthetic_zero_from is not None:
        logger.warning(
            "使用 --synthetic-zero 占位 baseline（全零）；仅适合启动阶段"
        )
        lbl = pd.read_parquet(synthetic_zero_from)
        if "label" not in lbl.columns:
            raise ValueError(
                f"synthetic 模式需要 label_parquet 含 'label' 列；实际 "
                f"{list(lbl.columns)}"
            )
        out = pd.DataFrame(
            {"prediction": 0.0}, index=lbl.index
        ).astype({"prediction": "float64"})
        out = _slice_window(out, start, end).sort_index()
    else:
        if src is None:
            raise ValueError("非 synthetic 模式必须传 --src")
        src = Path(src)
        if not src.exists():
            raise ValueError(f"src 不存在: {src}")
        df = pd.read_parquet(src)
        out = _normalize_dataframe(
            df,
            date_col=date_col,
            instrument_col=instrument_col,
            prediction_col=prediction_col,
        )
        out = _slice_window(out, start, end)

    if out.empty:
        raise RuntimeError(
            f"导出后 parquet 为空；检查 --start/--end 或 src 是否覆盖目标 OOS 窗"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out.to_parquet(tmp, engine="pyarrow")
    tmp.replace(out_path)

    n = len(out)
    n_non_null = int(out["prediction"].notna().sum())
    dt = out.index.get_level_values("datetime")
    n_inst = out.index.get_level_values("instrument").nunique()
    logger.info(
        "baseline_prediction parquet 写完：%s  rows=%d  non_null=%d (%.4f)  "
        "n_instr=%d  window=[%s, %s]",
        out_path,
        n,
        n_non_null,
        n_non_null / max(1, n),
        n_inst,
        pd.Timestamp(dt.min()).date(),
        pd.Timestamp(dt.max()).date(),
    )
    return out_path


# ------------------------------------------------------------------ CLI


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export production ensemble OOS predictions to a canonical parquet "
            "for L2 marginal / marginal_training checks."
        )
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="输入 parquet（OOF 或推理输出）；含 prediction 列",
    )
    p.add_argument(
        "--prediction-col",
        default=None,
        help=f"预测列名；默认在 {_PRED_COL_CANDIDATES} 里自动挑",
    )
    p.add_argument("--date-col", default=None, help="日期列名（flat schema）")
    p.add_argument(
        "--instrument-col", default=None, help="股票代码列名（flat schema）"
    )
    p.add_argument("--start", default=None, help="OOS 起始 YYYY-MM-DD（闭）")
    p.add_argument("--end", default=None, help="OOS 截止 YYYY-MM-DD（闭）")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径；默认 factor_validation/data/baseline_predictions_<tag>.parquet",
    )
    p.add_argument(
        "--tag",
        default="ensemble_v1",
        help="输出默认文件名中的 tag（当 --out 未给时生效）",
    )
    p.add_argument(
        "--synthetic-zero",
        type=Path,
        default=None,
        help=(
            "占位模式：不传 --src，改传 label_parquet 路径，"
            "生成全零 baseline 以便 marginal 先行跑通"
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    out_path = args.out or (
        DEFAULT_OUT_DIR / f"baseline_predictions_{args.tag}.parquet"
    )
    out = prepare_baseline(
        src=args.src,
        out_path=out_path,
        prediction_col=args.prediction_col,
        date_col=args.date_col,
        instrument_col=args.instrument_col,
        start=args.start,
        end=args.end,
        synthetic_zero_from=args.synthetic_zero,
    )
    print(str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
