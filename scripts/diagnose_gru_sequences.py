"""
诊断 GRU 序列构建问题：
- 检查每个股票在 valid 阶段是否能形成连续 seq_len 序列
- 输出每个 instrument 的有效序列数量与最大连续交易日长度
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# 添加项目根目录到路径，确保可直接运行脚本
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from trainer.trainer import RollingTrainer
from datasets.sequence_builder import _get_trading_calendar_auto


def _get_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return _get_trading_calendar_auto(start, end)


def _max_consecutive_run(pos: np.ndarray) -> int:
    if len(pos) == 0:
        return 0
    # pos 已按时间顺序
    diff = np.diff(pos)
    # 连续位置 diff==1 的最长段
    max_run = 1
    run = 1
    for d in diff:
        if d == 1:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    return max_run


def _count_sequences(
    pos: np.ndarray, valid_mask: np.ndarray, seq_len: int
) -> Tuple[int, int]:
    """
    返回：(total_seq, valid_seq)
    total_seq: 所有端点的序列数量
    valid_seq: 端点落在 valid 的序列数量
    """
    n = len(pos)
    if n < seq_len:
        return 0, 0
    total_seq = 0
    valid_seq = 0
    # 逐点判断窗口连续性
    for i in range(seq_len - 1, n):
        win = pos[i - seq_len + 1 : i + 1]
        if (win[-1] - win[0]) != (seq_len - 1):
            continue
        if not np.all(np.diff(win) == 1):
            continue
        total_seq += 1
        if valid_mask[i]:
            valid_seq += 1
    return total_seq, valid_seq


def main():
    ap = argparse.ArgumentParser(description="诊断 GRU 序列构建与覆盖率")
    ap.add_argument("--config", type=str, default="config/pipeline.yaml")
    ap.add_argument("--window", type=int, default=0, help="滚动窗口索引")
    ap.add_argument("--seq_len", type=int, default=None, help="覆盖 config 的 seq_len")
    ap.add_argument("--out", type=str, default="data/logs/gru_sequence_diagnosis.csv")
    args = ap.parse_args()

    trainer = RollingTrainer(args.config)
    trainer.pipeline.build()
    features, labels = trainer.pipeline.get_all()

    windows = list(trainer._generate_windows())
    if args.window < 0 or args.window >= len(windows):
        raise ValueError(f"window 超出范围: {args.window} / {len(windows)}")
    window = windows[args.window]

    train_feat, _ = trainer._slice(features, labels, window.train_start, window.train_end, is_validation=False)
    valid_feat, _ = trainer._slice(features, labels, window.valid_start, window.valid_end, is_validation=True)

    if args.seq_len is not None:
        seq_len = args.seq_len
    else:
        seq_len = int(trainer.cfg.get("model_gru", {}).get("seq_len", 60))

    # 拼接历史（与 GRU 预测逻辑一致）
    hist_tail = train_feat.groupby(level="instrument").tail(max(0, seq_len - 1))
    combined = pd.concat([hist_tail, valid_feat], axis=0).sort_index()

    # 交易日历
    dt_all = pd.to_datetime(combined.index.get_level_values("datetime")).normalize()
    calendar = _get_calendar(dt_all.min(), dt_all.max())

    rows = []
    instruments = combined.index.get_level_values("instrument").unique()
    for inst in instruments:
        sub = combined.xs(inst, level="instrument", drop_level=False).sort_index(level="datetime")
        sub_dt = pd.to_datetime(sub.index.get_level_values("datetime")).normalize()
        pos = calendar.get_indexer(sub_dt)
        keep = pos >= 0
        if not np.any(keep):
            continue
        sub_dt = sub_dt[keep]
        pos = pos[keep]
        # valid mask: endpoint 是否属于 valid
        valid_mask = sub.index.get_level_values("datetime").isin(valid_feat.index.get_level_values("datetime"))
        valid_mask = np.array(valid_mask)[keep]

        max_run = _max_consecutive_run(pos)
        total_seq, valid_seq = _count_sequences(pos, valid_mask, seq_len)

        rows.append(
            {
                "instrument": inst,
                "n_history": int(len(sub_dt) - valid_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "max_run_len": int(max_run),
                "total_seq": int(total_seq),
                "valid_seq": int(valid_seq),
                "valid_coverage": float(valid_seq / max(1, valid_mask.sum())),
            }
        )

    df = pd.DataFrame(rows).sort_values(["valid_seq", "max_run_len"], ascending=[True, True])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    # 汇总
    total_valid_seq = int(df["valid_seq"].sum()) if len(df) else 0
    total_valid = int(df["n_valid"].sum()) if len(df) else 0
    num_short = int((df["max_run_len"] < seq_len).sum()) if len(df) else 0
    print("=" * 80)
    print("GRU 序列诊断结果")
    print(f"window={args.window}  seq_len={seq_len}")
    print(f"valid样本总数={total_valid}  valid可用序列总数={total_valid_seq}")
    print(f"max_run_len < seq_len 的股票数: {num_short} / {len(df)}")
    print(f"输出文件: {args.out}")
    print("\nvalid_seq 最少的前10个股票:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

