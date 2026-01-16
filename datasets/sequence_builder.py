"""
序列构建器：将 “每天一个扁平特征向量 [D]” 组装为 GRU 输入的 “[T, D]” 序列。

需求约束：
- 对同一股票（instrument），取过去 T=seq_len 个交易日的特征堆叠为 [T, D]
- 严格按交易日排序；不允许未来泄露（窗口的 max(datetime) 必须等于样本 datetime）
- 缺失日期处理：默认仅保留“满窗且交易日连续”的样本（不做 padding/mask）
- 与原始 MultiIndex 对齐：返回 endpoint_index（每条序列对应的最后一天样本索引）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SequenceBuildResult:
    X: np.ndarray  # (N, T, D) float32
    endpoint_index: pd.Index  # MultiIndex entries of original panel (datetime,instrument)


def _get_trading_calendar_auto(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    优先使用 qlib 交易日历；失败则回退到工作日（Mon-Fri）。
    注意：回退日历不包含法定节假日信息，严格性会偏保守/偏宽松，建议生产环境使用 qlib。
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    try:
        from qlib.data import D  # type: ignore

        cal = D.calendar(start_time=start, end_time=end, freq="day")
        cal = pd.to_datetime(list(cal)).normalize()
        return pd.DatetimeIndex(cal).sort_values().unique()
    except Exception:
        return pd.bdate_range(start=start, end=end).normalize()


def build_panel_sequences(
    feat: pd.DataFrame,
    *,
    seq_len: int = 60,
    calendar: Optional[pd.DatetimeIndex] = None,
    require_consecutive_trading_days: bool = True,
    check: bool = True,
) -> SequenceBuildResult:
    """
    将 panel 特征构造成 GRU 输入序列。

    参数:
    - feat: DataFrame(index=MultiIndex[datetime,instrument], columns=[D])
    - seq_len: 序列长度 T
    - calendar: 交易日历；None 则自动推断（优先 qlib，否则 bdate_range）
    - require_consecutive_trading_days: True 则要求窗口覆盖 seq_len 个连续交易日（缺 1 天就丢弃）
    - check: True 则执行关键断言（last-day 对齐 & 无未来日期）
    """
    if not isinstance(feat.index, pd.MultiIndex):
        raise ValueError("feat.index 必须是 MultiIndex(datetime,instrument)")
    if "datetime" not in feat.index.names or "instrument" not in feat.index.names:
        raise ValueError(f"feat.index 必须包含 datetime/instrument，当前={feat.index.names}")
    if seq_len <= 1:
        raise ValueError("seq_len 必须 >= 2")

    # 全局日历
    dt_all = pd.to_datetime(feat.index.get_level_values("datetime")).normalize()
    if len(dt_all) == 0:
        return SequenceBuildResult(X=np.zeros((0, seq_len, feat.shape[1]), dtype=np.float32), endpoint_index=pd.Index([]))
    start = pd.Timestamp(dt_all.min()).normalize()
    end = pd.Timestamp(dt_all.max()).normalize()
    if calendar is None:
        calendar = _get_trading_calendar_auto(start, end)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar).normalize()).sort_values().unique()
    if len(calendar) == 0:
        raise RuntimeError("交易日历为空，无法构建序列")

    X_list: list[np.ndarray] = []
    idx_list: list = []

    # 每个 instrument 内部按 datetime 排序
    for inst, sub in feat.groupby(level="instrument"):
        sub = sub.sort_index(level="datetime")
        sub_dt = pd.to_datetime(sub.index.get_level_values("datetime")).normalize()
        # 过滤不在日历内的日期（pos=-1）
        pos = calendar.get_indexer(sub_dt)
        keep = pos >= 0
        if not np.any(keep):
            continue
        sub = sub.loc[keep]
        sub_dt = sub_dt[keep]
        pos = pos[keep]

        # 去重（若同一 instrument 同一天出现重复样本，保留最后一条）
        # 这能保证 pos 严格递增，避免 “同日多行”破坏窗口逻辑。
        if sub_dt.duplicated().any():
            dedup_mask = ~sub_dt.duplicated(keep="last")
            sub = sub.loc[dedup_mask]
            sub_dt = sub_dt[dedup_mask]
            pos = pos[dedup_mask]

        if len(sub) < seq_len:
            continue

        x = sub.values.astype(np.float32)
        n = len(sub)
        for i in range(seq_len - 1, n):
            win_pos = pos[i - seq_len + 1 : i + 1]
            if require_consecutive_trading_days:
                # 连续交易日：pos 必须严格步长为 1
                if (win_pos[-1] - win_pos[0]) != (seq_len - 1):
                    continue
                if not np.all(np.diff(win_pos) == 1):
                    continue
            X_seq = x[i - seq_len + 1 : i + 1]
            endpoint = sub.index[i]
            X_list.append(X_seq)
            idx_list.append(endpoint)

            if check:
                # 断言1：序列最后一天 = endpoint 当天特征
                last = X_seq[-1]
                cur = feat.loc[endpoint].values.astype(np.float32)
                assert np.allclose(last, cur, equal_nan=True), "序列最后一天特征不等于样本当天特征"
                # 断言2：不包含未来日期（窗口最大日期必须等于 endpoint 日期）
                assert sub_dt[i] == sub_dt[i - seq_len + 1 : i + 1].max(), "序列包含未来日期（最大日期不等于样本日期）"

    if not X_list:
        return SequenceBuildResult(X=np.zeros((0, seq_len, feat.shape[1]), dtype=np.float32), endpoint_index=pd.Index([]))

    X = np.stack(X_list, axis=0).astype(np.float32)
    endpoint_index = pd.Index(idx_list)
    return SequenceBuildResult(X=X, endpoint_index=endpoint_index)



