"""快查 qlib 本地数据 + csi300 instruments 范围。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import qlib
from qlib.data import D

qlib.init(provider_uri="D:/qlib_data/qlib_data", region="cn")

# 日历范围
cal = D.calendar(start_time="2020-01-01", end_time="2030-12-31", freq="day")
print(f"calendar: n={len(cal)} first={cal[0]} last={cal[-1]}")

# csi300 池
inst_cfg = D.instruments(market="csi300")
inst_list = D.list_instruments(instruments=inst_cfg, as_list=True)
print(f"csi300 instruments: n={len(inst_list)} sample={inst_list[:3]}")

# 尝试 D.features 大窗口
df = D.features(
    inst_cfg,
    fields=["$close_qfq"],
    start_time="2025-01-01",
    end_time="2026-04-07",
    freq="day",
    disk_cache=0,
)
print(f"D.features shape={df.shape}")
if not df.empty:
    dt = df.index.get_level_values("datetime")
    print(f"dt range: {dt.min()} ~ {dt.max()}")
    inst = df.index.get_level_values("instrument")
    print(f"inst unique: {inst.nunique()}")
