"""
A+B 准备阶段：排查 RD-Agent 原料数据的覆盖度
- daily_pv.h5：时间 / 股票范围
- RD-Agent_workspace：有多少 ws 带 factor.py / result.h5
- daily_pv 与 qlib_data/csi500+csi300 的 instrument 覆盖交集
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    pv_path = _ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "daily_pv.h5"
    ws_root = _ROOT / "git_ignore_folder" / "RD-Agent_workspace"

    print("=== daily_pv.h5 ===")
    if not pv_path.exists():
        print(f"MISSING: {pv_path}")
    else:
        pv = pd.read_hdf(str(pv_path))
        print(f"type           : {type(pv).__name__}")
        print(f"shape          : {pv.shape}")
        print(f"index names    : {pv.index.names}")
        print(f"columns        : {list(pv.columns)[:20]}")
        dt = pv.index.get_level_values("datetime")
        inst = pv.index.get_level_values("instrument")
        print(f"dt range       : {dt.min()}  ~  {dt.max()}")
        print(f"n_instruments  : {inst.nunique()}")
        sample = sorted(set(inst))
        print(f"sample head    : {sample[:8]}")
        print(f"sample tail    : {sample[-5:]}")

    print("\n=== RD-Agent_workspace ===")
    if not ws_root.exists():
        print(f"MISSING: {ws_root}")
    else:
        ws_dirs = [d for d in ws_root.iterdir() if d.is_dir()]
        has_factor = [d for d in ws_dirs if (d / "factor.py").exists()]
        has_result = [d for d in ws_dirs if (d / "result.h5").exists()]
        print(f"total ws       : {len(ws_dirs)}")
        print(f"with factor.py : {len(has_factor)}")
        print(f"with result.h5 : {len(has_result)}")
        print(f"sample ws dirs : {[d.name for d in ws_dirs[:5]]}")

    # --- instrument 重叠：daily_pv vs qlib_data csi500+csi300 ---
    print("\n=== instrument overlap (daily_pv vs qlib_data) ===")
    try:
        from feature.qlib_feature_pipeline import QlibFeaturePipeline
        import yaml

        data_cfg = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
        pipe = QlibFeaturePipeline(data_cfg)
        insts_csi = pipe._parse_instruments("csi500,csi300")
        pv_insts = set(pv.index.get_level_values("instrument"))
        csi_insts = set([i.upper() for i in insts_csi])
        pv_up = set([str(i).upper() for i in pv_insts])
        inter = csi_insts & pv_up
        print(f"csi500+csi300 count : {len(csi_insts)}")
        print(f"daily_pv count      : {len(pv_up)}")
        print(f"intersection        : {len(inter)}")
        print(f"only in csi(sample) : {sorted(csi_insts - pv_up)[:10]}")
        print(f"only in pv(sample)  : {sorted(pv_up - csi_insts)[:10]}")
    except Exception as e:
        print(f"overlap check failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
