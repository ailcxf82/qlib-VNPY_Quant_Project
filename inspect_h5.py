import qlib
from qlib.config import REG_CN
from qlib.data import D
import pandas as pd

PROVIDER_URI = "/mnt/d/qlib_data/qlib_data"
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# csi500.txt only has 2026-03-31 dates (future? or just snapshot)
# Check what start_time/end_time we need to use
# Try fetching from a specific instrument
fields = ["$close_qfq", "$volume"]
inst = "000009.SZ"
# Try different date ranges
for start, end in [("2024-01-01","2024-03-01"), ("2025-01-01","2025-06-01"), ("2026-01-01","2026-04-07")]:
    r = D.features([inst], fields, start_time=start, end_time=end, freq="day")
    print(f"{start}~{end}: shape={r.shape}")
    if not r.empty:
        print(r.head(3))
        break

# Also check all.txt
insts_all = D.instruments(market="all")
all_list = D.list_instruments(instruments=insts_all, start_time="2020-01-01", end_time="2020-06-01", as_list=True)
print("all market count:", len(all_list), "sample:", all_list[:3])
sample2 = D.features(all_list[:2], ["$close_qfq"], start_time="2020-01-01", end_time="2020-01-10", freq="day")
print("all market sample fetch:", sample2)
