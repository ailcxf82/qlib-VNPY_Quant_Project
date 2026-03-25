from backtest.msa.code_utils import rqalpha_to_tushare, qlib_to_rqalpha
from backtest.msa.filters import is_kcb_or_bj

# 测试代码转换
print("=== 代码转换测试 ===")
codes = ['1', '2', '63', '100', '157', '000001', '600000']
for code in codes:
    rq_code = qlib_to_rqalpha(code)
    ts_code = rqalpha_to_tushare(rq_code)
    is_kcb = is_kcb_or_bj(ts_code)
    print(f"原始: {code:6} -> RQ: {rq_code:12} -> TS: {ts_code:10} -> 科创/北交: {is_kcb}")

print("\n=== 问题分析 ===")
print("预测文件中的代码格式不正确，需要转换为标准6位代码")
print("例如: '1' -> '000001', '63' -> '000063'")
