from __future__ import annotations

import re


def rqalpha_to_tushare(code: str) -> str:
    """
    600000.XSHG -> 600000.SH
    000001.XSHE -> 000001.SZ
    """
    code = (code or "").strip()
    if code.endswith(".XSHG"):
        return code.replace(".XSHG", ".SH")
    if code.endswith(".XSHE"):
        return code.replace(".XSHE", ".SZ")
    return code


def tushare_to_rqalpha(ts_code: str) -> str:
    """
    600000.SH -> 600000.XSHG
    000001.SZ -> 000001.XSHE
    """
    ts_code = (ts_code or "").strip()
    if ts_code.endswith(".SH"):
        return ts_code.replace(".SH", ".XSHG")
    if ts_code.endswith(".SZ"):
        return ts_code.replace(".SZ", ".XSHE")
    return ts_code


def qlib_to_rqalpha(instrument: str) -> str:
    """
    支持：
    - SH600000 / SZ000001
    - 600000 / 000001
    - 600000.SH / 000001.SZ
    """
    s = str(instrument).strip()
    if s.startswith("SH"):
        code = s[2:].zfill(6)
        return f"{code}.XSHG"
    if s.startswith("SZ"):
        code = s[2:].zfill(6)
        return f"{code}.XSHE"
    if s.endswith(".SH"):
        return tushare_to_rqalpha(s)
    if s.endswith(".SZ"):
        return tushare_to_rqalpha(s)
    # 纯数字
    if re.fullmatch(r"\d+", s):
        code = s.zfill(6)
        if code.startswith(("688", "689", "6", "9")):
            return f"{code}.XSHG"
        return f"{code}.XSHG" if code.startswith(("6", "9")) else f"{code}.XSHE"
    return s


def is_kcb_or_bj(ts_code: str) -> bool:
    """
    过滤科创/北交等：
    - 68xxxx: 科创板
    - 4/8 开头：北交所常见（示意）
    """
    ts_code = (ts_code or "").strip()
    code = ts_code.split(".", 1)[0]
    return code.startswith("68") or code.startswith("4") or code.startswith("8")


