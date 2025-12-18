from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

def _try_load_token_from_secrets_file(path: str = "config/secrets.yaml") -> str:
    """
    从本地 secrets 文件读取 token（该文件应加入 .gitignore，避免泄露）。
    文件格式示例：
      tushare:
        token: "xxxx"
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        token = (((data.get("tushare") or {}).get("token")) if isinstance(data, dict) else "") or ""
        return str(token).strip()
    except Exception:
        return ""


def _dt_str(dt: pd.Timestamp) -> str:
    return pd.Timestamp(dt).strftime("%Y%m%d")


@dataclass
class TushareConfig:
    token: str
    cache_dir: str = "data/tushare_cache"


class TushareClient:
    """
    轻量 Tushare Pro 客户端（可选依赖）。
    - 未安装 tushare 或未设置 token 时：返回 None，策略会降级为“只做本地规则过滤”
    - 自带本地缓存：避免回测期间重复请求
    """

    def __init__(self, cfg: TushareConfig):
        self.cfg = cfg
        self.cache_root = Path(cfg.cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        try:
            import tushare as ts  # type: ignore
        except Exception as e:
            raise ImportError("未安装 tushare，请先 pip install tushare") from e

        self._ts = ts
        self._pro = ts.pro_api(cfg.token)

    @staticmethod
    def try_create(cache_dir: str = "data/tushare_cache") -> Optional["TushareClient"]:
        # 读取顺序：环境变量 > 本地 secrets 文件（不入库）
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not token:
            token = _try_load_token_from_secrets_file()
        if not token:
            logger.warning("未设置环境变量 TUSHARE_TOKEN，Tushare 过滤将被跳过（策略仍可运行）")
            return None
        try:
            return TushareClient(TushareConfig(token=token, cache_dir=cache_dir))
        except Exception as e:
            logger.warning("初始化 TushareClient 失败，将跳过 Tushare 过滤: %s", e)
            return None

    def _cache_path(self, name: str) -> Path:
        return self.cache_root / name

    def _load_cache_df(self, name: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(name)
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            try:
                return pd.read_csv(p)
            except Exception:
                return None

    def _save_cache_df(self, name: str, df: pd.DataFrame):
        p = self._cache_path(name)
        try:
            df.to_parquet(p, index=False)
        except Exception:
            df.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8")

    @staticmethod
    def _sanitize(s: str) -> str:
        s = (s or "").strip().replace("/", "-").replace("\\", "-").replace(":", "-")
        return s[:180] if len(s) > 180 else s

    def _cache_name(self, api: str, params: Dict[str, Any]) -> str:
        # 生成可读的缓存文件名（避免超长/不可用字符）
        parts = [api]
        for k in sorted(params.keys()):
            v = params[k]
            if v is None:
                continue
            parts.append(f"{k}={v}")
        name = "__".join(parts)
        return f"{self._sanitize(name)}.parquet"

    @staticmethod
    def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None and v != ""}

    def income(
        self,
        *,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Tushare Pro: income
        常见字段：ts_code, end_date, total_revenue, n_income_attr_p 等（具体以接口返回为准）
        参考文档：https://tushare.pro/document
        """
        params = self._drop_none(
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date, "period": period, "fields": fields}
        )
        cache_name = self._cache_name("income", params)
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.income(**params)
        self._save_cache_df(cache_name, df)
        return df

    def fina_indicator(
        self,
        *,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Tushare Pro: fina_indicator
        常见字段：ts_code, end_date, roa, roe 等（具体以接口返回为准）
        """
        params = self._drop_none(
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date, "period": period, "fields": fields}
        )
        cache_name = self._cache_name("fina_indicator", params)
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.fina_indicator(**params)
        self._save_cache_df(cache_name, df)
        return df

    def fina_audit(
        self,
        *,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Tushare Pro: fina_audit
        常见字段：ts_code, end_date, audit_result 等（具体以接口返回为准）
        """
        params = self._drop_none(
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date, "period": period, "fields": fields}
        )
        cache_name = self._cache_name("fina_audit", params)
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.fina_audit(**params)
        self._save_cache_df(cache_name, df)
        return df

    def stock_basic(self) -> pd.DataFrame:
        """
        股票基础信息（全量，缓存）。
        字段：ts_code, name, list_date, delist_date, market 等
        """
        cache_name = "stock_basic.parquet"
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached

        df = self._pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,market,list_date,delist_date"
        )
        self._save_cache_df(cache_name, df)
        return df

    def daily_basic(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        """
        每日指标（PB、市值等），按交易日缓存。
        字段：ts_code, pb, total_mv, circ_mv
        """
        d = _dt_str(trade_date)
        cache_name = f"daily_basic_{d}.parquet"
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached

        df = self._pro.daily_basic(
            trade_date=d,
            fields="ts_code,pb,total_mv,circ_mv"
        )
        self._save_cache_df(cache_name, df)
        return df

    def limit_list(self, trade_date: pd.Timestamp, *, limit_type: str = "U") -> pd.DataFrame:
        """
        涨跌停列表（可用于“近5日涨停过”过滤），按交易日缓存。
        limit_type: U(涨停) / D(跌停)
        字段：ts_code
        """
        d = _dt_str(trade_date)
        cache_name = f"limit_list_{limit_type}_{d}.parquet"
        cached = self._load_cache_df(cache_name)
        if cached is not None and not cached.empty:
            return cached

        df = self._pro.limit_list(trade_date=d, limit_type=limit_type, fields="ts_code")
        self._save_cache_df(cache_name, df)
        return df


