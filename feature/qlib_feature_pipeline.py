"""
基于 qlib 的特征提取流水线，负责：
1. 初始化 qlib 环境
2. 调用 D.features 获取行情与因子
3. 生成对齐标签 Ref($close, -5)/$close - 1
4. 进行基础标准化，并输出训练用 DataFrame
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

from utils import load_yaml_config

logger = logging.getLogger(__name__)
_QLIB_INITIALIZED = False


class QlibFeaturePipeline:
    """特征管线核心类。"""

    def __init__(self, config_path: str):
        self.config = load_yaml_config(config_path)
        self._init_qlib()
        self.feature_cfg = self.config["data"]
        self.features_df: pd.DataFrame | None = None
        self.label_series: pd.Series | None = None
        self._feature_mean: pd.Series | None = None
        self._feature_std: pd.Series | None = None

    def _init_qlib(self):
        qlib_cfg = self.config.get("qlib", {})
        global _QLIB_INITIALIZED
        already_initialized = False
        if hasattr(qlib, "is_initialized"):
            try:
                already_initialized = bool(qlib.is_initialized())
            except Exception:
                already_initialized = _QLIB_INITIALIZED
        else:
            already_initialized = _QLIB_INITIALIZED
        if already_initialized:
            # 在 notebook/调试环境中可能重复调用，避免重复初始化
            return
        logger.info("初始化 qlib，数据目录: %s", qlib_cfg.get("provider_uri"))
        qlib.init(
            provider_uri=qlib_cfg.get("provider_uri"),
            region=qlib_cfg.get("region", "cn"),
            expression_cache=None,
        )
        _QLIB_INITIALIZED = True

    def build(self):
        """执行特征提取。"""
        feats = self.feature_cfg["features"]
        instruments = self._parse_instruments(self.feature_cfg["instruments"])
        start = self.feature_cfg["start_time"]
        end = self.feature_cfg["end_time"]
        freq = self.feature_cfg.get("freq", "day")
        label_expr = self.feature_cfg.get("label", "Ref($close, -5)/$close - 1")

        logger.info("提取特征: %s", feats)
        feature_panel = D.features(instruments=instruments, fields=feats, start_time=start, end_time=end, freq=freq)
        label_panel = D.features(instruments=instruments, fields=[label_expr], start_time=start, end_time=end, freq=freq)

        feature_panel.columns = feats
        label_series = label_panel.iloc[:, 0].rename("label")

        feature_panel = self._normalize_index(feature_panel)
        label_series = self._normalize_index(label_series)

        # 基础对齐
        # inner join + dropna 保证特征、标签完全对齐
        combined = feature_panel.join(label_series, how="inner").dropna()
        features = combined.drop(columns=["label"])
        label = combined["label"]

        self._fit_norm(features)
        norm_feat = self._transform(features)

        self.features_df = norm_feat
        self.label_series = label
        
        # 记录实际构建的数据范围
        if len(norm_feat) > 0:
            datetime_level = norm_feat.index.get_level_values("datetime")
            actual_start = datetime_level.min()
            actual_end = datetime_level.max()
            logger.info("特征构建完成，样本量: %d", len(norm_feat))
            logger.info("实际数据日期范围: %s 到 %s (配置范围: %s 到 %s)", 
                       actual_start, actual_end, start, end)
            if pd.Timestamp(end) > actual_end:
                logger.warning("实际数据结束日期早于配置的结束日期，可能是 qlib 数据未更新或标签计算需要未来数据")
        else:
            logger.warning("特征构建完成，但样本量为 0，请检查数据配置和 qlib 数据")

    def _fit_norm(self, features: pd.DataFrame):
        """计算全局均值方差。"""
        self._feature_mean = features.mean()
        # 避免标准差为 0 导致除零
        std = features.std().replace(0, 1)
        self._feature_std = std

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """应用标准化。"""
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("标准化参数尚未拟合，请先调用 build()")
        arr = (features - self._feature_mean) / self._feature_std
        return arr.clip(-5, 5)  # 简单去极值，避免极端噪声

    def get_slice(self, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
        """按时间切片返回特征。"""
        if self.features_df is None or self.label_series is None:
            raise RuntimeError("尚未构建特征，请先调用 build()")
        idx = self.features_df.index
        datetime_level = idx.get_level_values("datetime")
        actual_start = datetime_level.min()
        actual_end = datetime_level.max()
        
        # 检查请求的日期范围是否超出实际数据范围
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts < actual_start:
            logger.warning(f"请求的起始日期 {start} 早于实际数据起始日期 {actual_start}，将使用实际起始日期")
        if end_ts > actual_end:
            logger.warning(f"请求的结束日期 {end} 晚于实际数据结束日期 {actual_end}，实际数据只到 {actual_end}")
            logger.warning(f"这可能是因为：1) qlib 数据未更新到该日期；2) 标签计算需要未来数据导致最后几天被过滤")
        
        mask = (datetime_level >= start) & (datetime_level <= end)
        feat = self.features_df.loc[mask]
        lbl = self.label_series.loc[mask]
        
        if len(feat) == 0:
            logger.error(f"在日期范围 [{start}, {end}] 内没有找到数据")
            logger.error(f"实际数据范围: [{actual_start}, {actual_end}]")
        else:
            slice_start = feat.index.get_level_values("datetime").min()
            slice_end = feat.index.get_level_values("datetime").max()
            logger.info(f"返回数据范围: {slice_start} 到 {slice_end}，共 {len(feat)} 个样本")
        
        return feat, lbl

    def get_all(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.features_df is None or self.label_series is None:
            raise RuntimeError("尚未构建特征，请先调用 build()")
        return self.features_df, self.label_series

    def stats(self) -> Dict[str, pd.Series]:
        """返回标准化统计量，供落地保存/加载。"""
        return {
            "mean": self._feature_mean,
            "std": self._feature_std,
        }

    @staticmethod
    def _parse_instruments(inst_conf: Union[str, Dict[str, Any], Tuple[str, ...], list[str]]) -> Union[Dict[str, Any], list[str]]:
        """
        将配置的股票池转换为 qlib 支持的输入。

        - 字符串默认视作市场别名（如 "csi300"），转换为 {"market": xxx, "filter_pipe": []}
        - 已是 dict 的情况下直接返回（便于自定义过滤器）
        - list/tuple 视为具体股票代码集合
        """
        if isinstance(inst_conf, str):
            return {"market": inst_conf, "filter_pipe": []}
        if isinstance(inst_conf, dict):
            return inst_conf
        if isinstance(inst_conf, (list, tuple)):
            return list(inst_conf)
        raise ValueError(f"不支持的股票池配置类型: {type(inst_conf)}")

    @staticmethod
    def _normalize_index(data: Union[pd.DataFrame, pd.Series]):
        """统一索引为 (datetime, instrument) 顺序，便于与预测/标签对齐。"""
        idx = data.index
        if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
            return data
        names = list(idx.names)
        if names == ["datetime", "instrument"]:
            return data.sort_index()
        if "datetime" in names and "instrument" in names:
            order = [names.index("datetime"), names.index("instrument")] + [
                i for i in range(len(names)) if i not in (names.index("datetime"), names.index("instrument"))
            ]
            data = data.reorder_levels(order)
        else:
            # 默认交换前两个层级
            data = data.reorder_levels(list(range(idx.nlevels))[::-1])
        new_names = list(data.index.names)
        if len(new_names) >= 2:
            new_names[0], new_names[1] = "datetime", "instrument"
            data.index = data.index.set_names(new_names)
        return data.sort_index()

