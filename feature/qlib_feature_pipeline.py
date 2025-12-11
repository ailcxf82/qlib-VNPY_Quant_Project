"""
基于 qlib 的特征提取流水线，负责：
1. 初始化 qlib 环境
2. 调用 D.features 获取行情与因子
3. 生成对齐标签 Ref($close, -5)/$close - 1
4. 进行基础标准化，并输出训练用 DataFrame
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Union, List

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

# 添加项目根目录到 Python 路径，确保可以导入 utils 模块
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils import load_yaml_config

logger = logging.getLogger(__name__)

# 尝试导入 158 因子相关模块
try:
    from qlib.contrib.data.loader import Alpha158DL
    HAS_ALPHA158 = True
except ImportError:
    try:
        # 某些版本的 qlib 可能使用不同的导入路径
        from qlib.contrib.data.handler import Alpha158
        HAS_ALPHA158 = True
    except ImportError:
        HAS_ALPHA158 = False
        logger.warning("无法导入 Alpha158DL，158 因子功能将不可用。可以通过配置文件手动指定因子列表。")
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
        self._label_is_rank: bool = False  # 标记标签是否为排名

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

    def _get_alpha158_factors(self) -> List[str]:
        """获取 158 因子表达式列表。
        
        根据 qlib 官方实现，158 因子通过 Alpha158DL.get_feature_config() 获取。
        支持通过配置自定义因子选择，减少计算资源消耗。
        """
        # 首先检查配置文件中是否手动指定了因子列表
        manual_factors = self.feature_cfg.get("alpha158_factors", None)
        if manual_factors and isinstance(manual_factors, list):
            logger.info("使用配置文件中手动指定的 158 因子列表，共 %d 个因子", len(manual_factors))
            return manual_factors
        
        if not HAS_ALPHA158:
            logger.warning("Alpha158DL 模块不可用，且配置文件中未指定因子列表。请在配置文件中添加 alpha158_factors 列表")
            return []
        
        # 检查是否有自定义配置（用于筛选因子）
        alpha158_config = self.feature_cfg.get("alpha158_config", None)
        
        try:
            # 方法1: 使用 Alpha158DL.get_feature_config() 获取因子表达式（推荐方法）
            # 如果提供了自定义配置，使用配置；否则使用默认配置
            if alpha158_config:
                logger.info("使用自定义 158 因子配置: %s", alpha158_config)
                fields, names = Alpha158DL.get_feature_config(alpha158_config)
            else:
                fields, names = Alpha158DL.get_feature_config()
            
            if isinstance(fields, list) and len(fields) > 0:
                logger.info("通过 Alpha158DL.get_feature_config() 成功获取因子列表，共 %d 个因子", len(fields))
                
                # 检查是否有因子筛选配置
                factor_filter = self.feature_cfg.get("alpha158_filter", None)
                if factor_filter:
                    fields = self._filter_factors(fields, names, factor_filter)
                    logger.info("筛选后因子数量: %d", len(fields))
                
                return fields
        except Exception as e:
            logger.debug("Alpha158DL.get_feature_config() 方法失败: %s", e)
        
        try:
            # 方法2: 尝试使用 Alpha158 处理器（备选方案）
            from qlib.contrib.data.handler import Alpha158
            alpha158_handler = Alpha158()
            # Alpha158 内部使用 Alpha158DL，尝试通过 get_feature_config 获取
            if hasattr(alpha158_handler, 'get_feature_config'):
                fields, names = alpha158_handler.get_feature_config()
                if isinstance(fields, list) and len(fields) > 0:
                    logger.info("通过 Alpha158.get_feature_config() 获取 158 因子，共 %d 个因子", len(fields))
                    return fields
        except Exception as e:
            logger.debug("Alpha158 类方法失败: %s", e)
        
        # 如果所有方法都失败，提示用户手动配置
        logger.warning(
            "无法自动获取 158 因子列表。请在配置文件的 data 部分添加 alpha158_factors 字段，"
            "手动指定因子表达式列表。参考: qlib/contrib/data/loader.py 中的 Alpha158DL.get_feature_config()"
        )
        return []
    
    def _filter_factors(self, fields: List[str], names: List[str], filter_config: Dict) -> List[str]:
        """根据配置筛选因子。
        
        filter_config 支持：
        - include_operators: 包含的操作符列表（如 ["ROC", "MA", "STD"]）
        - exclude_operators: 排除的操作符列表
        - include_windows: 包含的窗口列表（如 [5, 10, 20]）
        - exclude_windows: 排除的窗口列表
        - max_factors: 最大因子数量（按名称排序选择前N个）
        """
        filtered_fields = []
        filtered_names = []
        
        include_ops = filter_config.get("include_operators", None)
        exclude_ops = filter_config.get("exclude_operators", [])
        include_wins = filter_config.get("include_windows", None)
        exclude_wins = filter_config.get("exclude_windows", [])
        max_factors = filter_config.get("max_factors", None)
        
        for field, name in zip(fields, names):
            # 检查操作符
            operator = None
            for op in ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", 
                      "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD",
                      "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]:
                if name.startswith(op):
                    operator = op
                    break
            
            if operator:
                # 检查操作符过滤
                if include_ops and operator not in include_ops:
                    continue
                if operator in exclude_ops:
                    continue
                
                # 检查窗口过滤（从名称中提取数字）
                import re
                window_match = re.search(r'(\d+)', name)
                if window_match:
                    window = int(window_match.group(1))
                    if include_wins and window not in include_wins:
                        continue
                    if window in exclude_wins:
                        continue
            
            filtered_fields.append(field)
            filtered_names.append(name)
        
        # 如果设置了最大因子数量，按名称排序后选择前N个
        if max_factors and len(filtered_fields) > max_factors:
            # 按名称排序，保持 KBar 和 Price 因子优先
            sorted_pairs = sorted(zip(filtered_names, filtered_fields), key=lambda x: (
                0 if x[0].startswith(("KMID", "KLEN", "KUP", "KLOW", "KSFT", "OPEN", "HIGH", "LOW", "VWAP")) else 1,
                x[0]
            ))
            filtered_names, filtered_fields = zip(*sorted_pairs[:max_factors])
            filtered_fields = list(filtered_fields)
        
        return filtered_fields

    def build(self):
        """执行特征提取。"""
        feats = self.feature_cfg.get("features", []).copy()  # 使用 copy 避免修改原配置
        
        # 检查是否启用 158 因子
        use_alpha158 = self.feature_cfg.get("use_alpha158", False)
        if use_alpha158:
            alpha158_factors = self._get_alpha158_factors()
            if alpha158_factors:
                feats.extend(alpha158_factors)
                logger.info("已添加 158 因子，当前特征总数: %d (原有: %d, 158因子: %d)", 
                           len(feats), len(self.feature_cfg.get("features", [])), len(alpha158_factors))
            else:
                logger.warning("use_alpha158=True 但无法获取 158 因子，将仅使用自定义特征")
        
        instruments = self._parse_instruments(self.feature_cfg["instruments"])
        start = self.feature_cfg["start_time"]
        end = self.feature_cfg["end_time"]
        freq = self.feature_cfg.get("freq", "day")
        label_expr = self.feature_cfg.get("label", "Ref($close, -5)/$close - 1")

        logger.info("提取特征，共 %d 个特征表达式", len(feats))
        
        
        
        # 提取特征和标签
        try:
            feature_panel = D.features(instruments=instruments, fields=feats, start_time=start, end_time=end, freq=freq)
            label_panel = D.features(instruments=instruments, fields=[label_expr], start_time=start, end_time=end, freq=freq)
        except Exception as e:
            logger.error("特征提取失败: %s", e)
            logger.error("提示：可能是 158 因子中的某些表达式在当前数据源中不可用")
            logger.error("建议：1) 检查数据源是否包含 VWAP 字段；2) 使用 alpha158_config 筛选因子")
            raise
        
        # 设置列名，支持列名映射（将特定表达式映射为自定义名称）
        feature_names = []
        feature_name_mapping = self.feature_cfg.get("feature_name_mapping", {})
        for feat in feats:
            # 如果配置了列名映射，使用映射后的名称；否则使用原始表达式
            if feat in feature_name_mapping:
                feature_names.append(feature_name_mapping[feat])
            else:
                feature_names.append(feat)
        feature_panel.columns = feature_names
        label_series = label_panel.iloc[:, 0].rename("label")

        # 记录原始数据量
        logger.info("原始特征数据量: %d 行，%d 列", len(feature_panel), len(feature_panel.columns))
        logger.info("原始标签数据量: %d 行", len(label_series))
        
        # 检查缺失值情况
        feature_nan_count = feature_panel.isnull().sum().sum()
        feature_nan_pct = feature_nan_count / (len(feature_panel) * len(feature_panel.columns)) * 100 if len(feature_panel) > 0 else 0
        label_nan_count = label_series.isnull().sum()
        label_nan_pct = label_nan_count / len(label_series) * 100 if len(label_series) > 0 else 0
        logger.info("特征缺失值: %d (%.2f%%)，标签缺失值: %d (%.2f%%)", 
                   feature_nan_count, feature_nan_pct, label_nan_count, label_nan_pct)
        
        # 找出缺失值最多的特征（用于诊断）
        if len(feature_panel) > 0 and feature_nan_count > 0:
            nan_by_col = feature_panel.isnull().sum().sort_values(ascending=False)
            top_nan_cols = nan_by_col.head(10)
            logger.info("缺失值最多的前10个特征: %s", top_nan_cols.to_dict())

        feature_panel = self._normalize_index(feature_panel)
        label_series = self._normalize_index(label_series)

        # 基础对齐
        # inner join 先对齐索引
        combined = feature_panel.join(label_series, how="inner")
        logger.info("对齐后数据量: %d 行", len(combined))
        
        # 检查对齐后的缺失值
        if len(combined) > 0:
            combined_nan_count = combined.isnull().sum().sum()
            combined_nan_pct = combined_nan_count / (len(combined) * len(combined.columns)) * 100
            logger.info("对齐后缺失值: %d (%.2f%%)", combined_nan_count, combined_nan_pct)
            
            # 详细诊断：检查哪些列全为 NaN
            nan_by_col = combined.isnull().sum()
            all_nan_cols = nan_by_col[nan_by_col == len(combined)].index.tolist()
            if all_nan_cols:
                logger.warning("以下 %d 个特征全为 NaN（可能表达式不可用）: %s", 
                             len(all_nan_cols), all_nan_cols[:10])  # 只显示前10个
            
            # 检查标签缺失情况
            label_nan_count = combined["label"].isnull().sum()
            if label_nan_count > 0:
                logger.warning("标签缺失: %d 行 (%.2f%%)，这可能是由于标签计算需要未来数据（Ref($close, -5)）", 
                             label_nan_count, label_nan_count / len(combined) * 100)
            
            # 如果缺失值过多，使用更宽松的 dropna 策略
            # 只删除标签为 NaN 的行，特征中的 NaN 可以后续填充
            if combined_nan_pct > 50 or len(all_nan_cols) > 0:
                logger.warning("缺失值比例过高 (%.2f%%) 或存在全 NaN 特征 (%d 个)，使用宽松的清理策略", 
                             combined_nan_pct, len(all_nan_cols))
                
                # 先删除全为 NaN 的列（这些特征不可用）
                if all_nan_cols:
                    logger.info("删除 %d 个全为 NaN 的特征列", len(all_nan_cols))
                    combined = combined.drop(columns=all_nan_cols)
                
                # 只删除标签为 NaN 的行
                before_drop = len(combined)
                combined = combined.dropna(subset=["label"])
                logger.info("删除标签为 NaN 的行: %d -> %d", before_drop, len(combined))
                
                # 对于特征，使用前向填充和后向填充
                feature_cols = [col for col in combined.columns if col != "label"]
                if len(feature_cols) > 0:
                    # 按股票分组填充（避免跨股票填充）
                    if isinstance(combined.index, pd.MultiIndex) and "instrument" in combined.index.names:
                        combined[feature_cols] = combined.groupby(level="instrument")[feature_cols].ffill().bfill()
                    else:
                        combined[feature_cols] = combined[feature_cols].ffill().bfill()
                    
                    # 如果还有 NaN，用 0 填充（避免全部删除）
                    remaining_nan = combined[feature_cols].isnull().sum().sum()
                    if remaining_nan > 0:
                        logger.warning("填充后仍有 %d 个 NaN，使用 0 填充", remaining_nan)
                        combined[feature_cols] = combined[feature_cols].fillna(0)
            else:
                # 缺失值不多，使用严格的 dropna
                before_drop = len(combined)
                combined = combined.dropna()
                logger.info("使用严格清理策略: %d -> %d 行", before_drop, len(combined))
        
        if len(combined) == 0:
            logger.error("=" * 80)
            logger.error("清理后数据量为 0，详细诊断信息：")
            logger.error("=" * 80)
            logger.error("1. 原始特征数据量: %d 行", len(feature_panel) if 'feature_panel' in locals() else 0)
            logger.error("2. 原始标签数据量: %d 行", len(label_series) if 'label_series' in locals() else 0)
            logger.error("3. 对齐后数据量: %d 行", len(combined) if 'combined' in locals() else 0)
            
            if 'combined' in locals() and len(combined) == 0:
                logger.error("4. 可能原因：")
                logger.error("   a) 158 因子中的某些表达式在当前数据源中不可用（如 VWAP）")
                logger.error("   b) 标签计算需要未来数据（Ref($close, -5)），导致最后5天没有标签")
                logger.error("   c) 特征和标签的时间范围完全不匹配")
                logger.error("   d) 所有特征都包含 NaN，dropna() 删除了所有行")
                logger.error("5. 解决建议：")
                logger.error("   a) 检查数据源是否包含 VWAP: 在配置中去掉 VWAP")
                logger.error("   b) 使用 alpha158_config 筛选可用的因子")
                logger.error("   c) 先禁用 158 因子（use_alpha158: false），测试基础特征是否正常")
                logger.error("   d) 检查 qlib 数据源是否完整：$open, $high, $low, $close, $volume")
            logger.error("=" * 80)
            raise ValueError("特征提取后数据量为 0，请检查数据配置和 qlib 数据源")
        
        features = combined.drop(columns=["label"])
        label = combined["label"]
        
        logger.info("最终特征数据量: %d 行，%d 列", len(features), len(features.columns))
        logger.info("最终标签数据量: %d 行", len(label))

        # 标签转换：支持转换为排名百分位
        label_transform = self.feature_cfg.get("label_transform", {})
        if label_transform.get("enabled", False):
            from utils.label_transform import transform_to_rank
            method = label_transform.get("method", "percentile")
            groupby = label_transform.get("groupby", "datetime")
            label = transform_to_rank(label, method=method, groupby=groupby)
            logger.info("标签已转换为排名（方法: %s, 分组: %s）", method, groupby)
            self._label_is_rank = True
        else:
            self._label_is_rank = False

        # 修复：不再使用全局归一化，保存原始特征
        # 归一化将在训练时对每个窗口单独计算，避免数据泄露
        logger.info("特征构建完成，保存原始特征（未归一化），归一化将在训练时按窗口计算")
        self.features_df = features
        self.label_series = label
        
        # 记录实际构建的数据范围
        if len(features) > 0:
            datetime_level = features.index.get_level_values("datetime")
            actual_start = datetime_level.min()
            actual_end = datetime_level.max()
            logger.info("特征构建完成，样本量: %d", len(features))
            logger.info("实际数据日期范围: %s 到 %s (配置范围: %s 到 %s)", 
                       actual_start, actual_end, start, end)
            if pd.Timestamp(end) > actual_end:
                logger.warning("实际数据结束日期早于配置的结束日期，可能是 qlib 数据未更新或标签计算需要未来数据")
        else:
            logger.warning("特征构建完成，但样本量为 0，请检查数据配置和 qlib 数据")

    def _fit_norm(self, features: pd.DataFrame):
        """计算均值方差（用于单个窗口的归一化）。"""
        self._feature_mean = features.mean()
        # 避免标准差为 0 导致除零
        std = features.std().replace(0, 1)
        self._feature_std = std

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """应用标准化。"""
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("标准化参数尚未拟合，请先调用 _fit_norm()")
        arr = (features - self._feature_mean) / self._feature_std
        return arr.clip(-5, 5)  # 简单去极值，避免极端噪声
    
    @staticmethod
    def normalize_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        对特征进行归一化，返回归一化后的特征、均值和标准差。
        
        参数:
            features: 原始特征 DataFrame
        
        返回:
            normalized_features: 归一化后的特征
            mean: 均值 Series
            std: 标准差 Series
        """
        mean = features.mean()
        std = features.std().replace(0, 1)
        normalized = (features - mean) / std
        normalized = normalized.clip(-5, 5)
        return normalized, mean, std

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
    def _parse_instruments(inst_conf: Union[str, Dict[str, Any], Tuple[str, ...], list[str]]) -> list[str]:
        """
        将配置的股票池转换为股票代码列表。
        
        根据测试，D.features() 需要股票代码数组（如 ["000001", "000002", ...]），
        而不是字典配置。因此需要将市场别名转换为股票代码列表。

        - 字符串默认视作市场别名（如 "csi300"），通过 D.list_instruments() 获取股票列表
        - 字典配置（如 {"market": "csi300", "filter_pipe": []}），也转换为股票列表
        - list/tuple 视为具体股票代码集合，直接返回
        """
        if isinstance(inst_conf, str):
            # 如果是市场别名（如 "csi300"），先获取配置字典，再转换为股票列表
            try:
                market_config = D.instruments(inst_conf)
                # 使用 D.list_instruments() 获取股票代码列表
                stock_list = D.list_instruments(instruments=market_config, as_list=True)
                if isinstance(stock_list, list) and len(stock_list) > 0:
                    logger.info("从市场 '%s' 获取到 %d 只股票", inst_conf, len(stock_list))
                    # 确保返回的是纯数字股票代码（去掉 .SH 或 .SZ 后缀，如果存在）
                    cleaned_list = []
                    for code in stock_list:
                        # 如果代码包含点号，提取前面的数字部分
                        if '.' in str(code):
                            code = str(code).split('.')[0]
                        cleaned_list.append(str(code))
                    return cleaned_list
                else:
                    raise ValueError(f"无法从市场 '{inst_conf}' 获取股票列表，返回结果为空")
            except Exception as e:
                logger.error("无法从市场 '%s' 获取股票列表: %s", inst_conf, e)
                logger.error("请检查：1) qlib 数据源是否包含该市场定义；2) 市场名称是否正确")
                raise ValueError(f"无法解析股票池配置 '{inst_conf}': {e}")
        
        if isinstance(inst_conf, dict):
            # 如果是字典配置，也转换为股票列表
            try:
                stock_list = D.list_instruments(instruments=inst_conf, as_list=True)
                if isinstance(stock_list, list) and len(stock_list) > 0:
                    market_name = inst_conf.get("market", "未知市场")
                    logger.info("从市场配置 '%s' 获取到 %d 只股票", market_name, len(stock_list))
                    # 确保返回的是纯数字股票代码
                    cleaned_list = []
                    for code in stock_list:
                        if '.' in str(code):
                            code = str(code).split('.')[0]
                        cleaned_list.append(str(code))
                    return cleaned_list
                else:
                    raise ValueError(f"无法从市场配置获取股票列表，返回结果为空")
            except Exception as e:
                logger.error("无法从市场配置获取股票列表: %s", e)
                raise ValueError(f"无法从市场配置获取股票列表: {e}")
        
        if isinstance(inst_conf, (list, tuple)):
            # 如果是列表，确保格式正确（纯数字代码）
            result = []
            for code in inst_conf:
                code_str = str(code)
                # 如果包含点号，提取前面的数字部分
                if '.' in code_str:
                    code_str = code_str.split('.')[0]
                result.append(code_str)
            return result
        
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

