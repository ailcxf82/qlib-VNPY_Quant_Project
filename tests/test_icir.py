"""
ICIR测试脚本 - 模型预测能力评估

功能：
1. 计算各模型的IC（Information Coefficient）
2. 计算各模型的ICIR（Information Coefficient Information Ratio）
3. 计算Rank IC和Rank ICIR
4. 输出详细的测试报告

使用方法：
    python tests/test_icir.py --config config/pipeline.yaml --output data/logs/icir_report.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICIRTester:
    """ICIR测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_yaml_config(config_path)
        self.results = {}
        
    def calculate_ic(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        method: str = "spearman"
    ) -> float:
        """
        计算IC值
        
        Args:
            predictions: 预测值
            labels: 真实标签
            method: "spearman" 或 "pearson"
        
        Returns:
            IC值
        """
        valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
        if valid_mask.sum() < 10:
            return np.nan
        
        pred_valid = predictions[valid_mask]
        label_valid = labels[valid_mask]
        
        if method == "spearman":
            ic, _ = stats.spearmanr(pred_valid, label_valid)
        else:
            ic, _ = stats.pearsonr(pred_valid, label_valid)
        
        return ic
    
    def calculate_icir(self, ic_series: pd.Series) -> float:
        """
        计算ICIR
        
        ICIR = IC均值 / IC标准差
        """
        if len(ic_series) < 2 or ic_series.std() == 0:
            return np.nan
        return ic_series.mean() / ic_series.std()
    
    def calculate_rank_ic(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """计算Rank IC（使用Spearman相关系数）"""
        return self.calculate_ic(predictions, labels, method="spearman")
    
    def run_rolling_test(
        self,
        n_windows: int = 3,
        output_dir: str = "data/logs"
    ) -> Dict:
        """
        运行滚动窗口测试
        
        Args:
            n_windows: 测试窗口数量
            output_dir: 输出目录
        
        Returns:
            测试结果字典
        """
        logger.info("=" * 60)
        logger.info("开始ICIR滚动窗口测试")
        logger.info("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            from trainer.trainer import RollingTrainer
            trainer = RollingTrainer(self.config_path)
        except ImportError as e:
            logger.error(f"无法导入RollingTrainer: {e}")
            logger.info("使用配置验证模式...")
            return self._validate_config(output_dir)
        
        all_results = {
            "test_time": datetime.now().isoformat(),
            "config_path": self.config_path,
            "n_windows": n_windows,
            "windows": [],
            "summary": {}
        }
        
        ic_by_model = {}
        rank_ic_by_model = {}
        
        windows = list(trainer._generate_windows())
        test_windows = windows[-n_windows:] if len(windows) >= n_windows else windows
        
        for idx, window in enumerate(test_windows):
            logger.info(f"\n{'='*40}")
            logger.info(f"测试窗口 {idx + 1}/{len(test_windows)}")
            logger.info(f"训练期: {window.train_start} ~ {window.train_end}")
            logger.info(f"验证期: {window.valid_start} ~ {window.valid_end}")
            logger.info(f"{'='*40}")
            
            window_result = {
                "window_idx": idx,
                "train_start": str(window.train_start),
                "train_end": str(window.train_end),
                "valid_start": str(window.valid_start),
                "valid_end": str(window.valid_end),
                "models": {}
            }
            
            try:
                trainer.pipeline.build()
                features, labels = trainer.pipeline.get_all()
                
                train_feat, train_lbl = trainer._slice(
                    features, labels, window.train_start, window.train_end, is_validation=False
                )
                valid_feat, valid_lbl = trainer._slice(
                    features, labels, window.valid_start, window.valid_end, is_validation=True
                )
                
                if valid_feat.empty or valid_lbl.empty:
                    logger.warning(f"窗口 {idx} 验证集为空")
                    continue
                
                history_feat = train_feat
                if trainer.label_future_days > 0:
                    try:
                        train_end_actual = train_feat.index.get_level_values("datetime").max()
                        valid_start_actual = valid_feat.index.get_level_values("datetime").min()
                        gap_start = pd.Timestamp(train_end_actual) + pd.Timedelta(days=1)
                        gap_end = pd.Timestamp(valid_start_actual) - pd.Timedelta(days=1)
                        if gap_start <= gap_end:
                            gap_feat = trainer._slice_features_only(features, gap_start, gap_end)
                            if len(gap_feat) > 0:
                                history_feat = pd.concat([train_feat, gap_feat], axis=0).sort_index()
                    except Exception:
                        pass
                
                trainer.ensemble.fit(
                    train_feat, train_lbl,
                    valid_feat, valid_lbl,
                    history_feat=history_feat
                )
                
                _, valid_preds, _ = trainer.ensemble.predict(
                    valid_feat, history_feat=history_feat
                )
                
                if valid_preds is None or not valid_preds:
                    logger.warning(f"窗口 {idx} 预测结果为空")
                    continue
                
                labels_arr = valid_lbl.values
                
                for model_name, pred_series in valid_preds.items():
                    if model_name == 'ensemble':
                        continue
                    
                    pred = pred_series.values if isinstance(pred_series, pd.Series) else pred_series
                    
                    ic = self.calculate_ic(pred, labels_arr, method="pearson")
                    rank_ic = self.calculate_rank_ic(pred, labels_arr)
                    
                    if model_name not in ic_by_model:
                        ic_by_model[model_name] = []
                        rank_ic_by_model[model_name] = []
                    
                    ic_by_model[model_name].append(ic)
                    rank_ic_by_model[model_name].append(rank_ic)
                    
                    window_result["models"][model_name] = {
                        "ic": float(ic) if not np.isnan(ic) else None,
                        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else None,
                        "n_samples": int(len(pred))
                    }
                    
                    logger.info(f"  {model_name}: IC={ic:.4f}, RankIC={rank_ic:.4f}")
                
                all_results["windows"].append(window_result)
                
            except Exception as e:
                logger.error(f"窗口 {idx} 测试失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("\n" + "=" * 60)
        logger.info("ICIR汇总统计")
        logger.info("=" * 60)
        
        for model_name in ic_by_model:
            ic_series = pd.Series(ic_by_model[model_name])
            rank_ic_series = pd.Series(rank_ic_by_model[model_name])
            
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = self.calculate_icir(ic_series)
            
            rank_ic_mean = rank_ic_series.mean()
            rank_ic_std = rank_ic_series.std()
            rank_icir = self.calculate_icir(rank_ic_series)
            
            positive_ratio = (ic_series > 0).mean()
            rank_positive_ratio = (rank_ic_series > 0).mean()
            
            all_results["summary"][model_name] = {
                "ic_mean": float(ic_mean),
                "ic_std": float(ic_std),
                "icir": float(icir),
                "rank_ic_mean": float(rank_ic_mean),
                "rank_ic_std": float(rank_ic_std),
                "rank_icir": float(rank_icir),
                "ic_positive_ratio": float(positive_ratio),
                "rank_ic_positive_ratio": float(rank_positive_ratio),
                "n_windows": int(len(ic_series))
            }
            
            logger.info(f"\n模型: {model_name}")
            logger.info(f"  IC均值: {ic_mean:.4f}")
            logger.info(f"  IC标准差: {ic_std:.4f}")
            logger.info(f"  ICIR: {icir:.4f}")
            logger.info(f"  Rank IC均值: {rank_ic_mean:.4f}")
            logger.info(f"  Rank IC标准差: {rank_ic_std:.4f}")
            logger.info(f"  Rank ICIR: {rank_icir:.4f}")
            logger.info(f"  IC正向比例: {positive_ratio:.2%}")
            logger.info(f"  Rank IC正向比例: {rank_positive_ratio:.2%}")
        
        output_path = os.path.join(output_dir, f"icir_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n测试报告已保存: {output_path}")
        
        return all_results
    
    def _validate_config(self, output_dir: str) -> Dict:
        """验证配置是否正确"""
        logger.info("=" * 60)
        logger.info("配置验证")
        logger.info("=" * 60)
        
        results = {
            "test_time": datetime.now().isoformat(),
            "config_path": self.config_path,
            "validation": {},
            "summary": {}
        }
        
        base_models = self.config.get("base_models", [])
        model_features = self.config.get("model_features", {})
        model_train_days = self.config.get("rolling", {}).get("model_train_days", {})
        
        logger.info(f"基础模型: {base_models}")
        logger.info(f"模型特征配置: {model_features}")
        logger.info(f"模型训练天数: {model_train_days}")
        
        results["validation"] = {
            "base_models": base_models,
            "model_features": model_features,
            "model_train_days": model_train_days,
            "feature_normalization": self.config.get("rolling", {}).get("feature_normalization"),
            "window_mode": self.config.get("rolling", {}).get("window_mode")
        }
        
        data_config = load_yaml_config(self.config.get("data_config"))
        feature_sets = data_config.get("data", {}).get("feature_sets", {})
        
        logger.info(f"\n特征集定义:")
        for name, features in feature_sets.items():
            logger.info(f"  {name}: {len(features)} 个特征")
            results["validation"][f"feature_set_{name}"] = len(features)
        
        for model in base_models:
            feat_set = model_features.get(model)
            train_days = model_train_days.get(model)
            
            results["summary"][model] = {
                "feature_set": feat_set,
                "train_days": train_days,
                "n_features": len(feature_sets.get(feat_set, []))
            }
            
            logger.info(f"\n模型 {model}:")
            logger.info(f"  特征集: {feat_set}")
            logger.info(f"  训练天数: {train_days}")
            logger.info(f"  特征数量: {len(feature_sets.get(feat_set, []))}")
        
        output_path = os.path.join(output_dir, f"config_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n配置验证报告已保存: {output_path}")
        
        return results
    
    def run_quick_test(
        self,
        output_dir: str = "data/logs"
    ) -> Dict:
        """
        快速测试 - 只测试最后一个窗口
        """
        return self.run_rolling_test(n_windows=1, output_dir=output_dir)


def print_comparison_table(results: Dict):
    """打印模型对比表格"""
    if "summary" not in results or not results["summary"]:
        print("无汇总数据")
        return
    
    print("\n" + "=" * 80)
    print("模型ICIR对比表")
    print("=" * 80)
    
    first_value = list(results["summary"].values())[0]
    if "ic_mean" in first_value:
        print(f"{'模型':<10} {'IC均值':>10} {'IC标准差':>10} {'ICIR':>10} {'RankIC均值':>12} {'RankICIR':>12} {'正向比例':>10}")
        print("-" * 80)
        
        for model_name, metrics in results["summary"].items():
            print(f"{model_name:<10} "
                  f"{metrics['ic_mean']:>10.4f} "
                  f"{metrics['ic_std']:>10.4f} "
                  f"{metrics['icir']:>10.4f} "
                  f"{metrics['rank_ic_mean']:>12.4f} "
                  f"{metrics['rank_icir']:>12.4f} "
                  f"{metrics['ic_positive_ratio']:>9.1%}")
    else:
        print(f"{'模型':<10} {'特征集':>15} {'训练天数':>10} {'特征数量':>10}")
        print("-" * 80)
        
        for model_name, metrics in results["summary"].items():
            print(f"{model_name:<10} "
                  f"{metrics['feature_set']:>15} "
                  f"{metrics['train_days']:>10} "
                  f"{metrics['n_features']:>10}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="模型ICIR测试")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/pipeline.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/logs",
        help="输出目录"
    )
    parser.add_argument(
        "--n-windows", 
        type=int, 
        default=3,
        help="测试窗口数量"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="快速测试模式（只测试最后一个窗口）"
    )
    
    args = parser.parse_args()
    
    tester = ICIRTester(args.config)
    
    if args.quick:
        results = tester.run_quick_test(args.output)
    else:
        results = tester.run_rolling_test(args.n_windows, args.output)
    
    print_comparison_table(results)
    
    return results


if __name__ == "__main__":
    main()
