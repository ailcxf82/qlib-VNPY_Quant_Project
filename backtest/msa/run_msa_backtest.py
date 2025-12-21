"""
MSA 多策略回测启动脚本（RQAlpha）。

用法示例：
python backtest/msa/run_msa_backtest.py ^
  --rqalpha-config config/rqalpha_config.yaml ^
  --pred-csi101 data/predictions/pred_csi101_xxx.csv ^
  --pred-csi300 data/predictions/pred_csi300_xxx.csv

可选：
- 设置环境变量 TUSHARE_TOKEN 启用 Tushare 过滤
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import glob
from typing import Optional

# 允许直接运行本文件（python backtest/msa/run_msa_backtest.py）时也能正确导入项目包
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest.rqalpha_backtest import run_rqalpha_backtest as _run_single  # 复用配置/输出/数据预处理逻辑
from utils import load_yaml_config


def _resolve_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p).strip()
    # 兼容用户可能写成 "@pred_xxx.csv" 的情况
    if p.startswith("@"):
        p = p[1:]
    p = p.replace("/", os.sep)
    if os.path.isabs(p):
        return p
    # 相对路径统一相对于项目根目录解析（避免 chdir 影响）
    return os.path.join(_PROJECT_ROOT, p)


def _find_latest_prediction(pool: str) -> str:
    pred_dir = os.path.join(_PROJECT_ROOT, "data", "predictions")
    patterns = [
        os.path.join(pred_dir, f"pred_{pool}_*.csv"),
        os.path.join(pred_dir, f"*{pool}*pred*.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"未找到 {pool} 的预测文件，请在 data/predictions 下生成 pred_{pool}_*.csv 或手动传入 --pred-{pool}")
    return max(files, key=os.path.getmtime)


def parse_args():
    p = argparse.ArgumentParser(description="MSA 多策略 RQAlpha 回测")
    p.add_argument("--rqalpha-config", type=str, default="config/rqalpha_config.yaml")
    p.add_argument("--pred-csi101", type=str, default=None, help="CSI101 预测文件（csv，含 datetime/instrument/final），不传则自动选 data/predictions 下最新文件")
    p.add_argument("--pred-csi300", type=str, default=None, help="CSI300 预测文件（csv，含 datetime/instrument/final），不传则自动选 data/predictions 下最新文件")
    p.add_argument("--industry", type=str, default=None, help="行业映射文件（可选，暂未强制使用）")
    p.add_argument("--strategy", type=str, default="backtest/msa/rqalpha_msa_strategy.py", help="MSA 策略脚本路径")
    # allocations
    p.add_argument("--alloc1", type=float, default=0.5, help="策略1（csi101）资金占比")
    p.add_argument("--alloc2", type=float, default=0.5, help="策略2（csi300）资金占比")
    # risk
    p.add_argument("--drawdown-stop", type=float, default=0.08, help="个股回撤止损阈值（如0.08=8%）")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    # 工作目录固定为项目根目录，避免相对路径解析错误
    os.chdir(_PROJECT_ROOT)

    pred_csi101 = _resolve_path(args.pred_csi101) or _find_latest_prediction("csi101")
    pred_csi300 = _resolve_path(args.pred_csi300) or _find_latest_prediction("csi300")

    if not os.path.exists(pred_csi101):
        raise FileNotFoundError(f"csi101 预测文件不存在: {pred_csi101}")
    if not os.path.exists(pred_csi300):
        raise FileNotFoundError(f"csi300 预测文件不存在: {pred_csi300}")

    # 复用 rqalpha_backtest 的 run_file/config 逻辑，但通过 extra.context_vars 传入 MSA 所需参数
    cfg = load_yaml_config(_resolve_path(args.rqalpha_config) or args.rqalpha_config)
    cfg.setdefault("extra", {}).setdefault("context_vars", {})
    cv = cfg["extra"]["context_vars"]

    # 明确打印/校验回测周期：必须以 rqalpha_config.yaml 的 base.start_date/base.end_date 为准
    base_cfg = cfg.get("base", {}) if isinstance(cfg, dict) else {}
    start_date = (base_cfg.get("start_date") or "").strip() if isinstance(base_cfg, dict) else ""
    end_date = (base_cfg.get("end_date") or "").strip() if isinstance(base_cfg, dict) else ""
    if not start_date or not end_date:
        raise ValueError(
            "MSA 回测必须在 rqalpha_config.yaml 的 base.start_date/base.end_date 中显式设置回测周期（例如 2025-10-01~2025-12-18）"
        )
    logging.info("MSA 回测周期（来自 %s）: %s ~ %s", args.rqalpha_config, start_date, end_date)

    cv["pred_csi101"] = pred_csi101
    cv["pred_csi300"] = pred_csi300
    cv["alloc_strategy1"] = args.alloc1
    cv["alloc_strategy2"] = args.alloc2
    cv["drawdown_stop"] = args.drawdown_stop

    # 将修改后的配置写入临时文件，交给现有 runner 执行
    import tempfile
    import yaml

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()

    try:
        # 这里把 prediction_path 传一个占位（必须传），真正使用的是 context_vars 里的 pred_csi101/pred_csi300
        # 注意：strategy_path 指向 MSA 策略脚本
        _run_single(
            rqalpha_config_path=tmp.name,
            prediction_path=pred_csi101,
            industry_path=args.industry,
            strategy_path=_resolve_path(args.strategy) or args.strategy,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()


