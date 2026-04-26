"""
因子驱动模型训练脚本 — Windows 原生运行（无需 WSL）
=====================================================
整合 RD-Agent 产出的最优 AI 因子 + 34 个基础特征，
训练 LightGBM GBDT 模型并输出 IC / ICIR / 投组分析。

用法：
    conda activate qlib_zhengshi
    cd D:\\quant_project\\Qlib_Quant\\qlib-VNPY_Quant_Project
    python scripts/train_with_factors.py [--workspace <ws_id>] [--exp_name <名称>]

参数：
    --workspace  使用哪个 RD-Agent workspace 的 AI 因子（默认：ICIR 最高的那个）
    --exp_name   MLflow 实验名（默认：factor_prod_train）
    --train_end  训练集截止日（默认：2025-06-30）
    --valid_end  验证集截止日（默认：2025-12-31）
    --test_end   测试集截止日（默认：2026-04-23）
    --topk       组合持仓数量（默认：50）
    --no_port    跳过组合回测，只做信号分析
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ─── 项目根 ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"
QLIB_DATA = "D:/qlib_data/qlib_data"

# 优先使用已知的全覆盖 workspace（38418fc7）
PREFERRED_WS = "38418fc744744b9695e592f385e69e3c"


# ─── 选择最优 workspace ───────────────────────────────────────────────────────
def _pick_workspace(explicit: str | None) -> Path:
    """返回包含 combined_factors_df.parquet 的最优 workspace 目录。"""
    if explicit:
        ws = WS_BASE / explicit
        if not (ws / "combined_factors_df.parquet").exists():
            raise FileNotFoundError(f"workspace {explicit} 没有 parquet 文件")
        return ws

    # 优先使用已知全覆盖的那个
    preferred = WS_BASE / PREFERRED_WS
    if (preferred / "combined_factors_df.parquet").exists():
        return preferred

    # Fallback：扫描并按 ICIR 排序
    import pandas as pd
    best = None
    best_icir = -1.0
    for ws_dir in WS_BASE.iterdir():
        if not ws_dir.is_dir():
            continue
        csv = ws_dir / "qlib_res.csv"
        pq = ws_dir / "combined_factors_df.parquet"
        if not csv.exists() or not pq.exists():
            continue
        try:
            m = pd.read_csv(csv, index_col=0)
            m.columns = ["value"]
            icir = float(m.loc["ICIR", "value"])
            if icir > best_icir:
                best_icir = icir
                best = ws_dir
        except Exception:
            pass
    if best is None:
        raise RuntimeError("找不到任何含 parquet 的 workspace，请先运行 RD-Agent loop。")
    return best


# ─── 主训练流程 ───────────────────────────────────────────────────────────────
def train(
    workspace: str | None = None,
    exp_name: str = "factor_prod_train",
    train_end: str = "2025-06-30",
    valid_end: str = "2025-12-31",
    test_end: str = "2026-04-23",
    topk: int = 50,
    no_port: bool = False,
) -> None:
    ws_dir = _pick_workspace(workspace)
    print(f"\n{'='*60}")
    print(f"  使用 workspace : {ws_dir.name}")

    import pandas as pd
    pq = pd.read_parquet(ws_dir / "combined_factors_df.parquet")
    ai_factors = list(pq.columns.get_level_values(-1) if pq.columns.nlevels > 1 else pq.columns)
    dts = pq.index.get_level_values("datetime")
    print(f"  AI 因子数量   : {len(ai_factors)}")
    print(f"  AI 因子名称   : {ai_factors}")
    print(f"  因子数据日期  : {dts.min().date()} → {dts.max().date()}")
    print(f"  训练期        : 2022-01-01 → {train_end}")
    print(f"  验证期        : {train_end[:7].replace('-06','-07').replace('-12','-01')}-01 → {valid_end}")
    print(f"  测试期（OOS）  : 2026-01-01 → {test_end}")
    print(f"{'='*60}\n")

    # ── Step 1：切换到 workspace 目录（StaticDataLoader 使用相对路径）
    original_cwd = os.getcwd()
    os.chdir(ws_dir)
    print(f"[1/5] 切换工作目录到 workspace: {ws_dir.name}")

    try:
        # ── Step 2：初始化 Qlib ───────────────────────────────────────────────
        print(f"[2/5] 初始化 Qlib (provider_uri={QLIB_DATA}) ...")
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

        # ── Step 3：构建 DataHandler ─────────────────────────────────────────
        print("[3/5] 构建数据集（基础特征 + AI 因子）...")
        from qlib.data.dataset.loader import QlibDataLoader, StaticDataLoader, NestedDataLoader
        from qlib.contrib.data.handler import DataHandlerLP
        from qlib.data.dataset import DatasetH

        # 基础特征：34 个价量/估值/技术指标（与 conf_combined_factors.yaml 一致）
        BASE_EXPRS = [
            "$close_qfq/Ref($close_qfq,1)-1",
            "$close_qfq/Ref($close_qfq,5)-1",
            "$close_qfq/Ref($close_qfq,10)-1",
            "$close_qfq/Ref($close_qfq,20)-1",
            "Ref($close_qfq,1)/Ref($close_qfq,5)-1",
            "Std($close_qfq/Ref($close_qfq,1)-1,5)",
            "Std($close_qfq/Ref($close_qfq,1)-1,10)",
            "Std($close_qfq/Ref($close_qfq,1)-1,20)",
            "$vol/Mean($vol,5)",
            "$vol/Mean($vol,10)",
            "$vol/Mean($vol,20)",
            "($high_qfq-$low_qfq)/$close_qfq",
            "Mean(($high_qfq-$low_qfq)/$close_qfq,5)",
            "Mean(($high_qfq-$low_qfq)/$close_qfq,10)",
            "$turnover_rate",
            "$turnover_rate_f",
            "$turnover_rate/Mean($turnover_rate,20)-1",
            "$volume_ratio",
            "$pe_ttm",
            "$pb",
            "$ps_ttm",
            "Log($total_mv)",
            "$dv_ratio",
            "$roe",
            "$roa",
            "$q_profit_yoy",
            "$q_eps",
            "$rsi_qfq_12",
            "$macd_qfq",
            "$kdj_k_qfq-$kdj_d_qfq",
            "$atr_qfq",
            "$rzye/$total_mv",
            "$rqye/$total_mv",
        ]
        BASE_NAMES = [
            "RET1","MOM5","MOM10","MOM20","MOM1_5",
            "VOL5","VOL10","VOL20",
            "VRATIO5","VRATIO10","VRATIO20",
            "HLRANGE","HLRANGE5","HLRANGE10",
            "TURN","TURN_F","TURN_REL","VOLVOL",
            "PE_TTM","PB","PS_TTM","LOG_MV","DV",
            "ROE","ROA","PROFIT_YOY","EPS",
            "RSI12","MACD","KDJ_DIFF","ATR",
            "MARGIN_L","MARGIN_S",
        ]
        LABEL_EXPR = "Ref($close_qfq,-3)/Ref($close_qfq,1)-1"

        handler_kwargs = dict(
            start_time="2022-01-01",
            end_time=test_end,
            instruments="all",
            data_loader={
                "class": "NestedDataLoader",
                "kwargs": {
                    "dataloader_l": [
                        {
                            "class": "QlibDataLoader",
                            "kwargs": {
                                "config": {
                                    "feature": [BASE_EXPRS, BASE_NAMES],
                                    "label": [[LABEL_EXPR], ["LABEL0"]],
                                }
                            },
                        },
                        {
                            "class": "StaticDataLoader",
                            "kwargs": {"config": "combined_factors_df.parquet"},
                        },
                    ]
                },
            },
            infer_processors=[
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": "2022-01-01",
                    "fit_end_time": train_end,
                }},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        )

        # 推算 valid_start（train_end 后一天）
        from datetime import datetime, timedelta
        train_end_dt = datetime.strptime(train_end, "%Y-%m-%d")
        valid_start = (train_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        valid_end_dt = datetime.strptime(valid_end, "%Y-%m-%d")
        test_start = (valid_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        handler = DataHandlerLP(**handler_kwargs)
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": ("2022-01-01", train_end),
                "valid": (valid_start, valid_end),
                "test":  (test_start, test_end),
            },
        )

        # ── Step 4：训练模型 ─────────────────────────────────────────────────
        print("[4/5] 训练 LightGBM 模型...")
        import mlflow
        from qlib.workflow import R
        from qlib.contrib.model.gbdt import LGBModel

        model = LGBModel(
            loss="mse",
            colsample_bytree=0.72,
            learning_rate=0.05,
            subsample=0.8,
            lambda_l1=50.0,
            lambda_l2=100.0,
            max_depth=6,
            num_leaves=63,
            num_threads=min(20, os.cpu_count() or 4),
            n_estimators=500,
            early_stopping_rounds=50,
            min_child_samples=50,
        )

        # 用 Qlib R (Recorder) 管理实验
        with R.start(experiment_name=exp_name):
            R.log_params(
                workspace=ws_dir.name,
                ai_factors=str(ai_factors),
                n_base_features=len(BASE_NAMES),
                n_ai_factors=len(ai_factors),
                train_end=train_end,
                valid_end=valid_end,
                test_end=test_end,
                topk=topk,
            )

            model.fit(dataset)
            recorder = R.get_recorder()
            recorder.save_objects(**{"model.pkl": model})

            # ── Step 5：评估 ─────────────────────────────────────────────────
            print("[5/5] 评估模型...")
            from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

            sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
            sr.generate()

            sar = SigAnaRecord(recorder=recorder, ana_long_short=False, ann_scaler=252)
            sar.generate()

            # 组合回测（可选）
            if not no_port:
                try:
                    from qlib.workflow.record_temp import PortAnaRecord
                    from qlib.contrib.strategy import TopkDropoutStrategy
                    par = PortAnaRecord(
                        recorder=recorder,
                        config={
                            "strategy": {
                                "class": "TopkDropoutStrategy",
                                "module_path": "qlib.contrib.strategy",
                                "kwargs": {"signal": "<PRED>", "topk": topk, "n_drop": 5},
                            },
                            "backtest": {
                                "start_time": test_start,
                                "end_time": test_end,
                                "account": 100_000_000,
                                "benchmark": "SH000300",
                                "exchange_kwargs": {
                                    "limit_threshold": 0.095,
                                    "deal_price": "close",
                                    "open_cost": 0.0003,
                                    "close_cost": 0.0013,
                                    "min_cost": 5,
                                },
                            },
                        },
                    )
                    par.generate()
                    print("  组合回测完成。")
                except Exception as e:
                    print(f"  [跳过组合回测] {e}")

            # ── 打印核心指标 ────────────────────────────────────────────────

            print(f"\n{'='*60}")
            print("  训练结果摘要")
            print(f"{'='*60}")
            try:
                metrics = recorder.list_metrics()
                for k in ["IC", "ICIR", "Rank IC", "Rank ICIR",
                          "1day.excess_return_with_cost.annualized_return",
                          "1day.excess_return_with_cost.information_ratio",
                          "1day.excess_return_with_cost.max_drawdown"]:
                    v = metrics.get(k)
                    if v is not None:
                        print(f"  {k:50s}: {v:.4f}")
            except Exception as e:
                print(f"  指标读取失败: {e}")

            rid = recorder.id
            eid = recorder.experiment.id
            print(f"\n  实验 ID    : {eid}")
            print(f"  记录器 ID  : {rid}")
            print(f"  查看 MLflow: mlflow ui --backend-store-uri {ws_dir}/mlruns")
            print(f"{'='*60}\n")

    finally:
        os.chdir(original_cwd)


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 因子驱动 LightGBM 训练")
    parser.add_argument("--workspace", default=None, help="workspace ID（默认自动选择最优）")
    parser.add_argument("--exp_name", default="factor_prod_train", help="实验名称")
    parser.add_argument("--train_end", default="2025-06-30", help="训练集截止日")
    parser.add_argument("--valid_end", default="2025-12-31", help="验证集截止日")
    parser.add_argument("--test_end", default="2026-04-23", help="测试集截止日")
    parser.add_argument("--topk", type=int, default=50, help="组合持仓数量")
    parser.add_argument("--no_port", action="store_true", help="跳过组合回测")
    args = parser.parse_args()

    train(
        workspace=args.workspace,
        exp_name=args.exp_name,
        train_end=args.train_end,
        valid_end=args.valid_end,
        test_end=args.test_end,
        topk=args.topk,
        no_port=args.no_port,
    )
