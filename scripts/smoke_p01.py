"""
P0-1 冒烟脚本：验证 RD-Agent parquet → Qlib 特征合并 → LGB 取列 三步闭环。

- 在 qlib_zhengshi 环境里运行：
    conda run -n qlib_zhengshi python scripts/smoke_p01.py

- 与生产训练解耦：用 parquet 实际覆盖的时间范围（2022 年）+ csi300，
  避免因 data.yaml 的 2025-10-01~2026-04-07 与 parquet(2020~2022) 不重叠
  导致看不到合并效果。
"""
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from feature.qlib_feature_pipeline import QlibFeaturePipeline  # noqa: E402
from models.ensemble_manager import EnsembleModelManager  # noqa: E402
from utils import load_yaml_config  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log = logging.getLogger("smoke_p01")

    pq_path = _ROOT / "git_ignore_folder" / "combined_factors_df.parquet"
    if not pq_path.exists():
        log.error("parquet 不存在：%s", pq_path)
        sys.exit(1)

    pq = pd.read_parquet(str(pq_path))
    dt = pq.index.get_level_values("datetime")
    start = pd.Timestamp(dt.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(dt.max()).strftime("%Y-%m-%d")
    log.info("parquet: shape=%s, range=%s~%s, cols=%s", pq.shape, start, end, list(pq.columns))

    # 只取最后 60 个自然日做冒烟，避免 D.features 全量慢
    last = pd.Timestamp(dt.max())
    smoke_start = (last - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    data_cfg = copy.deepcopy(load_yaml_config(str(_ROOT / "config" / "data.yaml")))
    data_cfg["data"]["start_time"] = smoke_start
    data_cfg["data"]["end_time"] = end
    data_cfg["data"]["instruments"] = "csi300"
    data_cfg["data"]["active_feature_sets"] = ["lgb_short_cycle", "rdagent_exported"]
    data_cfg["data"]["label_transform"] = {"enabled": False}

    log.info("=== Step1: QlibFeaturePipeline.build ===")
    pipe = QlibFeaturePipeline(data_cfg)
    pipe.build(include_label=True)
    feats = pipe.features_df
    rd_cols = list(getattr(pipe, "rdagent_factor_columns", []) or [])
    log.info("features_df shape=%s", feats.shape)
    log.info("rdagent_factor_columns=%s", rd_cols)

    inter = [c for c in rd_cols if c in feats.columns]
    log.info("RD-Agent 列落入 features_df 数量=%d / %d", len(inter), len(rd_cols))
    if inter:
        sub = feats[inter]
        nan_ratio = sub.isna().mean().sort_values(ascending=False)
        log.info("合并后 RD-Agent 列 NaN 占比（top5）：\n%s", nan_ratio.head(5).to_string())
        valid_rows = sub.dropna(how="all").shape[0]
        log.info("至少有一个 RD-Agent 非空的行数=%d / %d", valid_rows, len(sub))

    log.info("=== Step2: EnsembleModelManager.update_feature_set ===")
    pipeline_cfg = load_yaml_config(str(_ROOT / "config" / "pipeline.yaml"))
    pipeline_cfg["data_config"] = str(_ROOT / "config" / "data.yaml")
    ensemble = EnsembleModelManager(pipeline_cfg, pipeline_cfg.get("ensemble"))
    ensemble.update_feature_set("rdagent_exported", rd_cols)

    log.info("=== Step3: LGB _resolve_feature_cols ===")
    cols_lgb = ensemble._resolve_feature_cols("lgb", list(feats.columns))
    log.info("lgb spec -> 解析列数=%d", len(cols_lgb or []))
    if cols_lgb:
        from_rd = [c for c in cols_lgb if c in set(rd_cols)]
        from_lgb_set = [c for c in cols_lgb if c not in set(rd_cols)]
        log.info("  - 来自 rdagent_exported: %d 列", len(from_rd))
        log.info("  - 来自 lgb_short_cycle : %d 列", len(from_lgb_set))
        log.info("  - 示例(后5 = RD-Agent 侧): %s", cols_lgb[-5:])

    cols_gru = ensemble._resolve_feature_cols("gru", list(feats.columns))
    has_rd_in_gru = any(c in set(rd_cols) for c in (cols_gru or []))
    log.info("gru spec -> 解析列数=%d, 是否含 RD-Agent 列=%s", len(cols_gru or []), has_rd_in_gru)

    log.info("=== SMOKE DONE ===")


if __name__ == "__main__":
    main()
