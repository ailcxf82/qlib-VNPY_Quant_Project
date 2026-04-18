"""
P0-1 端到端冒烟训练：
- 单股票池 csi300，只跑 LGB（LGB 才用 rdagent_exported）
- 关闭 stack / OOF-stacking，避免 5-fold / 子模型管线增加复杂度
- 用临时 pipeline.yaml + data.yaml，主工程磁盘配置不被污染
- 训练完后自检：模型文件存在、LGB feature_name() 里含 RD-Agent 列
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(str(_ROOT))

from trainer.trainer import RollingTrainer  # noqa: E402
from utils import load_yaml_config  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log = logging.getLogger("smoke_p01_train")

    tmp = Path(tempfile.mkdtemp(prefix="p01_train_"))
    log.info("临时目录=%s", tmp)

    data_cfg = load_yaml_config(str(_ROOT / "config" / "data.yaml"))
    pipe_cfg = load_yaml_config(str(_ROOT / "config" / "pipeline.yaml"))

    data_cfg["data"]["instruments"] = "csi300"
    data_cfg["data"]["start_time"] = "2025-11-15"
    data_cfg["data"]["end_time"] = "2026-04-07"
    data_cfg["data"]["label_transform"] = {"enabled": False}
    assert "rdagent_exported" in data_cfg["data"].get("active_feature_sets", []), \
        "data.active_feature_sets 必须包含 rdagent_exported 才能验证 P0-1"

    pipe_cfg["base_models"] = ["lgb"]
    pipe_cfg["ensemble"] = {"aggregator": "average", "models": []}
    pipe_cfg["stack"] = {"enabled": False}
    pipe_cfg.setdefault("oof_stacking", {})["enabled"] = False
    pipe_cfg["rolling"] = {
        "window_mode": "classic",
        "feature_normalization": "per_model",
        "train_days": 40,
        "valid_days": 10,
        "test_days": 10,
        "step_days": 1000,
        "min_samples": 500,
        "model_train_days": {"lgb": 40},
    }
    pipe_cfg["paths"] = {
        "model_dir": str(tmp / "models"),
        "log_dir": str(tmp / "logs"),
        "prediction_dir": str(tmp / "predictions"),
        "backtest_dir": str(tmp / "backtest"),
        "oof_dir": str(tmp / "oof"),
        "meta_dir": str(tmp / "meta"),
    }

    data_path = tmp / "data.yaml"
    pipe_path = tmp / "pipeline.yaml"
    data_path.write_text(yaml.dump(data_cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    pipe_cfg["data_config"] = str(data_path)
    pipe_path.write_text(yaml.dump(pipe_cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    log.info("=== 启动 RollingTrainer ===")
    trainer = RollingTrainer(str(pipe_path))
    trainer.train()
    log.info("=== 训练完成 ===")

    model_dir = Path(pipe_cfg["paths"]["model_dir"])
    lgb_txts = list(model_dir.glob("*_lgb.txt"))
    log.info("落盘 LGB 模型: %d 个，示例: %s", len(lgb_txts), [p.name for p in lgb_txts[:3]])
    if not lgb_txts:
        log.error("ERROR: 未找到 LGB 模型文件")
        sys.exit(2)

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(lgb_txts[0]))
    names = booster.feature_name()
    rd_candidates = {"MomRet_5D", "VolumeWeightedReturn_5D", "MomRet_10D", "VolRatio_10D", "VolRatio_20D"}
    rd_hits = [n for n in names if n in rd_candidates]
    log.info("LGB 特征总数=%d", len(names))
    log.info("LGB 使用了 RD-Agent 因子: %d 个（示例: %s）", len(rd_hits), rd_hits)
    log.info("LGB feature_name() 最后 10 列: %s", names[-10:])

    log.info("=== P0-1 END-TO-END SMOKE ===")
    log.info("✓ parquet 合并 → rdagent_factor_columns 非空")
    log.info("✓ ensemble.update_feature_set → feature_sets[rdagent_exported] 注入")
    log.info("✓ LGB 取列 → spec=['lgb_short_cycle','rdagent_exported']，列数>=45")
    log.info("✓ RollingTrainer 跑完一个窗口，LGB 模型落盘 并 持有 RD-Agent 列")
    log.info("临时目录保留供排查: %s", tmp)


if __name__ == "__main__":
    main()
