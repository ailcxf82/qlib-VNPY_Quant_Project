"""``run_predict_chan._load_ic_histories`` Phase 1 P1-1 回归测试。

不变量：含 ``ic_gru`` 列的 training_metrics.csv 加载后，返回 dict 必须包含
非空的 ``"gru"`` 键；缺失时回退到 ``ic_lgb``。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    """直接按文件路径加载 run_predict_chan，避免依赖 import path。"""
    project_root = Path(__file__).resolve().parents[2]
    py_path = project_root / "run_predict_chan.py"
    spec = importlib.util.spec_from_file_location("_run_predict_chan_for_test", str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_log(tmp_path, with_gru: bool):
    n = 5
    df = pd.DataFrame(
        {
            "window": list(range(n)),
            "train_start": ["2024-01-01"] * n,
            "train_end": ["2024-06-30"] * n,
            "valid_start": ["2024-07-01"] * n,
            "valid_end": [f"2024-{m:02d}-30" for m in range(7, 12)],
            "ic_lgb": [0.030, 0.025, 0.028, 0.022, 0.031],
            "ic_mlp": [0.020, 0.018, 0.021, 0.019, 0.020],
            "ic_stack": [0.032, 0.027, 0.030, 0.024, 0.033],
        }
    )
    if with_gru:
        df["ic_gru"] = [0.024, 0.020, 0.022, 0.018, 0.025]
    p = tmp_path / "training_metrics.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_ic_gru_loaded_when_present(tmp_path):
    mod = _load_module()
    p = _write_log(tmp_path, with_gru=True)
    histories = mod._load_ic_histories(p)
    assert "gru" in histories, "P1-1 回归：histories 必须包含 'gru' 键"
    assert len(histories["gru"]) == 5
    # 数值与写入一致
    assert abs(float(histories["gru"].iloc[0]) - 0.024) < 1e-9


def test_ic_gru_falls_back_to_lgb(tmp_path):
    mod = _load_module()
    p = _write_log(tmp_path, with_gru=False)
    histories = mod._load_ic_histories(p)
    assert "gru" in histories, "缺 ic_gru 时也应有 gru 兜底键"
    # 兜底实现：完全等于 ic_lgb
    assert (histories["gru"].values == histories["lgb"].values).all()


def test_default_when_log_missing(tmp_path):
    mod = _load_module()
    histories = mod._load_ic_histories(str(tmp_path / "missing.csv"))
    # 文件不存在时也必须有 gru
    assert "gru" in histories
    assert len(histories["gru"]) >= 1
