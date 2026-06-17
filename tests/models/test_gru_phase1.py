"""GRU Phase 1 工程修复回归测试。

覆盖：
- P1-2：``GRURegressor.load`` 的 ``dropout`` 默认值不再是 1.2（bug）。
- P1-4：双向 + 2 层网络前向 shape 正确，旧 ckpt 兼容加载。
- P1-6：per-instrument 时序 z-score 全链路 (fit→save→load→predict) 数值稳定。

设计要点：
- 用合成 panel data，不依赖 Qlib 数据初始化，方便在 CI 与本地快速跑通。
- 仅校验关键不变量（shape / 是否为 NaN / mean·std 数值范围），不验证模型预测准确性。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from models.gru_model import GRURegressor, _GRUNet, _GRUNetWithAttention


def _make_panel(n_dates: int = 60, n_inst: int = 4, n_feats: int = 6, seed: int = 0):
    """构造一个最小可训练的 panel。返回 (feat_df, label_series)。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    insts = [f"S{i:02d}" for i in range(n_inst)]
    rows_idx = []
    rows_X = []
    rows_y = []
    for inst in insts:
        scale = float(rng.uniform(0.5, 5.0))
        offset = float(rng.uniform(-3, 3))
        x = rng.normal(loc=offset, scale=scale, size=(n_dates, n_feats)).astype(np.float32)
        # y = 简单线性组合 + 噪声，保证 ListMLE 等 rank-loss 有信号
        coef = rng.normal(size=n_feats).astype(np.float32)
        y = (x @ coef) + rng.normal(scale=0.3, size=n_dates).astype(np.float32)
        for k, d in enumerate(dates):
            rows_idx.append((d, inst))
            rows_X.append(x[k])
            rows_y.append(float(y[k]))
    idx = pd.MultiIndex.from_tuples(rows_idx, names=["datetime", "instrument"])
    feat = pd.DataFrame(np.array(rows_X), index=idx, columns=[f"f{i}" for i in range(n_feats)])
    label = pd.Series(rows_y, index=idx, name="label")
    return feat, label


# --------------------------- P1-2 ---------------------------
def test_load_dropout_default_not_buggy(tmp_path):
    """P1-2：load() 的 dropout 默认值不应是 1.2（PyTorch 报错阈值=1.0）。"""
    # 构造一个最小 ckpt（无 dropout 字段）
    cfg_no_dropout = {
        "hidden_size": 16,
        "num_layers": 1,
        "attention_type": "none",
        "bidirectional": False,
    }
    net = _GRUNet(input_dim=4, hidden_size=16, num_layers=1, dropout=0.2, bidirectional=False)
    ckpt = {
        "state_dict": net.state_dict(),
        "config": cfg_no_dropout,
        "feature_names": ["f0", "f1", "f2", "f3"],
        "input_dim": 4,
        "best_metric": None,
        "history": [],
    }
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    path = out_dir / "tst_gru.pt"
    torch.save(ckpt, str(path))

    reg = GRURegressor(config=cfg_no_dropout)
    reg.load(str(out_dir), "tst")
    # 通过检查：load 没有抛异常 + 内部 config 的 dropout 默认值合理
    used_dropout = float(reg.config.get("dropout", 0.2))
    assert used_dropout < 0.5, f"dropout 默认值仍可能落到 1.2 这种 bug 区间: {used_dropout}"


# --------------------------- P1-4 ---------------------------
def test_bidirectional_forward_shape():
    """P1-4：双向 GRU 与双向 + self-attention 的前向 shape 正确。"""
    x = torch.randn(8, 20, 6)

    net_uni = _GRUNet(input_dim=6, hidden_size=16, num_layers=2, dropout=0.2, bidirectional=False)
    assert net_uni(x).shape == (8, 1)

    net_bi = _GRUNet(input_dim=6, hidden_size=16, num_layers=2, dropout=0.2, bidirectional=True)
    assert net_bi(x).shape == (8, 1)
    # head 输入维度应为 hidden_size * 2
    assert net_bi.head.in_features == 16 * 2

    net_attn = _GRUNetWithAttention(
        input_dim=6, hidden_size=16, num_layers=2, dropout=0.2,
        attention_type="self_attention", num_heads=4, bidirectional=True,
    )
    assert net_attn(x).shape == (8, 1)
    assert net_attn._out_dim == 16 * 2


def test_old_ckpt_compat_load(tmp_path):
    """P1-4：旧版（单向）ckpt 即使 config 里写 bidirectional=true，也能正确加载。"""
    # 构造一个单向 ckpt
    cfg_old = {"hidden_size": 16, "num_layers": 1, "attention_type": "none", "bidirectional": False}
    net = _GRUNet(input_dim=4, hidden_size=16, num_layers=1, dropout=0.2, bidirectional=False)
    ckpt = {
        "state_dict": net.state_dict(),
        "config": cfg_old,
        "feature_names": ["f0", "f1", "f2", "f3"],
        "input_dim": 4,
    }
    out_dir = tmp_path / "old_ckpt"
    out_dir.mkdir()
    torch.save(ckpt, str(out_dir / "old_gru.pt"))

    # 强制 config 里 bidirectional=true（模拟用户先升级 yaml 再加载旧 ckpt）
    cfg_new = {"hidden_size": 16, "num_layers": 1, "attention_type": "none", "bidirectional": True}
    reg = GRURegressor(config=cfg_new)
    reg.load(str(out_dir), "old")  # 不应抛异常；自动 fallback 到 bidirectional=False
    assert reg.model is not None


# --------------------------- P1-6 ---------------------------
def test_per_instrument_norm_roundtrip(tmp_path):
    """P1-6：fit -> save -> load -> predict 全链路保留 per-instrument 统计量。"""
    feat, label = _make_panel(n_dates=80, n_inst=3, n_feats=4, seed=7)
    # 划分 train/valid，为了快收敛 epochs 设小一点
    tr_dates = feat.index.get_level_values("datetime").unique()[:60]
    va_dates = feat.index.get_level_values("datetime").unique()[60:]
    tr_mask = feat.index.get_level_values("datetime").isin(tr_dates)
    va_mask = feat.index.get_level_values("datetime").isin(va_dates)
    tr_feat, tr_label = feat[tr_mask], label[tr_mask]
    va_feat, va_label = feat[va_mask], label[va_mask]

    cfg = {
        "seq_len": 8,
        "hidden_size": 8,
        "num_layers": 1,
        "bidirectional": False,
        "attention_type": "none",
        "dropout": 0.0,
        "lr": 1e-3,
        "batch_size": 64,
        "max_epochs": 1,
        "patience": 1,
        "loss": "mse",
        "early_stopping_metric": "loss",
        "per_instrument_norm": True,
        "amp": False,
    }
    reg = GRURegressor(config=cfg)
    reg.fit(tr_feat, tr_label, va_feat, va_label)
    assert reg._inst_norm_stats is not None, "fit 后未生成 per-instrument 统计"
    assert "__global__" in reg._inst_norm_stats
    insts_in_stats = {k for k in reg._inst_norm_stats.keys() if k != "__global__"}
    assert insts_in_stats == set(tr_feat.index.get_level_values("instrument").unique().astype(str))

    # 预测：测试保存/加载链路
    out_dir = tmp_path / "gru_p16"
    out_dir.mkdir()
    reg.save(str(out_dir), "test")
    reg2 = GRURegressor(config=cfg)
    reg2.load(str(out_dir), "test")
    assert reg2._inst_norm_stats is not None, "load 后丢失 per-instrument 统计"

    pred = reg2.predict(va_feat, history_feat=tr_feat)
    # 至少有部分预测非 NaN（因 seq_len=8，valid 起始 7 行可能 NaN）
    assert pred.notna().sum() > 0, "predict 全 NaN，per-instrument 归一化或序列构造异常"


def test_per_instrument_norm_disabled_path(tmp_path):
    """P1-6：当 per_instrument_norm=False，行为与旧版一致（不破坏既有用例）。"""
    feat, label = _make_panel(n_dates=40, n_inst=2, n_feats=3, seed=9)
    cfg = {
        "seq_len": 6,
        "hidden_size": 4,
        "num_layers": 1,
        "bidirectional": False,
        "attention_type": "none",
        "dropout": 0.0,
        "lr": 1e-3,
        "batch_size": 32,
        "max_epochs": 1,
        "patience": 1,
        "loss": "mse",
        "early_stopping_metric": "loss",
        "per_instrument_norm": False,
        "amp": False,
    }
    reg = GRURegressor(config=cfg)
    reg.fit(feat, label)
    assert reg._inst_norm_stats is None
