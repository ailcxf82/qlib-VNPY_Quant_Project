"""
GRU 基模型：将 panel 特征按 (instrument, datetime) 组装为序列，用 GRU 预测 label。

接口保持 sklearn-like：
  - fit(train_feat, train_label, valid_feat=None, valid_label=None)
  - predict(feat) -> pd.Series（索引为 MultiIndex[datetime,instrument]，不足 seq_len 的位置为 NaN）
  - save/load
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.sequence_builder import build_panel_sequences
from utils import load_yaml_config
from utils.torch_utils import set_global_seed

logger = logging.getLogger(__name__)


class _GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)  # (B, T, H)
        last = out[:, -1, :]  # (B, H)
        y = self.head(last)   # (B, 1)
        return y


class GRURegressor:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            raw = load_yaml_config(config)
        else:
            raw = config
        self.config = raw.get("model", raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None
        self._feature_names: Optional[list[str]] = None
        self._best_state: Optional[dict] = None
        self._best_metric: Optional[float] = None
        self._history: list[dict] = []

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        # 记录 NaN 特征分布，便于后续数据优化
        try:
            report_dir = str(self.config.get("nan_report_dir", os.path.join("data", "logs")))
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "gru_nan_report.csv")

            def _nan_report(df: pd.DataFrame, split: str):
                if df is None or len(df) == 0:
                    return
                total = len(df)
                nan_count = df.isna().sum()
                nan_ratio = (nan_count / max(1, total)).astype(float)
                # 只记录有 NaN 的特征
                mask = nan_count > 0
                ts = df.index.get_level_values("datetime")
                window_start = pd.to_datetime(ts.min()).strftime("%Y-%m-%d")
                window_end = pd.to_datetime(ts.max()).strftime("%Y-%m-%d")
                if mask.any():
                    rows = pd.DataFrame(
                        {
                            "split": split,
                            "feature": nan_count.index[mask],
                            "nan_count": nan_count.values[mask],
                            "total": total,
                            "nan_ratio": nan_ratio.values[mask],
                            "window_start": window_start,
                            "window_end": window_end,
                        }
                    ).sort_values("nan_ratio", ascending=False)
                    # 打印 Top10
                    top_k = int(self.config.get("nan_report_topk", 10))
                    top = rows.head(top_k)
                    logger.warning("GRU %s 特征 NaN Top%d:\n%s", split, top_k, top.to_string(index=False))
                else:
                    # 仍然记录一个摘要行，避免报告文件缺失
                    rows = pd.DataFrame(
                        {
                            "split": [split],
                            "feature": ["__no_nan__"],
                            "nan_count": [0],
                            "total": [total],
                            "nan_ratio": [0.0],
                            "window_start": [window_start],
                            "window_end": [window_end],
                        }
                    )
                    logger.warning("GRU %s 特征无 NaN（已写入摘要行）", split)
                # 追加落盘（便于全局统计）
                header = not os.path.exists(report_path)
                rows.to_csv(report_path, mode="a", index=False, header=header)

            _nan_report(train_feat, "train")
            if valid_feat is not None:
                _nan_report(valid_feat, "valid")
        except Exception as e:
            logger.warning("GRU NaN 报告生成失败：%s", e)

        # GRU 无法处理 NaN；统一用 0 填充（特征已归一化，0≈均值）
        train_feat = train_feat.copy().fillna(0.0)
        if valid_feat is not None:
            valid_feat = valid_feat.copy().fillna(0.0)
        # 固定随机种子（可复现）
        seed = self.config.get("seed", None)
        if seed is not None:
            set_global_seed(int(seed), deterministic=bool(self.config.get("deterministic", True)))

        seq_len = int(self.config.get("seq_len", 60))
        self._input_dim = int(train_feat.shape[1])
        self._feature_names = list(train_feat.columns)
        # 训练前打印 GRU 实际使用的特征列（便于确认按模型特征集合生效）
        try:
            limit = int(self.config.get("log_feature_names_limit", 200))
        except Exception:
            limit = 200
        if self._feature_names:
            preview = self._feature_names[:limit]
            logger.info(
                "GRU 训练特征列数=%d，示例=%s%s",
                len(self._feature_names),
                preview,
                "" if len(self._feature_names) <= limit else " ...",
            )
        # 若窗口太短（尤其考虑 label_future_days 的 gap 后），GRU 无法构造满窗序列。
        # 这里不抛异常，避免中断整个训练流程；改为“跳过该窗口的 GRU 训练”，并在日志里给出清晰提示。
        try:
            n_days = int(pd.to_datetime(train_feat.index.get_level_values("datetime")).normalize().nunique())
        except Exception:
            n_days = None
        if n_days is not None and n_days < seq_len:
            logger.warning(
                "GRU 跳过训练：当前窗口可用交易日=%d，小于 seq_len=%d。"
                "建议增大 rolling.train_days（至少 >= seq_len + label_gap，例如 60+5=65，更推荐 >=120）。",
                n_days,
                seq_len,
            )
            self.model = None
            self._best_state = None
            self._best_metric = None
            self._history = []
            return

        res_tr = build_panel_sequences(
            train_feat,
            seq_len=seq_len,
            require_consecutive_trading_days=True,
            check=True,
        )
        X_tr = res_tr.X
        idx_tr = res_tr.endpoint_index
        logger.info("GRU 训练序列: %d（train样本=%d, seq_len=%d）", len(X_tr), len(train_feat), seq_len)
        y_tr_series = train_label.reindex(idx_tr)
        if len(X_tr) == 0 or len(y_tr_series) == 0:
            logger.warning(
                "GRU 跳过训练：构造出的满窗序列为 0（seq_len=%d）。"
                "这通常是训练期太短、或中间缺失交易日导致窗口不连续。",
                seq_len,
            )
            self.model = None
            self._best_state = None
            self._best_metric = None
            self._history = []
            return
        y_tr = y_tr_series.values.astype(np.float32)

        X_va = y_va = None
        if valid_feat is not None and valid_label is not None and len(valid_feat) > 0 and len(valid_label) > 0:
            # 关键修复：验证集序列需要 train 的历史拼接，否则 valid 前期缺少 seq_len-1 天历史会导致预测为空/NaN
            # 仅拼接每个 instrument 的最后 seq_len-1 天作为历史，避免无谓扩大数据量
            hist_tail = train_feat.groupby(level="instrument").tail(max(0, seq_len - 1))
            combined = pd.concat([hist_tail, valid_feat], axis=0).sort_index()
            res_va_all = build_panel_sequences(
                combined,
                seq_len=seq_len,
                require_consecutive_trading_days=True,
                check=False,
            )
            if len(res_va_all.endpoint_index) > 0:
                mask = res_va_all.endpoint_index.isin(valid_feat.index)
                X_va = res_va_all.X[mask]
                idx_va = res_va_all.endpoint_index[mask]
                y_va_series = valid_label.reindex(idx_va)
                if len(X_va) == 0 or len(y_va_series) == 0:
                    X_va = y_va = None
                else:
                    y_va = y_va_series.values.astype(np.float32)
            else:
                X_va = y_va = None
        if X_va is None or y_va is None:
            if valid_feat is not None and len(valid_feat) > 0:
                logger.warning(
                    "GRU 验证序列为空：valid样本=%d, seq_len=%d。可能因交易日不连续或历史不足。",
                    len(valid_feat),
                    seq_len,
                )
        else:
            logger.info("GRU 验证序列: %d（valid样本=%d）", len(X_va), len(valid_feat) if valid_feat is not None else 0)

        self.model = _GRUNet(
            input_dim=self._input_dim,
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(self.device)

        loss_type = str(self.config.get("loss", "mse")).lower()
        loss_params = self.config.get("loss_params", {}) or {}
        if loss_type == "asymmetric_mse":
            from utils.loss_functions import AsymmetricMSELoss
            criterion = AsymmetricMSELoss(gamma=float(loss_params.get("gamma", 2.0)))
        elif loss_type == "weighted_mse":
            from utils.loss_functions import WeightedMSELoss
            criterion = WeightedMSELoss(
                w_positive=float(loss_params.get("w_positive", 2.0)),
                w_negative=float(loss_params.get("w_negative", 0.5)),
            )
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

        batch_size = int(self.config.get("batch_size", 1024))
        max_epochs = int(self.config.get("max_epochs", 20))
        patience = int(self.config.get("patience", 5))
        grad_clip = float(self.config.get("grad_clip_norm", 1.0))
        early_metric = str(self.config.get("early_stopping_metric", "loss")).strip().lower()  # loss | rankic
        minimize = True if early_metric == "loss" else False

        # AMP（仅 CUDA 生效；CPU 下自动关闭）
        amp_enabled = bool(self.config.get("amp", False)) and (self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(-1)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = None
        if X_va is not None and y_va is not None:
            valid_loader = DataLoader(
                TensorDataset(torch.tensor(X_va), torch.tensor(y_va).unsqueeze(-1)),
                batch_size=batch_size,
                shuffle=False,
            )

        # early stopping
        best_metric = float("inf") if minimize else float("-inf")
        wait = 0
        best_state: Optional[dict] = None
        self._history = []

        for epoch in range(max_epochs):
            self.model.train()
            total = 0.0
            n = 0
            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred = self.model(bx)
                    loss = criterion(pred, by)
                scaler.scale(loss).backward()
                # 梯度裁剪（对 AMP 需要先 unscale）
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item()) * len(bx)
                n += len(bx)
            train_loss = total / max(1, n)

            val_loss = train_loss
            val_rankic = None
            if valid_loader is not None:
                self.model.eval()
                total = 0.0
                n = 0
                # 如果需要 RankIC，则同时收集预测与标签
                all_pred = []
                all_y = []
                with torch.no_grad():
                    for vx, vy in valid_loader:
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        with torch.cuda.amp.autocast(enabled=amp_enabled):
                            pred = self.model(vx)
                            loss = criterion(pred, vy)
                        total += float(loss.item()) * len(vx)
                        n += len(vx)
                        if early_metric == "rankic":
                            all_pred.append(pred.detach().float().cpu().numpy().reshape(-1))
                            all_y.append(vy.detach().float().cpu().numpy().reshape(-1))
                val_loss = total / max(1, n)
                if early_metric == "rankic":
                    if all_pred and all_y:
                        p = np.concatenate(all_pred, axis=0)
                        yv = np.concatenate(all_y, axis=0)
                        # Spearman：对 rank 后求 Pearson
                        try:
                            val_rankic = float(pd.Series(p).rank().corr(pd.Series(yv), method="pearson"))
                        except Exception:
                            val_rankic = None

            # 选择监控指标
            monitor_value = val_loss if early_metric == "loss" else (val_rankic if val_rankic is not None else float("-inf"))
            # 训练日志（每 epoch）
            if val_rankic is None:
                logger.info("GRU epoch %d train_loss=%.6f valid_loss=%.6f", epoch, train_loss, val_loss)
            else:
                logger.info(
                    "GRU epoch %d train_loss=%.6f valid_loss=%.6f valid_rankic=%.6f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_rankic,
                )
            self._history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "valid_loss": float(val_loss),
                    "valid_rankic": float(val_rankic) if val_rankic is not None else None,
                    "monitor": float(monitor_value),
                }
            )

            improved = (monitor_value < best_metric) if minimize else (monitor_value > best_metric)
            if improved:
                best_metric = float(monitor_value)
                wait = 0
                best_state = self.model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    logger.info("GRU 早停触发，最佳监控指标=%.6f (metric=%s)", best_metric, early_metric)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._best_state = best_state
        self._best_metric = best_metric

    def predict(self, feat: pd.DataFrame, history_feat: Optional[pd.DataFrame] = None) -> pd.Series:
        # 若该窗口未训练出 GRU（例如窗口太短），返回全 NaN（后续 IC/OOF/meta 会统一过滤）
        if self.model is None:
            return pd.Series(np.nan, index=feat.index, name="gru_pred")
        if self._feature_names is None:
            self._feature_names = list(feat.columns)

        # 对齐特征列顺序（缺失列填 0，额外列忽略）
        aligned = pd.DataFrame(index=feat.index, columns=self._feature_names, dtype=float)
        for c in self._feature_names:
            aligned[c] = feat[c] if c in feat.columns else 0.0
        aligned = aligned.fillna(0.0)

        seq_len = int(self.config.get("seq_len", 60))
        # 若提供 history_feat，则拼接 history 的尾部，保证起始阶段也能构造满窗序列（不泄露未来）
        if history_feat is not None and len(history_feat) > 0:
            hist_aligned = pd.DataFrame(index=history_feat.index, columns=self._feature_names, dtype=float)
            for c in self._feature_names:
                hist_aligned[c] = history_feat[c] if c in history_feat.columns else 0.0
            hist_aligned = hist_aligned.fillna(0.0)
            hist_tail = hist_aligned.groupby(level="instrument").tail(max(0, seq_len - 1))
            combined = pd.concat([hist_tail, aligned], axis=0).sort_index()
        else:
            combined = aligned

        res = build_panel_sequences(
            combined,
            seq_len=seq_len,
            require_consecutive_trading_days=True,
            check=False,
        )
        X = res.X
        idx = res.endpoint_index
        if len(X) == 0 or len(idx) == 0:
            logger.warning(
                "GRU 预测序列为空：feat样本=%d, history样本=%d, seq_len=%d",
                len(feat),
                len(history_feat) if history_feat is not None else 0,
                seq_len,
            )
            return pd.Series(np.nan, index=feat.index, name="gru_pred")
        # 只取 endpoint 在目标 feat.index 内的序列（避免把 history 的 endpoint 也输出）
        mask = idx.isin(feat.index)
        X = X[mask]
        idx = idx[mask]
        if len(X) == 0 or len(idx) == 0:
            logger.warning(
                "GRU 预测序列对齐为空：feat样本=%d, 端点=%d, seq_len=%d",
                len(feat),
                len(res.endpoint_index),
                seq_len,
            )
            return pd.Series(np.nan, index=feat.index, name="gru_pred")

        batch_size = int(self.config.get("predict_batch_size", 4096))
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.tensor(X[i : i + batch_size]).to(self.device)
                py = self.model(bx).detach().cpu().numpy().reshape(-1)
                preds.append(py)
        pred_arr = np.concatenate(preds, axis=0)
        s = pd.Series(pred_arr, index=idx, name="gru_pred")
        # 对齐回原 feat.index，历史不足的行置 NaN（后续 stacking 会取交集避免错位）
        out = s.reindex(feat.index)
        if out.notna().sum() == 0:
            logger.warning(
                "GRU 预测全为 NaN：feat样本=%d, 端点=%d, seq_len=%d",
                len(feat),
                len(idx),
                seq_len,
            )
        return out

    def save(self, output_dir: str, model_name: str):
        if self.model is None:
            raise RuntimeError("无可保存模型")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}_gru.pt")
        torch.save(
            {
                # 保存 best 权重（若存在），否则保存当前权重
                "state_dict": self._best_state if self._best_state is not None else self.model.state_dict(),
                "config": self.config,
                "feature_names": self._feature_names,
                "input_dim": self._input_dim,
                "best_metric": self._best_metric,
                "history": self._history,
            },
            path,
        )
        meta_path = os.path.join(output_dir, f"{model_name}_gru_meta.json")
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "feature_names": self._feature_names,
                    "input_dim": self._input_dim,
                    "seq_len": int(self.config.get("seq_len", 60)),
                    "best_metric": self._best_metric,
                    "early_stopping_metric": str(self.config.get("early_stopping_metric", "loss")),
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, output_dir: str, model_name: str):
        path = os.path.join(output_dir, f"{model_name}_gru.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt.get("config", self.config)
        self._feature_names = ckpt.get("feature_names")
        self._input_dim = ckpt.get("input_dim")
        self._best_metric = ckpt.get("best_metric")
        self._history = ckpt.get("history", []) or []
        if self._input_dim is None:
            raise RuntimeError("GRU ckpt 缺少 input_dim")

        self.model = _GRUNet(
            input_dim=int(self._input_dim),
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])


