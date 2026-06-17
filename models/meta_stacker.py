"""
Meta Stacking 模型：使用 OOF 预测训练一个二层模型（Ridge/Linear）。
用于：final = meta_model([pred_lgb, pred_mlp, pred_gru, ...])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class MetaStackerConfig:
    model_type: str = "ridge"  # "ridge" | "linear"
    alpha: float = 1.0


class MetaStacker:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cfg = MetaStackerConfig(
            model_type=str(cfg.get("model_type", "ridge")).lower(),
            alpha=float(cfg.get("alpha", 1.0)),
        )
        self.scaler = StandardScaler()
        if self.cfg.model_type == "linear":
            self.model = LinearRegression()
        elif self.cfg.model_type == "ridge":
            self.model = Ridge(alpha=self.cfg.alpha)
        else:
            raise ValueError(f"不支持的 meta model_type={self.cfg.model_type}")

        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X_meta: pd.DataFrame, y: pd.Series):
        if X_meta is None or len(X_meta) == 0:
            raise ValueError("X_meta 为空，无法训练 MetaStacker")
        if y is None or len(y) == 0:
            raise ValueError("y 为空，无法训练 MetaStacker")

        X, y2 = X_meta.align(y, join="inner", axis=0)
        mask = ~(X.isna().any(axis=1) | y2.isna())
        X = X.loc[mask]
        y2 = y2.loc[mask]
        if len(X) < 10:
            raise ValueError(f"有效样本过少 ({len(X)})，无法训练 MetaStacker")

        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y2.values)
        self.is_fitted = True

    def predict(self, X_meta: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("MetaStacker 未训练/未加载")
        if self.feature_names is None:
            raise RuntimeError("MetaStacker 缺少 feature_names")
        # 对齐列，缺失列填 0（对 Meta 模型来说含义是“无贡献”）
        X = pd.DataFrame(index=X_meta.index, columns=self.feature_names, dtype=float)
        for c in self.feature_names:
            if c in X_meta.columns:
                X[c] = X_meta[c]
            else:
                X[c] = 0.0
        X_scaled = self.scaler.transform(X.values)
        pred = self.model.predict(X_scaled)
        return pd.Series(pred, index=X.index, name="meta_pred")

    def save(self, output_dir: str, tag: str):
        os.makedirs(output_dir, exist_ok=True)
        pkl = os.path.join(output_dir, f"{tag}_meta.pkl")
        meta = os.path.join(output_dir, f"{tag}_meta_meta.json")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "cfg": self.cfg.__dict__,
            },
            pkl,
        )
        with open(meta, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "cfg": self.cfg.__dict__,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, output_dir: str, tag: str):
        pkl = os.path.join(output_dir, f"{tag}_meta.pkl")
        if not os.path.exists(pkl):
            raise FileNotFoundError(pkl)
        obj = joblib.load(pkl)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.feature_names = obj.get("feature_names")
        self.is_fitted = True



