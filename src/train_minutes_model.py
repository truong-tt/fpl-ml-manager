"""Expected minutes / 90 model. Replace flat `chance_of_playing_next_round` haircut.

Single shared XGBoost regressor with `reg:logistic` objective. Predictions bounded
[0, 1]. Position one-hot encoded. Rotation patterns differ — DEFs nailed, FWDs
rotated more. Shared model captures position effects without per-position fragmentation.

Inference: at projection time prediction multiplied onto q10/q50/q90 quantiles
per-fixture. Replaces flat FPL `chance_of_playing` field for future GWs. For
immediate next GW, FPL hint = hard upper bound. FPL knows specific injuries
model cannot infer from history alone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import (build_match_features, build_player_features,
                      minutes_feature_cols)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = "xgb_minutes.json"


def train_minutes_model() -> None:
    """Train + serialize minutes / 90 regressor under data/."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats)
    if train.empty:
        return

    # Target: minutes / 90 clipped [0, 1]. DGW totals clipped at 1. Player who
    # played both DGW fixtures = "fully available", not 2.0× available.
    train["mins_target"] = (train["minutes"].clip(upper=90) / 90.0).clip(0.0, 1.0)
    cols = minutes_feature_cols()
    X = train[cols].astype(float).fillna(0.0)
    y = train["mins_target"].astype(float)

    params = dict(objective="reg:logistic", learning_rate=0.05, max_depth=4,
                  subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=20, reg_alpha=0.3, reg_lambda=1.5,
                  verbosity=0)
    m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=300)
    m.save_model(DATA_DIR / MODEL_FILE)


def load_minutes_model() -> xgb.Booster | None:
    """Load serialized minutes model. Return None if missing."""
    p = DATA_DIR / MODEL_FILE
    if not p.exists():
        return None
    b = xgb.Booster()
    b.load_model(p)
    return b


def predict_minutes(model: xgb.Booster, X: pd.DataFrame,
                    apply_recalib: bool = True) -> pd.Series:
    """Predict minutes / 90 ∈ [0, 1] for every row in X. Must contain feature cols.

    If `apply_recalib` and `data/minutes_recalib.json` exists, apply per-pos
    isotonic map fit by `recalibrate_minutes.py` after raw prediction.
    """
    cols = minutes_feature_cols()
    Xf = X[cols].astype(float).fillna(0.0)
    raw = model.predict(xgb.DMatrix(Xf))
    if apply_recalib:
        from recalibrate_minutes import apply_recalib as _apply, load_recalib
        coef = load_recalib()
        if coef is not None:
            pos_id = X[[f"pos_{p}" for p in (1, 2, 3, 4)]].values.argmax(axis=1) + 1
            raw = _apply(coef, pos_id, raw)
    return pd.Series(raw, index=X.index).clip(0.0, 1.0)


if __name__ == "__main__":
    train_minutes_model()
