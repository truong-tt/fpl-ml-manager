"""Expected minutes / 90 model. Replaces flat `chance_of_playing_next_round` haircut.

Single shared XGBoost regressor with `reg:logistic` objective so predictions are bounded
in [0, 1]. Position is one-hot encoded (rotation patterns differ — DEFs nailed, FWDs
rotated more) so a shared model captures position effects without per-position fragmentation.

Inference: at projection time the prediction is multiplied onto the q10/q50/q90 quantiles
per-fixture, replacing the flat FPL `chance_of_playing` field for future GWs. For the
immediate next GW the FPL hint is taken as a hard upper bound (FPL knows about specific
injuries the model cannot infer from history alone).
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
    """Trains and serializes the minutes / 90 regressor under data/."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats)
    if train.empty:
        return

    # Target: minutes / 90 clipped [0, 1]. DGW totals get clipped at 1 — a player who
    # played both DGW fixtures is "fully available", not 2.0× available.
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
    """Loads the serialized minutes model; returns None if missing."""
    p = DATA_DIR / MODEL_FILE
    if not p.exists():
        return None
    b = xgb.Booster()
    b.load_model(p)
    return b


def predict_minutes(model: xgb.Booster, X: pd.DataFrame) -> pd.Series:
    """Predicts minutes / 90 ∈ [0, 1] for every row in X (must contain feature cols)."""
    cols = minutes_feature_cols()
    Xf = X[cols].astype(float).fillna(0.0)
    return pd.Series(model.predict(xgb.DMatrix(Xf)), index=X.index).clip(0.0, 1.0)


if __name__ == "__main__":
    train_minutes_model()
