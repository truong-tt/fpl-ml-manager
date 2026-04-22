"""Learned per-player FPL points model using XGBoost quantile regression (q10/q50/q90)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import build_match_features, build_player_features, points_feature_cols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUANTILES = [0.10, 0.50, 0.90]
MODEL_FILES = {q: f"xgb_points_q{int(q*100):02d}.json" for q in QUANTILES}


def train_points_models() -> None:
    """Trains and serializes the three quantile regressors to data/."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats).dropna(subset=["target"])
    if train.empty:
        return

    X = train[points_feature_cols()].astype(float).fillna(0.0)
    y = train["target"].astype(float)
    for q in QUANTILES:
        params = dict(objective="reg:quantileerror", quantile_alpha=q,
                      learning_rate=0.05, max_depth=5, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, verbosity=0)
        m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=400)
        m.save_model(DATA_DIR / MODEL_FILES[q])


def load_points_models() -> dict[float, xgb.Booster] | None:
    """Loads the three quantile boosters; returns None if any file is missing."""
    out: dict[float, xgb.Booster] = {}
    for q, fn in MODEL_FILES.items():
        p = DATA_DIR / fn
        if not p.exists():
            return None
        b = xgb.Booster()
        b.load_model(p)
        out[q] = b
    return out


def predict_quantiles(
    models: dict[float, xgb.Booster], X: pd.DataFrame
) -> pd.DataFrame:
    """Runs all three boosters, enforces non-crossing, and clips at -3."""
    dm = xgb.DMatrix(X.astype(float).fillna(0.0))
    out = pd.DataFrame(index=X.index)
    for q, m in models.items():
        out[f"q{int(q*100):02d}"] = m.predict(dm)
    vals = out[["q10", "q50", "q90"]].values
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals
    return out.clip(lower=-3.0)


if __name__ == "__main__":
    train_points_models()