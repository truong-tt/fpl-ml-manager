"""Per-position XGBoost quantile regressors (q10/q50/q90 × {GK, DEF, MID, FWD})."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import build_match_features, build_player_features, points_feature_cols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUANTILES = [0.10, 0.50, 0.90]
POSITIONS = [1, 2, 3, 4]


def _model_file(q: float, pos: int) -> str:
    return f"xgb_points_q{int(q * 100):02d}_p{pos}.json"


def _pos_feature_cols() -> list[str]:
    """Drops pos one-hots — constant within each per-position model."""
    return [c for c in points_feature_cols() if not c.startswith("pos_")]


def _row_pos(df: pd.DataFrame) -> pd.Series:
    """Recovers pos_id 1..4 from the pos_{1..4} one-hots."""
    return df[[f"pos_{p}" for p in POSITIONS]].idxmax(axis=1).str.replace("pos_", "").astype(int)


def train_points_models() -> None:
    """Trains and serializes 4 positions × 3 quantiles = 12 boosters under data/."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats).dropna(subset=["target"])
    if train.empty:
        return

    train["_pos"] = _row_pos(train)
    feat_cols = _pos_feature_cols()

    for pos in POSITIONS:
        sub = train[train["_pos"] == pos]
        if len(sub) < 200:
            continue
        X = sub[feat_cols].astype(float).fillna(0.0)
        y = sub["target"].astype(float)
        for q in QUANTILES:
            # Per-pos sets are small (~3k for FWD); strong regularization keeps q90 credible.
            params = dict(objective="reg:quantileerror", quantile_alpha=q,
                          learning_rate=0.03, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=30,
                          reg_alpha=0.5, reg_lambda=2.0, verbosity=0)
            m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=400)
            m.save_model(DATA_DIR / _model_file(q, pos))


def load_points_models() -> dict[int, dict[float, xgb.Booster]] | None:
    """Loads {pos: {q: booster}}; returns None if any file is missing."""
    out: dict[int, dict[float, xgb.Booster]] = {}
    for pos in POSITIONS:
        out[pos] = {}
        for q in QUANTILES:
            p = DATA_DIR / _model_file(q, pos)
            if not p.exists():
                return None
            b = xgb.Booster()
            b.load_model(p)
            out[pos][q] = b
    return out


def predict_quantiles(
    models: dict[int, dict[float, xgb.Booster]], X: pd.DataFrame
) -> pd.DataFrame:
    """Routes each row to its position's booster; enforces non-crossing q10≤q50≤q90."""
    feat_cols = _pos_feature_cols()
    out = pd.DataFrame(0.0, index=X.index, columns=["q10", "q50", "q90"])
    pos_series = _row_pos(X)
    Xf = X[feat_cols].astype(float).fillna(0.0)
    for pos in POSITIONS:
        mask = (pos_series == pos)
        if not mask.any() or pos not in models:
            continue
        dm = xgb.DMatrix(Xf.loc[mask])
        for q, m in models[pos].items():
            out.loc[mask, f"q{int(q * 100):02d}"] = m.predict(dm)
    vals = out[["q10", "q50", "q90"]].values
    vals = vals.copy()
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals
    # Sanity ceiling — credible single-GW boom tops ~25 (hat-trick+assist+bonus).
    return out.clip(lower=-3.0, upper=25.0)


if __name__ == "__main__":
    train_points_models()
