"""Two-stage minutes / 90 head.

Replaces single reg:logistic regressor — target was bimodal (0 spike DNP, ~0.95
mass when playing) and reg:logistic regressing single point compressed both
modes toward middle, blurring "nailed-but-subbed at 70'" vs "rotation risk".

Two heads, disjoint loss surfaces:

1. plays head (xgb_minutes_plays.json): binary:logistic, target = (minutes > 0).
   AUC-friendly. Availability / rotation prediction.
2. mins-given-played head (xgb_minutes_when_played.json): reg:logistic, fit only
   on minutes > 0, target = min(minutes, 90)/90. Stay-on-pitch given start.

Combined output `predict_minutes`:
- plays = P(plays | features), [0, 1]
- mins_when_played = E[mins/90 | plays=1, features], [0, 1]
- mins_pred = plays * mins_when_played — drop-in for prior availability
  multiplier. Engine multiplies onto q10/q50/q90.

Inference also emits stage outputs. Engine uses `plays` alone for q90 ceiling
(haul prob ~ P(plays), hauls usually pre-sub).

Recalib (data/minutes_recalib.json) fits isotonic on combined mins_pred —
fitted curves remain interpretable as P(played) reliability post-7.4 audit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import (build_match_features, build_player_features,
                      minutes_feature_cols)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLAYS_FILE = "xgb_minutes_plays.json"
MINS_FILE = "xgb_minutes_when_played.json"
LEGACY_FILE = "xgb_minutes.json"  # single-head, fallback during migration


def train_minutes_model() -> None:
    """Train + serialize plays head + mins-given-played head under data/."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats)
    if train.empty:
        return

    cols = minutes_feature_cols()
    X = train[cols].astype(float).fillna(0.0)
    train["played"] = (train["minutes"] > 0).astype(int)
    train["mins_target"] = (train["minutes"].clip(upper=90) / 90.0).clip(0.0, 1.0)

    plays_params = dict(objective="binary:logistic", eval_metric="logloss",
                        learning_rate=0.05, max_depth=4,
                        subsample=0.85, colsample_bytree=0.85,
                        min_child_weight=20, reg_alpha=0.3, reg_lambda=1.5,
                        verbosity=0)
    plays_m = xgb.train(plays_params,
                        xgb.DMatrix(X, label=train["played"].astype(int)),
                        num_boost_round=300)
    plays_m.save_model(DATA_DIR / PLAYS_FILE)

    played_mask = train["played"] == 1
    if played_mask.sum() < 200:
        return
    Xp = X.loc[played_mask]
    yp = train.loc[played_mask, "mins_target"].astype(float)
    mins_params = dict(objective="reg:logistic", learning_rate=0.05, max_depth=4,
                       subsample=0.85, colsample_bytree=0.85,
                       min_child_weight=20, reg_alpha=0.3, reg_lambda=1.5,
                       verbosity=0)
    mins_m = xgb.train(mins_params, xgb.DMatrix(Xp, label=yp),
                       num_boost_round=300)
    mins_m.save_model(DATA_DIR / MINS_FILE)


def load_minutes_model() -> dict | None:
    """Load both heads. Return {plays, mins} dict, None if all missing.

    Fallback: legacy single-head artifact under "mins" — keeps inference
    compatible during migration.
    """
    plays_p = DATA_DIR / PLAYS_FILE
    mins_p = DATA_DIR / MINS_FILE
    legacy_p = DATA_DIR / LEGACY_FILE
    if plays_p.exists() and mins_p.exists():
        plays = xgb.Booster()
        plays.load_model(plays_p)
        mins = xgb.Booster()
        mins.load_model(mins_p)
        return {"plays": plays, "mins": mins}
    if legacy_p.exists():
        b = xgb.Booster()
        b.load_model(legacy_p)
        return {"plays": None, "mins": b, "legacy": True}
    return None


def predict_minutes(model: dict, X: pd.DataFrame,
                    apply_recalib: bool = True,
                    return_components: bool = False
                    ) -> pd.Series | pd.DataFrame:
    """Predict expected minutes / 90.

    return_components=True → frame with plays / mins_when_played / mins_pred
    (= plays * mins_when_played). Else mins_pred Series matching legacy single-head.
    """
    cols = minutes_feature_cols()
    Xf = X[cols].astype(float).fillna(0.0)
    dm = xgb.DMatrix(Xf)
    if model.get("legacy"):
        mins_pred = model["mins"].predict(dm)
        plays = mins_pred  # no separate signal in legacy artifact
        mins_when_played = mins_pred
    else:
        plays = model["plays"].predict(dm) if model.get("plays") is not None else np.ones_like(Xf.values[:, 0])
        mins_when_played = model["mins"].predict(dm)
        mins_pred = plays * mins_when_played

    if apply_recalib:
        from recalibrate_minutes import apply_recalib as _apply, load_recalib
        coef = load_recalib()
        if coef is not None:
            pos_id = X[[f"pos_{p}" for p in (1, 2, 3, 4)]].values.argmax(axis=1) + 1
            mins_pred = _apply(coef, pos_id, mins_pred)

    mins_pred = np.clip(mins_pred, 0.0, 1.0)
    plays = np.clip(plays, 0.0, 1.0)
    mins_when_played = np.clip(mins_when_played, 0.0, 1.0)

    if return_components:
        return pd.DataFrame({
            "plays": pd.Series(plays, index=X.index),
            "mins_when_played": pd.Series(mins_when_played, index=X.index),
            "mins_pred": pd.Series(mins_pred, index=X.index),
        })
    return pd.Series(mins_pred, index=X.index)


if __name__ == "__main__":
    train_minutes_model()
