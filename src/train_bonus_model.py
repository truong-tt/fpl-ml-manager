"""Direct BPS / bonus head — quantile booster stack on FPL `bonus` column.

Bonus (FPL `bonus` in {0, 1, 2, 3}, top-3 BPS scorers per match) = highest-
variance fragment of total_points. Main points head sees rolling BPS as feature
+ learns contribution implicitly, but signal diluted across other targets baked
into total_points (goals, assists, CS, BPS-driven bonus, deductions). Result:
flat q90 ceiling on bonus-heavy archetypes (CS+block defender lands closer to
population mean).

Separate quantile booster on bonus directly preserves discrete 0/1/2/3 mass +
asymmetric tail. Engine sums onto points quantiles with damping factor
(BONUS_BLEND in fpl_engine.py) — full additivity double-counts partial bonus
already inside points head. Future cleanup: retrain points head on
`total_points - bonus` to remove double count entirely (Future Work).

Single shared model. Pos one-hots in points_feature_cols + bonus dist sparse →
splitting into 4 per-pos heads shrinks each subset below regularization budget.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import (build_match_features, build_player_features,
                      points_feature_cols)
from train_points_model import _monotone_constraints

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUANTILES = [0.10, 0.50, 0.90]
PARAMS = dict(objective="reg:quantileerror", learning_rate=0.03, max_depth=3,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
              reg_alpha=0.5, reg_lambda=2.0, verbosity=0)
ROUNDS = 400


def _model_path(alpha: float) -> Path:
    return DATA_DIR / f"xgb_bonus_q{int(alpha * 100):02d}.json"


def train_bonus_model() -> None:
    """Train + serialize 3 quantile boosters on bonus target.

    Played-only filter. DNP rows have bonus=0 by definition (BPS requires
    minutes). Including them collapses quantiles toward 0. Engine gates final
    bonus contribution via P(plays) from minutes head.
    """
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats)
    if train.empty or "bonus" not in hist.columns:
        return
    if "bonus" not in train.columns:
        train = train.merge(hist[["player_id", "fixture", "bonus"]],
                            on=["player_id", "fixture"], how="left")
    if "minutes" in train.columns:
        train = train[train["minutes"] > 0]
    if train.empty:
        return

    cols = points_feature_cols()
    X = train[cols].astype(float).fillna(0.0)
    y = pd.to_numeric(train["bonus"], errors="coerce").fillna(0.0).astype(float)
    mono = _monotone_constraints(cols)

    for alpha in QUANTILES:
        params = dict(PARAMS, quantile_alpha=alpha, monotone_constraints=mono)
        m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=ROUNDS)
        m.save_model(_model_path(alpha))


def load_bonus_models() -> dict[float, xgb.Booster] | None:
    """Load 3 quantile boosters. None if any missing."""
    out: dict[float, xgb.Booster] = {}
    for alpha in QUANTILES:
        p = _model_path(alpha)
        if not p.exists():
            return None
        b = xgb.Booster()
        b.load_model(p)
        out[alpha] = b
    return out


def predict_bonus_quantiles(models: dict[float, xgb.Booster],
                            X: pd.DataFrame) -> pd.DataFrame:
    """Per-row q10/q50/q90 of bonus (in [0, 3]). Non-crossing enforced."""
    cols = points_feature_cols()
    Xf = X[cols].astype(float).fillna(0.0)
    dm = xgb.DMatrix(Xf)
    out = pd.DataFrame(0.0, index=X.index, columns=["q10", "q50", "q90"])
    for alpha, m in models.items():
        out[f"q{int(alpha * 100):02d}"] = m.predict(dm)
    vals = out[["q10", "q50", "q90"]].values.copy()
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals
    # Bonus floor 0, ceiling 3 per fixture.
    return out.clip(lower=0.0, upper=3.0)


if __name__ == "__main__":
    train_bonus_model()
