"""XGBoost Poisson match model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

from features import build_match_features, match_feature_cols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def train_match_models(
    fixtures: pd.DataFrame, history: pd.DataFrame, teams: pd.DataFrame
) -> None:
    """Train + serialize home / away goal Poisson regressors."""
    df = build_match_features(fixtures, history, teams)
    past = df[df["finished"] == True].dropna(subset=["team_h_score", "team_a_score"])
    if past.empty:
        return
    X = past[match_feature_cols()].astype(float)
    params = dict(objective="count:poisson", learning_rate=0.05, max_depth=4,
                  subsample=0.85, colsample_bytree=0.85, verbosity=0)
    for label, side in (("team_h_score", "home"), ("team_a_score", "away")):
        m = xgb.train(params, xgb.DMatrix(X, label=past[label].astype(int)), num_boost_round=200)
        m.save_model(DATA_DIR / f"xgb_{side}_goals.json")


def score_matrix(lh: float, la: float, max_goals: int = 8) -> np.ndarray:
    """Joint score PMF from independent Poisson marginals. Indexed [home_goals, away_goals]."""
    ph = poisson.pmf(np.arange(max_goals + 1), lh)
    pa = poisson.pmf(np.arange(max_goals + 1), la)
    return np.outer(ph, pa)


def clean_sheet_probs(lh: float, la: float) -> tuple[float, float]:
    """(home_CS, away_CS) = independent Poisson P(opp goals = 0)."""
    return float(poisson.pmf(0, la)), float(poisson.pmf(0, lh))


def _load_match_boosters() -> dict[str, xgb.Booster] | None:
    """Load saved home/away goal boosters. None if any missing."""
    out: dict[str, xgb.Booster] = {}
    for side in ("home", "away"):
        p = DATA_DIR / f"xgb_{side}_goals.json"
        if not p.exists():
            return None
        b = xgb.Booster()
        b.load_model(p)
        out[side] = b
    return out


def compute_fixture_lambdas(fixtures: pd.DataFrame, history: pd.DataFrame,
                            teams: pd.DataFrame) -> pd.DataFrame | None:
    """Predict λ_h / λ_a + CS probs per fixture. Writes
    data/fixture_lambdas.csv [id, lambda_h, lambda_a, cs_h_p, cs_a_p].

    Used as features in points head (own_lambda_for / against, own_cs_p).

    Mild leakage in backtest walk-forward points: λ for round G fixtures
    computed using match model trained on all rounds (incl. G). Acceptable;
    walk-forward isolation possible later.
    """
    boosters = _load_match_boosters()
    if boosters is None:
        return None
    df = build_match_features(fixtures, history, teams)
    cols = match_feature_cols()
    X = df[cols].astype(float).fillna(0.0)
    dm = xgb.DMatrix(X)
    df = df.copy()
    df["lambda_h"] = np.maximum(boosters["home"].predict(dm), 1e-3)
    df["lambda_a"] = np.maximum(boosters["away"].predict(dm), 1e-3)
    df["cs_h_p"] = np.exp(-df["lambda_a"].values)
    df["cs_a_p"] = np.exp(-df["lambda_h"].values)
    out = df[["id", "lambda_h", "lambda_a", "cs_h_p", "cs_a_p"]].copy()
    out.to_csv(DATA_DIR / "fixture_lambdas.csv", index=False)
    return out


if __name__ == "__main__":
    train_match_models(
        pd.read_csv(DATA_DIR / "fixtures.csv"),
        pd.read_csv(DATA_DIR / "history.csv"),
        pd.read_csv(DATA_DIR / "teams.csv"),
    )
