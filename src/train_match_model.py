"""XGBoost Poisson match model + Dixon-Coles low-score correction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

from features import build_match_features, match_feature_cols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DC_RHO = -0.10


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


def _dc_tau(x: int, y: int, lh: float, la: float, rho: float = DC_RHO) -> float:
    """Dixon-Coles τ correction. Low scoring outcomes (0-0, 0-1, 1-0, 1-1)."""
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def score_matrix(lh: float, la: float, max_goals: int = 8) -> np.ndarray:
    """DC-adjusted joint score probability matrix. Indexed [home_goals, away_goals]."""
    ph = poisson.pmf(np.arange(max_goals + 1), lh)
    pa = poisson.pmf(np.arange(max_goals + 1), la)
    M = np.outer(ph, pa)
    for x in range(2):
        for y in range(2):
            M[x, y] *= _dc_tau(x, y, lh, la)
    return M / M.sum() if M.sum() > 0 else M


def clean_sheet_probs(lh: float, la: float) -> tuple[float, float]:
    """Return (home_CS, away_CS) analytically from DC score matrix."""
    M = score_matrix(lh, la)
    return float(M[:, 0].sum()), float(M[0, :].sum())


if __name__ == "__main__":
    train_match_models(
        pd.read_csv(DATA_DIR / "fixtures.csv"),
        pd.read_csv(DATA_DIR / "history.csv"),
        pd.read_csv(DATA_DIR / "teams.csv"),
    )