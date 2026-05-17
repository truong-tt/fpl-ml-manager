"""Fit empirical team-rho for joint MC aggregator (§4.9).

Walks player history through the trained points + bonus heads, computes
standardised residuals (total_points - mu) / sigma per row, then estimates the
Pearson correlation across same-team-same-GW residual pairs. Replaces the
hand-set MC_TEAM_RHO = 0.4 with a data-driven scalar.

Outputs `data/team_rho.json`:
    {
      "rho_global": <float>,        # used by fpl_engine._joint_mc_aggregate
      "n_pairs":    <int>,
      "by_pos_pair": {
          "1-1": {"rho": ..., "n_pairs": ...},
          "1-2": {...},
          ...
      }
    }

The position-pair breakdown is consumed by `fpl_engine._load_team_corr_matrix`:
diag = same-position correlation, off-diag = cross-position correlation, PSD-
repaired (eigen-clip) before Cholesky. `rho_global` remains the scalar fallback
used when this JSON is absent.

CLI:
    python src/fit_team_rho.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_match_features, build_player_features, points_feature_cols
from fpl_engine import BONUS_BLEND, SWANSON_W10, SWANSON_W50, SWANSON_W90
from train_bonus_model import load_bonus_models, predict_bonus_quantiles
from train_points_model import load_points_models, predict_quantiles

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "team_rho.json"

# Minimum same-team residual pairs required before we publish a per-pos-pair
# correlation estimate. Below this the sampling noise drowns the signal.
MIN_PAIR_SAMPLES = 200


def _residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row mu / sigma / z from points + bonus heads."""
    pts_models = load_points_models()
    if pts_models is None:
        raise RuntimeError("points models missing — run train_points_model.py first")
    feat_cols = points_feature_cols()
    X = df[feat_cols].astype(float).fillna(0.0)
    qpts = predict_quantiles(pts_models, X, apply_recalib=True)

    bon_models = load_bonus_models()
    if bon_models is not None:
        qbon = predict_bonus_quantiles(bon_models, X)
        for c in ("q10", "q50", "q90"):
            qpts[c] = qpts[c].values + BONUS_BLEND * qbon[c].values

    q10 = qpts["q10"].astype(float).values
    q50 = qpts["q50"].astype(float).values
    q90 = qpts["q90"].astype(float).values
    mu = SWANSON_W10 * q10 + SWANSON_W50 * q50 + SWANSON_W90 * q90
    sd = np.maximum((q90 - q10) / 2.56, 1e-3)

    out = df.copy()
    out["mu_pred"] = mu
    out["sd_pred"] = sd
    out["z"] = (out["total_points"].astype(float) - mu) / sd
    return out


def fit_team_rho() -> dict:
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fixtures, history, teams)
    df = build_player_features(history, players, fixture_feats)
    df = df.dropna(subset=["target", "total_points"])
    if "minutes" in df.columns:
        df = df[df["minutes"] > 0]
    if df.empty:
        raise RuntimeError("no rows for residual fit")

    df = _residuals(df)

    # Drop residual outliers > 6 sigma. Otherwise one DGW haul per season skews
    # the Pearson estimator (it weights extremes quadratically). Same-team-same-
    # GW pair structure is preserved.
    df = df[df["z"].abs() <= 6.0]

    # Group by (team_id, season, round) — same-team players in the same GW form
    # the residual-pair set. DGW: a player has two fixture rows in one GW; we
    # treat each (player, fixture) row separately so the joint shock samples are
    # consistent with how `_joint_mc_aggregate` keys per (team_id, fixture_gw).
    z_a: list[float] = []
    z_b: list[float] = []
    pos_a: list[int] = []
    pos_b: list[int] = []
    for _, g in df.groupby(["team_id", "season", "round"], sort=False):
        if len(g) < 2:
            continue
        zz = g["z"].astype(float).values
        pp = g["pos_id"].astype(int).values
        for i, j in itertools.combinations(range(len(g)), 2):
            z_a.append(zz[i]); z_b.append(zz[j])
            pos_a.append(pp[i]); pos_b.append(pp[j])

    if not z_a:
        raise RuntimeError("no within-team pairs found")

    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    pa = np.asarray(pos_a, dtype=np.int8)
    pb = np.asarray(pos_b, dtype=np.int8)

    rho_global = float(np.corrcoef(a, b)[0, 1])

    by_pair: dict[str, dict[str, float]] = {}
    for p1, p2 in itertools.combinations_with_replacement((1, 2, 3, 4), 2):
        mask = ((pa == p1) & (pb == p2)) | ((pa == p2) & (pb == p1))
        n = int(mask.sum())
        if n < MIN_PAIR_SAMPLES:
            continue
        r = float(np.corrcoef(a[mask], b[mask])[0, 1])
        by_pair[f"{p1}-{p2}"] = {"rho": r, "n_pairs": n}

    out = {
        "rho_global": rho_global,
        "n_pairs": int(len(a)),
        "by_pos_pair": by_pair,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = fit_team_rho()
    print(f"team-rho global = {r['rho_global']:+.4f}  (n={r['n_pairs']:,})")
    for k, v in sorted(r["by_pos_pair"].items()):
        print(f"  {k}: {v['rho']:+.4f}  (n={v['n_pairs']:,})")
    print(f"wrote {OUT_PATH}")
