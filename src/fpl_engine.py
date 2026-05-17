"""FPLEngine. Per-(player, GW) projection frame for optimizer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import SEASON
from features import (CUP_COLS, CUP_DEFAULTS, build_match_features,
                      build_player_features, minutes_feature_cols,
                      points_feature_cols)
from train_bonus_model import load_bonus_models, predict_bonus_quantiles
from train_minutes_model import load_minutes_model, predict_minutes
from train_points_model import load_points_models, predict_quantiles

# Bonus blend factor. Applied at the MOMENT level (mu_b *= b, sigma_b *= b)
# before variance-additive combine with the points head — not as a quantile
# multiplier. Points head is trained on `total_points - bonus`; the bonus head
# fills the missing component. 1.0 = full inclusion. <1 damps the bonus
# contribution to both expected value and dispersion uniformly.
BONUS_BLEND = 1.0

# Joint MC aggregation. Per-(team, GW) shock + per-row idiosyncratic.
#
# Two correlation models:
#   1. Scalar MC_TEAM_RHO — legacy single ρ applied uniformly across positions.
#      Cov(X_i, X_j | same team, GW) = ρ · σ_i · σ_j. Used as fallback when
#      the position-pair matrix is absent.
#   2. 4×4 within-team position-pair correlation matrix C — diag = same-pos
#      correlation, off-diag = cross-pos correlation, both from
#      `data/team_rho.json::by_pos_pair` (`fit_team_rho.py`). Decomposed via
#      Cholesky C = L L^T. Per (team, GW) we draw a 4-dim factor f = L z,
#      z ~ N(0, I_4); a row at position p uses f[p-1] as its team-correlated
#      component. Idiosyncratic remainder σ · √(1 − C[p,p]) ε keeps Var(X_i)
#      = σ_i^2. Recovers Cov(X_i, X_j) = σ_i σ_j · C[p_i, p_j] exactly — the
#      scalar path's σ_i σ_j ρ^2 (variance-explained, not correlation) is an
#      under-estimate. Unlocks GK-DEF/2-2 (~0.15–0.23 empirically) vs. 3-3
#      (~0.01) and 4-4 (small, sometimes negative — clipped to 0).
MC_SAMPLES = 800
MC_TEAM_RHO_FALLBACK = 0.4
_TEAM_RHO_PATH = Path(__file__).resolve().parent.parent / "data" / "team_rho.json"


def _load_team_rho_default() -> float:
    """Read empirical rho from `data/team_rho.json` (written by
    `fit_team_rho.py`). Falls back to MC_TEAM_RHO_FALLBACK when missing or
    malformed so the engine remains usable on a fresh checkout. Negative
    values are clipped to 0 (the joint-MC math assumes 0 ≤ rho ≤ 1)."""
    if not _TEAM_RHO_PATH.exists():
        return MC_TEAM_RHO_FALLBACK
    try:
        d = json.loads(_TEAM_RHO_PATH.read_text(encoding="utf-8"))
        v = float(d.get("rho_global", MC_TEAM_RHO_FALLBACK))
        return float(np.clip(v, 0.0, 1.0))
    except (ValueError, OSError, TypeError):
        return MC_TEAM_RHO_FALLBACK


def _load_team_corr_matrix() -> tuple[np.ndarray, np.ndarray] | None:
    """Build a 4×4 within-team position correlation matrix C (and its Cholesky
    factor L) from `data/team_rho.json::by_pos_pair`. Returns None on missing /
    malformed file so the engine falls back to the scalar path.

    Diag C[p,p] = same-position correlation (e.g. 2-2 for two DEFs same team
    same GW). Off-diag C[p,q] = cross-position correlation. Missing same-pos
    entries (rare — 1-1 absent because only one GK plays per match) fall back
    to rho_global. Negative diagonals (4-4 sometimes flips signs on small n)
    are clipped to 0 — the factor model needs Var(f_p) ≥ 0.

    PSD repair: eigen-clip to a small floor (1e-6) so Cholesky succeeds. The
    empirical matrix can be indefinite due to sampling noise + clipping.
    """
    if not _TEAM_RHO_PATH.exists():
        return None
    try:
        d = json.loads(_TEAM_RHO_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError, TypeError):
        return None
    by_pair = d.get("by_pos_pair") or {}
    if not by_pair:
        return None
    rho_g = float(np.clip(d.get("rho_global", 0.0), 0.0, 0.9))
    C = np.full((4, 4), rho_g, dtype=float)
    for k, v in by_pair.items():
        try:
            p1_s, p2_s = k.split("-")
            p1, p2 = int(p1_s), int(p2_s)
            r = float(v["rho"])
        except (ValueError, KeyError, TypeError):
            continue
        if not (1 <= p1 <= 4 and 1 <= p2 <= 4):
            continue
        C[p1 - 1, p2 - 1] = r
        C[p2 - 1, p1 - 1] = r
    for i in range(4):
        # Diagonal must be non-negative (variance of factor) and bounded by 1
        # (correlation upper bound). 4-4 empirical can be negative due to
        # within-team forward substitution dynamics on small n.
        C[i, i] = float(np.clip(C[i, i], 0.0, 1.0))
    # PSD repair via eigenvalue clipping. The empirical matrix is sample-
    # estimated + diag-clipped → can drift slightly indefinite. Clip negative
    # eigvals to a small floor, re-symmetrise.
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C_psd = (V * w) @ V.T
    C_psd = (C_psd + C_psd.T) / 2.0
    try:
        L = np.linalg.cholesky(C_psd + 1e-9 * np.eye(4))
    except np.linalg.LinAlgError:
        return None
    return C_psd, L


MC_TEAM_RHO = _load_team_rho_default()
MC_TEAM_CORR = _load_team_corr_matrix()


# Swanson / Keefer-Bodily 3-point estimator weights. Calibrated to skewed
# (lognormal-family) distributions; outperforms Simpson (1,4,1)/6 on right-
# skewed FPL points where median systematically under-shoots mean. Sums to 1.
SWANSON_W10, SWANSON_W50, SWANSON_W90 = 0.3, 0.4, 0.3


def _row_quantile_to_moments(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Swanson (Keefer-Bodily) mean + Gaussian-bracket std from 3 quantiles."""
    mu = SWANSON_W10 * q10 + SWANSON_W50 * q50 + SWANSON_W90 * q90
    sd = np.maximum((q90 - q10) / 2.56, 0.0)
    return mu, sd


def _mixture_quantile(q10_played: np.ndarray, q50_played: np.ndarray,
                      q90_played: np.ndarray, p: np.ndarray,
                      alpha: float) -> np.ndarray:
    """Unconditional alpha-quantile under zero-inflated mixture.

    Distribution: with prob (1-p) the outcome is 0 (DNP); with prob p the
    outcome is drawn from F_played. CDF:

        F_unc(y) = (1-p) * 1[y >= 0] + p * F_played(y)

    Inversion at level alpha:

        q_alpha_unc = 0                              if alpha <= 1 - p
                    = F_played^{-1}((alpha - (1-p))/p)   otherwise

    F_played^{-1} approximated by piecewise-linear interpolation through the
    known 3-knot grid (0.1, q10_played), (0.5, q50_played), (0.9, q90_played);
    clamped to [q10, q90] outside [0.1, 0.9]. Same routine for points + bonus
    heads -- both trained on minutes > 0 so their predicted quantiles ARE
    F_played quantiles, not unconditional.

    Replaces the prior `q_alpha * p` heuristic, which is the right operation
    for E[X] (linearity) but not for q_alpha (no analogous identity). The old
    rule under-counted ceiling for high-rotation high-ceiling profiles and
    over-counted floor for the same (q10 > 0 even when alpha < 1-p).
    """
    p = np.asarray(p, dtype=float)
    p_safe = np.clip(p, 1e-9, 1.0)
    threshold = 1.0 - p_safe                 # mass at zero
    mask_zero = alpha <= threshold
    alpha_played = np.clip((alpha - threshold) / p_safe, 0.0, 1.0)
    q_played = np.where(
        alpha_played <= 0.1,
        q10_played,
        np.where(
            alpha_played <= 0.5,
            q10_played + (alpha_played - 0.1) / 0.4 * (q50_played - q10_played),
            q50_played + (alpha_played - 0.5) / 0.4 * (q90_played - q50_played),
        ),
    )
    q_played = np.minimum(np.maximum(q_played, q10_played), q90_played)
    return np.where(mask_zero, 0.0, q_played)


def _joint_mc_aggregate(rows: pd.DataFrame, n_samples: int = MC_SAMPLES,
                        team_rho: float = MC_TEAM_RHO,
                        team_corr: tuple[np.ndarray, np.ndarray] | None
                        = MC_TEAM_CORR,
                        seed: int = 17) -> pd.DataFrame:
    """Sample correlated player totals per (player, fixture_gw).

    Same (team_id, fixture_gw) rows share team-shock per draw — Liverpool CS
    shock lifts Salah + Virgil together; goal blitz lifts Salah + Diaz.
    Idiosyncratic remainder keeps individual upside unrolled.

    Correlation models (see module-level note):
      * `team_corr=(C, L)` — 4×4 position-pair Cholesky path. Per (team, GW)
        draw a 4-dim factor f = L z; row at position p picks f[p-1]. Idio
        scale = √(1 − C[p,p]) so total per-row variance stays σ_i^2 and
        Cov(X_i, X_j | same team, GW) = σ_i σ_j · C[p_i, p_j].
      * `team_corr=None` — scalar legacy path. Single shared η per (team, GW)
        scaled by ρ; idio scale = √(1 − ρ²). Recovers σ_i σ_j · ρ² covariance
        (NB this is variance-explained, not correlation; the matrix path
        recovers the empirical correlation directly).

    Points + bonus heads combine via independence-assumption variance addition
    on the (mu, sigma) plane rather than linear quantile addition. Quantile
    arithmetic is wrong: for independent X, Y, q_a(X+Y) != q_a(X) + q_a(Y)
    (sub-additive variance, super-additive linear-quantile-sum); the latter
    inflates q90 by up to sqrt(2) in the equal-variance limit. Combining via
    mu_c = mu_p + mu_b, sigma_c^2 = sigma_p^2 + sigma_b^2 reproduces the joint
    quantile under the Gaussian envelope already used by Pearson-Tukey here.

    Returns: player_id, fixture_gw, mean_xp, std_xp, cap_xp, q10_mc, q90_mc.
    """
    n = len(rows)
    if n == 0:
        return pd.DataFrame(columns=["player_id", "fixture_gw", "mean_xp",
                                     "std_xp", "cap_xp", "q10_mc", "q90_mc"])
    rng = np.random.default_rng(seed)

    q10 = rows["q10"].astype(float).values
    q50 = rows["q50"].astype(float).values
    q90 = rows["q90"].astype(float).values
    mu_p, sd_p = _row_quantile_to_moments(q10, q50, q90)
    if {"q10_b", "q50_b", "q90_b"}.issubset(rows.columns):
        q10_b = rows["q10_b"].astype(float).values
        q50_b = rows["q50_b"].astype(float).values
        q90_b = rows["q90_b"].astype(float).values
        mu_b, sd_b_bonus = _row_quantile_to_moments(q10_b, q50_b, q90_b)
        mu = mu_p + mu_b
        sd = np.sqrt(sd_p ** 2 + sd_b_bonus ** 2)
    else:
        mu, sd = mu_p, sd_p

    # (team_id, fixture_gw) → unique shock index.
    keys = list(zip(rows["team_id"].astype(int).tolist(),
                    rows["fixture_gw"].astype(int).tolist()))
    uniq = {}
    for k in keys:
        if k not in uniq:
            uniq[k] = len(uniq)
    n_shocks = len(uniq)
    shock_idx = np.array([uniq[k] for k in keys], dtype=np.int32)

    eps = rng.standard_normal((n_samples, n))
    sd_b = sd[None, :]

    if team_corr is not None:
        # Cholesky path. Per (team, GW) factor f = L z, z ~ N(0, I_4).
        # Row at pos p picks f[p-1] as team-correlated component; idio
        # scale √(1 − C[p,p]) keeps total Var(X_i) = σ_i^2.
        C, L = team_corr
        # pos_id derived from one-hot dummies in inference rows.
        pos_dummies = rows[["pos_1", "pos_2", "pos_3", "pos_4"]].values
        pos = pos_dummies.argmax(axis=1)            # 0..3
        z = rng.standard_normal((n_samples, n_shocks, 4))
        f = z @ L.T                                 # (S, K, 4)
        # Index per-row team-correlated factor: f_row[s, i] = f[s, shock_idx[i], pos[i]]
        f_row = f[:, shock_idx, pos]                # (S, n)
        diag_C = np.diag(C)[pos]                    # (n,)
        idio_scale = np.sqrt(np.clip(1.0 - diag_C, 0.0, 1.0))[None, :]
        correlated = sd_b * f_row
        idiosync = sd_b * idio_scale * eps
    else:
        eta = rng.standard_normal((n_samples, n_shocks))
        rho = float(np.clip(team_rho, 0.0, 1.0))
        correlated = rho * sd_b * eta[:, shock_idx]
        idiosync = np.sqrt(max(1.0 - rho * rho, 0.0)) * sd_b * eps
    draws = mu[None, :] + correlated + idiosync  # (n_samples, n)

    # Aggregate per (player_id, fixture_gw). DGW = same player, same GW, two
    # fixtures, summed within each draw before stats.
    pid = rows["player_id"].astype(int).values
    gw = rows["fixture_gw"].astype(int).values
    pair = pid.astype(np.int64) * 1000 + gw.astype(np.int64)
    uniq_pairs, inverse = np.unique(pair, return_inverse=True)
    summed = np.zeros((n_samples, len(uniq_pairs)), dtype=np.float64)
    np.add.at(summed.T, inverse, draws.T)

    mean_xp = summed.mean(axis=0)
    std_xp = summed.std(axis=0, ddof=0)
    q10_mc = np.quantile(summed, 0.10, axis=0)
    q90_mc = np.quantile(summed, 0.90, axis=0)
    # Captaincy: anchor mean, half-weight on tail premium. 0.3 matches legacy
    # CAP_UPSIDE_WEIGHT so behaviour comparable when MC_TEAM_RHO=0.
    cap_xp = mean_xp + 0.3 * (q90_mc - mean_xp)

    out_pid = (uniq_pairs // 1000).astype(int)
    out_gw = (uniq_pairs % 1000).astype(int)
    return pd.DataFrame({
        "player_id": out_pid, "fixture_gw": out_gw,
        "mean_xp": mean_xp, "std_xp": std_xp,
        "cap_xp": cap_xp, "q10_mc": q10_mc, "q90_mc": q90_mc,
    })


class FPLEngine:
    """Load trained models. Produce wide projection frame for optimizer."""

    def __init__(self, fixtures: pd.DataFrame, history: pd.DataFrame,
                 players: pd.DataFrame, teams: pd.DataFrame) -> None:
        """Store inputs. Eager load points / minutes / bonus heads."""
        self.fixtures, self.players, self.teams = fixtures, players, teams
        self.history = history.copy()
        if "season" not in self.history.columns:
            self.history["season"] = SEASON
        id2team = players.set_index("id")["team"].to_dict()
        if "team" not in self.history.columns:
            self.history["team"] = self.history["player_id"].map(id2team)
        self.points_models = load_points_models()
        self.minutes_model = load_minutes_model()
        self.bonus_models = load_bonus_models()

    def _latest_rolling(self) -> pd.DataFrame:
        """Per-player most-recent current-season feature row. One row per player_id.

        Filter SEASON before tail(1). Stop historical-season rows becoming
        inference baseline for player not yet appeared in current season.
        """
        fx = build_match_features(self.fixtures, self.history, self.teams)
        past = build_player_features(self.history, self.players, fx)
        if past.empty:
            return pd.DataFrame()
        if "season" in past.columns:
            past = past[past["season"] == SEASON]
        if past.empty:
            return pd.DataFrame()
        return past.sort_values(["player_id", "round"]).groupby("player_id").tail(1).set_index("player_id")

    def _inference_rows(self, current_gw: int, horizon: int) -> pd.DataFrame:
        """One row per (player, upcoming fixture). DGWs yield multiple rows."""
        latest = self._latest_rolling()
        if latest.empty:
            return pd.DataFrame()
        fx_all = build_match_features(self.fixtures, self.history, self.teams)
        if "season" in fx_all.columns:
            fx_all = fx_all[fx_all["season"] == SEASON]
        fx_up = fx_all[(fx_all["event"] >= current_gw) &
                       (fx_all["event"] < current_gw + horizon)]
        if fx_up.empty:
            return pd.DataFrame()

        cols = points_feature_cols()
        # (side_tag, is_home, team_col, opp_xg_col, opp_xga_col,
        #  opp_elo_col, own_elo_col, lam_for_col, lam_against_col, cs_p_col)
        sides = (
            ("h", 1, "team_h", "a_xg_5", "a_xga_5", "elo_a_pre", "elo_h_pre",
             "lambda_h", "lambda_a", "cs_h_p"),
            ("a", 0, "team_a", "h_xg_5", "h_xga_5", "elo_h_pre", "elo_a_pre",
             "lambda_a", "lambda_h", "cs_a_p"),
        )
        keep_cols = list(dict.fromkeys(
            cols + minutes_feature_cols() + ["player_id", "fixture_gw", "team_id"]
        ))
        rows: list[dict[str, Any]] = []

        for _, fx in fx_up.iterrows():
            for (side_tag, home, team_col, opp_xg, opp_xga, opp_elo, own_elo,
                 lam_for, lam_against, cs_p) in sides:
                tid = int(fx[team_col])
                for _, p in self.players[self.players["team"] == tid].iterrows():
                    pid = int(p["id"])
                    if pid not in latest.index:
                        continue
                    r = latest.loc[pid].to_dict()
                    pos = int(p["element_type"])
                    r.update({
                        "is_home": home,
                        "opp_xg_5": float(fx[opp_xg]), "opp_xga_5": float(fx[opp_xga]),
                        "opp_elo": float(fx[opp_elo]), "own_elo": float(fx[own_elo]),
                        "elo_gap": float(fx[own_elo]) - float(fx[opp_elo]),
                        "own_lambda_for": float(fx[lam_for]),
                        "own_lambda_against": float(fx[lam_against]),
                        "own_cs_p": float(fx[cs_p]),
                        "pos_1": int(pos == 1), "pos_2": int(pos == 2),
                        "pos_3": int(pos == 3), "pos_4": int(pos == 4),
                        "is_pen_taker": int(p.get("penalties_order", 0) == 1),
                        "is_fk_taker": int(p.get("direct_freekicks_order", 0) == 1),
                        "player_id": pid, "fixture_gw": int(fx["event"]),
                        "team_id": tid,
                    })
                    # Cup congestion: pivot fx own-side cup_* (refresh from upcoming
                    # fixture, not stale historical row). Minutes head consumes.
                    for c in CUP_COLS:
                        src = f"{side_tag}_{c}"
                        r[f"own_{c}"] = (float(fx[src]) if src in fx
                                         else CUP_DEFAULTS[c])
                    rows.append({k: r.get(k, 0.0) for k in keep_cols})
        return pd.DataFrame(rows)

    def build_projections(self, current_gw: int, horizon: int = 5,
                          mc_samples: int = MC_SAMPLES,
                          team_rho: float = MC_TEAM_RHO,
                          team_corr: tuple[np.ndarray, np.ndarray] | None
                          = MC_TEAM_CORR) -> pd.DataFrame:
        """Wide df. xp_t / var_t / cap_xp_t per player + convenience totals.

        mc_samples > 0 → joint MC aggregation (within-club correlation via
        shared team shocks per draw). mc_samples=0 → deterministic Pearson-Tukey.
        `team_corr=(C, L)` enables the per-position Cholesky path (default when
        `data/team_rho.json::by_pos_pair` is loadable). Pass `team_corr=None`
        to force the legacy scalar `team_rho` path.
        """
        if self.points_models is None:
            return pd.DataFrame()
        rows = self._inference_rows(current_gw, horizon)
        if rows.empty:
            return pd.DataFrame()

        rows = rows.join(predict_quantiles(self.points_models, rows[points_feature_cols()]))

        # Bonus head: kept SEPARATE from points quantiles. Combination via
        # variance-additive Gaussian moments downstream (see _joint_mc_aggregate
        # + the non-MC branch below). Linear quantile addition is statistically
        # invalid for independent components; the former code over-estimated q90
        # by up to sqrt(2) and biased cap_xp toward high-bonus-history players.
        # BONUS_BLEND scales the bonus contribution at the moment level
        # (mu_b *= b, sigma_b *= b), preserving the meaning of the knob.
        if self.bonus_models is not None:
            bonus_q = predict_bonus_quantiles(self.bonus_models,
                                              rows[points_feature_cols()])
            rows["q10_b"] = BONUS_BLEND * bonus_q["q10"].values
            rows["q50_b"] = BONUS_BLEND * bonus_q["q50"].values
            rows["q90_b"] = BONUS_BLEND * bonus_q["q90"].values
        else:
            rows["q10_b"] = 0.0
            rows["q50_b"] = 0.0
            rows["q90_b"] = 0.0

        pmeta = self.players.set_index("id")
        bad = pmeta["status"].isin(["s", "n", "u"]).to_dict()
        # Two-stage minutes. plays = P(on pitch) → discount q90 ceiling. mins_pred
        # = plays * mins_when_played = E[mins/90] → discount q10/q50 (mean mass
        # needs minutes on pitch). Ceiling realized once on the pitch — hauls
        # land before subs.
        if self.minutes_model is not None:
            comp = predict_minutes(self.minutes_model, rows, return_components=True)
            plays = comp["plays"]
            mins_pred = comp["mins_pred"]
        else:
            plays = pd.Series(1.0, index=rows.index)
            mins_pred = pd.Series(1.0, index=rows.index)
        # FPL chance_of_playing_next_round authoritative for IMMEDIATE next GW.
        # FPL knows specific injuries model can't infer from history. Apply as
        # hard upper bound on plays + mins_pred for that GW only; applying
        # across horizon double-counts injuries for likely-recovered weeks.
        chance = (pd.to_numeric(pmeta["chance_of_playing_next_round"],
                                errors="coerce").fillna(100.0) / 100.0).to_dict()
        next_gw = rows["fixture_gw"].min()
        is_next = (rows["fixture_gw"] == next_gw).values
        fpl_hint = rows["player_id"].map(chance).fillna(1.0).values
        for s in (plays, mins_pred):
            s.loc[is_next] = pd.Series(
                [min(v, h) for v, h in zip(s.values[is_next], fpl_hint[is_next])],
                index=s.index[is_next],
            )
        # Hard-bad statuses ('s' suspended, 'n' not avail, 'u' unavail). Zero all GWs.
        bad_mask = rows["player_id"].map(bad).fillna(False).values
        plays.loc[bad_mask] = 0.0
        mins_pred.loc[bad_mask] = 0.0
        rows["chance"] = mins_pred.values
        rows["plays"] = plays.values
        # Zero-inflated mixture quantile transform. Both heads trained on
        # minutes > 0, so their q10/q50/q90 are F_played quantiles. Compose
        # with P(plays) = `plays` via _mixture_quantile to get unconditional
        # quantiles. This replaces the old q*p heuristic which was correct
        # only for the mean (linearity of expectation) and wrong for arbitrary
        # quantiles. Same probability `plays` applied to points + bonus since
        # bonus is conditional on the same event (any appearance, BPS earned).
        p_vec = rows["plays"].values
        for prefix in ("", "_b"):
            q10_col, q50_col, q90_col = f"q10{prefix}", f"q50{prefix}", f"q90{prefix}"
            q10v = rows[q10_col].values.astype(float)
            q50v = rows[q50_col].values.astype(float)
            q90v = rows[q90_col].values.astype(float)
            rows[q10_col] = _mixture_quantile(q10v, q50v, q90v, p_vec, 0.10)
            rows[q50_col] = _mixture_quantile(q10v, q50v, q90v, p_vec, 0.50)
            rows[q90_col] = _mixture_quantile(q10v, q50v, q90v, p_vec, 0.90)

        if mc_samples and mc_samples > 0:
            mc = _joint_mc_aggregate(rows, n_samples=mc_samples,
                                     team_rho=team_rho, team_corr=team_corr)
            mc = mc.rename(columns={"std_xp": "variance"})
            agg = mc[["player_id", "fixture_gw", "mean_xp", "variance", "cap_xp"]]
        else:
            # Non-MC: deterministic Pearson-Tukey on per-row moments, then
            # variance-additive combine of points + bonus heads. Aggregate
            # across DGW fixtures by summing means + adding variances under
            # independence (within-row, across fixtures the same player faces).
            # The `variance` column holds sigma (not sigma^2) to match the MC
            # branch's downstream contract.
            mu_p, sd_p = _row_quantile_to_moments(rows["q10"].values,
                                                  rows["q50"].values,
                                                  rows["q90"].values)
            mu_b, sd_b = _row_quantile_to_moments(rows["q10_b"].values,
                                                  rows["q50_b"].values,
                                                  rows["q90_b"].values)
            rows = rows.assign(
                _mu=(mu_p + mu_b),
                _var=(sd_p ** 2 + sd_b ** 2),
            )
            grouped = rows.groupby(["player_id", "fixture_gw"], as_index=False).agg(
                mean_xp=("_mu", "sum"), var_sum=("_var", "sum"))
            grouped["variance"] = np.sqrt(grouped["var_sum"].clip(lower=0.0))
            CAP_UPSIDE_WEIGHT = 0.3
            # cap_xp = mean + gamma * (q90 - mean) under Gaussian envelope
            # -> mean + gamma * 1.28 * sigma.
            grouped["cap_xp"] = (grouped["mean_xp"]
                                 + CAP_UPSIDE_WEIGHT * 1.28 * grouped["variance"])
            agg = grouped[["player_id", "fixture_gw", "mean_xp",
                           "variance", "cap_xp"]]

        xp = agg.pivot(index="player_id", columns="fixture_gw", values="mean_xp").fillna(0.0)
        xp.columns = [f"xp_{int(c)}" for c in xp.columns]
        var = agg.pivot(index="player_id", columns="fixture_gw", values="variance").fillna(0.0)
        var.columns = [f"var_{int(c)}" for c in var.columns]
        cap = agg.pivot(index="player_id", columns="fixture_gw", values="cap_xp").fillna(0.0)
        cap.columns = [f"cap_xp_{int(c)}" for c in cap.columns]

        meta = self.players[["id", "web_name", "team", "element_type",
                             "now_cost", "selected_by_percent"]].rename(
            columns={"web_name": "name", "team": "team_id", "element_type": "pos_id"})
        meta["price"] = meta["now_cost"] / 10.0
        meta["eo"] = pd.to_numeric(meta["selected_by_percent"], errors="coerce").fillna(0.0) / 100.0
        meta = meta[["id", "name", "team_id", "pos_id", "price", "eo"]]

        out = (meta.merge(xp, left_on="id", right_index=True)
                   .merge(var, left_on="id", right_index=True)
                   .merge(cap, left_on="id", right_index=True))
        out = out[out["pos_id"].isin([1, 2, 3, 4])].reset_index(drop=True)

        xp_cols = sorted([c for c in out.columns if c.startswith("xp_")],
                         key=lambda c: int(c.split("_")[1]))
        out["horizon_xp"] = out[xp_cols].sum(axis=1)
        out["next_gw_xp"] = out[xp_cols[0]] if xp_cols else 0.0
        return out
