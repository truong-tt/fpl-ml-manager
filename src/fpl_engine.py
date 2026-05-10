"""FPLEngine. Per-(player, GW) projection frame for optimizer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import SEASON
from features import build_match_features, build_player_features, points_feature_cols
from train_bonus_model import load_bonus_models, predict_bonus_quantiles
from train_minutes_model import load_minutes_model, predict_minutes
from train_points_model import load_points_models, predict_quantiles

# Bonus blend factor. 1.0 = full additivity. Points head now trained on
# `total_points - bonus` (target excludes bonus), bonus head adds back. Clean
# decomposition. Pre-fix used 0.5 damp + total_points target → double-count.
BONUS_BLEND = 1.0

# Joint MC aggregation. Per-(team, GW) shock + per-row idiosyncratic. Shock-
# correlated fraction = MC_TEAM_RHO; rest idiosyncratic. Rho=0 → independence
# (matches prior diagonal aggregation). Rho>0 captures within-club covariance:
# clean sheet correlates DEF + GK upside; strong attack correlates MID + FWD.
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


MC_TEAM_RHO = _load_team_rho_default()


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


def _joint_mc_aggregate(rows: pd.DataFrame, n_samples: int = MC_SAMPLES,
                        team_rho: float = MC_TEAM_RHO,
                        seed: int = 17) -> pd.DataFrame:
    """Sample correlated player totals per (player, fixture_gw).

    Same (team_id, fixture_gw) rows share team-shock per draw — Liverpool CS
    shock lifts Salah + Virgil together; goal blitz lifts Salah + Diaz.
    Idiosyncratic remainder keeps individual upside unrolled.

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
    mu, sd = _row_quantile_to_moments(q10, q50, q90)

    # (team_id, fixture_gw) → unique shock index.
    keys = list(zip(rows["team_id"].astype(int).tolist(),
                    rows["fixture_gw"].astype(int).tolist()))
    uniq = {}
    for k in keys:
        if k not in uniq:
            uniq[k] = len(uniq)
    n_shocks = len(uniq)
    shock_idx = np.array([uniq[k] for k in keys], dtype=np.int32)

    # (n_samples, n_shocks) team shocks; (n_samples, n) idiosyncratic eps.
    eta = rng.standard_normal((n_samples, n_shocks))
    eps = rng.standard_normal((n_samples, n))

    sd_b = sd[None, :]
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
        rows: list[dict[str, Any]] = []

        for _, fx in fx_up.iterrows():
            for (_, home, team_col, opp_xg, opp_xga, opp_elo, own_elo,
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
                    rows.append({k: r.get(k, 0.0) for k in
                                 cols + ["player_id", "fixture_gw", "team_id"]})
        return pd.DataFrame(rows)

    def build_projections(self, current_gw: int, horizon: int = 5,
                          mc_samples: int = MC_SAMPLES,
                          team_rho: float = MC_TEAM_RHO) -> pd.DataFrame:
        """Wide df. xp_t / var_t / cap_xp_t per player + convenience totals.

        mc_samples > 0 → joint MC aggregation (within-club correlation via
        shared team shocks per draw). mc_samples=0 → deterministic Pearson-Tukey.
        """
        if self.points_models is None:
            return pd.DataFrame()
        rows = self._inference_rows(current_gw, horizon)
        if rows.empty:
            return pd.DataFrame()

        rows = rows.join(predict_quantiles(self.points_models, rows[points_feature_cols()]))

        # Bonus head q-quantiles, blend factor BONUS_BLEND. Lifts q90 mainly for
        # bonus-heavy archetypes (CS-keeping defenders, save-rich GKs) without
        # disturbing q10 floor.
        if self.bonus_models is not None:
            bonus_q = predict_bonus_quantiles(self.bonus_models,
                                              rows[points_feature_cols()])
            for c in ("q10", "q50", "q90"):
                rows[c] = rows[c] + BONUS_BLEND * bonus_q[c].values

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
        rows["q10"] = rows["q10"] * rows["chance"]
        rows["q50"] = rows["q50"] * rows["chance"]
        rows["q90"] = rows["q90"] * rows["plays"]

        if mc_samples and mc_samples > 0:
            mc = _joint_mc_aggregate(rows, n_samples=mc_samples, team_rho=team_rho)
            mc = mc.rename(columns={"std_xp": "variance"})
            agg = mc[["player_id", "fixture_gw", "mean_xp", "variance", "cap_xp"]]
        else:
            agg = rows.groupby(["player_id", "fixture_gw"], as_index=False).agg(
                q10=("q10", "sum"), q50=("q50", "sum"), q90=("q90", "sum"))
            agg["mean_xp"] = (SWANSON_W10 * agg["q10"]
                              + SWANSON_W50 * agg["q50"]
                              + SWANSON_W90 * agg["q90"])
            agg["variance"] = (agg["q90"] - agg["q10"]) / 2.56
            CAP_UPSIDE_WEIGHT = 0.3
            agg["cap_xp"] = agg["mean_xp"] + CAP_UPSIDE_WEIGHT * (agg["q90"] - agg["mean_xp"])

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
