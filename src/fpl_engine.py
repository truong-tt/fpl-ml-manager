"""FPLEngine. Build per-player per-GW projection frame for optimizer."""
from __future__ import annotations

from typing import Any

import pandas as pd

from data_loader import SEASON
from features import build_match_features, build_player_features, points_feature_cols
from train_minutes_model import load_minutes_model, predict_minutes
from train_points_model import load_points_models, predict_quantiles


class FPLEngine:
    """Load trained models. Produce wide projection frame for optimizer."""

    def __init__(self, fixtures: pd.DataFrame, history: pd.DataFrame,
                 players: pd.DataFrame, teams: pd.DataFrame) -> None:
        """Store inputs. Eager load quantile points + minutes models."""
        self.fixtures, self.players, self.teams = fixtures, players, teams
        self.history = history.copy()
        if "season" not in self.history.columns:
            self.history["season"] = SEASON
        id2team = players.set_index("id")["team"].to_dict()
        if "team" not in self.history.columns:
            self.history["team"] = self.history["player_id"].map(id2team)
        self.points_models = load_points_models()
        self.minutes_model = load_minutes_model()

    def _latest_rolling(self) -> pd.DataFrame:
        """Per-player most-recent current-season feature row. One row per player_id.

        Filter to SEASON before tail(1). Stop historical-season rows becoming
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
        sides = (("h", 1, "team_h", "a_xg_5", "a_xga_5", "elo_a_pre", "elo_h_pre"),
                 ("a", 0, "team_a", "h_xg_5", "h_xga_5", "elo_h_pre", "elo_a_pre"))
        rows: list[dict[str, Any]] = []

        for _, fx in fx_up.iterrows():
            for _, home, team_col, opp_xg, opp_xga, opp_elo, own_elo in sides:
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
                        "pos_1": int(pos == 1), "pos_2": int(pos == 2),
                        "pos_3": int(pos == 3), "pos_4": int(pos == 4),
                        "is_pen_taker": int(p.get("penalties_order", 0) == 1),
                        "is_fk_taker": int(p.get("direct_freekicks_order", 0) == 1),
                        "player_id": pid, "fixture_gw": int(fx["event"]),
                    })
                    rows.append({k: r.get(k, 0.0) for k in cols + ["player_id", "fixture_gw"]})
        return pd.DataFrame(rows)

    def build_projections(self, current_gw: int, horizon: int = 5) -> pd.DataFrame:
        """Return wide df. xp_{t}, var_{t} per player + convenience totals."""
        if self.points_models is None:
            return pd.DataFrame()
        rows = self._inference_rows(current_gw, horizon)
        if rows.empty:
            return pd.DataFrame()

        rows = rows.join(predict_quantiles(self.points_models, rows[points_feature_cols()]))

        pmeta = self.players.set_index("id")
        bad = pmeta["status"].isin(["s", "n", "u"]).to_dict()
        # Learned availability multiplier: predicted minutes / 90 per (player, fixture).
        # Fallback 1.0 if model not trained yet.
        if self.minutes_model is not None:
            mins_pred = predict_minutes(self.minutes_model, rows)
        else:
            mins_pred = pd.Series(1.0, index=rows.index)
        # FPL `chance_of_playing_next_round` authoritative for IMMEDIATE next GW.
        # FPL knows specific injuries model cannot infer from history alone.
        # Use as hard upper bound that GW only. Apply across horizon double-counts
        # injuries for weeks player likely recovered.
        chance = (pd.to_numeric(pmeta["chance_of_playing_next_round"],
                                errors="coerce").fillna(100.0) / 100.0).to_dict()
        next_gw = rows["fixture_gw"].min()
        is_next = (rows["fixture_gw"] == next_gw).values
        fpl_hint = rows["player_id"].map(chance).fillna(1.0).values
        mins_pred.loc[is_next] = pd.Series(
            [min(m, h) for m, h in zip(mins_pred.values[is_next], fpl_hint[is_next])],
            index=mins_pred.index[is_next],
        )
        # Hard-bad statuses ('s' suspended, 'n' not available, 'u' unavailable). Zero all GWs.
        bad_mask = rows["player_id"].map(bad).fillna(False).values
        mins_pred.loc[bad_mask] = 0.0
        rows["chance"] = mins_pred.values
        for c in ("q10", "q50", "q90"):
            rows[c] = rows[c] * rows["chance"]

        agg = rows.groupby(["player_id", "fixture_gw"], as_index=False).agg(
            q10=("q10", "sum"), q50=("q50", "sum"), q90=("q90", "sum"))
        # Pearson-Tukey 3-quantile mean estimator. q50 is median; FPL points
        # right-skewed (most weeks 1-2, hauls 8-15) so median << mean. Use mean
        # for XP so per-GW totals match expected scoring, not median scoring.
        agg["mean_xp"] = (agg["q10"] + 4.0 * agg["q50"] + agg["q90"]) / 6.0
        # Use std not variance. Penalty scale linear with ceiling, not quadratic.
        # Stop solver dodging high-ceiling players.
        agg["variance"] = (agg["q90"] - agg["q10"]) / 2.56
        # Captaincy = mean anchor + small upside premium. Pure q90 crowned
        # 1.99-mean ceiling players over high-mean MIDs. alpha=0.3 keeps mean dominant.
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