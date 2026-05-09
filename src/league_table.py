"""Per-(season, event) league table state and stakes features.

Computes pre-match standings (pts, GD, GF, rank) plus mathematically-normalised
distance to each tier cutoff: title (1st), Champions League (4th), European cups
band (6th, covers UEL + UECL in normal years), and relegation safety (17th).
Encodes late-season motivation that EMA rolling form lags on — title-chasers
ramping up, mid-table teams on beach mode, drop-zone fights.

A fixture's pre-match row reflects only events strictly earlier in the same
season (cumsum.shift(1)). No leakage from the fixture's own outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_GW = 38  # Premier League

# tier label -> rank position whose pts define that tier's cutoff.
# Convention: pts_to_<tier>_norm = (cutoff_pts - own_pts) / (3 * gws_remaining).
# Positive = need that many pts to reach tier; negative = clear of tier by that.
TIER_RANKS = {"title": 1, "top4": 4, "top6": 6, "safety": 17}

PRE_COLS = ("pts", "gf", "ga", "gd", "played")


def build_pre_match_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team_id, event) cumulative pre-match standings.

    Cumulative aggregation over finished fixtures, then `shift(1)` within
    (season, team) so the row stamped at event E carries state from events
    {1..E-1}. Teams without prior fixtures default to all-zero pre-state.
    """
    fx = fixtures.copy()
    if "season" not in fx.columns:
        fx["season"] = "current"
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
    fx["team_h_score"] = pd.to_numeric(fx["team_h_score"], errors="coerce")
    fx["team_a_score"] = pd.to_numeric(fx["team_a_score"], errors="coerce")

    finished = fx[fx["finished"].astype(str).str.lower().isin(("true", "1"))].copy()
    finished = finished.dropna(subset=["event", "team_h_score", "team_a_score"])
    finished["event"] = finished["event"].astype(int)
    if finished.empty:
        return pd.DataFrame(columns=["season", "event", "team_id", *PRE_COLS])

    h = finished[["season", "event", "team_h", "team_h_score", "team_a_score"]].rename(
        columns={"team_h": "team_id", "team_h_score": "gf", "team_a_score": "ga"})
    a = finished[["season", "event", "team_a", "team_a_score", "team_h_score"]].rename(
        columns={"team_a": "team_id", "team_a_score": "gf", "team_h_score": "ga"})
    long = pd.concat([h, a], ignore_index=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3,
                            np.where(long["gf"] == long["ga"], 1, 0)).astype(float)
    long["played"] = 1.0

    # Sum DGW legs per (season, team_id, event) so cumsum across events is clean.
    agg = (long.groupby(["season", "team_id", "event"], as_index=False)
                .agg(pts=("pts", "sum"), gf=("gf", "sum"),
                     ga=("ga", "sum"), played=("played", "sum")))
    agg["gd"] = agg["gf"] - agg["ga"]

    # Reindex: emit one row per (season, team, event_in_season) so pre-state is
    # available for opponents who blank that GW. Missing (team didn't play GW E)
    # gets a zero delta row, but cumsum below still propagates earlier state.
    fx_cur = fx.dropna(subset=["event", "season"]).copy()
    fx_cur["event"] = fx_cur["event"].astype(int)
    teams_per_season: dict[str, set[int]] = {}
    for s, sub in fx_cur.groupby("season"):
        teams_per_season[s] = set(pd.concat([sub["team_h"], sub["team_a"]]).dropna().astype(int))
    grid_rows: list[tuple] = []
    for season, team_set in teams_per_season.items():
        for t in team_set:
            for e in range(1, SEASON_GW + 1):
                grid_rows.append((season, int(t), e))
    grid = pd.DataFrame(grid_rows, columns=["season", "team_id", "event"])
    full = grid.merge(agg, on=["season", "team_id", "event"], how="left").fillna(0.0)

    full = full.sort_values(["season", "team_id", "event"]).reset_index(drop=True)
    grp = full.groupby(["season", "team_id"], group_keys=False)
    for c in PRE_COLS:
        full[c] = grp[c].cumsum()
    for c in PRE_COLS:
        full[c] = grp[c].shift(1).fillna(0.0)
    return full[["season", "event", "team_id", *PRE_COLS]]


def _rank_and_cuts(table: pd.DataFrame) -> pd.DataFrame:
    """Vectorised: per (season, event) rank teams + emit each tier's cutoff pts."""
    if table.empty:
        return table.assign(rank=pd.Series(dtype=float),
                             **{f"cut_{l}": pd.Series(dtype=float) for l in TIER_RANKS})
    t = table.sort_values(["season", "event", "pts", "gd", "gf"],
                           ascending=[True, True, False, False, False]).copy()
    t["rank"] = t.groupby(["season", "event"]).cumcount() + 1
    for label, k in TIER_RANKS.items():
        kth = (t[t["rank"] == k][["season", "event", "pts"]]
               .rename(columns={"pts": f"cut_{label}"}))
        t = t.merge(kth, on=["season", "event"], how="left")
        # If league smaller than k (early test fixtures only), fall back to
        # last available rank's pts.
        last_pts = t.groupby(["season", "event"])["pts"].transform("min")
        t[f"cut_{label}"] = t[f"cut_{label}"].fillna(last_pts)
    return t


def attach_stakes(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures with stakes cols attached per side.

    New cols (per side prefix h_/a_):
      pts, played, rank, ppg, in_drop_zone,
      pts_to_{title,top4,top6,safety}_norm
    Plus scalar `gws_remaining`.

    Normalised gap = (tier_cutoff_pts - own_pts) / (3 * gws_remaining), so
    positive ↔ team needs that fraction of remaining max-pts to reach the
    tier; negative ↔ already clear of it.
    """
    fx = fixtures.copy()
    if "season" not in fx.columns:
        fx["season"] = "current"

    table = build_pre_match_table(fx)
    if table.empty:
        # Cold start (e.g. GW1 first run). Emit neutral defaults so feature
        # cols exist for the booster.
        for s in ("h", "a"):
            fx[f"{s}_pts"] = 0.0
            fx[f"{s}_played"] = 0.0
            fx[f"{s}_rank"] = 10.0
            fx[f"{s}_ppg"] = 0.0
            fx[f"{s}_in_drop_zone"] = 0.0
            for label in TIER_RANKS:
                fx[f"{s}_pts_to_{label}_norm"] = 0.0
        fx["gws_remaining"] = (SEASON_GW - pd.to_numeric(fx["event"], errors="coerce")
                                ).clip(lower=0).fillna(SEASON_GW)
        return fx

    enriched = _rank_and_cuts(table)

    side_cols = ["pts", "played", "rank", *[f"cut_{l}" for l in TIER_RANKS]]
    for s, key in (("h", "team_h"), ("a", "team_a")):
        right = enriched[["season", "event", "team_id", *side_cols]].rename(
            columns={"team_id": key, **{c: f"{s}_{c}" for c in side_cols}})
        fx = fx.merge(right, on=["season", "event", key], how="left")

    gws_left = (SEASON_GW - pd.to_numeric(fx["event"], errors="coerce")).clip(lower=1)
    for s in ("h", "a"):
        fx[f"{s}_pts"] = fx[f"{s}_pts"].fillna(0.0)
        fx[f"{s}_played"] = fx[f"{s}_played"].fillna(0.0)
        fx[f"{s}_rank"] = fx[f"{s}_rank"].fillna(10.0)
        fx[f"{s}_ppg"] = fx[f"{s}_pts"] / fx[f"{s}_played"].clip(lower=1)
        fx[f"{s}_in_drop_zone"] = (fx[f"{s}_rank"] >= 18).astype(float)
        for label in TIER_RANKS:
            cut_col = f"{s}_cut_{label}"
            cut = fx[cut_col].fillna(0.0) if cut_col in fx.columns else 0.0
            gap = cut - fx[f"{s}_pts"]
            fx[f"{s}_pts_to_{label}_norm"] = gap / (3.0 * gws_left)
            if cut_col in fx.columns:
                fx = fx.drop(columns=[cut_col])

    fx["gws_remaining"] = (SEASON_GW - pd.to_numeric(fx["event"], errors="coerce")
                            ).clip(lower=0).fillna(SEASON_GW)
    return fx


def stakes_match_cols() -> list[str]:
    """Match-model side-prefixed stakes cols."""
    cols: list[str] = []
    for s in ("h", "a"):
        cols += [f"{s}_pts", f"{s}_played", f"{s}_rank", f"{s}_ppg",
                 f"{s}_in_drop_zone"]
        cols += [f"{s}_pts_to_{label}_norm" for label in TIER_RANKS]
    cols.append("gws_remaining")
    return cols


def stakes_player_cols() -> list[str]:
    """Points-model own/opp side-conditioned stakes cols."""
    cols: list[str] = []
    for s in ("own", "opp"):
        cols += [f"{s}_rank", f"{s}_ppg", f"{s}_in_drop_zone"]
        cols += [f"{s}_pts_to_{label}_norm" for label in TIER_RANKS]
    cols.append("gws_remaining")
    return cols
