"""Chip scheduling heuristics for TC / BB / FH / WC over the projection horizon."""
from __future__ import annotations

import pandas as pd


def recommend_triple_captain(proj: pd.DataFrame, squad_ids: set[int]) -> dict:
    """Best (gw, MID/FWD player) by cap_xp among owned — chip lives or dies on the boom."""
    gws = sorted(int(c.split("_")[-1]) for c in proj.columns if c.startswith("cap_xp_"))
    owned = proj[proj["id"].isin(squad_ids) & proj["pos_id"].isin([3, 4])]
    best = {"gw": None, "player_id": None, "bonus": 0.0}
    for t in gws:
        r = owned.sort_values(f"cap_xp_{t}", ascending=False).head(1)
        if not r.empty and float(r[f"cap_xp_{t}"].iloc[0]) > best["bonus"]:
            best = {"gw": t, "player_id": int(r["id"].iloc[0]),
                    "bonus": float(r[f"cap_xp_{t}"].iloc[0])}
    return best


def recommend_bench_boost(proj: pd.DataFrame, squad_ids: set[int], xi_ids: set[int]) -> dict:
    """Picks the horizon GW where the 4 bench players project the highest sum."""
    gws = sorted(int(c.split("_")[1]) for c in proj.columns if c.startswith("xp_"))
    bench = proj[proj["id"].isin(squad_ids - xi_ids)]
    best = {"gw": None, "bonus": 0.0}
    for t in gws:
        total = float(bench[f"xp_{t}"].sum())
        if total > best["bonus"]:
            best = {"gw": t, "bonus": total}
    return best


def recommend_free_hit(fixtures: pd.DataFrame, current_gw: int, horizon: int) -> dict:
    """Finds the GW with the most teams blanking inside the horizon (current season only)."""
    fx_curr = fixtures
    if "season" in fx_curr.columns:
        from data_loader import SEASON
        fx_curr = fx_curr[fx_curr["season"] == SEASON]
    teams_all = set(fx_curr["team_h"].tolist() + fx_curr["team_a"].tolist())
    best = {"gw": None, "blanks": 0}
    for t in range(current_gw, current_gw + horizon):
        fx = fx_curr[fx_curr["event"] == t]
        missing = len(teams_all - set(fx["team_h"].tolist() + fx["team_a"].tolist()))
        if missing > best["blanks"]:
            best = {"gw": t, "blanks": missing}
    return best


def recommend_wildcard(transfers_in: list, hits: int) -> dict:
    """Heuristic: wildcard if the RHC wants >=4 transfers or >=2 hits this GW."""
    return {"recommend": len(transfers_in) >= 4 or hits >= 2,
            "n_transfers": len(transfers_in), "hits": hits}