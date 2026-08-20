"""Chip heuristics. TC / BB / FH / WC over horizon.

FPL 2026/27 chip rules: two of each chip per season. Set 1 (tc1/bb1/fh1/wc1)
covers GW1..19 and expires unused at the GW19 deadline; set 2 covers GW20..38.
FH cannot be played in consecutive GWs.

The pipeline *recommends* chips, it never plays them, so used-chip inventory
cannot be derived from pipeline output -- it is read from data/chip_state.json,
which the manager maintains. Absent file = nothing used yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HALF1_END = 19
HALF2_END = 38

CHIP_STATE = Path(__file__).resolve().parent.parent / "data" / "chip_state.json"


def chip_set(gw: int) -> int:
    """1 for GW1..19, 2 for GW20..38."""
    return 1 if int(gw) <= HALF1_END else 2


def set_last_gw(gw: int) -> int:
    """Final GW of the chip set `gw` falls in — the expiry deadline."""
    return HALF1_END if chip_set(gw) == 1 else HALF2_END


def load_chip_state(path: Path | None = None) -> dict[str, int]:
    """{token: gw_played} from data/chip_state.json. Missing/corrupt -> {}.

    Shape: {"used": {"tc1": 5, "fh1": 12}}. Never written by the pipeline.
    """
    p = CHIP_STATE if path is None else path
    try:
        used = json.loads(p.read_text()).get("used", {})
        return {str(k): int(v) for k, v in used.items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def chip_token(kind: str, gw: int) -> str:
    """e.g. ("tc", 12) -> "tc1"; ("tc", 25) -> "tc2"."""
    return f"{kind}{chip_set(gw)}"


def chip_available(kind: str, gw: int, used: dict[str, int]) -> bool:
    """True when this half's copy of `kind` has not been played."""
    return chip_token(kind, gw) not in used


def _set_gws(proj_gws: list[int], gw: int) -> list[int]:
    """Horizon GWs that belong to the same chip set as `gw`.

    Without this an H=8 scan from GW16 can propose a set-1 chip for GW22,
    which is a set-2 GW — the set-1 copy has expired by then.
    """
    return [t for t in proj_gws if chip_set(t) == chip_set(gw)]


def recommend_triple_captain(proj: pd.DataFrame, squad_ids: set[int],
                             gw: int, used: dict[str, int] | None = None) -> dict:
    """Best (gw, owned MID/FWD) by cap_xp. Chip lives/dies on boom.

    Restricted to GWs in the same chip set as `gw`; returns gw=None when this
    half's TC is already played.
    """
    used = used or {}
    if not chip_available("tc", gw, used):
        return {"gw": None, "player_id": None, "bonus": 0.0}
    gws = sorted(int(c.split("_")[-1]) for c in proj.columns if c.startswith("cap_xp_"))
    gws = _set_gws(gws, gw)
    owned = proj[proj["id"].isin(squad_ids) & proj["pos_id"].isin([3, 4])]
    best = {"gw": None, "player_id": None, "bonus": 0.0}
    for t in gws:
        r = owned.sort_values(f"cap_xp_{t}", ascending=False).head(1)
        if not r.empty and float(r[f"cap_xp_{t}"].iloc[0]) > best["bonus"]:
            best = {"gw": t, "player_id": int(r["id"].iloc[0]),
                    "bonus": float(r[f"cap_xp_{t}"].iloc[0])}
    return best


def recommend_bench_boost(proj: pd.DataFrame, squad_ids: set[int], xi_ids: set[int],
                          gw: int, used: dict[str, int] | None = None) -> dict:
    """GW where bench-4 sum xp peaks. Same-chip-set GWs only."""
    used = used or {}
    if not chip_available("bb", gw, used):
        return {"gw": None, "bonus": 0.0}
    gws = sorted(int(c.split("_")[1]) for c in proj.columns if c.startswith("xp_"))
    gws = _set_gws(gws, gw)
    bench = proj[proj["id"].isin(squad_ids - xi_ids)]
    best = {"gw": None, "bonus": 0.0}
    for t in gws:
        total = float(bench[f"xp_{t}"].sum())
        if total > best["bonus"]:
            best = {"gw": t, "bonus": total}
    return best


def recommend_free_hit(fixtures: pd.DataFrame, current_gw: int, horizon: int,
                       used: dict[str, int] | None = None) -> dict:
    """GW with most teams blanking inside horizon. Current season only.

    Clamped to the current chip set's GWs, and FH cannot follow the GW its
    own set-mate was played in (no consecutive Free Hits).
    """
    used = used or {}
    if not chip_available("fh", current_gw, used):
        return {"gw": None, "blanks": 0}
    fx_curr = fixtures
    if "season" in fx_curr.columns:
        from data_loader import SEASON
        fx_curr = fx_curr[fx_curr["season"] == SEASON]
    teams_all = set(fx_curr["team_h"].tolist() + fx_curr["team_a"].tolist())
    best = {"gw": None, "blanks": 0}
    # Any recorded FH GW blocks the GW immediately after it.
    last_fh = max((g for k, g in used.items() if k.startswith("fh")), default=-10)
    hi = min(current_gw + horizon, set_last_gw(current_gw) + 1)
    for t in range(current_gw, hi):
        if t == last_fh + 1:
            continue
        fx = fx_curr[fx_curr["event"] == t]
        if fx.empty:
            continue
        missing = len(teams_all - set(fx["team_h"].tolist() + fx["team_a"].tolist()))
        if missing > best["blanks"]:
            best = {"gw": t, "blanks": missing}
    return best


def recommend_wildcard(transfers_in: list, hits: int, gw: int,
                       used: dict[str, int] | None = None) -> dict:
    """Fire if RHC wants >=4 transfers or >=2 hits. Proxy: squad far from optimal.

    Never fires once this half's WC is spent.
    """
    used = used or {}
    have = chip_available("wc", gw, used or {})
    return {"recommend": have and (len(transfers_in) >= 4 or hits >= 2),
            "n_transfers": len(transfers_in), "hits": hits, "available": have}
