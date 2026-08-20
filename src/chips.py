"""Chip heuristics. TC / BB / FH / WC over horizon.

FPL 2026/27 chip rules: two of each chip per season. Set 1 (tc1/bb1/fh1/wc1)
covers GW1..19 and expires unused at the GW19 deadline; set 2 covers GW20..38.
FH cannot be played in consecutive GWs.

There is no linked FPL entry to sync from, so the pipeline is the manager and
keeps its own ledger of plays in data/chip_state.json. Absent, malformed, or
stamped with a previous season = nothing used yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HALF1_END = 19
HALF2_END = 38

# Firing thresholds, shared with season_replay.py so the live path and the
# replay commit chips on the same evidence.
TC_TRIGGER_PREMIUM = 4.5    # fire TC when (cap_xp - xp) for the captain >= this
BB_TRIGGER_BENCH_EV = 10.0  # fire BB when bench mu sum >= this
FH_TRIGGER_BLANKS = 2       # fire FH when this many teams blank

CHIP_STATE = Path(__file__).resolve().parent.parent / "data" / "chip_state.json"


def chip_set(gw: int) -> int:
    """1 for GW1..19, 2 for GW20..38."""
    return 1 if int(gw) <= HALF1_END else 2


def set_last_gw(gw: int) -> int:
    """Final GW of the chip set `gw` falls in — the expiry deadline."""
    return HALF1_END if chip_set(gw) == 1 else HALF2_END


def _season() -> str:
    """Lazy import — keeps `chips` usable without the data_loader dependency."""
    from data_loader import SEASON
    return SEASON


def load_chip_state(path: Path | None = None,
                    season: str | None = None) -> dict[str, int]:
    """{token: gw_played} from data/chip_state.json. Missing/corrupt -> {}.

    Shape: {"season": "2026-2027", "used": {"tc1": 5, "fh1": 12}}.

    Chips reset every season, so a ledger stamped with a previous season reads
    as empty — without this the 8 tokens spent in one season would suppress
    every chip in the next. Matches the season guards on model_state.json and
    squad_snapshot.csv. An unstamped ledger is treated as the current season
    so a hand-written file does not silently evaporate.
    """
    p = CHIP_STATE if path is None else path
    want = _season() if season is None else season
    try:
        doc = json.loads(p.read_text())
        if str(doc.get("season", want)) != want:
            return {}
        return {str(k): int(v) for k, v in doc.get("used", {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def save_chip_state(used: dict[str, int], path: Path | None = None,
                    season: str | None = None) -> None:
    """Write the ledger back, stamped with the season, keeping the comment."""
    p = CHIP_STATE if path is None else path
    try:
        doc = json.loads(p.read_text())
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    doc["season"] = _season() if season is None else season
    doc["used"] = dict(sorted(used.items()))
    p.write_text(json.dumps(doc, indent=2) + chr(10))


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


def withdraw_pending(used: dict[str, int], gw: int) -> bool:
    """Drop any chip recorded for `gw`, in place. True if one was dropped.

    The current GW has not kicked off — its deadline is still ahead, so an
    entry against it is intent, not a spent chip. Midweek news must be able to
    change or withdraw the call, and the recommenders would otherwise treat it
    as spent and refuse to re-propose it. Entries for earlier GWs are past
    their deadline and frozen.

    Call this before the recommenders run; commit_chip then re-decides and
    persists. At most one chip per GW survives either way.
    """
    stale = [k for k, v in used.items() if v == gw]
    for k in stale:
        del used[k]
    return bool(stale)


def commit_chip(gw: int, proj: pd.DataFrame, tc: dict, bb: dict, fh: dict,
                wc: dict, used: dict[str, int],
                path: Path | None = None) -> str | None:
    """Record at most one chip as played for the *current* GW.

    The pipeline is the manager: nothing external records what it played, so
    a recommendation that lands on the current GW and clears its trigger is
    the play. Recommendations for a future GW stay advisory and are never
    recorded — they are still free to change next run.

    Returns the token committed, or None. The pipeline runs twice daily; a
    later run in the same GW revises its own not-yet-locked call rather than
    stacking a second chip on top of it.
    """
    cands: list[tuple[str, float]] = []

    # WC and FH are structural: WC fires when the squad is far from optimal,
    # FH only on a blank week. Both outrank the points-uplift chips.
    if wc.get("recommend"):
        cands.append(("wc", float("inf")))
    if fh.get("gw") == gw and fh.get("blanks", 0) >= FH_TRIGGER_BLANKS:
        cands.append(("fh", float(fh["blanks"]) * 1e6))

    if bb.get("gw") == gw and float(bb.get("bonus", 0.0)) >= BB_TRIGGER_BENCH_EV:
        cands.append(("bb", float(bb["bonus"])))

    # TC premium = captain's q90 minus its mean, matching season_replay.
    if tc.get("gw") == gw and tc.get("player_id") is not None:
        row = proj[proj["id"] == tc["player_id"]]
        cap_col, mu_col = f"cap_xp_{gw}", f"xp_{gw}"
        if not row.empty and cap_col in row.columns and mu_col in row.columns:
            premium = float(row[cap_col].iloc[0]) - float(row[mu_col].iloc[0])
            if premium >= TC_TRIGGER_PREMIUM:
                cands.append(("tc", premium))

    token = None
    if cands:
        token = chip_token(max(cands, key=lambda c: c[1])[0], gw)
        used[token] = gw

    # Covers a withdrawal too: if withdraw_pending dropped this GW's call and
    # nothing replaced it, `used` still differs from disk and must be written.
    if used != load_chip_state(path):
        save_chip_state(used, path)
    return token
