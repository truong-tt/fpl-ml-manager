"""FPL data loader. Multi-season ingest from FPL-Core-Insights + live FPL API price overlay.

Current season: By-Gameweek slicing. Historical seasons: entity-folder layout
(2024-2025+). Per-GW deltas reconstructed from cumulative playerstats.csv plus
per-match playermatchstats.csv joined to matches.csv for GW attribution.

Cross-season joins stable on FPL player_code + team.code. Players/teams absent
from current season pool (left PL, relegated) dropped from history.
"""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# https://github.com/olbauday/FPL-Core-Insights. Refresh 2x/day, 05:00 / 17:00 UTC.
FPL_CI_REF = "main"
FPL_CI_BASE = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/{FPL_CI_REF}/data"
SEASON = "2025-2026"
HISTORICAL_SEASONS: list[str] = ["2024-2025"]
ALL_SEASONS = HISTORICAL_SEASONS + [SEASON]

FPL_API_BASE = "https://fantasy.premierleague.com/api/"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / ".fpl_ci_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HIST_NUM = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "bps", "ict_index", "saves",
    "clearances_blocks_interceptions", "tackles", "recoveries",
    "minutes", "goals_scored", "goals_conceded", "total_points",
]

# Excludes player-meta in players.csv. Merging both in features.py would collide
# on penalties_order, status, etc.
HIST_OUTPUT_COLS = [
    "player_id", "round", "season", "fixture", "opponent_team", "team",
    "minutes", "goals_scored", "goals_conceded", "assists",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "bps", "ict_index", "saves",
    "clearances_blocks_interceptions", "tackles", "recoveries",
    "total_points", "bonus", "clean_sheets",
    "yellow_cards", "red_cards", "own_goals",
    "penalties_saved", "penalties_missed",
    "defensive_contribution", "starts",
    "pm_xg", "pm_xa", "pm_cc", "pm_tob", "pm_shots", "pm_drib",
]

OPTA_PM_COLS = {
    "xg": "pm_xg",
    "xa": "pm_xa",
    "chances_created": "pm_cc",
    "touches_opposition_box": "pm_tob",
    "total_shots": "pm_shots",
    "successful_dribbles": "pm_drib",
}

# Cup tournaments — kept separate from PL fixtures. Used to derive congestion
# features (rotation risk) for minutes head. Folder names per FPL-Core-Insights.
CUP_TOURNAMENTS = [
    "EFL Cup", "Champions League", "Europa League", "Conference League",
]

# FPL-CI uses position strings. FPL API contract uses element_type 1..4.
POSITION_TO_ELEMENT_TYPE = {
    "Goalkeeper": 1, "GKP": 1, "GK": 1,
    "Defender": 2, "DEF": 2,
    "Midfielder": 3, "MID": 3,
    "Forward": 4, "FWD": 4,
}

# Cumulative cols in 2024-25 playerstats.csv. Diff for per-GW deltas.
HIST_CUMULATIVE_COLS = {
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "bps": "bps",
    "ict_index": "ict_index",
    "bonus": "bonus",
}


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce cols → float. Fill NaN with 0."""
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    return df


def _fetch_csv(rel_path: str, cache: bool = True, retries: int = 3) -> Optional[pd.DataFrame]:
    """GET <FPL_CI_BASE>/<rel_path>. Cache under data/.fpl_ci_cache."""
    cache_path = CACHE_DIR / rel_path
    if cache and cache_path.exists():
        return pd.read_csv(cache_path)
    url = f"{FPL_CI_BASE}/{quote(rel_path)}"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
            return df
        except requests.RequestException:
            time.sleep(1.5 * (i + 1))
    return None


def _fetch_gw_csv(gw: int, filename: str, cache_history: bool) -> Optional[pd.DataFrame]:
    """Per-GW fetch. cache_history=False forces re-fetch for current/future GWs."""
    rel = f"{SEASON}/By Gameweek/GW{gw}/{filename}"
    if not cache_history:
        cache_path = CACHE_DIR / rel
        if cache_path.exists():
            cache_path.unlink()
    return _fetch_csv(rel, cache=cache_history)


def _discover_gw_bounds() -> tuple[int, int]:
    """Return (current_gw, max_gw). Read gameweek_summaries. Probe if absent."""
    summary = _fetch_csv(f"{SEASON}/gameweek_summaries.csv", cache=False)
    if summary is not None and "is_current" in summary.columns:
        cur_rows = summary[summary["is_current"].astype(str) == "True"]
        cur = int(cur_rows["id"].iloc[0]) if not cur_rows.empty else 1
        max_gw = int(summary["id"].max())
        return cur, max_gw
    cur = 1
    for gw in range(1, 39):
        if _fetch_gw_csv(gw, "fixtures.csv", cache_history=True) is None:
            break
        cur = gw
    return cur, cur


def _build_teams() -> pd.DataFrame:
    """Rename FPL-CI teams.csv → FPL-API col contract. Current season only."""
    src = _fetch_csv(f"{SEASON}/teams.csv", cache=False)
    if src is None or src.empty:
        raise RuntimeError("FPL-CI teams.csv unavailable")
    out = src.rename(columns={"id": "team_id", "name": "team_name"})
    required = ["team_id", "team_name", "short_name", "strength",
                "strength_overall_home", "strength_overall_away"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"teams.csv missing required cols: {missing}")
    return out


def _normalize_fixture_columns(raw: pd.DataFrame, code_to_id: dict[int, int]) -> pd.DataFrame:
    """Rename FPL-CI matches/fixtures cols → fixtures.csv schema. Remap team codes → ids."""
    raw = raw.dropna(subset=["match_id", "gameweek", "home_team", "away_team"]).copy()
    raw = raw.drop_duplicates(subset=["match_id"], keep="last")
    raw["gameweek"] = raw["gameweek"].astype(float).astype(int)
    raw["home_team"] = raw["home_team"].astype(float).astype(int).map(code_to_id)
    raw["away_team"] = raw["away_team"].astype(float).astype(int).map(code_to_id)
    raw = raw.dropna(subset=["home_team", "away_team"]).copy()
    raw["home_team"] = raw["home_team"].astype(int)
    raw["away_team"] = raw["away_team"].astype(int)
    raw["finished"] = raw["finished"].astype(str).str.lower() == "true"
    return raw


def _build_fixtures_current(
    current_gw: int, max_gw: int, code_to_id: dict[int, int]
) -> pd.DataFrame:
    """Concat current-season per-GW fixtures into one frame. Filter to PL."""
    parts: list[pd.DataFrame] = []
    for gw in range(1, max_gw + 1):
        df = _fetch_gw_csv(gw, "fixtures.csv", cache_history=(gw < current_gw))
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        raise RuntimeError("no fixtures found for current season")
    raw = pd.concat(parts, ignore_index=True)
    if "tournament" in raw.columns:
        raw = raw[raw["tournament"].astype(str).str.lower() == "prem"].copy()
    raw = _normalize_fixture_columns(raw, code_to_id)
    raw["season"] = SEASON
    return raw


def _build_fixtures_historical(
    season: str, code_to_id: dict[int, int]
) -> pd.DataFrame:
    """Load <season>/matches/matches.csv. Remap team codes → current team_ids. Drop non-PL."""
    raw = _fetch_csv(f"{season}/matches/matches.csv", cache=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if "tournament" in raw.columns:
        raw = raw[raw["tournament"].astype(str).str.lower() == "prem"].copy()
    raw = _normalize_fixture_columns(raw, code_to_id)
    raw["season"] = season
    return raw


def _assign_global_fixture_ids(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort by (season, kickoff_time, match_id). Globally unique `id`. Return lookup."""
    fx = fixtures.sort_values(["season", "kickoff_time", "match_id"]).reset_index(drop=True)
    fx["id"] = range(1, len(fx) + 1)
    rename = {
        "gameweek": "event",
        "home_team": "team_h",
        "away_team": "team_a",
        "home_score": "team_h_score",
        "away_score": "team_a_score",
    }
    fx = fx.rename(columns=rename)
    keep_first = ["id", "season", "event", "finished", "kickoff_time",
                  "team_h", "team_a", "team_h_score", "team_a_score", "match_id"]
    extras = [c for c in fx.columns if c not in keep_first
              and (c.startswith("home_") or c.startswith("away_")
                   or c in ("home_team_elo", "away_team_elo"))]
    fx = fx[keep_first + extras]

    lookup: dict[tuple[int, int, str], int] = {}
    for fid, season, team_h, team_a, event in zip(
        fx["id"], fx["season"], fx["team_h"], fx["team_a"], fx["event"]
    ):
        lookup.setdefault((int(team_h), int(event), str(season)), int(fid))
        lookup.setdefault((int(team_a), int(event), str(season)), int(fid))
    return fx, lookup


def _build_players(teams: pd.DataFrame, current_gw: int) -> pd.DataFrame:
    """Join current-season players.csv + latest playerstats.csv. Recreate FPL element shape."""
    base = _fetch_csv(f"{SEASON}/players.csv", cache=False)
    if base is None or base.empty:
        raise RuntimeError("FPL-CI players.csv unavailable")

    stats = _fetch_gw_csv(current_gw, "playerstats.csv", cache_history=False)
    if stats is None or stats.empty:
        raise RuntimeError(f"playerstats.csv unavailable for GW{current_gw}")

    pos = base["position"].map(POSITION_TO_ELEMENT_TYPE)
    if pos.isna().any():
        unknown = base.loc[pos.isna(), "position"].unique().tolist()
        raise RuntimeError(f"unknown position values in players.csv: {unknown}")

    code_to_id = dict(zip(teams["code"].astype(int), teams["team_id"].astype(int)))
    team_ids = base["team_code"].astype(int).map(code_to_id)
    if team_ids.isna().any():
        bad = base.loc[team_ids.isna(), "team_code"].unique().tolist()
        raise RuntimeError(f"team_code(s) not in teams.csv: {bad}")

    players = pd.DataFrame({
        "id": base["player_id"].astype(int),
        "code": base["player_code"].astype(int),
        "web_name": base["web_name"],
        "first_name": base["first_name"],
        "second_name": base["second_name"],
        "element_type": pos.astype(int),
        "team": team_ids.astype(int),
        "team_code": base["team_code"].astype(int),
    })

    meta_cols = [
        "now_cost", "selected_by_percent", "status",
        "chance_of_playing_next_round", "chance_of_playing_this_round",
        "penalties_order", "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
        "minutes", "total_points", "form", "ep_next", "ep_this",
        "transfers_in_event", "transfers_out_event",
        "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "ict_index", "bps", "bonus",
    ]
    avail = [c for c in meta_cols if c in stats.columns]
    meta = stats[["id"] + avail].copy()
    players = players.merge(meta, on="id", how="left")
    players = _num(players, [
        "now_cost", "selected_by_percent",
        "penalties_order", "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
        "minutes", "total_points",
    ])
    return players


def _overlay_live_fpl_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Override `gameweek` for unfinished current-season fixtures from live FPL API.
    Source of truth for postponements / DGW reschedules that lag in FPL-CI snapshot.
    Silent fallback on network failure."""
    if fixtures.empty or "finished" not in fixtures.columns:
        return fixtures
    try:
        r = requests.get(f"{FPL_API_BASE}fixtures/", timeout=10)
        r.raise_for_status()
        live = pd.DataFrame(r.json())
    except Exception:
        return fixtures
    if live.empty or not {"event", "team_h", "team_a"}.issubset(live.columns):
        return fixtures
    live = live.dropna(subset=["event", "team_h", "team_a"]).copy()
    live["event"] = live["event"].astype(int)
    live["team_h"] = live["team_h"].astype(int)
    live["team_a"] = live["team_a"].astype(int)
    key_to_event = {(int(row.team_h), int(row.team_a)): int(row.event)
                    for row in live.itertuples(index=False)}
    out = fixtures.copy()
    mask = (~out["finished"].astype(bool)) & \
           out["home_team"].notna() & out["away_team"].notna()
    if not mask.any():
        return out
    new_gw = out.loc[mask].apply(
        lambda r: key_to_event.get((int(r["home_team"]), int(r["away_team"]))),
        axis=1,
    )
    valid = new_gw.notna()
    if valid.any():
        idx = new_gw.index[valid]
        out.loc[idx, "gameweek"] = new_gw.loc[idx].astype(int).values
    return out


def _overlay_live_fpl_api(players: pd.DataFrame) -> pd.DataFrame:
    """Best-effort price/status refresh from live FPL API. Silent fallback on failure."""
    try:
        r = requests.get(f"{FPL_API_BASE}bootstrap-static/", timeout=10)
        r.raise_for_status()
        live = pd.DataFrame(r.json()["elements"])
    except Exception:
        return players
    cols = ["id", "now_cost", "selected_by_percent", "status",
            "chance_of_playing_next_round"]
    avail = [c for c in cols if c in live.columns]
    if "id" not in avail or len(avail) == 1:
        return players
    overlay = live[avail].set_index("id")
    out = players.set_index("id")
    for c in avail[1:]:
        live_col = overlay[c].reindex(out.index)
        if c in out.columns:
            out[c] = live_col.where(live_col.notna(), out[c])
        else:
            out[c] = live_col
    return out.reset_index()


def _build_opta_per_gw_current(current_gw: int) -> pd.DataFrame:
    """Aggregate current-season per-match Opta → per-(player, round) sums. DGWs add."""
    parts: list[pd.DataFrame] = []
    src_cols = list(OPTA_PM_COLS.keys())
    for gw in range(1, current_gw + 1):
        df = _fetch_gw_csv(gw, "playermatchstats.csv",
                           cache_history=(gw < current_gw))
        if df is None or df.empty:
            continue
        avail = [c for c in src_cols if c in df.columns]
        if "player_id" not in df.columns or not avail:
            continue
        d = df[["player_id"] + avail].copy()
        d["round"] = gw
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["player_id", "round"] + list(OPTA_PM_COLS.values()))
    raw = pd.concat(parts, ignore_index=True)
    for c in OPTA_PM_COLS:
        if c not in raw.columns:
            raw[c] = 0.0
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    agg = raw.groupby(["player_id", "round"], as_index=False)[list(OPTA_PM_COLS.keys())].sum()
    agg = agg.rename(columns=OPTA_PM_COLS)
    return agg


def _build_cup_fixtures(
    current_gw: int, max_gw: int, code_to_id: dict[int, int]
) -> pd.DataFrame:
    """Pull EFL Cup / UCL / UEL / UECL fixtures for current season.

    Per memory `dataset_fpl_core_insights.md`: each cup folder
    `By Tournament/<cup>/GW{n}/` contains fixtures.csv (upcoming) + matches.csv
    (finished). GW{n} keys to the PL gameweek window — aligned with our `event`.
    Non-PL tournaments only include English clubs' matches; opponent may be a
    foreign club not in our teams.csv (drop in remap; keep raw code only).

    Output one row per (English team, cup match): season, event, kickoff_time,
    tournament, team_id, opponent_code, match_id. Future-window cup matches
    (not yet played) come from fixtures.csv with finished=False.
    """
    rows: list[dict] = []
    for tour in CUP_TOURNAMENTS:
        for gw in range(1, max_gw + 1):
            for fname in ("matches.csv", "fixtures.csv"):
                rel = f"{SEASON}/By Tournament/{tour}/GW{gw}/{fname}"
                df = _fetch_csv(rel, cache=(gw < current_gw))
                if df is None or df.empty:
                    continue
                req = {"kickoff_time", "home_team", "away_team", "match_id"}
                if not req.issubset(df.columns):
                    continue
                for _, r in df.iterrows():
                    try:
                        hcode = int(float(r["home_team"]))
                        acode = int(float(r["away_team"]))
                    except (ValueError, TypeError):
                        continue
                    for team_code, opp_code in ((hcode, acode), (acode, hcode)):
                        team_id = code_to_id.get(team_code)
                        if team_id is None:
                            continue  # foreign club — skip; keeps row tied to English side only
                        rows.append({
                            "season": SEASON,
                            "event": int(gw),
                            "kickoff_time": r["kickoff_time"],
                            "tournament": tour,
                            "team_id": int(team_id),
                            "opponent_code": int(opp_code),
                            "match_id": r["match_id"],
                        })
    cols = ["season", "event", "kickoff_time", "tournament",
            "team_id", "opponent_code", "match_id"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)[cols]
    # matches.csv + fixtures.csv may overlap (same fixture, after the match is
    # played both can be present). Keep first (matches.csv → finished kickoff).
    return out.drop_duplicates(subset=["match_id", "team_id"], keep="first")


def _build_history_current(
    current_gw: int,
    players: pd.DataFrame,
    fixture_lookup: dict[tuple[int, int, str], int],
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Concat current-season per-GW player_gameweek_stats.csv. Tag season."""
    parts: list[pd.DataFrame] = []
    for gw in range(1, current_gw + 1):
        df = _fetch_gw_csv(gw, "player_gameweek_stats.csv",
                           cache_history=(gw < current_gw))
        if df is not None and not df.empty:
            d = df.copy()
            d["round"] = gw
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=HIST_OUTPUT_COLS)
    hist = pd.concat(parts, ignore_index=True)
    hist = hist.rename(columns={"id": "player_id"})

    pteam = players.set_index("id")["team"].to_dict()
    team_series = hist["player_id"].map(pteam)
    hist = hist.loc[team_series.notna()].copy()
    hist["team"] = team_series.loc[hist.index].astype(int)
    hist["season"] = SEASON

    hist["fixture"] = [
        fixture_lookup.get((int(t), int(r), SEASON), 0)
        for t, r in zip(hist["team"], hist["round"])
    ]
    fix_idx = fixtures.set_index("id")
    hist["opponent_team"] = [
        (int(fix_idx.at[f, "team_a"]) if f in fix_idx.index and int(fix_idx.at[f, "team_h"]) == int(t)
         else (int(fix_idx.at[f, "team_h"]) if f in fix_idx.index else 0))
        for f, t in zip(hist["fixture"], hist["team"])
    ]

    opta = _build_opta_per_gw_current(current_gw)
    if not opta.empty:
        hist = hist.merge(opta, on=["player_id", "round"], how="left")
        for c in OPTA_PM_COLS.values():
            if c in hist.columns:
                hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)

    hist = _num(hist, HIST_NUM)
    cols = [c for c in HIST_OUTPUT_COLS if c in hist.columns]
    return hist[cols]


def _build_history_historical(
    season: str,
    players_current: pd.DataFrame,
    teams_current: pd.DataFrame,
    fixture_lookup: dict[tuple[int, int, str], int],
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct per-(player, GW) history for finished season.

    Strategy:
    1. Load historical players.csv. Map season-local player_id → player_code, team_code.
    2. Diff cumulative cols in playerstats.csv for per-GW deltas. event_points = target.
    3. Aggregate playermatchstats.csv per (player_id, gw) via match_id → matches.csv join.
    4. Remap player_id (season-local) → current player_id via player_code.
    5. Remap team_id (season-local) → current team_id via team_code.
    6. Drop rows for players/teams not in current pool (left PL, relegated).
    """
    base = _fetch_csv(f"{season}/players/players.csv", cache=True)
    pstats = _fetch_csv(f"{season}/playerstats/playerstats.csv", cache=True)
    pms = _fetch_csv(f"{season}/playermatchstats/playermatchstats.csv", cache=True)
    matches = _fetch_csv(f"{season}/matches/matches.csv", cache=True)
    if any(d is None or d.empty for d in (base, pstats, pms, matches)):
        return pd.DataFrame(columns=HIST_OUTPUT_COLS)

    # Map season-local player_id → current player_id via player_code.
    code_to_current_pid = dict(zip(
        players_current["code"].astype(int), players_current["id"].astype(int)
    ))
    pid_local_to_code = dict(zip(
        base["player_id"].astype(int), base["player_code"].astype(int)
    ))
    pid_local_to_team_code = dict(zip(
        base["player_id"].astype(int), base["team_code"].astype(int)
    ))

    # Map team_code → current team_id.
    code_to_current_tid = dict(zip(
        teams_current["code"].astype(int), teams_current["team_id"].astype(int)
    ))

    # Step 1. Per-GW deltas from cumulative playerstats.csv.
    ps = pstats[["id", "gw", "event_points"] + list(HIST_CUMULATIVE_COLS.keys()) + [
        "transfers_in_event", "transfers_out_event"
    ]].copy()
    ps["gw"] = pd.to_numeric(ps["gw"], errors="coerce").astype("Int64")
    ps = ps.dropna(subset=["gw"]).copy()
    ps["gw"] = ps["gw"].astype(int)
    ps = ps.sort_values(["id", "gw"]).reset_index(drop=True)
    for raw, _ in HIST_CUMULATIVE_COLS.items():
        ps[raw] = pd.to_numeric(ps[raw], errors="coerce").fillna(0.0)
        delta = ps.groupby("id")[raw].diff().fillna(ps[raw])
        ps[raw] = delta.clip(lower=0.0)
    ps = ps.rename(columns={"id": "player_id_local", "gw": "round",
                            "event_points": "total_points"})
    ps["total_points"] = pd.to_numeric(ps["total_points"], errors="coerce").fillna(0.0)

    # Step 2. Aggregate playermatchstats per (player_id, gw) via match_id → matches gameweek.
    m_gw = matches[["match_id", "gameweek"]].copy()
    m_gw["gameweek"] = pd.to_numeric(m_gw["gameweek"], errors="coerce").astype("Int64")
    m_gw = m_gw.dropna(subset=["gameweek"])
    m_gw["gameweek"] = m_gw["gameweek"].astype(int)
    pm = pms.merge(m_gw, on="match_id", how="inner")
    pm = pm.rename(columns={"player_id": "player_id_local", "gameweek": "round"})

    # Per-match → per-(player, GW) sum aggregations.
    agg_cols: dict[str, str] = {
        "minutes_played": "minutes",
        "goals": "goals_scored",
        "assists": "assists",
        "goals_conceded": "goals_conceded",
        "saves": "saves",
        "tackles_won": "tackles",
        "recoveries": "recoveries",
        "interceptions": "interceptions_raw",
        "blocks": "blocks_raw",
        "clearances": "clearances_raw",
        "penalties_missed": "penalties_missed",
    }
    for src in agg_cols:
        if src not in pm.columns:
            pm[src] = 0.0
        pm[src] = pd.to_numeric(pm[src], errors="coerce").fillna(0.0)
    # Opta per-match → pm_* aggregated.
    for src in OPTA_PM_COLS:
        if src not in pm.columns:
            pm[src] = 0.0
        pm[src] = pd.to_numeric(pm[src], errors="coerce").fillna(0.0)
    # start_min → starts indicator. 1 if start_min ≤ 1 (on pitch from kickoff).
    if "start_min" in pm.columns:
        pm["starts"] = (pd.to_numeric(pm["start_min"], errors="coerce").fillna(99) <= 1).astype(int)
    else:
        pm["starts"] = 0

    sum_src = list(agg_cols.keys()) + list(OPTA_PM_COLS.keys()) + ["starts"]
    pm_agg = pm.groupby(["player_id_local", "round"], as_index=False)[sum_src].sum()
    pm_agg = pm_agg.rename(columns=agg_cols)
    pm_agg = pm_agg.rename(columns=OPTA_PM_COLS)
    # CBI = interceptions + blocks + clearances. clean_sheets = goals_conceded==0 ∧ minutes≥60.
    pm_agg["clearances_blocks_interceptions"] = (
        pm_agg["interceptions_raw"] + pm_agg["blocks_raw"] + pm_agg["clearances_raw"]
    )
    pm_agg["clean_sheets"] = (
        (pm_agg["goals_conceded"] == 0) & (pm_agg["minutes"] >= 60)
    ).astype(int)
    pm_agg = pm_agg.drop(columns=["interceptions_raw", "blocks_raw", "clearances_raw"])

    # Step 3. Merge playerstats deltas + playermatchstats aggregates.
    # Outer merge. Player with playerstats row but no playermatchstats row (DNP)
    # gets NaN for per-match agg cols. Fill 0 so trainer doesn't drop them.
    hist = pm_agg.merge(ps, on=["player_id_local", "round"], how="outer")
    pm_filled = (list(agg_cols.values()) + list(OPTA_PM_COLS.values())
                 + ["clearances_blocks_interceptions", "clean_sheets", "starts"])
    for c in pm_filled:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)
    for raw in HIST_CUMULATIVE_COLS:
        if raw in hist.columns:
            hist[raw] = hist[raw].fillna(0.0)
    for c in ("total_points", "transfers_in_event", "transfers_out_event"):
        if c in hist.columns:
            hist[c] = hist[c].fillna(0.0)

    # Step 4. Remap player_id_local → current player_id. Drop unmappable.
    hist["player_code"] = hist["player_id_local"].astype(int).map(pid_local_to_code)
    hist["player_id"] = hist["player_code"].map(code_to_current_pid)
    hist = hist.dropna(subset=["player_id"]).copy()
    hist["player_id"] = hist["player_id"].astype(int)

    # Step 5. Remap team via player terminal team_code from historical players.csv.
    hist["team_code_hist"] = hist["player_id_local"].astype(int).map(pid_local_to_team_code)
    hist["team"] = hist["team_code_hist"].map(code_to_current_tid)
    hist = hist.dropna(subset=["team"]).copy()
    hist["team"] = hist["team"].astype(int)

    # Step 6. Fixture id + opponent_team via global lookup.
    hist["season"] = season
    hist["fixture"] = [
        fixture_lookup.get((int(t), int(r), season), 0)
        for t, r in zip(hist["team"], hist["round"])
    ]
    fix_idx = fixtures.set_index("id")
    hist["opponent_team"] = [
        (int(fix_idx.at[f, "team_a"]) if f in fix_idx.index and int(fix_idx.at[f, "team_h"]) == int(t)
         else (int(fix_idx.at[f, "team_h"]) if f in fix_idx.index else 0))
        for f, t in zip(hist["fixture"], hist["team"])
    ]

    # Fields absent in 2024-25 player-level data. Fill 0 to match output schema.
    for c in ("bonus", "yellow_cards", "red_cards", "own_goals",
              "penalties_saved", "defensive_contribution"):
        if c not in hist.columns:
            hist[c] = 0.0

    hist = _num(hist, HIST_NUM)
    cols = [c for c in HIST_OUTPUT_COLS if c in hist.columns]
    return hist[cols]


def main() -> None:
    """Refresh players / teams / fixtures / history CSVs under data/ across all seasons."""
    current_gw, max_gw = _discover_gw_bounds()
    print(f"[data_loader] current_gw={current_gw} max_gw={max_gw} season={SEASON} "
          f"historical={HISTORICAL_SEASONS}")

    teams = _build_teams()
    code_to_id = dict(zip(teams["code"].astype(int), teams["team_id"].astype(int)))

    # Phase 1. Fixtures across all seasons → global ids, season-aware lookup.
    fx_curr = _build_fixtures_current(current_gw, max_gw, code_to_id)
    fx_curr = _overlay_live_fpl_fixtures(fx_curr)
    fx_parts: list[pd.DataFrame] = [fx_curr]
    for s in HISTORICAL_SEASONS:
        fx_h = _build_fixtures_historical(s, code_to_id)
        if not fx_h.empty:
            fx_parts.append(fx_h)
    fixtures_all = pd.concat(fx_parts, ignore_index=True)
    fixtures, fixture_lookup = _assign_global_fixture_ids(fixtures_all)

    # Phase 2. Current-season players. Join pool for all seasons.
    players = _build_players(teams, current_gw)
    players = _overlay_live_fpl_api(players)

    # Phase 3. History across all seasons.
    h_parts: list[pd.DataFrame] = [
        _build_history_current(current_gw, players, fixture_lookup, fixtures)
    ]
    for s in HISTORICAL_SEASONS:
        h_h = _build_history_historical(s, players, teams, fixture_lookup, fixtures)
        if not h_h.empty:
            h_parts.append(h_h)
    history = pd.concat(h_parts, ignore_index=True)

    teams_out = teams[[
        "team_id", "team_name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
    ] + [c for c in ("strength_attack_home", "strength_attack_away",
                     "strength_defence_home", "strength_defence_away",
                     "elo", "code", "fotmob_name") if c in teams.columns]]

    # Phase 4. Cup fixtures (current season only). Powers minutes-head
    # congestion features (rotation risk for Chelsea / UCL clubs etc.).
    cup_fixtures = _build_cup_fixtures(current_gw, max_gw, code_to_id)

    teams_out.to_csv(DATA_DIR / "teams.csv", index=False)
    players.to_csv(DATA_DIR / "players.csv", index=False)
    fixtures.to_csv(DATA_DIR / "fixtures.csv", index=False)
    history.to_csv(DATA_DIR / "history.csv", index=False)
    cup_fixtures.to_csv(DATA_DIR / "cup_fixtures.csv", index=False)
    season_counts = history.groupby("season").size().to_dict() if not history.empty else {}
    cup_counts = (cup_fixtures.groupby("tournament").size().to_dict()
                  if not cup_fixtures.empty else {})
    print(f"[data_loader] wrote teams={len(teams_out)} players={len(players)} "
          f"fixtures={len(fixtures)} history={len(history)} per-season={season_counts} "
          f"cup_fixtures={len(cup_fixtures)} per-cup={cup_counts}")


if __name__ == "__main__":
    main()
