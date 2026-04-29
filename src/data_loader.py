"""FPL data loader sourced from FPL-Core-Insights CSVs, with a thin live-API price overlay."""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

# https://github.com/olbauday/FPL-Core-Insights — refreshed 2x/day, 05:00 / 17:00 UTC.
FPL_CI_REF = "main"
FPL_CI_BASE = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/{FPL_CI_REF}/data"
SEASON = "2025-2026"

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

# Excludes player-meta fields that live in players.csv — merging both in
# features.py would collide on penalties_order, status, etc.
HIST_OUTPUT_COLS = [
    "player_id", "round", "fixture", "opponent_team", "team",
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

# FPL-CI uses position strings; FPL API contract uses element_type 1..4.
POSITION_TO_ELEMENT_TYPE = {
    "Goalkeeper": 1, "GKP": 1, "GK": 1,
    "Defender": 2, "DEF": 2,
    "Midfielder": 3, "MID": 3,
    "Forward": 4, "FWD": 4,
}


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerces cols to float, fills NaN with 0."""
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    return df


def _fetch_csv(rel_path: str, cache: bool = True, retries: int = 3) -> Optional[pd.DataFrame]:
    """GETs <FPL_CI_BASE>/<rel_path> with retry; caches under data/.fpl_ci_cache."""
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
    """Per-GW fetch; cache_history=False forces a re-fetch for current/future GWs."""
    rel = f"{SEASON}/By Gameweek/GW{gw}/{filename}"
    if not cache_history:
        cache_path = CACHE_DIR / rel
        if cache_path.exists():
            cache_path.unlink()
    return _fetch_csv(rel, cache=cache_history)


def _discover_gw_bounds() -> tuple[int, int]:
    """Returns (current_gw, max_gw); reads gameweek_summaries, probes if absent."""
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
    """Renames FPL-CI teams.csv to the FPL-API column contract."""
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


def _build_fixtures(current_gw: int, max_gw: int, teams: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Concats PL fixtures, synthesizes numeric `id`; returns (df, (team,gw)->id lookup)."""
    parts: list[pd.DataFrame] = []
    for gw in range(1, max_gw + 1):
        df = _fetch_gw_csv(gw, "fixtures.csv", cache_history=(gw < current_gw))
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        raise RuntimeError("no fixtures found in FPL-CI dataset")
    raw = pd.concat(parts, ignore_index=True)

    if "tournament" in raw.columns:
        raw = raw[raw["tournament"].astype(str).str.lower() == "prem"].copy()
    raw = raw.dropna(subset=["match_id", "gameweek", "home_team", "away_team"])
    raw = raw.drop_duplicates(subset=["match_id"], keep="last")

    # FPL-CI fixtures encode home/away_team as team `code`, not `id`.
    code_to_id = dict(zip(teams["code"].astype(int), teams["team_id"].astype(int)))
    raw["gameweek"] = raw["gameweek"].astype(float).astype(int)
    raw["home_team"] = raw["home_team"].astype(float).astype(int).map(code_to_id)
    raw["away_team"] = raw["away_team"].astype(float).astype(int).map(code_to_id)
    if raw["home_team"].isna().any() or raw["away_team"].isna().any():
        raise RuntimeError("fixtures contain team codes not present in teams.csv")
    raw["home_team"] = raw["home_team"].astype(int)
    raw["away_team"] = raw["away_team"].astype(int)
    raw["finished"] = raw["finished"].astype(str).str.lower() == "true"
    # Deterministic synthetic id: (kickoff_time, match_id) is stable across runs.
    raw = raw.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    raw["id"] = range(1, len(raw) + 1)

    out = raw.rename(columns={
        "gameweek": "event",
        "home_team": "team_h",
        "away_team": "team_a",
        "home_score": "team_h_score",
        "away_score": "team_a_score",
    })
    keep_first = ["id", "event", "finished", "kickoff_time",
                  "team_h", "team_a", "team_h_score", "team_a_score", "match_id"]
    extras = [c for c in out.columns if c not in keep_first
              and (c.startswith("home_") or c.startswith("away_")
                   or c in ("home_team_elo", "away_team_elo"))]
    out = out[keep_first + extras]

    home_lookup = (out.sort_values("kickoff_time")
                      .groupby(["team_h", "event"])["id"].first())
    away_lookup = (out.sort_values("kickoff_time")
                      .groupby(["team_a", "event"])["id"].first())
    lookup: dict[tuple[int, int], int] = {}
    for (team, event), fid in home_lookup.items():
        lookup[(int(team), int(event))] = int(fid)
    for (team, event), fid in away_lookup.items():
        lookup.setdefault((int(team), int(event)), int(fid))
    return out, lookup


def _build_players(teams: pd.DataFrame, current_gw: int) -> pd.DataFrame:
    """Joins players.csv with latest playerstats.csv to recreate the FPL element shape."""
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


def _overlay_live_fpl_api(players: pd.DataFrame) -> pd.DataFrame:
    """Best-effort price/status refresh from the live FPL API; silent fallback on failure."""
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


def _build_opta_per_gw(current_gw: int) -> pd.DataFrame:
    """Aggregates per-match Opta into per-(player, round) sums (DGWs add across matches)."""
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


def _build_history(
    current_gw: int,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    fixture_lookup: dict[tuple[int, int], int],
) -> pd.DataFrame:
    """Concats per-GW player stats and derives team / opponent_team / fixture columns."""
    parts: list[pd.DataFrame] = []
    for gw in range(1, current_gw + 1):
        df = _fetch_gw_csv(gw, "player_gameweek_stats.csv",
                           cache_history=(gw < current_gw))
        if df is not None and not df.empty:
            d = df.copy()
            d["round"] = gw
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    hist = pd.concat(parts, ignore_index=True)
    hist = hist.rename(columns={"id": "player_id"})

    pteam = players.set_index("id")["team"].to_dict()
    team_series = hist["player_id"].map(pteam)
    hist = hist.loc[team_series.notna()].copy()
    hist["team"] = team_series.loc[hist.index].astype(int)

    hist["fixture"] = [
        fixture_lookup.get((int(t), int(r)), 0)
        for t, r in zip(hist["team"], hist["round"])
    ]
    fix_idx = fixtures.set_index("id")
    hist["opponent_team"] = [
        (int(fix_idx.at[f, "team_a"]) if f in fix_idx.index and int(fix_idx.at[f, "team_h"]) == int(t)
         else (int(fix_idx.at[f, "team_h"]) if f in fix_idx.index else 0))
        for f, t in zip(hist["fixture"], hist["team"])
    ]

    opta = _build_opta_per_gw(current_gw)
    if not opta.empty:
        hist = hist.merge(opta, on=["player_id", "round"], how="left")
        for c in OPTA_PM_COLS.values():
            if c in hist.columns:
                hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)

    hist = _num(hist, HIST_NUM)
    cols = [c for c in HIST_OUTPUT_COLS if c in hist.columns]
    return hist[cols]


def main() -> None:
    """Refreshes players / teams / fixtures / history CSVs under data/."""
    current_gw, max_gw = _discover_gw_bounds()
    print(f"[data_loader] current_gw={current_gw} max_gw={max_gw} season={SEASON}")

    teams = _build_teams()
    fixtures, fixture_lookup = _build_fixtures(current_gw, max_gw, teams)
    players = _build_players(teams, current_gw)
    players = _overlay_live_fpl_api(players)
    history = _build_history(current_gw, players, fixtures, fixture_lookup)

    teams_out = teams[[
        "team_id", "team_name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
    ] + [c for c in ("strength_attack_home", "strength_attack_away",
                     "strength_defence_home", "strength_defence_away",
                     "elo", "code", "fotmob_name") if c in teams.columns]]

    teams_out.to_csv(DATA_DIR / "teams.csv", index=False)
    players.to_csv(DATA_DIR / "players.csv", index=False)
    fixtures.to_csv(DATA_DIR / "fixtures.csv", index=False)
    history.to_csv(DATA_DIR / "history.csv", index=False)
    print(f"[data_loader] wrote teams={len(teams_out)} players={len(players)} "
          f"fixtures={len(fixtures)} history={len(history)}")


if __name__ == "__main__":
    main()
