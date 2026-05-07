"""Rolling team / player features. Elo comes from ClubElo (FPL-CI); replay is the fallback.

All rolling state is partitioned by `season` to prevent cross-season leakage — a player's
or team's GW1 row in season N never sees data from season N-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_WINDOWS = [3, 5, 10]
# Used only when FPL-CI home_team_elo/away_team_elo are absent or null.
INIT_ELO, K, HFA = 1500.0, 20.0, 60.0


def _ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    """Backfills `season` col with a single-season default for callers passing legacy frames."""
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = "current"
    return df


def _elo_replay(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Chronological Elo replay over finished fixtures (fallback path). Resets per season."""
    sort_col = "kickoff_time" if "kickoff_time" in fixtures.columns else "event"
    out = fixtures.sort_values(["season", sort_col]).copy().reset_index(drop=True)
    out["elo_h_pre"] = 0.0
    out["elo_a_pre"] = 0.0
    ratings: dict[tuple, float] = {}
    for i, f in out.iterrows():
        season = f["season"]
        h, a = int(f["team_h"]), int(f["team_a"])
        rh = ratings.get((season, h), INIT_ELO)
        ra = ratings.get((season, a), INIT_ELO)
        out.at[i, "elo_h_pre"], out.at[i, "elo_a_pre"] = rh, ra
        if bool(f.get("finished")) and pd.notna(f.get("team_h_score")) and pd.notna(f.get("team_a_score")):
            gh, ga = int(f["team_h_score"]), int(f["team_a_score"])
            eh = 1.0 / (1.0 + 10 ** (-(rh + HFA - ra) / 400.0))
            sh = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            mov = (abs(gh - ga) + 1) ** 0.4
            ratings[(season, h)] = rh + K * mov * (sh - eh)
            ratings[(season, a)] = ra + K * mov * ((1 - sh) - (1 - eh))
    return out


def elo_snapshot_series(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Stamps pre-match Elo (ClubElo if present on fixtures.csv, replay otherwise)."""
    fixtures = _ensure_season(fixtures)
    has_elo = "home_team_elo" in fixtures.columns and "away_team_elo" in fixtures.columns
    if not has_elo:
        return _elo_replay(fixtures, teams)

    sort_col = "kickoff_time" if "kickoff_time" in fixtures.columns else "event"
    out = fixtures.sort_values(["season", sort_col]).copy().reset_index(drop=True)
    out["elo_h_pre"] = pd.to_numeric(out["home_team_elo"], errors="coerce")
    out["elo_a_pre"] = pd.to_numeric(out["away_team_elo"], errors="coerce")
    if out[["elo_h_pre", "elo_a_pre"]].isna().any().any():
        replayed = _elo_replay(fixtures, teams).set_index("id")[["elo_h_pre", "elo_a_pre"]]
        idx = out.set_index("id").index
        for col in ("elo_h_pre", "elo_a_pre"):
            mask = out[col].isna()
            out.loc[mask, col] = replayed.loc[idx[mask], col].values
    return out


def _rolling_team_stats(history: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, season, GW) xG/xGA rolled forward at TEAM_WINDOWS horizons."""
    if history.empty or "team" not in history.columns:
        return pd.DataFrame()
    history = _ensure_season(history)
    g = history.groupby(["season", "team", "round"], as_index=False)[
        ["expected_goals", "expected_goals_conceded", "goals_scored", "goals_conceded"]
    ].sum().sort_values(["season", "team", "round"])
    for w in TEAM_WINDOWS:
        for raw, short in [("expected_goals", "xg"), ("expected_goals_conceded", "xga"),
                           ("goals_scored", "gf"), ("goals_conceded", "ga")]:
            g[f"roll_{short}_{w}"] = g.groupby(["season", "team"])[raw].transform(
                lambda x: x.shift().rolling(w, min_periods=1).mean()
            )
    return g


# Match-level Opta from fixtures.csv — true on-the-ball xG, distinct from
# _rolling_team_stats which sums per-player FPL xG.
OPTA_STATS = [
    ("expected_goals_xg", "oxg"),
    ("big_chances", "obc"),
    ("total_shots", "osh"),
]
OPTA_WINDOW = 5
OPTA_DEFAULTS = {"oxg": 1.2, "oxga": 1.2, "obc": 1.5, "obca": 1.5, "osh": 11.0, "osha": 11.0}


def _rolling_fixture_team_stats(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, season, GW) rolling Opta stats; each fixture contributes a row per side."""
    needed = [f"{p}_{r}" for r, _ in OPTA_STATS for p in ("home", "away")]
    if not all(c in fixtures.columns for c in needed):
        return pd.DataFrame()
    fixtures = _ensure_season(fixtures)
    parts = []
    for is_home in (True, False):
        side = "home" if is_home else "away"
        opp = "away" if is_home else "home"
        d = pd.DataFrame({
            "team": fixtures["team_h" if is_home else "team_a"].values,
            "round": fixtures["event"].values,
            "season": fixtures["season"].values,
            "kickoff_time": fixtures["kickoff_time"].values,
        })
        for raw, short in OPTA_STATS:
            d[short] = pd.to_numeric(fixtures[f"{side}_{raw}"], errors="coerce").values
            d[f"{short}a"] = pd.to_numeric(fixtures[f"{opp}_{raw}"], errors="coerce").values
        parts.append(d)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["team"])
    long = long.sort_values(["season", "team", "kickoff_time"]).reset_index(drop=True)
    short_cols = [c for _, c in OPTA_STATS] + [f"{c}a" for _, c in OPTA_STATS]
    for c in short_cols:
        long[f"roll_{c}_{OPTA_WINDOW}"] = long.groupby(["season", "team"])[c].transform(
            lambda x: x.shift().rolling(OPTA_WINDOW, min_periods=1).mean()
        )
    keep = [f"roll_{c}_{OPTA_WINDOW}" for c in short_cols]
    return long.groupby(["season", "team", "round"], as_index=False)[keep].mean()


def build_match_features(
    fixtures: pd.DataFrame, history: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Joins team rolling stats + Elo diff onto every fixture row."""
    fx = elo_snapshot_series(fixtures, teams)
    fx = _ensure_season(fx)
    history = _ensure_season(history)
    tg = _rolling_team_stats(history)
    for side, team_col in (("h", "team_h"), ("a", "team_a")):
        if tg.empty:
            for w in TEAM_WINDOWS:
                for s in ("xg", "xga", "gf", "ga"):
                    fx[f"{side}_{s}_{w}"] = 1.2
            continue
        m = fx.merge(tg, left_on=[team_col, "event", "season"],
                     right_on=["team", "round", "season"], how="left")
        for w in TEAM_WINDOWS:
            for s in ("xg", "xga", "gf", "ga"):
                fx[f"{side}_{s}_{w}"] = m[f"roll_{s}_{w}"].fillna(1.2).values

    og = _rolling_fixture_team_stats(fixtures)
    opta_short = [c for _, c in OPTA_STATS] + [f"{c}a" for _, c in OPTA_STATS]
    for side, team_col in (("h", "team_h"), ("a", "team_a")):
        if og.empty:
            for c in opta_short:
                fx[f"{side}_{c}_{OPTA_WINDOW}"] = OPTA_DEFAULTS.get(c, 1.0)
            continue
        m = fx.merge(og, left_on=[team_col, "event", "season"],
                     right_on=["team", "round", "season"],
                     how="left", suffixes=("", "_o"))
        for c in opta_short:
            fx[f"{side}_{c}_{OPTA_WINDOW}"] = (
                m[f"roll_{c}_{OPTA_WINDOW}"].fillna(OPTA_DEFAULTS.get(c, 1.0)).values
            )

    fx["elo_diff"] = fx["elo_h_pre"] - fx["elo_a_pre"]
    fx["xg_diff_5"] = fx["h_xg_5"] - fx["a_xg_5"]
    fx["xga_diff_5"] = fx["h_xga_5"] - fx["a_xga_5"]
    fx["oxg_diff"] = fx[f"h_oxg_{OPTA_WINDOW}"] - fx[f"a_oxg_{OPTA_WINDOW}"]
    fx["oxga_diff"] = fx[f"h_oxga_{OPTA_WINDOW}"] - fx[f"a_oxga_{OPTA_WINDOW}"]
    return fx


def match_feature_cols() -> list[str]:
    """Canonical feature-column order for the match model."""
    cols = [f"{s}_{stat}_{w}"
            for s in ("h", "a") for w in TEAM_WINDOWS for stat in ("xg", "xga", "gf", "ga")]
    opta_short = [c for _, c in OPTA_STATS] + [f"{c}a" for _, c in OPTA_STATS]
    cols += [f"{s}_{c}_{OPTA_WINDOW}" for s in ("h", "a") for c in opta_short]
    return cols + ["elo_h_pre", "elo_a_pre", "elo_diff",
                   "xg_diff_5", "xga_diff_5", "oxg_diff", "oxga_diff"]


def build_player_features(
    history: pd.DataFrame, players: pd.DataFrame, fixture_feats: pd.DataFrame
) -> pd.DataFrame:
    """Per (player, past GW) training rows for the points model. Target = total_points.

    Rolling and lag features are partitioned by (player_id, season) so a player's first
    GW in a new season starts cold (no prior-season form leakage).
    """
    if history.empty:
        return pd.DataFrame()
    history = _ensure_season(history)
    df = history.copy().sort_values(["player_id", "season", "round"]).reset_index(drop=True)

    for lag in (1, 2, 3):
        df[f"lag{lag}_min"] = df.groupby(["player_id", "season"])["minutes"].shift(lag).fillna(0.0)

    # `total_points` is intentionally excluded — rolling it creates a feedback loop
    # where a premium's bad recent GW projects them lower forever.
    roll_map = {"expected_goals": "xg", "expected_assists": "xa",
                "expected_goal_involvements": "xgi", "bps": "bps", "ict_index": "ict",
                "saves": "saves", "clearances_blocks_interceptions": "cbi",
                "tackles": "tkl", "recoveries": "rec",
                "pm_xg": "oxg", "pm_xa": "oxa", "pm_cc": "occ",
                "pm_tob": "otob", "pm_shots": "osh", "pm_drib": "odrib"}
    for raw, short in roll_map.items():
        if raw not in df.columns:
            df[raw] = 0.0
        for w in (5, 10):
            df[f"roll{w}_{short}"] = (
                df.groupby(["player_id", "season"])[raw]
                  .transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
                  .fillna(0.0)
            )

    meta = players[[
        "id", "element_type", "team",
        "penalties_order", "direct_freekicks_order",
    ]].rename(columns={"id": "player_id", "element_type": "pos_id", "team": "team_id"})
    df = df.merge(meta, on="player_id", how="left")
    df["pos_id"] = df["pos_id"].fillna(3).astype(int)
    # One-hot — treating pos_id as ordinal conflates GK<->FWD scoring distributions.
    for p in (1, 2, 3, 4):
        df[f"pos_{p}"] = (df["pos_id"] == p).astype(int)
    df["is_pen_taker"] = (df["penalties_order"].fillna(0) == 1).astype(int)
    df["is_fk_taker"] = (df["direct_freekicks_order"].fillna(0) == 1).astype(int)

    fx = fixture_feats[["id", "event", "team_h", "team_a",
                        "h_xg_5", "a_xg_5", "h_xga_5", "a_xga_5",
                        "elo_h_pre", "elo_a_pre"]].rename(columns={"id": "fixture"})
    df = df.merge(fx, on="fixture", how="left")
    df["is_home"] = (df["team_id"] == df["team_h"]).astype(int)
    df["opp_xg_5"] = np.where(df["is_home"] == 1, df["a_xg_5"], df["h_xg_5"]).astype(float)
    df["opp_xga_5"] = np.where(df["is_home"] == 1, df["a_xga_5"], df["h_xga_5"]).astype(float)
    df["own_elo"] = np.where(df["is_home"] == 1, df["elo_h_pre"], df["elo_a_pre"]).astype(float)
    df["opp_elo"] = np.where(df["is_home"] == 1, df["elo_a_pre"], df["elo_h_pre"]).astype(float)
    df["elo_gap"] = df["own_elo"] - df["opp_elo"]
    df["target"] = df["total_points"].astype(float)
    return df


def points_feature_cols() -> list[str]:
    """Canonical feature-column order for the quantile points model."""
    base = ["pos_1", "pos_2", "pos_3", "pos_4",
            "is_home", "is_pen_taker", "is_fk_taker",
            "lag1_min", "lag2_min", "lag3_min",
            "opp_xg_5", "opp_xga_5", "opp_elo", "own_elo", "elo_gap"]
    for w in (5, 10):
        base += [f"roll{w}_{k}" for k in
                 ("xg", "xa", "xgi", "bps", "ict", "saves", "cbi", "tkl", "rec",
                  "oxg", "oxa", "occ", "otob", "osh", "odrib")]
    return base


def minutes_feature_cols() -> list[str]:
    """Feature subset for the availability / expected-minutes model.

    Drops set-piece flags and per-action rolling stats (cbi, tkl, saves) — these
    correlate strongly with playing time but are circular for predicting it. Keeps
    lag/roll minutes, form proxies (xg/xa/ict), position, and fixture-side context.
    """
    return ["pos_1", "pos_2", "pos_3", "pos_4",
            "is_home",
            "lag1_min", "lag2_min", "lag3_min",
            "roll5_xg", "roll5_xa", "roll5_xgi", "roll5_ict",
            "roll10_xg", "roll10_xa", "roll10_xgi", "roll10_ict",
            "own_elo", "opp_elo", "elo_gap"]
