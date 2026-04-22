"""Rolling team / player features + incremental team Elo."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_WINDOWS = [3, 5, 10]
INIT_ELO, K, HFA = 1500.0, 20.0, 60.0


def elo_snapshot_series(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Replays finished fixtures chronologically, stamping pre-match Elo on every row."""
    ratings = {int(t): INIT_ELO for t in teams["team_id"]}
    sort_col = "kickoff_time" if "kickoff_time" in fixtures.columns else "event"
    out = fixtures.sort_values(sort_col).copy().reset_index(drop=True)
    out["elo_h_pre"] = 0.0
    out["elo_a_pre"] = 0.0
    for i, f in out.iterrows():
        h, a = int(f["team_h"]), int(f["team_a"])
        rh, ra = ratings.get(h, INIT_ELO), ratings.get(a, INIT_ELO)
        out.at[i, "elo_h_pre"], out.at[i, "elo_a_pre"] = rh, ra
        if bool(f.get("finished")) and pd.notna(f.get("team_h_score")) and pd.notna(f.get("team_a_score")):
            gh, ga = int(f["team_h_score"]), int(f["team_a_score"])
            eh = 1.0 / (1.0 + 10 ** (-(rh + HFA - ra) / 400.0))
            sh = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            mov = (abs(gh - ga) + 1) ** 0.4
            ratings[h] = rh + K * mov * (sh - eh)
            ratings[a] = ra + K * mov * ((1 - sh) - (1 - eh))
    return out


def _rolling_team_stats(history: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, GW) xG/xGA rolled forward at the TEAM_WINDOWS horizons."""
    if history.empty or "team" not in history.columns:
        return pd.DataFrame()
    g = history.groupby(["team", "round"], as_index=False)[
        ["expected_goals", "expected_goals_conceded", "goals_scored", "goals_conceded"]
    ].sum().sort_values(["team", "round"])
    for w in TEAM_WINDOWS:
        for raw, short in [("expected_goals", "xg"), ("expected_goals_conceded", "xga"),
                           ("goals_scored", "gf"), ("goals_conceded", "ga")]:
            g[f"roll_{short}_{w}"] = g.groupby("team")[raw].transform(
                lambda x: x.shift().rolling(w, min_periods=1).mean()
            )
    return g


def build_match_features(
    fixtures: pd.DataFrame, history: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Joins team rolling stats + Elo diff onto every fixture row."""
    fx = elo_snapshot_series(fixtures, teams)
    tg = _rolling_team_stats(history)
    for side, team_col in (("h", "team_h"), ("a", "team_a")):
        if tg.empty:
            for w in TEAM_WINDOWS:
                for s in ("xg", "xga", "gf", "ga"):
                    fx[f"{side}_{s}_{w}"] = 1.2
            continue
        m = fx.merge(tg, left_on=[team_col, "event"], right_on=["team", "round"], how="left")
        for w in TEAM_WINDOWS:
            for s in ("xg", "xga", "gf", "ga"):
                fx[f"{side}_{s}_{w}"] = m[f"roll_{s}_{w}"].fillna(1.2).values
    fx["elo_diff"] = fx["elo_h_pre"] - fx["elo_a_pre"]
    fx["xg_diff_5"] = fx["h_xg_5"] - fx["a_xg_5"]
    fx["xga_diff_5"] = fx["h_xga_5"] - fx["a_xga_5"]
    return fx


def match_feature_cols() -> list[str]:
    """Canonical feature-column order for the match model."""
    cols = [f"{s}_{stat}_{w}"
            for s in ("h", "a") for w in TEAM_WINDOWS for stat in ("xg", "xga", "gf", "ga")]
    return cols + ["elo_h_pre", "elo_a_pre", "elo_diff", "xg_diff_5", "xga_diff_5"]


def build_player_features(
    history: pd.DataFrame, players: pd.DataFrame, fixture_feats: pd.DataFrame
) -> pd.DataFrame:
    """Per (player, past GW) training rows for the points model. Target = total_points."""
    if history.empty:
        return pd.DataFrame()
    df = history.copy().sort_values(["player_id", "round"]).reset_index(drop=True)

    for lag in (1, 2, 3):
        df[f"lag{lag}_min"] = df.groupby("player_id")["minutes"].shift(lag).fillna(0.0)

    roll_map = {"expected_goals": "xg", "expected_assists": "xa",
                "expected_goal_involvements": "xgi", "bps": "bps", "ict_index": "ict",
                "saves": "saves", "clearances_blocks_interceptions": "cbi",
                "tackles": "tkl", "recoveries": "rec", "total_points": "pts"}
    for raw, short in roll_map.items():
        if raw not in df.columns:
            df[raw] = 0.0
        for w in (5, 10):
            df[f"roll{w}_{short}"] = (
                df.groupby("player_id")[raw]
                  .transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
                  .fillna(0.0)
            )

    meta = players[[
        "id", "element_type", "team",
        "penalties_order", "direct_freekicks_order",
    ]].rename(columns={"id": "player_id", "element_type": "pos_id", "team": "team_id"})
    df = df.merge(meta, on="player_id", how="left")
    df["pos_id"] = df["pos_id"].fillna(3).astype(int)
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
    base = ["pos_id", "is_home", "is_pen_taker", "is_fk_taker",
            "lag1_min", "lag2_min", "lag3_min",
            "opp_xg_5", "opp_xga_5", "opp_elo", "own_elo", "elo_gap"]
    for w in (5, 10):
        base += [f"roll{w}_{k}" for k in
                 ("xg", "xa", "xgi", "bps", "ict", "saves", "cbi", "tkl", "rec", "pts")]
    return base