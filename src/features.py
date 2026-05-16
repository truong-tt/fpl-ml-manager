"""Rolling team / player features. Elo from ClubElo (FPL-CI). Replay = fallback.

All rolling state partitioned by season. Stop cross-season leakage. Player or
team GW1 row in season N never sees season N-1 data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from league_table import attach_stakes, stakes_match_cols, stakes_player_cols

# Match-model-predicted fixture λ + DC CS prob. Optional. Built by
# train_match_model.compute_fixture_lambdas → main pipeline writes after match
# trains. Absent on cold start → neutral defaults (avg PL goals, baseline CS).
FIXTURE_LAMBDAS_FILE = Path(__file__).resolve().parent.parent / "data" / "fixture_lambdas.csv"
LAMBDA_DEFAULT = 1.4
CS_PROB_DEFAULT = 0.30

# Cup fixtures (EFL Cup / UCL / UEL / UECL). Powers minutes-head congestion
# features. Absent on cold start → all-zero congestion / sentinel days_to.
CUP_FIXTURES_FILE = Path(__file__).resolve().parent.parent / "data" / "cup_fixtures.csv"
CUP_WINDOW_DAYS = 3.0
DAYS_TO_NEXT_CUP_SENTINEL = 999.0

TEAM_WINDOWS = [3, 5, 10]
# Used only when FPL-CI home_team_elo / away_team_elo absent or null.
INIT_ELO, K, HFA = 1500.0, 20.0, 60.0

# EMA rolling, not equal-weight. Halflife scales with nominal window: roll{w}
# → halflife = w / 2. Recent GWs weight more without hard window cutoff.
# Catches manager / form swings faster than rolling(w).mean() — there a 1-pre-GW
# shock got 1/w weight until window slid past.
EMA_HALFLIFE_FACTOR = 0.5


def _ema_shifted(s: pd.Series, halflife: float) -> pd.Series:
    """Shift-1 then EMA. Shift-1 prevents round-t target leak into round-t feature."""
    return s.shift().ewm(halflife=halflife, adjust=False, ignore_na=True).mean()


def _ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill `season` col with single-season default. Legacy frames."""
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = "current"
    return df


def _elo_replay(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Chronological Elo replay over finished fixtures. Fallback path. Reset per season."""
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
    """Stamp pre-match Elo. ClubElo if on fixtures.csv, else replay."""
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
    """Per-(team, season, GW) xG/xGA EMA at TEAM_WINDOWS horizons."""
    if history.empty or "team" not in history.columns:
        return pd.DataFrame()
    history = _ensure_season(history)
    g = history.groupby(["season", "team", "round"], as_index=False)[
        ["expected_goals", "expected_goals_conceded", "goals_scored", "goals_conceded"]
    ].sum().sort_values(["season", "team", "round"])
    for w in TEAM_WINDOWS:
        hl = w * EMA_HALFLIFE_FACTOR
        for raw, short in [("expected_goals", "xg"), ("expected_goals_conceded", "xga"),
                           ("goals_scored", "gf"), ("goals_conceded", "ga")]:
            g[f"roll_{short}_{w}"] = g.groupby(["season", "team"])[raw].transform(
                lambda x, hl=hl: _ema_shifted(x, hl)
            )
    return g


# Match-level Opta from fixtures.csv. True on-the-ball xG. Distinct from
# _rolling_team_stats which sums per-player FPL xG.
OPTA_STATS = [
    ("expected_goals_xg", "oxg"),
    ("big_chances", "obc"),
    ("total_shots", "osh"),
]
OPTA_WINDOW = 5
OPTA_DEFAULTS = {"oxg": 1.2, "oxga": 1.2, "obc": 1.5, "obca": 1.5, "osh": 11.0, "osha": 11.0}


def _rolling_fixture_team_stats(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, season, GW) Opta EMA. Each fixture = row per side."""
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
    hl = OPTA_WINDOW * EMA_HALFLIFE_FACTOR
    for c in short_cols:
        long[f"roll_{c}_{OPTA_WINDOW}"] = long.groupby(["season", "team"])[c].transform(
            lambda x, hl=hl: _ema_shifted(x, hl)
        )
    keep = [f"roll_{c}_{OPTA_WINDOW}" for c in short_cols]
    return long.groupby(["season", "team", "round"], as_index=False)[keep].mean()


def _team_cup_congestion(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, event, season) cup-congestion features.

    Sources `data/cup_fixtures.csv` (built by data_loader from FPL-Core-Insights
    By Tournament/ folders for EFL Cup / UCL / UEL / UECL). For each PL fixture
    the team plays, count cup matches that team plays within ±CUP_WINDOW_DAYS
    of the PL kickoff. Captures the rotation signal the minutes head currently
    misses (e.g. Chelsea resting EPL XI before UECL final).

    Cols:
      cup_pre: cup matches in [-CUP_WINDOW_DAYS, 0) days (recent fatigue).
      cup_post: cup matches in [0, +CUP_WINDOW_DAYS] days (upcoming priority).
      cup_total: pre + post.
      days_to_next_cup: min(positive delta_days) → sentinel if none.

    Returns empty frame if cup_fixtures.csv absent (cold start). Wired into
    build_match_features as `{h,a}_{col}`; build_player_features converts to
    `own_{col}` via is_home pivot.
    """
    cols = ["team", "event", "season",
            "cup_pre", "cup_post", "cup_total", "days_to_next_cup"]
    if not CUP_FIXTURES_FILE.exists():
        return pd.DataFrame(columns=cols)
    cf = pd.read_csv(CUP_FIXTURES_FILE)
    if cf.empty or "kickoff_time" not in cf.columns:
        return pd.DataFrame(columns=cols)
    cf = cf.rename(columns={"team_id": "team", "kickoff_time": "cup_kickoff"})
    cf["cup_kickoff"] = pd.to_datetime(cf["cup_kickoff"], utc=True, errors="coerce")
    cf = cf.dropna(subset=["cup_kickoff", "team"])

    fx = _ensure_season(fixtures).copy()
    if "kickoff_time" not in fx.columns:
        return pd.DataFrame(columns=cols)
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], utc=True, errors="coerce")
    fx = fx.dropna(subset=["kickoff_time"])

    sides = []
    for team_col in ("team_h", "team_a"):
        d = fx[["event", "season", "kickoff_time", team_col]].rename(
            columns={team_col: "team"})
        sides.append(d)
    long = pd.concat(sides, ignore_index=True)

    j = long.merge(cf[["team", "season", "cup_kickoff"]],
                   on=["team", "season"], how="left")
    j["delta_days"] = (
        (j["cup_kickoff"] - j["kickoff_time"]).dt.total_seconds() / 86400.0
    )
    in_win = j["delta_days"].abs() <= CUP_WINDOW_DAYS
    j["pre"] = (in_win & (j["delta_days"] < 0)).astype(int)
    j["post"] = (in_win & (j["delta_days"] >= 0)).astype(int)

    agg = (j.groupby(["team", "event", "season"], as_index=False)
            .agg(cup_pre=("pre", "sum"), cup_post=("post", "sum")))
    agg["cup_total"] = agg["cup_pre"] + agg["cup_post"]

    pos = j.loc[j["delta_days"] >= 0].copy()
    if pos.empty:
        agg["days_to_next_cup"] = DAYS_TO_NEXT_CUP_SENTINEL
    else:
        nxt = (pos.groupby(["team", "event", "season"], as_index=False)["delta_days"]
                  .min().rename(columns={"delta_days": "days_to_next_cup"}))
        agg = agg.merge(nxt, on=["team", "event", "season"], how="left")
        agg["days_to_next_cup"] = agg["days_to_next_cup"].fillna(
            DAYS_TO_NEXT_CUP_SENTINEL)
    return agg[cols]


CUP_COLS = ("cup_pre", "cup_post", "cup_total", "days_to_next_cup")
CUP_DEFAULTS = {
    "cup_pre": 0.0, "cup_post": 0.0, "cup_total": 0.0,
    "days_to_next_cup": DAYS_TO_NEXT_CUP_SENTINEL,
}


def build_match_features(
    fixtures: pd.DataFrame, history: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Join team rolling + Elo diff onto every fixture row."""
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

    cc = _team_cup_congestion(fixtures)
    for side, team_col in (("h", "team_h"), ("a", "team_a")):
        if cc.empty:
            for c in CUP_COLS:
                fx[f"{side}_{c}"] = CUP_DEFAULTS[c]
            continue
        m = fx.merge(cc, left_on=[team_col, "event", "season"],
                     right_on=["team", "event", "season"],
                     how="left", suffixes=("", "_cc"))
        for c in CUP_COLS:
            fx[f"{side}_{c}"] = m[c].fillna(CUP_DEFAULTS[c]).values

    fx["elo_diff"] = fx["elo_h_pre"] - fx["elo_a_pre"]
    fx["xg_diff_5"] = fx["h_xg_5"] - fx["a_xg_5"]
    fx["xga_diff_5"] = fx["h_xga_5"] - fx["a_xga_5"]
    fx["oxg_diff"] = fx[f"h_oxg_{OPTA_WINDOW}"] - fx[f"a_oxg_{OPTA_WINDOW}"]
    fx["oxga_diff"] = fx[f"h_oxga_{OPTA_WINDOW}"] - fx[f"a_oxga_{OPTA_WINDOW}"]

    # Match-model λ + DC CS prob. Structurally novel (DC correction not derivable
    # from rolling team stats). Neutral default if file absent.
    if FIXTURE_LAMBDAS_FILE.exists() and "id" in fx.columns:
        lam = pd.read_csv(FIXTURE_LAMBDAS_FILE)
        fx = fx.merge(lam, on="id", how="left", suffixes=("", "_lam"))
    for col, default in (("lambda_h", LAMBDA_DEFAULT), ("lambda_a", LAMBDA_DEFAULT),
                          ("cs_h_p", CS_PROB_DEFAULT), ("cs_a_p", CS_PROB_DEFAULT)):
        if col not in fx.columns:
            fx[col] = default
        else:
            fx[col] = pd.to_numeric(fx[col], errors="coerce").fillna(default)

    # Stakes features: pre-match league position + tier-distance gaps. Encodes
    # late-season motivation (title race, top-4, top-6, drop fight) that EMA
    # rolling form lags by halflife window. See src/league_table.py.
    fx = attach_stakes(fx)
    return fx


def match_feature_cols() -> list[str]:
    """Canonical match feature col order."""
    cols = [f"{s}_{stat}_{w}"
            for s in ("h", "a") for w in TEAM_WINDOWS for stat in ("xg", "xga", "gf", "ga")]
    opta_short = [c for _, c in OPTA_STATS] + [f"{c}a" for _, c in OPTA_STATS]
    cols += [f"{s}_{c}_{OPTA_WINDOW}" for s in ("h", "a") for c in opta_short]
    cols += ["elo_h_pre", "elo_a_pre", "elo_diff",
             "xg_diff_5", "xga_diff_5", "oxg_diff", "oxga_diff"]
    return cols + stakes_match_cols()


def build_player_features(
    history: pd.DataFrame, players: pd.DataFrame, fixture_feats: pd.DataFrame
) -> pd.DataFrame:
    """Per (player, past GW) training rows. Target = total_points.

    Rolling/lag partitioned by (player_id, season). Player first GW in new
    season starts cold. No prior-season form leakage.
    """
    if history.empty:
        return pd.DataFrame()
    history = _ensure_season(history)
    df = history.copy().sort_values(["player_id", "season", "round"]).reset_index(drop=True)

    for lag in (1, 2, 3):
        df[f"lag{lag}_min"] = df.groupby(["player_id", "season"])["minutes"].shift(lag).fillna(0.0)

    # total_points excluded on purpose. Rolling = feedback loop. Premium with one
    # bad recent GW projects lower forever.
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
            hl = w * EMA_HALFLIFE_FACTOR
            df[f"roll{w}_{short}"] = (
                df.groupby(["player_id", "season"])[raw]
                  .transform(lambda x, hl=hl: _ema_shifted(x, hl))
                  .fillna(0.0)
            )

    # Player attack share. share = player_roll{w}_X / sum_team_roll{w}_X. Net new
    # info booster cannot derive from per-player rolling alone — captures role
    # within team's offense (15% of Liverpool's xGI ≠ 15% of Burnley's).
    if "team" in df.columns:
        share_eps = 1e-3
        for w in (5, 10):
            for short in ("xg", "xa", "xgi"):
                col = f"roll{w}_{short}"
                team_sum = df.groupby(["season", "team", "round"])[col].transform("sum")
                df[f"{col}_share"] = df[col] / (team_sum + share_eps)

    meta = players[[
        "id", "element_type", "team",
        "penalties_order", "direct_freekicks_order",
    ]].rename(columns={"id": "player_id", "element_type": "pos_id", "team": "team_id"})
    df = df.merge(meta, on="player_id", how="left")
    df["pos_id"] = df["pos_id"].fillna(3).astype(int)
    # One-hot. pos_id ordinal would conflate GK<->FWD scoring distributions.
    for p in (1, 2, 3, 4):
        df[f"pos_{p}"] = (df["pos_id"] == p).astype(int)
    df["is_pen_taker"] = (df["penalties_order"].fillna(0) == 1).astype(int)
    df["is_fk_taker"] = (df["direct_freekicks_order"].fillna(0) == 1).astype(int)

    stakes_side_cols = []
    for s in ("h", "a"):
        stakes_side_cols += [f"{s}_rank", f"{s}_ppg", f"{s}_in_drop_zone"]
        stakes_side_cols += [f"{s}_pts_to_title_norm", f"{s}_pts_to_top4_norm",
                              f"{s}_pts_to_top6_norm", f"{s}_pts_to_safety_norm"]
    cup_side_cols = [f"{s}_{c}" for s in ("h", "a") for c in CUP_COLS]
    fx_take = ["id", "event", "team_h", "team_a",
               "h_xg_5", "a_xg_5", "h_xga_5", "a_xga_5",
               "elo_h_pre", "elo_a_pre",
               "lambda_h", "lambda_a", "cs_h_p", "cs_a_p",
               "gws_remaining"] + stakes_side_cols + cup_side_cols
    fx_take = [c for c in fx_take if c in fixture_feats.columns]
    fx = fixture_feats[fx_take].rename(columns={"id": "fixture"})
    df = df.merge(fx, on="fixture", how="left")
    df["is_home"] = (df["team_id"] == df["team_h"]).astype(int)
    df["opp_xg_5"] = np.where(df["is_home"] == 1, df["a_xg_5"], df["h_xg_5"]).astype(float)
    df["opp_xga_5"] = np.where(df["is_home"] == 1, df["a_xga_5"], df["h_xga_5"]).astype(float)
    df["own_elo"] = np.where(df["is_home"] == 1, df["elo_h_pre"], df["elo_a_pre"]).astype(float)
    df["opp_elo"] = np.where(df["is_home"] == 1, df["elo_a_pre"], df["elo_h_pre"]).astype(float)
    df["elo_gap"] = df["own_elo"] - df["opp_elo"]
    # Side-conditioned match-model λ. own_lambda_for = expected goals BY my team
    # (offensive proxy). own_lambda_against = expected goals AGAINST my team
    # (defensive proxy, drives CS prob). own_cs_p = DC-corrected clean sheet prob.
    df["own_lambda_for"] = np.where(df["is_home"] == 1, df["lambda_h"], df["lambda_a"]).astype(float)
    df["own_lambda_against"] = np.where(df["is_home"] == 1, df["lambda_a"], df["lambda_h"]).astype(float)
    df["own_cs_p"] = np.where(df["is_home"] == 1, df["cs_h_p"], df["cs_a_p"]).astype(float)

    # Stakes: side-condition h_/a_ pre-match-table cols → own_/opp_. Late-season
    # motivation signal: opp_pts_to_title near 0 = title chaser fully engaged;
    # own_pts_to_safety strongly positive = mid-table no-stakes.
    stakes_pairs = [("rank", 10.0), ("ppg", 0.0), ("in_drop_zone", 0.0),
                    ("pts_to_title_norm", 0.0), ("pts_to_top4_norm", 0.0),
                    ("pts_to_top6_norm", 0.0), ("pts_to_safety_norm", 0.0)]
    for name, default in stakes_pairs:
        h_col, a_col = f"h_{name}", f"a_{name}"
        if h_col not in df.columns or a_col not in df.columns:
            df[f"own_{name}"] = default
            df[f"opp_{name}"] = default
            continue
        df[h_col] = df[h_col].fillna(default).astype(float)
        df[a_col] = df[a_col].fillna(default).astype(float)
        df[f"own_{name}"] = np.where(df["is_home"] == 1, df[h_col], df[a_col]).astype(float)
        df[f"opp_{name}"] = np.where(df["is_home"] == 1, df[a_col], df[h_col]).astype(float)
    if "gws_remaining" not in df.columns:
        df["gws_remaining"] = 0.0
    df["gws_remaining"] = df["gws_remaining"].fillna(0.0).astype(float)

    # Cup congestion: pivot h_/a_ → own_/opp_ via is_home. own_cup_total >> 0
    # = rotation likely (Euro tie / cup final close to PL kickoff). opp_cup_*
    # exposed for symmetry; minutes head consumes own_* only.
    for c in CUP_COLS:
        h_col, a_col = f"h_{c}", f"a_{c}"
        default = CUP_DEFAULTS[c]
        if h_col not in df.columns or a_col not in df.columns:
            df[f"own_{c}"] = default
            df[f"opp_{c}"] = default
            continue
        df[h_col] = df[h_col].fillna(default).astype(float)
        df[a_col] = df[a_col].fillna(default).astype(float)
        df[f"own_{c}"] = np.where(df["is_home"] == 1, df[h_col], df[a_col]).astype(float)
        df[f"opp_{c}"] = np.where(df["is_home"] == 1, df[a_col], df[h_col]).astype(float)
    # target = total_points - bonus. Bonus head adds back at engine w/ BONUS_BLEND=1.0.
    # Removes double-count (points head learning partial bonus + bonus head adding it).
    bonus = pd.to_numeric(df.get("bonus", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["target"] = df["total_points"].astype(float) - bonus
    return df


def points_feature_cols() -> list[str]:
    """Canonical points feature col order."""
    base = ["pos_1", "pos_2", "pos_3", "pos_4",
            "is_home", "is_pen_taker", "is_fk_taker",
            "lag1_min", "lag2_min", "lag3_min",
            "opp_xg_5", "opp_xga_5", "opp_elo", "own_elo", "elo_gap",
            "own_lambda_for", "own_lambda_against", "own_cs_p"]
    for w in (5, 10):
        base += [f"roll{w}_{k}" for k in
                 ("xg", "xa", "xgi", "bps", "ict", "saves", "cbi", "tkl", "rec",
                  "oxg", "oxa", "occ", "otob", "osh", "odrib")]
        base += [f"roll{w}_{k}_share" for k in ("xg", "xa", "xgi")]
    return base + stakes_player_cols()


def minutes_feature_cols() -> list[str]:
    """Feature subset for minutes head.

    Drop set-piece flags + per-action rolling (cbi/tkl/saves). Strong correlation
    with playing time but circular for predicting it. Keep lag/roll minutes,
    form proxies (xg/xa/ict), pos, fixture-side context.

    Cup-congestion cols (own_cup_pre/post/total, own_days_to_next_cup) feed
    rotation signal: team facing a near-term cup tie tends to rest XI in EPL.
    Lacking before — minutes head couldn't anticipate Chelsea UECL-final
    rotation in advance (only saw lag minutes after the first benching).
    """
    return ["pos_1", "pos_2", "pos_3", "pos_4",
            "is_home",
            "lag1_min", "lag2_min", "lag3_min",
            "roll5_xg", "roll5_xa", "roll5_xgi", "roll5_ict",
            "roll10_xg", "roll10_xa", "roll10_xgi", "roll10_ict",
            "own_elo", "opp_elo", "elo_gap",
            "own_cup_pre", "own_cup_post", "own_cup_total",
            "own_days_to_next_cup"]
