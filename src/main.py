"""GH Actions entrypoint. Refresh data, train missing models, solve, write lineup.md."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from chips import (recommend_bench_boost, recommend_free_hit,
                   recommend_triple_captain, recommend_wildcard)
from data_loader import SEASON, main as refresh_data
from features import match_feature_cols, minutes_feature_cols, points_feature_cols
from fpl_engine import FPLEngine
from optimizer import solve_initial_squad, solve_rhc_transfers
from train_bonus_model import train_bonus_model
from train_match_model import compute_fixture_lambdas, train_match_models
from train_minutes_model import train_minutes_model
from train_points_model import _pos_feature_cols, train_points_models

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 8
# Linear-std variance penalty (was variance^2). 0.05 = equivalent scale.
# Set 0.0 if solver still under-invests in premiums.
LAMBDA_VAR, LAMBDA_EO, BENCH_WEIGHT = 0.05, 0.0, 0.15
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Auto-recalibration cadence. Pipeline runs 2×/day on data-source refresh, but
# walk-forward retrain is expensive (~minutes) and recalib drift is slow. Re-fit
# only when JSON older than N days OR missing. Manual override: delete the JSON.
RECALIB_STALE_DAYS = 14
RECALIB_HOLDOUT_K = 8
POINTS_RECALIB_PATH = DATA_DIR / "points_recalib.json"
MINUTES_RECALIB_PATH = DATA_DIR / "minutes_recalib.json"


def _md_table(df: pd.DataFrame) -> str:
    """Render df as GitHub-flavored markdown table. No tabulate dep."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _current_gw(fixtures: pd.DataFrame) -> int:
    """Smallest GW <50% finished. Robust to lingering postponed past matches."""
    fx = fixtures
    if "season" in fx.columns:
        fx = fx[fx["season"] == SEASON]
    g = fx.groupby("event")["finished"].agg(sum_="sum", size_="size")
    upcoming = g[(g["sum_"] / g["size_"]) < 0.5]
    return int(upcoming.index.min()) if not upcoming.empty else 38


# Live-match window guard. Daily cron fires at 05:30 + 17:30 UTC; skip when a
# GW is mid-flight so we don't burn CI minutes producing a lineup the user
# can't act on (transfers locked from deadline through last final whistle).
LIVE_KICKOFF_LOOKBACK_HOURS = 3.0  # 90' + ET + stoppage cushion
LIVE_KICKOFF_LOOKAHEAD_HOURS = 2.0  # post-deadline lockout before first match


def _gw_in_play(fixtures: pd.DataFrame) -> bool:
    """True when any current-season fixture is live or imminent.

    Live: kickoff in [now − LOOKBACK, now] and finished=False.
    Imminent: kickoff in [now, now + LOOKAHEAD] (deadline already passed,
    first match about to start; transfers locked, lineup advice moot).
    """
    fx = fixtures
    if "season" in fx.columns:
        fx = fx[fx["season"] == SEASON]
    if fx.empty:
        return False
    ko = pd.to_datetime(fx["kickoff_time"], utc=True, errors="coerce")
    fin = fx["finished"].astype(str).str.lower().isin(["true", "1"])
    now = pd.Timestamp.now(tz="UTC")
    lookback = now - pd.Timedelta(hours=LIVE_KICKOFF_LOOKBACK_HOURS)
    lookahead = now + pd.Timedelta(hours=LIVE_KICKOFF_LOOKAHEAD_HOURS)
    live = (~fin) & (ko >= lookback) & (ko <= now)
    imminent = (ko > now) & (ko <= lookahead)
    return bool((live | imminent).any())


def _booster_features(path: Path) -> list[str] | None:
    """Booster's stored feature_names. None if file missing or unreadable."""
    if not path.exists():
        return None
    try:
        b = xgb.Booster()
        b.load_model(path)
        return list(b.feature_names or [])
    except Exception:
        return None


def _schema_drift(path: Path, expected: list[str]) -> bool:
    """True if booster's feature_names diverge from expected set."""
    feats = _booster_features(path)
    if feats is None:
        return False  # file absent — caller's existence check handles retrain
    return set(feats) != set(expected)


def _invalidate_recalib(*paths: Path) -> None:
    """Drop stale recalib JSONs after retrain. New raw quantiles → old map invalid."""
    for p in paths:
        if p.exists():
            p.unlink()


def _ensure_models(fx: pd.DataFrame, hist: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Train missing match / points / minutes / bonus artifacts.

    Match → fixture λ → points + bonus. λ feeds points feature schema, so
    refresh fixture_lambdas.csv any time match models present (cheap). Also
    detect feature-schema drift on cached on-disk boosters (e.g. features.py
    grew new cols since last train) and force retrain.
    """
    match_files = ("xgb_home_goals.json", "xgb_away_goals.json")
    match_probe = DATA_DIR / "xgb_home_goals.json"
    match_have = all((DATA_DIR / f).exists() for f in match_files)
    if (not match_have
            or _schema_drift(match_probe, match_feature_cols())):
        print("[models] (re)training match head — schema drift or missing artifacts")
        train_match_models(fx, hist, teams)
    compute_fixture_lambdas(fx, hist, teams)
    points_files = [f"xgb_points_q{q:02d}_p{p}.json"
                    for q in (10, 50, 90) for p in (1, 2, 3, 4)]
    points_probe = DATA_DIR / "xgb_points_q10_p1.json"
    if (not all((DATA_DIR / f).exists() for f in points_files)
            or _schema_drift(points_probe, _pos_feature_cols())):
        print("[models] (re)training points head — schema drift or missing artifacts")
        train_points_models()
        _invalidate_recalib(POINTS_RECALIB_PATH)
    minutes_files = ("xgb_minutes_plays.json", "xgb_minutes_when_played.json")
    minutes_probe = DATA_DIR / "xgb_minutes_plays.json"
    minutes_have_two_stage = all((DATA_DIR / f).exists() for f in minutes_files)
    minutes_have_legacy = (DATA_DIR / "xgb_minutes.json").exists()
    if ((not minutes_have_two_stage and not minutes_have_legacy)
            or (minutes_have_two_stage and _schema_drift(minutes_probe, minutes_feature_cols()))):
        print("[models] (re)training minutes head — schema drift or missing artifacts")
        train_minutes_model()
        _invalidate_recalib(MINUTES_RECALIB_PATH)
    bonus_files = [f"xgb_bonus_q{q:02d}.json" for q in (10, 50, 90)]
    bonus_probe = DATA_DIR / "xgb_bonus_q10.json"
    if (not all((DATA_DIR / f).exists() for f in bonus_files)
            or _schema_drift(bonus_probe, points_feature_cols())):
        print("[models] (re)training bonus head — schema drift or missing artifacts")
        train_bonus_model()


def _recalib_stale(path: Path, max_age_days: int = RECALIB_STALE_DAYS) -> bool:
    """True if recalib JSON missing or older than max_age_days."""
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days > max_age_days


def _maybe_recalibrate(fixtures: pd.DataFrame, history: pd.DataFrame,
                       players: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Re-fit points + minutes recalib JSONs if stale.

    Walk-forward retrain runs on last RECALIB_HOLDOUT_K finished GWs. Outputs
    consumed by predict_quantiles / predict_minutes auto-load on next inference
    pass within same process — but they cache via load_*_models, so recalib is
    re-read at next model-load. Engine constructed AFTER this call.
    """
    points_stale = _recalib_stale(POINTS_RECALIB_PATH)
    minutes_stale = _recalib_stale(MINUTES_RECALIB_PATH)
    if not (points_stale or minutes_stale):
        return

    # Lazy imports. Backtest pulls xgb + heavy training stack.
    from backtest import (_resolve_holdout, walk_forward_minutes,
                          walk_forward_points)
    from recalibrate_minutes import fit_minutes_recalib
    from recalibrate_minutes import save_recalib as save_min_recalib
    from recalibrate_points import fit_points_recalib
    from recalibrate_points import save_recalib as save_pts_recalib

    try:
        holdout = _resolve_holdout(fixtures, RECALIB_HOLDOUT_K, None, None)
    except RuntimeError:
        print("recalib skipped: no finished GWs available")
        return
    if not holdout:
        print("recalib skipped: empty holdout")
        return
    print(f"recalib: walk-forward over GWs {holdout}")

    if points_stale:
        pred = walk_forward_points(holdout, history, fixtures, players, teams)
        if not pred.empty:
            coef = fit_points_recalib(pred, played_only=True)
            save_pts_recalib(coef, POINTS_RECALIB_PATH)
            print(f"recalib points -> {POINTS_RECALIB_PATH}")

    if minutes_stale:
        pred = walk_forward_minutes(holdout, history, fixtures, players, teams)
        if not pred.empty:
            coef = fit_minutes_recalib(pred)
            save_min_recalib(coef, MINUTES_RECALIB_PATH)
            print(f"recalib minutes -> {MINUTES_RECALIB_PATH}")


def _load_prior() -> tuple[set[int], float, int] | None:
    """Read last-week squad snapshot for RHC. None on cold start."""
    snap = OUT_DIR / "squad_snapshot.csv"
    if not snap.exists():
        return None
    df = pd.read_csv(snap)
    return (set(df["id"].astype(int)), float(df["bank"].iloc[0]),
            int(df["free_transfers"].iloc[0]))


def _persist(squad: pd.DataFrame, bank: float, ft: int) -> None:
    """Save GW snapshot (squad + bank + FT) for next run."""
    out = squad.copy()
    out["bank"], out["free_transfers"] = bank, ft
    out.to_csv(OUT_DIR / "squad_snapshot.csv", index=False)


def _render(
    gw: int, squad: pd.DataFrame, xi: set[int], cap: int, vice: int,
    bank: float, hits: int, ins: list[int], outs: list[int],
    players: pd.DataFrame, teams: pd.DataFrame,
    tc: dict, bb: dict, fh: dict, wc: dict,
) -> str:
    """Weekly lineup + transfers + chips → single markdown doc."""
    tmap = teams.set_index("team_id")["short_name"].to_dict()
    nmap = players.set_index("id")["web_name"].to_dict()

    d = squad.copy()
    d["Team"] = d["team_id"].map(tmap)
    d["Pos"] = d["pos_id"].map(POS)
    d["Price"] = d["price"].round(1)
    # Captain row XP doubled so XI col-sum reflects actual GW total.
    cap_mult_1 = d["id"].map({cap: 2.0}).fillna(1.0)
    d["XP(1)"] = (d["next_gw_xp"] * cap_mult_1).round(2)
    d["XP(H)"] = d["horizon_xp"].round(2)
    d["Role"] = ""
    d.loc[d["id"] == cap, "Role"] = "(C)"
    d.loc[d["id"] == vice, "Role"] = "(VC)"
    d = d.rename(columns={"name": "Name"})
    cols = ["Name", "Team", "Pos", "Price", "XP(1)", "XP(H)", "Role"]

    xi_df = d[d["id"].isin(xi)].sort_values("pos_id")[cols]
    bench_df = d[~d["id"].isin(xi)].sort_values("pos_id")[cols]
    xi_total_xp = float(xi_df["XP(1)"].sum())

    lines = [
        f"# GW{gw} Lineup", "",
        f"- **Bank:** £{bank:.1f}m",
        f"- **Hits:** -{hits * 4} pts" if hits else "- **Hits:** 0",
        f"- **Squad Value:** £{squad['price'].sum():.1f}m",
        f"- **XI Expected Points (incl. captain):** {xi_total_xp:.1f}",
        "", "## Starting XI", "", _md_table(xi_df),
        "", "## Bench", "", _md_table(bench_df),
    ]

    if ins or outs:
        tr = pd.DataFrame({
            "Out": [nmap.get(o, str(o)) for o in outs],
            "In":  [nmap.get(i, str(i)) for i in ins],
        })
        lines += ["", "## Transfers", "", _md_table(tr)]
    else:
        lines += ["", "## Transfers", "", "_Hold — no transfer beats a 4-pt hit._"]

    chips = ["", "## Chip Recommendations", ""]
    if tc.get("gw") is not None:
        chips.append(f"- **Triple Captain:** GW{tc['gw']} — "
                     f"{nmap.get(tc['player_id'], '?')} (+{tc['bonus']:.1f} pts)")
    if bb.get("gw") is not None:
        chips.append(f"- **Bench Boost:** GW{bb['gw']} (+{bb['bonus']:.1f} pts)")
    if fh.get("gw") is not None and fh["blanks"] >= 2:
        chips.append(f"- **Free Hit:** GW{fh['gw']} ({fh['blanks']} teams blank)")
    chips.append(f"- **Wildcard:** {'PLAY NOW' if wc['recommend'] else 'hold'}"
                 f" ({wc['n_transfers']} suggested transfers, {wc['hits']} hits)")
    lines += chips
    return "\n".join(lines) + "\n"


def main() -> None:
    """End-to-end weekly pipeline. Invoked by GH Actions workflow."""
    refresh_data()

    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    if _gw_in_play(fixtures):
        print("[main] GW in play — skip pipeline. "
              "No actionable transfer/lineup work between deadline and final whistle.")
        return

    _ensure_models(fixtures, history, teams)
    _maybe_recalibrate(fixtures, history, players, teams)
    engine = FPLEngine(fixtures, history, players, teams)
    gw = _current_gw(fixtures)
    proj = engine.build_projections(gw, horizon=HORIZON)
    if proj.empty:
        raise RuntimeError("empty projections; check data and model artifacts")

    prior = _load_prior()
    if prior is None:
        sq = solve_initial_squad(proj, lambda_var=LAMBDA_VAR,
                                 lambda_eo=LAMBDA_EO, bench_weight=BENCH_WEIGHT)
        if sq.empty:
            raise RuntimeError("initial squad solver failed")
        squad_ids = set(sq["id"].astype(int))
        squad = proj[proj["id"].isin(squad_ids)].copy()
        xi_ids = set(sq[sq["in_xi"] == 1]["id"].astype(int))
        cap = int(sq[sq["is_captain"] == 1]["id"].iloc[0])
        vice = int(sq[sq["is_vice"] == 1]["id"].iloc[0])
        bank = round(100.0 - float(squad["price"].sum()), 1)
        hits, ins, outs = 0, [], []
        next_ft = 1
    else:
        prior_ids, bank_prior, ft = prior
        rec = solve_rhc_transfers(proj, prior_ids, bank_prior, ft,
                                  lambda_var=LAMBDA_VAR, lambda_eo=LAMBDA_EO,
                                  bench_weight=BENCH_WEIGHT)
        if rec["status"] != "ok":
            raise RuntimeError(f"RHC failed: {rec['status']}")
        squad_ids = rec["squad_ids"]
        squad = proj[proj["id"].isin(squad_ids)].copy()
        xi_ids = rec["xi_ids"]
        cap = int(rec["captain"])
        vice = int(rec["vice"])
        hits = rec["hits"]
        ins, outs = rec["transfers_in"], rec["transfers_out"]
        bank = round(100.0 - float(squad["price"].sum()), 1)
        next_ft = min(5, ft + 1) if not ins else 1

    tc = recommend_triple_captain(proj, squad_ids)
    bb = recommend_bench_boost(proj, squad_ids, xi_ids)
    fh = recommend_free_hit(fixtures, gw, HORIZON)
    wc = recommend_wildcard(ins, hits)

    md = _render(gw, squad, xi_ids, cap, vice, bank, hits, ins, outs,
                 players, teams, tc, bb, fh, wc)
    (OUT_DIR / "lineup.md").write_text(md)

    snap = squad.copy()
    snap["in_xi"] = snap["id"].isin(xi_ids).astype(int)
    snap["is_captain"] = (snap["id"] == cap).astype(int)
    snap["is_vice"] = (snap["id"] == vice).astype(int)
    _persist(snap, bank, next_ft)


if __name__ == "__main__":
    main()
