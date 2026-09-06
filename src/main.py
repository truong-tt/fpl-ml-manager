"""GH Actions entrypoint. Refresh data, train missing models, solve, write lineup.md."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from chips import decide_chips
from data_loader import SEASON, main as refresh_data
from gameweeks import from_frame
from model_preparation import prepare_engine
from optimizer import solve_initial_squad, solve_rhc_transfers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 8
# Linear-std variance penalty (was variance^2). 0.05 = equivalent scale.
# Set 0.0 if solver still under-invests in premiums.
LAMBDA_VAR, LAMBDA_EO, BENCH_WEIGHT = 0.05, 0.0, 0.15
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


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


def _last_finished_gw(fixtures: pd.DataFrame) -> int:
    return from_frame(fixtures, SEASON).last_finalized


def _last_completed_gw(fixtures: pd.DataFrame) -> int:
    return from_frame(fixtures, SEASON).last_completed


def _season_complete(fixtures: pd.DataFrame) -> bool:
    return bool(from_frame(fixtures, SEASON).status()["season_done"])


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


def _load_prior() -> tuple[set[int], float, int] | None:
    """Read last-week squad snapshot for RHC. None on cold start."""
    snap = OUT_DIR / "squad_snapshot.csv"
    if not snap.exists():
        return None
    df = pd.read_csv(snap)
    if ("season" not in df.columns or df.empty
            or not df["season"].astype(str).eq(SEASON).all()):
        return None
    return (set(df["id"].astype(int)), float(df["bank"].iloc[0]),
            int(df["free_transfers"].iloc[0]))


def _persist(squad: pd.DataFrame, bank: float, ft: int) -> None:
    """Save GW snapshot (squad + bank + FT) for next run."""
    out = squad.copy()
    out["season"] = SEASON
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
    if not wc.get("available", True):
        chips.append("- **Wildcard:** spent for this half of the season")
    else:
        chips.append(f"- **Wildcard:** {'PLAY NOW' if wc['recommend'] else 'hold'}"
                     f" ({wc['n_transfers']} suggested transfers, {wc['hits']} hits)")
    lines += chips
    return "\n".join(lines) + "\n"


def _pipeline_outcome(reason: str) -> None:
    """Expose whether this invocation produced a lineup, including successful skips."""
    generated = reason == "generated"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"generated={str(generated).lower()}\nreason={reason}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"Pipeline outcome: **{reason}**.\n\n")
    print(f"[pipeline] {reason}")


def main() -> None:
    """End-to-end weekly pipeline. Invoked by GH Actions workflow."""
    fixtures_path = DATA_DIR / "fixtures.csv"
    if fixtures_path.exists():
        try:
            cached_fixtures = pd.read_csv(fixtures_path)
        except Exception:
            cached_fixtures = None
        if cached_fixtures is not None and _season_complete(cached_fixtures):
            _pipeline_outcome("season_complete")
            print("[main] Season complete in cached fixtures — stop scheduled pipeline.")
            return

    refresh_data()

    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    if _season_complete(fixtures):
        _pipeline_outcome("season_complete")
        print("[main] Season complete after refresh — stop scheduled pipeline.")
        return

    if _last_completed_gw(fixtures) > _last_finished_gw(fixtures):
        _pipeline_outcome("review_pending")
        print("[main] Latest GW is awaiting official score review — defer models and lineup.")
        return

    if _gw_in_play(fixtures):
        _pipeline_outcome("in_play")
        print("[main] GW in play — skip pipeline. "
              "No actionable transfer/lineup work between deadline and final whistle.")
        return

    engine = prepare_engine(fixtures, history, players, teams)
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

    decision = decide_chips(
        proj, fixtures, gw, squad_ids, xi_ids, horizon=HORIZON,
        transfers_in=ins, hits=hits,
        free_transfers=None if prior is None else ft,
    )
    if decision.played:
        print(f"[chips] committed {decision.played} at GW{gw}")

    md = _render(gw, squad, xi_ids, cap, vice, bank, hits, ins, outs,
                 players, teams, decision.tc, decision.bb, decision.fh, decision.wc)
    (OUT_DIR / "lineup.md").write_text(md, encoding="utf-8")

    snap = squad.copy()
    snap["in_xi"] = snap["id"].isin(xi_ids).astype(int)
    snap["is_captain"] = (snap["id"] == cap).astype(int)
    snap["is_vice"] = (snap["id"] == vice).astype(int)
    _persist(snap, bank, decision.next_ft)
    _pipeline_outcome("generated")


if __name__ == "__main__":
    main()
