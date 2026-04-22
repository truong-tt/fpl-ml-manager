"""GH Actions entrypoint: refresh data, train missing models, solve, write lineup.md."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from chips import (recommend_bench_boost, recommend_free_hit,
                   recommend_triple_captain, recommend_wildcard)
from data_loader import main as refresh_data
from fpl_engine import FPLEngine
from optimizer import solve_initial_squad, solve_rhc_transfers
from train_match_model import train_match_models
from train_points_model import train_points_models

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
LAMBDA_VAR, LAMBDA_EO, BENCH_WEIGHT = 0.02, 0.0, 0.15
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _md_table(df: pd.DataFrame) -> str:
    """Renders df as a GitHub-flavored markdown table (no tabulate dep)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _current_gw(fixtures: pd.DataFrame) -> int:
    """Next unfinished gameweek; defaults to 38 if season complete."""
    up = fixtures[~fixtures["finished"]]
    return int(up["event"].min()) if not up.empty else 38


def _ensure_models(fx: pd.DataFrame, hist: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Trains any missing match / points model artifacts."""
    if not all((DATA_DIR / f).exists() for f in ("xgb_home_goals.json", "xgb_away_goals.json")):
        train_match_models(fx, hist, teams)
    if not all((DATA_DIR / f"xgb_points_q{q:02d}.json").exists() for q in (10, 50, 90)):
        train_points_models()


def _load_prior() -> tuple[set[int], float, int] | None:
    """Reads last week's squad snapshot for RHC, or None on cold start."""
    snap = OUT_DIR / "squad_snapshot.csv"
    if not snap.exists():
        return None
    df = pd.read_csv(snap)
    return (set(df["id"].astype(int)), float(df["bank"].iloc[0]),
            int(df["free_transfers"].iloc[0]))


def _persist(squad: pd.DataFrame, bank: float, ft: int) -> None:
    """Saves this GW's snapshot (squad + bank + FT) for next run."""
    out = squad.copy()
    out["bank"], out["free_transfers"] = bank, ft
    out.to_csv(OUT_DIR / "squad_snapshot.csv", index=False)


def _render(
    gw: int, squad: pd.DataFrame, xi: set[int], cap: int, vice: int,
    bank: float, hits: int, ins: list[int], outs: list[int],
    players: pd.DataFrame, teams: pd.DataFrame,
    tc: dict, bb: dict, fh: dict, wc: dict,
) -> str:
    """Renders the weekly lineup + transfers + chips as a single markdown document."""
    tmap = teams.set_index("team_id")["short_name"].to_dict()
    nmap = players.set_index("id")["web_name"].to_dict()

    d = squad.copy()
    d["Team"] = d["team_id"].map(tmap)
    d["Pos"] = d["pos_id"].map(POS)
    d["Price"] = d["price"].round(1)
    d["XP(1)"] = d["next_gw_xp"].round(2)
    d["XP(H)"] = d["horizon_xp"].round(2)
    d["Role"] = ""
    d.loc[d["id"] == cap, "Role"] = "(C)"
    d.loc[d["id"] == vice, "Role"] = "(VC)"
    d = d.rename(columns={"name": "Name"})
    cols = ["Name", "Team", "Pos", "Price", "XP(1)", "XP(H)", "Role"]

    xi_df = d[d["id"].isin(xi)].sort_values("pos_id")[cols]
    bench_df = d[~d["id"].isin(xi)].sort_values("pos_id")[cols]

    lines = [
        f"# GW{gw} Lineup", "",
        f"- **Bank:** £{bank:.1f}m",
        f"- **Hits:** -{hits * 4} pts" if hits else "- **Hits:** 0",
        f"- **Squad Value:** £{squad['price'].sum():.1f}m",
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
    """End-to-end weekly pipeline invoked by the GH Actions workflow."""
    refresh_data()

    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    _ensure_models(fixtures, history, teams)
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