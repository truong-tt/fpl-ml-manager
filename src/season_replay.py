"""Season replay: simulate model's GW-by-GW manager decisions and score
actual FPL points against the chosen XI + captain.

Walk-forward by default. At each GW G the engine sees only history with
`round < G` in the current season (full prior seasons retained). Models
are NOT re-fit per GW — uses the production points/match/minutes/bonus
heads trained on full data. This is *partly leaky*: the booster has
seen rounds >= G, so projected μ for GW G inherits some of the future
in its parameters even if the rolling features are filtered. Treat the
total as an upper bound on a clean walk-forward season.

Output: data/processed/season_replay.md plus a per-GW CSV.

CLI:
    python src/season_replay.py --start 1 [--end 36] [--budget 100]
"""
from __future__ import annotations

import argparse
import builtins
import functools
from pathlib import Path

import pandas as pd

# CI captures stdout via pipe → Python buffers by default and the log appears
# empty until the run ends. Force every replay print to flush.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

from data_loader import SEASON
from fpl_engine import FPLEngine
from optimizer import solve_initial_squad, solve_rhc_transfers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"
HORIZON = 8


def _filter_history(history: pd.DataFrame, season: str, before_gw: int) -> pd.DataFrame:
    """Keep all prior seasons + current-season rows with `round < before_gw`.

    GW1 edge case: current-season slice is empty. FPLEngine._latest_rolling
    filters to SEASON before tail(1), so it would return an empty baseline
    and projections would be empty. Graft the most recent per-player row
    from prior seasons, relabelled `season=SEASON`, so the booster sees a
    cross-season carry-forward baseline (same handling fans expect for
    new-season GW1: lean on last-season's xG/xA, accept transfer noise).
    """
    if "season" not in history.columns:
        return history[history["round"] < before_gw]
    cur = history[(history["season"] == season) & (history["round"] < before_gw)]
    prior = history[history["season"] != season]
    if cur.empty and not prior.empty:
        last = (prior.sort_values(["player_id", "season", "round"])
                     .groupby("player_id").tail(1).copy())
        last["season"] = season
        return pd.concat([prior, last], ignore_index=True)
    return pd.concat([prior, cur], ignore_index=True)


def _last_finished_gw(fixtures: pd.DataFrame, season: str) -> int:
    fx = fixtures[fixtures.get("season", season) == season]
    fin = fx["finished"].astype(str).str.lower().isin(("true", "1"))
    done = fx[fin]
    return int(done["event"].max()) if not done.empty else 0


def _score_xi(history: pd.DataFrame, season: str, gw: int,
              xi: set[int], captain: int, vice: int) -> dict[str, float]:
    """Sum actual FPL points for chosen XI. Captain doubled if played, else vice."""
    actual = history[(history["season"] == season) & (history["round"] == gw)]
    actual = actual.groupby("player_id", as_index=False)["total_points"].sum()
    pts_map = dict(zip(actual["player_id"].astype(int), actual["total_points"].astype(float)))
    cap_pts = pts_map.get(int(captain), 0.0)
    if cap_pts <= 0.0 and pts_map.get(int(vice), 0.0) > 0.0:
        cap_id, cap_pts = int(vice), pts_map[int(vice)]
    else:
        cap_id = int(captain)
    xi_pts = sum(pts_map.get(int(i), 0.0) for i in xi)
    return {"xi_pts": xi_pts, "cap_id": cap_id, "cap_pts": cap_pts}


def replay(start_gw: int = 1, end_gw: int | None = None,
           season: str = SEASON, budget: float = 100.0) -> pd.DataFrame:
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    if end_gw is None:
        end_gw = _last_finished_gw(fixtures, season)
    if end_gw < start_gw:
        raise RuntimeError(f"no finished GWs in season {season}")

    print(f"[replay] season={season} GW{start_gw}..{end_gw} horizon={HORIZON}")

    prior_squad: set[int] | None = None
    bank = budget
    ft = 1
    rows: list[dict] = []

    for G in range(start_gw, end_gw + 1):
        hist_pre = _filter_history(history, season, G)
        engine = FPLEngine(fixtures, hist_pre, players, teams)
        proj = engine.build_projections(current_gw=G, horizon=HORIZON)
        if proj.empty:
            print(f"GW{G}: empty projection — skip")
            continue

        if prior_squad is None:
            sq = solve_initial_squad(proj, budget=budget)
            if sq.empty:
                print(f"GW{G}: initial squad solve failed")
                continue
            squad_ids = set(sq["id"].astype(int))
            xi_ids = set(sq[sq["in_xi"] == 1]["id"].astype(int))
            cap = int(sq[sq["is_captain"] == 1]["id"].iloc[0])
            vice = int(sq[sq["is_vice"] == 1]["id"].iloc[0])
            squad_val = float(proj[proj["id"].isin(squad_ids)]["price"].sum())
            bank = round(budget - squad_val, 1)
            hits = 0
            n_in = 0
        else:
            res = solve_rhc_transfers(proj, prior_squad, bank, ft)
            if res.get("status") != "ok":
                print(f"GW{G}: RHC status={res.get('status')}")
                continue
            squad_ids = set(res["squad_ids"])
            xi_ids = set(res["xi_ids"])
            cap = int(res["captain"])
            vice = int(res["vice"])
            hits = int(res["hits"])
            n_in = len(res["transfers_in"])
            squad_val = float(proj[proj["id"].isin(squad_ids)]["price"].sum())
            bank = round(budget - squad_val, 1)

        score = _score_xi(history, season, G, xi_ids, cap, vice)
        gw_total = score["xi_pts"] + score["cap_pts"] - 4 * hits

        rows.append({
            "gw": G,
            "xi_pts": round(score["xi_pts"], 1),
            "cap_id": int(score["cap_id"]),
            "cap_pts": round(score["cap_pts"], 1),
            "hits": hits,
            "transfers_in": n_in,
            "gw_total": round(gw_total, 1),
            "bank": bank,
        })
        print(f"GW{G:2d}: xi={score['xi_pts']:5.1f} cap={score['cap_pts']:4.1f}*2 "
              f"hits=-{4 * hits:2d} -> {gw_total:5.1f}  in={n_in} bank={bank}")

        prior_squad = squad_ids
        ft = min(5, ft + 1) if n_in == 0 else 1

    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, season: str) -> str:
    if df.empty:
        return f"# Season Replay {season}\n\nNo GWs replayed.\n"
    cum = df["gw_total"].cumsum().round(1)
    df = df.assign(cum_total=cum)
    avg = df["gw_total"].mean()
    lines = [
        f"# Season Replay — {season}",
        "",
        f"- **GWs replayed:** {len(df)}",
        f"- **Total points:** {df['gw_total'].sum():.0f}",
        f"- **Avg per GW:** {avg:.1f}",
        f"- **Hits taken:** {df['hits'].sum()} ({-4 * df['hits'].sum()} pts)",
        "",
        "## Per-GW",
        "",
        "| GW | XI | Cap×2 | Hits | In | Total | Cumulative | Bank |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r.gw)} | {r.xi_pts:.1f} | {r.cap_pts:.1f} | "
            f"{-4 * int(r.hits):d} | {int(r.transfers_in)} | "
            f"{r.gw_total:.1f} | {r.cum_total:.1f} | £{r.bank:.1f} |"
        )
    lines += ["",
              "> **Note:** Production models trained on full season fit each GW's "
              "rolling state, so the booster's parameters have already seen rounds "
              ">= G even when the per-GW feature row is filtered to history "
              "< G. Treat the total as an upper bound on a strict walk-forward run."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--season", type=str, default=SEASON)
    p.add_argument("--budget", type=float, default=100.0)
    args = p.parse_args()

    df = replay(args.start, args.end, args.season, args.budget)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "season_replay.csv"
    md_path = OUT_DIR / "season_replay.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(render_report(df, args.season), encoding="utf-8")
    print(f"\n[replay] wrote {csv_path} + {md_path}")
    if not df.empty:
        print(f"[replay] total: {df['gw_total'].sum():.0f} pts over {len(df)} GWs "
              f"(avg {df['gw_total'].mean():.1f}/GW)")


if __name__ == "__main__":
    main()
