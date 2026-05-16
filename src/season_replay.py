"""Season replay: simulate model's GW-by-GW manager decisions and score
actual FPL points against the chosen XI + captain.

Walk-forward by default. At each GW G the engine sees only history with
`round < G` in the current season (full prior seasons retained). Models
are NOT re-fit per GW — uses the production points/match/minutes/bonus
heads trained on full data. This is *partly leaky*: the booster has
seen rounds >= G, so projected μ for GW G inherits some of the future
in its parameters even if the rolling features are filtered. Treat the
total as an upper bound on a clean walk-forward season.

All four chips are simulated: TC, BB, WC, FH. WC re-runs the cold-start
solver with current cash (unlimited free transfers); FH re-runs it on a
single-GW projection slice and the squad reverts the next GW. WC + FH
alt-solves are gated by an attempt-window list per half plus the
half-deadline force-fire to keep the replay loop's cost bounded. At
most one chip fires per GW (FPL rule). When multiple chips are eligible
the highest-uplift candidate wins.

Output:
- data/processed/season_replay.md — human-readable per-GW table.
- data/season_replay.csv — structured per-GW state. Lives in data/ (not
  processed/) because the CI gate in .github/workflows/season_replay.yml
  reads it to compute `last_replayed`. processed/ reserved for outputs
  consumed by humans.

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

# FPL 2025/26 chip rules. Two of each chip per season; first half (GW1..19)
# uses set 1, second half (GW20..38) uses set 2. Set 1 expires at GW19 if
# unused. FH cannot be played in consecutive GWs. TC = captain pts x3
# (instead of x2). BB = bench pts also count.
TC_TRIGGER_PREMIUM = 4.5   # use TC when (q90_cap - mu_cap) >= this
BB_TRIGGER_BENCH_EV = 10.0  # use BB when bench mu sum >= this
WC_TRIGGER = 8.0           # use WC when horizon-discounted XI EV uplift >= this
FH_TRIGGER = 6.0           # use FH when single-GW XI EV uplift >= this
WC_FH_TIME_LIMIT = 30      # CBC seconds for WC/FH alt-solves (vs 60 default)
HALF1_END = 19
HALF2_END = 38

# Window of GWs at which we attempt a WC/FH alt-solve. Keeps the replay loop
# bounded — each alt-solve is a full cold-start MILP. The half-deadline GWs
# are always added at runtime to guarantee force-fire still works.
WC_ATTEMPT_GWS = {1: {8, 12, 16, 19}, 2: {26, 30, 34, 38}}
FH_ATTEMPT_GWS = {1: {6, 8, 14, 18, 19}, 2: {26, 28, 33, 36, 38}}


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
    """Max event N where EVERY fixture row with event==N is finished.

    Strict "fully-finished" rule. Handles DGW (>10 fixtures for one event) and
    BGW (<10) naturally. Must match the GW detection in
    .github/workflows/season_replay.yml — otherwise the workflow skips runs
    forever because the CSV's last_replayed gets pinned to a partial GW that
    the workflow never reaches with its stricter rule.
    """
    fx = fixtures[fixtures.get("season", season) == season].copy()
    if fx.empty:
        return 0
    fx["fin"] = fx["finished"].astype(str).str.lower().isin(("true", "1"))
    by_event = fx.groupby("event")["fin"].all()
    done = by_event[by_event].index.astype(int)
    return int(done.max()) if len(done) else 0


def _one_gw_proj(proj: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Slice projection to a single GW. Drops `xp_*`, `cap_xp_*`, `var_*` cols
    for any other GW so `_gws()` collapses to `[gw]` and the optimizer solves a
    one-GW lineup. Used for FH simulation where the temp squad lives one GW
    only and reverts. All non-GW columns (price/pos_id/team_id/eo) preserved."""
    keep = {f"xp_{gw}", f"cap_xp_{gw}", f"var_{gw}"}
    drop = [c for c in proj.columns
            if c.startswith(("xp_", "cap_xp_", "var_")) and c not in keep]
    return proj.drop(columns=drop)


def _xi_one_gw_ev(proj: pd.DataFrame, xi_ids: set[int], cap_id: int, gw: int) -> float:
    """Sum of XI's `xp_{gw}` plus captain bonus (`cap_xp - xp` for the picked
    captain). Used as one-GW EV proxy for FH uplift trigger."""
    if not xi_ids:
        return 0.0
    xp_col = f"xp_{gw}"
    cap_col = f"cap_xp_{gw}"
    if xp_col not in proj.columns:
        return 0.0
    s = float(proj.loc[proj["id"].isin(xi_ids), xp_col].sum())
    if cap_col in proj.columns and cap_id in xi_ids:
        cap_row = proj.loc[proj["id"] == cap_id]
        if not cap_row.empty:
            s += float(cap_row[cap_col].iloc[0]) - float(cap_row[xp_col].iloc[0])
    return s


def _xi_horizon_ev(proj: pd.DataFrame, xi_ids: set[int], cap_id: int) -> float:
    """Geometric-discounted (γ=0.85) horizon sum of XI EV with captain bonus.
    Matches optimizer's `RHC_DISCOUNT`. Approximation: assumes XI fixed across
    horizon (real RHC rotates per GW). OK as trigger proxy — both branches
    (current and WC) use the same approximation, so the *difference* is
    structurally fair."""
    gws = sorted(int(c.split("_")[1]) for c in proj.columns if c.startswith("xp_"))
    s = 0.0
    for k, t in enumerate(gws):
        w = 0.85 ** k
        s += w * _xi_one_gw_ev(proj, xi_ids, cap_id, t)
    return s


def _score_xi(history: pd.DataFrame, season: str, gw: int,
              xi: set[int], bench: set[int], captain: int, vice: int,
              tc_active: bool = False, bb_active: bool = False
              ) -> dict[str, float]:
    """Sum actual FPL points for chosen XI. Captain doubled if played, else vice.

    tc_active: triple captain this GW (multiplier 3 instead of 2).
    bb_active: bench points also count.
    """
    actual = history[(history["season"] == season) & (history["round"] == gw)]
    actual = actual.groupby("player_id", as_index=False)["total_points"].sum()
    pts_map = dict(zip(actual["player_id"].astype(int), actual["total_points"].astype(float)))
    cap_base = pts_map.get(int(captain), 0.0)
    if cap_base <= 0.0 and pts_map.get(int(vice), 0.0) > 0.0:
        cap_id, cap_base = int(vice), pts_map[int(vice)]
    else:
        cap_id = int(captain)
    cap_mult = 3 if tc_active else 2
    cap_extra = cap_base * (cap_mult - 1)
    xi_pts = sum(pts_map.get(int(i), 0.0) for i in xi)
    bench_pts = sum(pts_map.get(int(i), 0.0) for i in bench) if bb_active else 0.0
    return {
        "xi_pts": xi_pts, "cap_id": cap_id, "cap_pts": cap_extra,
        "cap_base": cap_base, "cap_mult": cap_mult,
        "bench_pts": bench_pts,
    }


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
    # Chip inventory. Key = chip-half token. True = available.
    # FH non-consecutive: tracked via last_fh_gw.
    chips = {"tc1": True, "tc2": True, "bb1": True, "bb2": True,
             "fh1": True, "fh2": True, "wc1": True, "wc2": True}
    last_fh_gw = -10

    # Effective deadlines respect both FPL's GW19/GW38 cutoffs AND the replay
    # window. Without this, a replay ending before HALF2_END (e.g. mid-season
    # at last_finished_gw=27) never force-fires set-2 chips → TC2/WC2/FH2/BB2
    # stay unused. We force on the last GW the replay actually visits within
    # each half so every available chip gets a fair chance to fire.
    half1_deadline = min(HALF1_END, end_gw) if start_gw <= HALF1_END else None
    half2_deadline = min(HALF2_END, end_gw) if end_gw > HALF1_END else None
    half2_open_gw = max(start_gw, HALF1_END + 1)

    # Always extend WC/FH attempt windows with their respective deadlines so a
    # force fires the alt-solve even when the deadline GW isn't in the static
    # WC_ATTEMPT_GWS / FH_ATTEMPT_GWS sets above.
    wc_windows = {1: set(WC_ATTEMPT_GWS.get(1, set())),
                  2: set(WC_ATTEMPT_GWS.get(2, set()))}
    fh_windows = {1: set(FH_ATTEMPT_GWS.get(1, set())),
                  2: set(FH_ATTEMPT_GWS.get(2, set()))}
    if half1_deadline is not None:
        wc_windows[1].add(half1_deadline)
        fh_windows[1].add(half1_deadline)
    if half2_deadline is not None and half2_deadline >= half2_open_gw:
        wc_windows[2].add(half2_deadline)
        fh_windows[2].add(half2_deadline)

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

        # Chip decision. FPL rule: at most 1 chip per GW. Pick highest-uplift
        # candidate above its trigger; force-fire one at the half-deadline so
        # unused chips don't expire. Deadlines clamped to the replay window so
        # mid-season replays still force-fire set-2 chips at end_gw.
        half = 1 if G <= HALF1_END else 2
        force_use_half1 = (half1_deadline is not None and G == half1_deadline)
        force_use_half2 = (half2_deadline is not None and G == half2_deadline)
        force = force_use_half1 or force_use_half2
        bench_ids = squad_ids - xi_ids

        # TC: captain upside premium = q90_cap - mu_cap.
        gw_col_q90 = f"cap_xp_{G}" if f"cap_xp_{G}" in proj.columns else f"xp_{G}"
        cap_q90 = float(proj.loc[proj["id"] == cap, gw_col_q90].iloc[0])
        cap_mu = float(proj.loc[proj["id"] == cap, f"xp_{G}"].iloc[0])
        tc_premium = cap_q90 - cap_mu
        # BB: bench mu sum.
        bench_ev = float(proj.loc[proj["id"].isin(bench_ids), f"xp_{G}"].sum())

        tc_key, bb_key = f"tc{half}", f"bb{half}"
        wc_key, fh_key = f"wc{half}", f"fh{half}"

        # WC alt-solve. Only at attempt-window GWs (or at force) and only when
        # chip available. Re-solves cold-start with current cash; full transfer
        # freedom modelled by treating squad as fresh.
        wc_uplift = float("-inf")
        wc_squad_ids: set[int] | None = None
        wc_xi_ids: set[int] | None = None
        wc_cap_id: int | None = None
        wc_vice_id: int | None = None
        try_wc = (chips.get(wc_key, False) and prior_squad is not None
                  and (G in wc_windows.get(half, set()) or force))
        if try_wc:
            wc_df = solve_initial_squad(proj, budget=squad_val + bank,
                                        time_limit=WC_FH_TIME_LIMIT)
            if not wc_df.empty:
                wc_squad_ids = set(wc_df["id"].astype(int))
                wc_xi_ids = set(wc_df[wc_df["in_xi"] == 1]["id"].astype(int))
                wc_cap_id = int(wc_df[wc_df["is_captain"] == 1]["id"].iloc[0])
                wc_vice_id = int(wc_df[wc_df["is_vice"] == 1]["id"].iloc[0])
                wc_uplift = (_xi_horizon_ev(proj, wc_xi_ids, wc_cap_id)
                             - _xi_horizon_ev(proj, xi_ids, cap))

        # FH alt-solve. One-GW slice; squad reverts after this GW.
        fh_uplift = float("-inf")
        fh_xi_ids: set[int] | None = None
        fh_cap_id: int | None = None
        fh_vice_id: int | None = None
        try_fh = (chips.get(fh_key, False) and prior_squad is not None
                  and (G - last_fh_gw) > 1
                  and (G in fh_windows.get(half, set()) or force))
        if try_fh:
            fh_df = solve_initial_squad(_one_gw_proj(proj, G),
                                        budget=squad_val + bank,
                                        time_limit=WC_FH_TIME_LIMIT)
            if not fh_df.empty:
                fh_xi_ids = set(fh_df[fh_df["in_xi"] == 1]["id"].astype(int))
                fh_cap_id = int(fh_df[fh_df["is_captain"] == 1]["id"].iloc[0])
                fh_vice_id = int(fh_df[fh_df["is_vice"] == 1]["id"].iloc[0])
                fh_uplift = (_xi_one_gw_ev(proj, fh_xi_ids, fh_cap_id, G)
                             - _xi_one_gw_ev(proj, xi_ids, cap, G))

        # Candidate list: (chip_token, uplift, met_trigger_flag).
        cands = []
        if chips.get(tc_key, False):
            cands.append(("tc", tc_premium, tc_premium >= TC_TRIGGER_PREMIUM))
        if chips.get(bb_key, False):
            cands.append(("bb", bench_ev, bench_ev >= BB_TRIGGER_BENCH_EV))
        if chips.get(wc_key, False) and wc_uplift > float("-inf"):
            cands.append(("wc", wc_uplift, wc_uplift >= WC_TRIGGER))
        if chips.get(fh_key, False) and fh_uplift > float("-inf"):
            cands.append(("fh", fh_uplift, fh_uplift >= FH_TRIGGER))

        chip_choice: str | None = None
        if force and cands:
            # Pick best regardless of trigger so unused chips don't expire.
            chip_choice = max(cands, key=lambda c: c[1])[0]
        else:
            triggered = [c for c in cands if c[2]]
            if triggered:
                chip_choice = max(triggered, key=lambda c: c[1])[0]

        tc_active = chip_choice == "tc"
        bb_active = chip_choice == "bb"
        wc_active = chip_choice == "wc"
        fh_active = chip_choice == "fh"

        # WC: replace squad path going forward.
        if wc_active and wc_squad_ids is not None:
            squad_ids = wc_squad_ids
            xi_ids = wc_xi_ids or xi_ids
            cap = wc_cap_id if wc_cap_id is not None else cap
            vice = wc_vice_id if wc_vice_id is not None else vice
            n_in = len(squad_ids - (prior_squad or set()))
            hits = 0
            squad_val = float(proj[proj["id"].isin(squad_ids)]["price"].sum())
            bank = round(budget - squad_val, 1)
            bench_ids = squad_ids - xi_ids
            chips[wc_key] = False

        # FH: temp XI for scoring; squad/bank/ft revert.
        score_xi_ids = xi_ids
        score_bench_ids = bench_ids
        score_cap = cap
        score_vice = vice
        if fh_active and fh_xi_ids is not None:
            score_xi_ids = fh_xi_ids
            score_bench_ids = set()  # FH bench irrelevant; no BB stack.
            score_cap = fh_cap_id if fh_cap_id is not None else cap
            score_vice = fh_vice_id if fh_vice_id is not None else vice
            chips[fh_key] = False
            last_fh_gw = G

        if tc_active:
            chips[tc_key] = False
        if bb_active:
            chips[bb_key] = False

        score = _score_xi(history, season, G, score_xi_ids, score_bench_ids,
                          score_cap, score_vice,
                          tc_active=tc_active, bb_active=bb_active)
        gw_total = (score["xi_pts"] + score["cap_pts"] + score["bench_pts"]
                    - 4 * hits)

        chip_tag = chip_choice.upper() if chip_choice else ""

        rows.append({
            "gw": G,
            "xi_pts": round(score["xi_pts"], 1),
            "cap_id": int(score["cap_id"]),
            "cap_pts": round(score["cap_pts"], 1),
            "bench_pts": round(score["bench_pts"], 1),
            "chip": chip_tag,
            "hits": hits,
            "transfers_in": n_in,
            "gw_total": round(gw_total, 1),
            "bank": bank,
        })
        cap_note = f"x{score['cap_mult']}" if tc_active else "x2"
        bb_note = f"+bb{score['bench_pts']:.0f}" if bb_active else ""
        print(f"GW{G:2d}: xi={score['xi_pts']:5.1f} "
              f"cap={score['cap_base']:4.1f}{cap_note} "
              f"{bb_note} hits=-{4 * hits:2d} -> {gw_total:5.1f}  "
              f"in={n_in} bank={bank} {chip_tag}")

        # FH does not change the persistent squad / bank / FT.
        if not fh_active:
            prior_squad = squad_ids
            # WC resets FT to 1 next GW (FPL rule). Otherwise standard banking.
            if wc_active:
                ft = 1
            else:
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
        "| GW | XI | Cap+ | Bench | Chip | Hits | In | Total | Cumulative | Bank |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r.gw)} | {r.xi_pts:.1f} | {r.cap_pts:.1f} | "
            f"{r.get('bench_pts', 0.0):.1f} | {r.get('chip', '') or '-'} | "
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
    csv_path = DATA_DIR / "season_replay.csv"
    md_path = OUT_DIR / "season_replay.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(render_report(df, args.season), encoding="utf-8")
    print(f"\n[replay] wrote {csv_path} + {md_path}")
    if not df.empty:
        print(f"[replay] total: {df['gw_total'].sum():.0f} pts over {len(df)} GWs "
              f"(avg {df['gw_total'].mean():.1f}/GW)")


if __name__ == "__main__":
    main()
