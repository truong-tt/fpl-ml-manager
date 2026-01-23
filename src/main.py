import os
import pandas as pd
from fpl_engine import FPLEngine

# Processed data inputs/outputs
BASE = "data/processed"
FIXTURES = os.path.join(BASE, "fpl_fixtures.csv")
HISTORY = os.path.join(BASE, "fpl_gameweek_history.csv")
PLAYERS = os.path.join(BASE, "fpl_players_summary.csv")
OUTPUT_SQUAD = os.path.join(BASE, "optimal_squad_live.csv")


def print_team(starters, bench, cap_id, vc_id):
    """Pretty-print XI + bench with captain/vice."""
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    print("\n" + "=" * 60)
    print(f"{'POS':<5} {'PLAYER':<25} {'TEAM':<5} {'PRICE':<6} {'XP':<5}")
    print("=" * 60 + "\n--- STARTING XI ---")

    for _, p in starters.iterrows():
        status = "(C)" if p["id"] == cap_id else "(V)" if p["id"] == vc_id else ""
        print(
            f"{pos_map[p['pos']]:<5} {p['name']} {status:<25} "
            f"{p['team']:<5} £{p['price']:<5} {p['next_gw_xp']:.1f}"
        )

    print("\n--- BENCH ---")
    for _, p in bench.iterrows():
        print(f"{pos_map[p['pos']]:<5} {p['name']:<25} {p['team']:<5} £{p['price']:<5} {p['next_gw_xp']:.1f}")
    print("=" * 60)


def main():
    """Load data -> predict xP -> optimize squad -> pick XI."""
    print("\n=== FPL AI MANAGER ===")

    if not os.path.exists(FIXTURES):
        print("Error: Run src/data_loader.py first.")
        return

    fixtures = pd.read_csv(FIXTURES)
    next_gw = fixtures[~fixtures["finished"]]["event"].min()

    if pd.isna(next_gw):
        print("No upcoming fixtures found.")
        return

    print(f"Targeting GW: {int(next_gw)}")

    engine = FPLEngine(FIXTURES, HISTORY, PLAYERS)

    print("Training model & predicting...")
    preds = engine.train_and_predict(next_gw, horizon=3)
    if preds.empty:
        print("Error: No predictions generated.")
        return

    print("Optimizing squad...")
    squad = engine.optimize_squad(preds)
    if squad.empty:
        print("Error: Optimizer failed.")
        return

    result = engine.pick_team_sheet(squad)
    if result is None:
        print("Error: Team selection failed.")
        return

    starters, bench, cap, vc = result
    print_team(starters, bench, cap, vc)

    squad.to_csv(OUTPUT_SQUAD, index=False)
    print(f"\nSquad saved to: {OUTPUT_SQUAD}")


if __name__ == "__main__":
    main()