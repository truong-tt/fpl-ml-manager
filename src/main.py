import os
import pandas as pd
from fpl_engine import FPLEngine

# Paths
BASE = "data/processed"
FIXTURES = os.path.join(BASE, "fpl_fixtures.csv")
HISTORY = os.path.join(BASE, "fpl_gameweek_history.csv")
PLAYERS = os.path.join(BASE, "fpl_players_summary.csv")
OUTPUT_SQUAD = os.path.join(BASE, "optimal_squad_live.csv")


def print_team(starters, bench, cap_id, vc_id):
    """Display formatted team sheet."""
    print("\n" + "="*70)
    print(f"{'POS':<5} {'PLAYER':<25} {'TEAM':<15} {'PRICE':<6} {'XP'}")
    print("="*70)
    
    print("\n--- STARTING XI ---")
    for _, p in starters.iterrows():
        status = "(C)" if p['id'] == cap_id else "(V)" if p['id'] == vc_id else ""
        name = f"{p['name']} {status}".strip()
        print(f"{p['position']:<5} {name:<25} {p['team']:<15} £{p['price']:<5} {p['next_gw_xp']}")

    print("\n--- BENCH ---")
    for _, p in bench.iterrows():
        print(f"{p['position']:<5} {p['name']:<25} {p['team']:<15} £{p['price']:<5} {p['next_gw_xp']}")
    
    print("="*70)


def main():
    print("\n=== FPL AI MANAGER ===")
    
    # Check data exists
    if not os.path.exists(FIXTURES):
        print("Error: Data not found. Run data_loader.py first.")
        return

    # Find next gameweek
    fixtures = pd.read_csv(FIXTURES)
    next_gw = fixtures[~fixtures['finished']]['event'].min()
    
    if pd.isna(next_gw):
        print("Error: No upcoming fixtures.")
        return
    
    print(f"Targeting GW{int(next_gw)}")
    
    # Initialize engine and predict
    engine = FPLEngine(FIXTURES, HISTORY, PLAYERS)
    
    print("Predicting points...")
    preds = engine.train_and_predict(next_gw, horizon=3)
    if preds.empty:
        print("Error: Prediction failed.")
        return

    print("Optimizing squad...")
    squad = engine.optimize_squad(preds)
    if squad.empty:
        print("Error: Optimization failed.")
        return

    # Pick team and format
    starters, bench, cap_id, vc_id = engine.pick_team_sheet(squad)
    starters_fmt = engine.format_squad(starters)
    bench_fmt = engine.format_squad(bench)
    
    # Add IDs back for captain marking
    starters_fmt['id'] = starters['id'].values
    bench_fmt['id'] = bench['id'].values
    
    # Display and save
    print_team(starters_fmt, bench_fmt, cap_id, vc_id)
    
    full_squad = engine.format_squad(squad)
    full_squad.to_csv(OUTPUT_SQUAD, index=False)
    print(f"\nSquad saved to: {OUTPUT_SQUAD}")


if __name__ == "__main__":
    main()