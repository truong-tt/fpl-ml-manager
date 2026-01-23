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
    """Display formatted team sheet with clean visuals."""
    print("\n" + "─" * 80)
    print(" " * 25 + "FPL Squad Selection using ML Optimization")
    print("─" * 80)
    
    print("\nSTARTING XI (11 Players)")
    print("─" * 80)
    print(f"{'POS':<6} {'PLAYER':<24} {'TEAM':<14} {'PRICE':<8} {'XP':<8} {'ROLE':<8}")
    print("─" * 80)
    
    for _, p in starters.iterrows():
        role = "CAPTAIN" if p['id'] == cap_id else "VICE-CAPTAIN" if p['id'] == vc_id else ""
        print(f"{p['position']:<6} {p['name']:<24} {p['team']:<14} £{p['price']:<7} {p['next_gw_xp']:<7.1f} {role:<8}")
    
    print("\nSUBSTITUTES (4 Players)")
    print("─" * 80)
    print(f"{'POS':<6} {'PLAYER':<24} {'TEAM':<14} {'PRICE':<8} {'XP':<8}")
    print("─" * 80)
    
    for _, p in bench.iterrows():
        print(f"{p['position']:<6} {p['name']:<24} {p['team']:<14} £{p['price']:<7} {p['next_gw_xp']:<7.1f}")
    
    print("─" * 80 + "\n")


def main():
    # Check data exists
    if not os.path.exists(FIXTURES):
        print("Error: Data files not found.")
        return

    # Find next gameweek
    fixtures = pd.read_csv(FIXTURES)
    next_gw = fixtures[~fixtures['finished']]['event'].min()
    
    if pd.isna(next_gw):
        return
    
    # Initialize engine and predict
    engine = FPLEngine(FIXTURES, HISTORY, PLAYERS)
    
    preds = engine.train_and_predict(next_gw, horizon=3)
    if preds.empty:
        return

    squad = engine.optimize_squad(preds)
    if squad.empty:
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


if __name__ == "__main__":
    main()