import os
import pandas as pd
from fpl_engine import FPLEngine

# File paths
BASE = "data/processed"
FIXTURES = os.path.join(BASE, "fpl_fixtures.csv")
HISTORY = os.path.join(BASE, "fpl_gameweek_history.csv")
PLAYERS = os.path.join(BASE, "fpl_players_summary.csv")
OUTPUT = os.path.join(BASE, "optimal_squad_live.csv")


def print_team(starters, bench, cap_id, vc_id):
    """Display formatted team sheet."""
    line = "─" * 80
    header = f"{'POS':<6} {'PLAYER':<24} {'TEAM':<14} {'PRICE':<8} {'XP':<8}"

    print(f"\n{line}\n{' ' * 25}FPL Squad Selection using ML Predictions\n{line}")
    print(f"\nSTARTING XI\n{line}\n{header} {'ROLE':<8}\n{line}")

    for _, p in starters.iterrows():
        role = "CAPTAIN" if p['id'] == cap_id else "VICE-CAP" if p['id'] == vc_id else ""
        print(f"{p['position']:<6} {p['name']:<24} {p['team']:<14} £{p['price']:<7} {p['next_gw_xp']:<7.1f} {role}")

    print(f"\nSUBSTITUTES\n{line}\n{header}\n{line}")
    for _, p in bench.iterrows():
        print(f"{p['position']:<6} {p['name']:<24} {p['team']:<14} £{p['price']:<7} {p['next_gw_xp']:<7.1f}")
    print(line)


def main():
    if not os.path.exists(FIXTURES):
        return print("Error: Data files not found.")

    # Find next gameweek
    fixtures = pd.read_csv(FIXTURES)
    next_gw = fixtures[~fixtures['finished']]['event'].min()
    if pd.isna(next_gw):
        return

    # Run prediction pipeline
    engine = FPLEngine(FIXTURES, HISTORY, PLAYERS)
    preds = engine.train_and_predict(next_gw, horizon=3)
    if preds.empty:
        return

    squad = engine.optimize_squad(preds)
    if squad.empty:
        return

    # Format and display
    team_result = engine.pick_team_sheet(squad)
    if team_result is None:
        return
    starters, bench, cap_id, vc_id = team_result
    starters_fmt = engine.format_squad(starters).assign(id=starters['id'].values)
    bench_fmt = engine.format_squad(bench).assign(id=bench['id'].values)

    print_team(starters_fmt, bench_fmt, cap_id, vc_id)
    engine.format_squad(squad).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()