from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import main as update_data
from fpl_engine import FPLEngine

DATA_DIR = Path("data")
FIXTURES_PATH = DATA_DIR / "fixtures.csv"
HISTORY_PATH = DATA_DIR / "history.csv"
PLAYERS_PATH = DATA_DIR / "players.csv"
TEAMS_PATH = DATA_DIR / "teams.csv"


def get_current_gw(fixtures_df: pd.DataFrame) -> int:
    """Get next upcoming Gameweek."""
    upcoming = fixtures_df[~fixtures_df['finished']]
    return 38 if upcoming.empty else int(upcoming.iloc[0]['event'])


def main() -> None:
    print("\nAI FPL Manager")

    if PLAYERS_PATH.exists() and HISTORY_PATH.exists() and FIXTURES_PATH.exists():
        print("Data found")
    else:
        print("Downloading data...")
        try:
            update_data()
        except Exception as e:
            print(f"Failed: {e}")
            return

    try:
        fixtures = pd.read_csv(FIXTURES_PATH)
        history = pd.read_csv(HISTORY_PATH)
        players = pd.read_csv(PLAYERS_PATH)
        teams = pd.read_csv(TEAMS_PATH) if TEAMS_PATH.exists() else None
    except Exception as e:
        print(f"Load error: {e}")
        return

    engine = FPLEngine(fixtures, history, players, teams_df=teams)
    current_gw = get_current_gw(fixtures)
    print(f"Training for GW {current_gw}...")
    projections = engine.train_and_predict(current_gw, horizon=5)

    if projections.empty:
        print("No projections")
        return

    print("Optimizing squad...")
    best_squad = engine.optimize_squad(projections, budget=100.0)

    if best_squad.empty:
        print("Optimization failed")
        return

    starting_xi, bench, cap, vice = engine.pick_team_sheet(best_squad)
    engine.display_squad(starting_xi, bench, cap, vice)

    print("\nTransfers")
    transfer_rec = engine.recommend_transfers(best_squad, bank=0.5, free_transfers=1)
    engine.print_transfer_recommendation(transfer_rec)


if __name__ == "__main__":
    main()