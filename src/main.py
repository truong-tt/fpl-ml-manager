import os
import pandas as pd
from data_loader import main as update_data
from fpl_engine import FPLEngine

DATA_DIR = "data"
FIXTURES_PATH = os.path.join(DATA_DIR, "fixtures.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
PLAYERS_PATH = os.path.join(DATA_DIR, "players.csv")

def get_current_gw(fixtures_df):
    """Get next upcoming Gameweek."""
    upcoming = fixtures_df[fixtures_df['finished'] == False]
    return 38 if upcoming.empty else upcoming.iloc[0]['event']

def main():
    print("\n=== AI FPL MANAGER ===")

    if os.path.exists(PLAYERS_PATH) and os.path.exists(HISTORY_PATH) and os.path.exists(FIXTURES_PATH):
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
    except Exception as e:
        print(f"Load error: {e}")
        return

    engine = FPLEngine(fixtures, history, players)
    current_gw = get_current_gw(fixtures)
    print(f"Training for GW {current_gw}...")
    projections = engine.train_and_predict(current_gw, horizon=3)
    
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

    print("\n=== TRANSFERS ===")
    transfer_rec = engine.recommend_transfers(best_squad, bank=0.5, free_transfers=1)
    engine.print_transfer_recommendation(transfer_rec)

if __name__ == "__main__":
    main()