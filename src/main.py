from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from typing import cast

import pandas as pd
from data_loader import main as update_data
from fpl_engine import FPLEngine
from train_minutes_model import main as train_xgb_model

os.environ.update(
    {"PATH": r"C:\msys64\mingw64\bin" + os.pathsep + os.environ.get("PATH", ""), "pytensor_FLAGS": "cxx=g++"})
warnings.filterwarnings("ignore")
for logger in ["pymc", "pytensor"]:
    logging.getLogger(logger).setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PATHS = {k: DATA_DIR / f"{k}.csv" for k in ["fixtures", "history", "players", "teams"]}

RISK_AVERSION_COEF = 0.05
STACK_BONUS = 2.5
SYNERGY_PAIRS = []


def get_current_gw(fixtures_df: pd.DataFrame) -> int:
    """
    Determines the next upcoming Gameweek (GW) based on fixture status.

    Args:
        fixtures_df (pd.DataFrame): The global schedule.

    Returns:
        int: The integer ID of the next Gameweek.
    """
    upcoming = fixtures_df[~fixtures_df['finished']]
    return int(upcoming.iloc[0]['event']) if not upcoming.empty else 38


def main() -> None:
    """
    Primary Orchestrator.
    Handles data ingestion, initiates XGBoost model training, runs Bayesian
    predictions, and executes the MILP knapsack solver to generate the optimal team.
    Prints ONLY the final squad and transfer output for GitHub Actions capture.
    """
    try:
        update_data()
    except Exception:
        return

    model_path = DATA_DIR / "xgboost_minutes_model.json"
    if not model_path.exists():
        try:
            train_xgb_model()
        except Exception:
            pass

    try:
        fixtures = cast(pd.DataFrame, pd.read_csv(str(PATHS["fixtures"])))
        history = cast(pd.DataFrame, pd.read_csv(str(PATHS["history"])))
        players = cast(pd.DataFrame, pd.read_csv(str(PATHS["players"])))
        teams = cast(pd.DataFrame, pd.read_csv(str(PATHS["teams"]))) if PATHS["teams"].exists() else None
    except Exception:
        return

    engine = FPLEngine(fixtures, history, players, teams_df=teams)
    current_gw = get_current_gw(fixtures)

    projections = engine.train_and_predict(current_gw, horizon=5)

    if projections.empty:
        return

    best_squad = engine.optimize_squad(projections, budget=100.0, risk_aversion=RISK_AVERSION_COEF,
                                       stack_bonus=STACK_BONUS)

    if not best_squad.empty:
        processed_dir = DATA_DIR / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        csv_path = processed_dir / "optimal_squad_live.csv"
        txt_path = processed_dir / "lineup_output.txt"

        best_squad.to_csv(csv_path, index=False)

        squad_str = engine.get_squad_str(*engine.pick_team_sheet(best_squad))

        current_value = best_squad['price'].sum()
        actual_bank = round(100.0 - current_value, 1)

        transfer_str = engine.get_transfer_recommendation_str(
            engine.recommend_transfers(best_squad, bank=actual_bank, gw=current_gw,
                                       proj=projections, risk=RISK_AVERSION_COEF)
        )


        final_output = f"AI FPL Manager - Weekly Summary\n{'=' * 30}\n{squad_str}\n{transfer_str}"
        with open(txt_path, 'w') as f:
            f.write(final_output)
        print(final_output)


if __name__ == "__main__":
    main()