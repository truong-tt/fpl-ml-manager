from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api/"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(endpoint: str, retries: int = 3) -> dict | None:
    """
    Fetches JSON data from the official FPL API with basic retry logic.

    Args:
        endpoint (str): The specific API endpoint to query.
        retries (int, optional): Number of connection attempts. Defaults to 3.

    Returns:
        dict | None: Parsed JSON response, or None if the request fails.
    """
    for attempt in range(retries):
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None


def save_to_csv(df: pd.DataFrame | None, filename: str) -> None:
    """
    Saves a DataFrame to the local data directory for offline processing.

    Args:
        df (pd.DataFrame | None): The data to save.
        filename (str): Target filename (e.g., 'players.csv').
    """
    if df is not None and not df.empty:
        df.to_csv(DATA_DIR / filename, index=False)


def clean_expected_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes underlying metrics.

    Args:
        df: Raw input dataframe.

    Returns:
        Dataframe with numeric values.
    """
    cols = ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'recoveries', 'yellow_cards', 'red_cards', 'penalties_missed', 'own_goals']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    return df


def get_static_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Extracts base entity data (Teams and Players) from the bootstrap-static endpoint.

    Returns:
        tuple: (players_df, teams_df) containing sanitized master records.
    """
    data = fetch_json("bootstrap-static/")
    if not data:
        return None, None

    teams_df = pd.DataFrame(data["teams"])[
        ["id", "name", "short_name", "strength", "strength_overall_home", "strength_overall_away"]
    ]
    teams_df = teams_df.rename(columns={"id": "team_id", "name": "team_name"})

    players_df = pd.DataFrame(data["elements"])
    players_df["team_name"] = players_df["team"].map(teams_df.set_index("team_id")["team_name"].to_dict())
    players_df = clean_expected_stats(players_df)

    return players_df, teams_df


def get_fixtures() -> pd.DataFrame:
    """
    Retrieves the global fixture schedule and match outcomes.

    Returns:
        pd.DataFrame: Fixture timeline detailing home/away teams and results.
    """
    data = fetch_json("fixtures/")
    if not data:
        return pd.DataFrame()

    fixtures_df = pd.DataFrame(data)
    cols = ["id", "event", "finished", "kickoff_time", "team_h", "team_a",
            "team_h_score", "team_a_score", "team_h_difficulty", "team_a_difficulty"]

    fixtures_df = fixtures_df[[c for c in cols if c in fixtures_df.columns]]
    fixtures_df = fixtures_df.dropna(subset=['event']).copy()
    fixtures_df['event'] = fixtures_df['event'].astype(int)

    return fixtures_df


def get_all_player_histories(player_ids: list[int]) -> pd.DataFrame:
    """
    Compiles gameweek-by-gameweek historical data for active players.

    Args:
        player_ids (list[int]): List of active player IDs to fetch histories for.

    Returns:
        pd.DataFrame: Aggregated time-series data for match simulation and modeling.
    """
    all_history: list[pd.DataFrame] = []

    for _, pid in enumerate(player_ids):
        data = fetch_json(f"element-summary/{pid}/")
        if data and "history" in data:
            history_df = pd.DataFrame(data["history"])
            history_df["player_id"] = pid
            all_history.append(history_df)

    if not all_history:
        return pd.DataFrame()

    return clean_expected_stats(pd.concat(all_history, ignore_index=True))


def main() -> None:
    """Orchestrates the data pipeline: fetch, transform, and persist to disk."""
    players_df, teams_df = get_static_data()
    if players_df is None:
        return

    fixtures_df = get_fixtures()
    active_players = players_df[(players_df['minutes'] > 0) | (players_df['now_cost'] > 40)]['id'].tolist()
    history_df = get_all_player_histories(active_players)

    save_to_csv(players_df, "players.csv")
    save_to_csv(teams_df, "teams.csv")
    save_to_csv(fixtures_df, "fixtures.csv")
    save_to_csv(history_df, "history.csv")


if __name__ == "__main__":
    main()