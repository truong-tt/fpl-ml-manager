import requests
import pandas as pd
import os
import json

BASE_URL = "https://fantasy.premierleague.com/api/"
RAW_PATH = os.path.join("data", "raw")
PROCESSED_PATH = os.path.join("data", "processed")

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)


def fetch_json(endpoint):
    """GET endpoint -> JSON (or None)."""
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint}: {e}")
        return None


def save_raw(data, filename):
    """Write raw JSON to data/raw."""
    with open(os.path.join(RAW_PATH, filename), "w") as f:
        json.dump(data, f)


def get_static_data():
    """Download players/teams (bootstrap-static)."""
    print("Fetching static data...")
    data = fetch_json("bootstrap-static/")
    if not data:
        return None, None

    save_raw(data, "bootstrap_static.json")

    # Teams table
    teams_df = pd.DataFrame(data["teams"])[
        ["id", "name", "short_name", "strength", "strength_overall_home", "strength_overall_away"]
    ].rename(columns={"id": "team_id", "name": "team_name"})

    # Players table (+ team name)
    players_df = pd.DataFrame(data["elements"])
    team_map = teams_df.set_index("team_id")["team_name"].to_dict()
    players_df["team_name"] = players_df["team"].map(team_map)

    return players_df, teams_df


def get_fixtures():
    """Download fixtures (+ scores when available)."""
    print("Fetching fixtures...")
    data = fetch_json("fixtures/")
    if not data:
        return pd.DataFrame()

    save_raw(data, "fixtures.json")

    cols = [
        "id",
        "event",
        "finished",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
        "team_h_difficulty",
        "team_a_difficulty",
    ]
    fixtures_df = pd.DataFrame(data)
    return fixtures_df[[c for c in cols if c in fixtures_df.columns]]


def get_all_player_histories(player_ids):
    """Download per-player match history (element-summary)."""
    print(f"Fetching history for {len(player_ids)} players...")
    all_history = []

    for i, pid in enumerate(player_ids):
        if i > 0 and i % 50 == 0:
            print(f"  Processed {i}/{len(player_ids)}...")

        data = fetch_json(f"element-summary/{pid}/")
        if data and "history" in data:
            history_df = pd.DataFrame(data["history"])
            history_df["player_id"] = pid
            all_history.append(history_df)

    return pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()


def save_csv(df, filename):
    """Write DataFrame to data/processed (if non-empty)."""
    if df is not None and not df.empty:
        path = os.path.join(PROCESSED_PATH, filename)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
        return True
    print(f"Warning: No data for {filename}")
    return False


def main():
    """End-to-end data pull -> CSVs."""
    print("--- Starting Data Pipeline ---")

    players_df, teams_df = get_static_data()
    save_csv(players_df, "fpl_players_summary.csv")
    save_csv(teams_df, "fpl_teams.csv")

    save_csv(get_fixtures(), "fpl_fixtures.csv")

    if players_df is not None:
        history_df = get_all_player_histories(players_df["id"].tolist())
        save_csv(history_df, "fpl_gameweek_history.csv")

    print("\n--- Data Extraction Complete ---")


if __name__ == "__main__":
    main()