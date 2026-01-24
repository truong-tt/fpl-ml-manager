import requests
import pandas as pd
import os
import time

BASE_URL = "https://fantasy.premierleague.com/api/"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_json(endpoint, retries=3):
    """Fetch JSON from endpoint with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Retry {attempt+1}/{retries}: {endpoint}")
            time.sleep(2)
    return None

def save_to_csv(df, filename):
    """Save DataFrame to CSV."""
    if df is not None and not df.empty:
        df.to_csv(os.path.join(DATA_DIR, filename), index=False)
        print(f"{filename}: {len(df)} rows")
    else:
        print(f"No data: {filename}")

def clean_expected_stats(df):
    """Convert xG/xA columns to floats."""
    for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

def get_static_data():
    """Download Players and Teams."""
    data = fetch_json("bootstrap-static/")
    if not data:
        return None, None

    teams_df = pd.DataFrame(data["teams"])[["id", "name", "short_name", "strength", "strength_overall_home", "strength_overall_away"]]
    teams_df.rename(columns={"id": "team_id", "name": "team_name"}, inplace=True)

    players_df = pd.DataFrame(data["elements"])
    players_df["team_name"] = players_df["team"].map(teams_df.set_index("team_id")["team_name"].to_dict())
    players_df = clean_expected_stats(players_df)

    return players_df, teams_df

def get_fixtures():
    """Download Fixtures."""
    data = fetch_json("fixtures/")
    if not data:
        return pd.DataFrame()

    fixtures_df = pd.DataFrame(data)
    cols = ["id", "event", "finished", "kickoff_time", "team_h", "team_a", 
            "team_h_score", "team_a_score", "team_h_difficulty", "team_a_difficulty"]
    return fixtures_df[[c for c in cols if c in fixtures_df.columns]]

def get_all_player_histories(player_ids):
    """Download match history for all players."""
    print(f"Fetching {len(player_ids)} players...")
    all_history = []
    
    for i, pid in enumerate(player_ids):
        if i > 0 and i % 100 == 0:
            print(f"{i}/{len(player_ids)}")
        
        data = fetch_json(f"element-summary/{pid}/")
        if data and "history" in data:
            history_df = pd.DataFrame(data["history"])
            history_df["player_id"] = pid
            all_history.append(history_df)
    
    if not all_history:
        return pd.DataFrame()

    return clean_expected_stats(pd.concat(all_history, ignore_index=True))

def main():
    """Main execution."""
    print("\n=== FPL DATA PIPELINE ===")

    players_df, teams_df = get_static_data()
    if players_df is None:
        print("Failed to fetch data")
        return

    fixtures_df = get_fixtures()
    active_players = players_df[(players_df['minutes'] > 0) | (players_df['now_cost'] > 40)]['id'].tolist()
    history_df = get_all_player_histories(active_players)

    save_to_csv(players_df, "players.csv")
    save_to_csv(teams_df, "teams.csv")
    save_to_csv(fixtures_df, "fixtures.csv")
    save_to_csv(history_df, "history.csv")
    print("Done\n")

if __name__ == "__main__":
    main()