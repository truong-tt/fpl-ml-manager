import pandas as pd

def build_match_features(fixtures_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Builds rolling features for match simulation.

    Args:
        fixtures_df: Fixture schedule data.
        history_df: Historical player match data.

    Returns:
        DataFrame containing merged match features.
    """
    team_gw = history_df.groupby(['team', 'round'])[['expected_goals', 'expected_goals_conceded']].sum().reset_index()
    team_gw['roll_xg'] = team_gw.groupby('team')['expected_goals'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_gw['roll_xga'] = team_gw.groupby('team')['expected_goals_conceded'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())

    df = fixtures_df.copy()
    df = df.merge(team_gw, left_on=['team_h', 'event'], right_on=['team', 'round'], how='left')
    df = df.rename(columns={'roll_xg': 'h_xg', 'roll_xga': 'h_xga'})

    df = df.merge(team_gw, left_on=['team_a', 'event'], right_on=['team', 'round'], how='left')
    df = df.rename(columns={'roll_xg': 'a_xg', 'roll_xga': 'a_xga'})

    df['strength_diff'] = df['team_h_difficulty'] - df['team_a_difficulty']
    return df.dropna(subset=['h_xg', 'a_xg'])