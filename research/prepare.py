import pandas as pd
from pathlib import Path

# Resolves to fpl-ml-manager/data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_START_GW = 28 # Gameweek where the holdout test set begins

def load_backtest_data():
    """Loads raw data for the agent to engineer features from."""
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    return fx, hist, players, teams

def split_data(df: pd.DataFrame, feature_cols: list[str]):
    """Splits the engineered dataframe into Train and Test sets."""
    # Ensure target exists
    df = df.dropna(subset=["target"])
    
    train = df[df['round'] < TEST_START_GW]
    test = df[df['round'] >= TEST_START_GW]

    X_train = train[feature_cols].astype(float).fillna(0.0)
    y_train = train["target"].astype(float)
    X_test = test[feature_cols].astype(float).fillna(0.0)
    y_test = test["target"].astype(float)
    
    return X_train, y_train, X_test, y_test