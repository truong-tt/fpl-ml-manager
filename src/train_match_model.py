import pandas as pd
import xgboost as xgb
from pathlib import Path
from features import build_match_features

def train_match_models(fixtures: pd.DataFrame, history: pd.DataFrame) -> None:
    """Trains XGBoost Poisson models for match outcome simulation.

    Args:
        fixtures: Fixture schedule data.
        history: Historical player match data.
    """
    df = build_match_features(fixtures, history)
    past = df[df['finished'] == True]

    features = ['h_xg', 'h_xga', 'a_xg', 'a_xga', 'strength_diff']

    X = past[features]
    y_h, y_a = past['team_h_score'], past['team_a_score']

    params = {'objective': 'count:poisson', 'learning_rate': 0.1, 'max_depth': 4}

    model_h = xgb.train(params, xgb.DMatrix(X, label=y_h), num_boost_round=100)
    model_a = xgb.train(params, xgb.DMatrix(X, label=y_a), num_boost_round=100)

    path = Path(__file__).resolve().parent.parent / "data"
    model_h.save_model(path / "xgb_home_goals.json")
    model_a.save_model(path / "xgb_away_goals.json")

if __name__ == "__main__":
    fix = pd.read_csv("../data/fixtures.csv")
    hist = pd.read_csv("../data/history.csv")
    train_match_models(fix, hist)