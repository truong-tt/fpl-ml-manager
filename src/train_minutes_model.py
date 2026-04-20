from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw historical gameweek data into time-series features.

    To prevent data leakage, predictions for Gameweek N are modeled exclusively
    using lagged metrics from Gameweek N-1, N-2, and N-3.

    Args:
        df (pd.DataFrame): Raw historical match records.

    Returns:
        pd.DataFrame: DataFrame enriched with rolling averages and state flags.
    """
    df = df.sort_values(['player_id', 'round']).copy()

    df['target'] = np.where(df['minutes'] >= 60, 2, np.where(df['minutes'] > 0, 1, 0))

    df['lag_1_minutes'] = df.groupby('player_id')['minutes'].shift(1).fillna(0)
    df['lag_2_minutes'] = df.groupby('player_id')['minutes'].shift(2).fillna(0)
    df['lag_3_minutes'] = df.groupby('player_id')['minutes'].shift(3).fillna(0)

    df['roll_3_avg'] = df.groupby('player_id')['lag_1_minutes'].rolling(3, min_periods=1).mean().reset_index(0,
                                                                                                             drop=True).fillna(
        0)
    df['roll_5_avg'] = df.groupby('player_id')['lag_1_minutes'].rolling(5, min_periods=1).mean().reset_index(0,
                                                                                                             drop=True).fillna(
        0)

    df['played_last_week'] = (df['lag_1_minutes'] > 0).astype(int)

    return df


def main() -> None:
    """
    Data pipeline executable for the Minutes ML Model.
    Ingests history, engineers features, trains the XGBClassifier, and persists
    the serialized model to disk for fast inference during the optimization loop.
    """
    history_path = DATA_DIR / "history.csv"
    if not history_path.exists():
        return

    history = pd.read_csv(history_path)
    df = engineer_features(history)

    train_df = df.dropna(subset=['lag_1_minutes'])

    features = ['lag_1_minutes', 'lag_2_minutes', 'lag_3_minutes', 'roll_3_avg', 'roll_5_avg', 'played_last_week']
    X = train_df[features]
    y = train_df['target']

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)

    model_path = DATA_DIR / "xgboost_minutes_model.json"
    model.save_model(model_path)


if __name__ == "__main__":
    main()