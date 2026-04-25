import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from prepare import load_backtest_data, split_data
import pandas as pd
import numpy as np

# --- AGENT CAN EDIT EVERYTHING BELOW THIS LINE ---

def engineer_features(fx, hist, players, teams):
    """
    Agent can modify feature engineering here.
    This version adds more rolling features, lagged features,
    player-specific attributes, and opponent difficulty.
    Increased robustness by handling potential missing columns,
    using np.where for conditional logic, and explicitly dropping rows with NaN targets.
    """
    df = hist.copy().sort_values(["player_id", "round"])
    
    # Ensure target is float and drop rows where target is NaN
    df["target"] = df["total_points"].astype(float)
    df = df.dropna(subset=['target']) # Crucial to drop rows with NaN targets before splitting

    # Ensure 'was_home' is numeric and fill NaNs
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].fillna(0).astype(int)
    else:
        df['was_home'] = 0 
    
    # Lagged features: points and minutes from the previous round
    # Fill with 0.0 for initial rounds where there's no previous data
    if 'total_points' in df.columns:
        df["lagged_points"] = df.groupby("player_id")["total_points"].shift(1).fillna(0.0)
    else:
        df["lagged_points"] = 0.0

    if 'minutes' in df.columns:
        df["lagged_minutes"] = df.groupby("player_id")["minutes"].shift(1).fillna(0.0)
    else:
        df["lagged_minutes"] = 0.0

    # Rolling features over different window sizes
    rolling_features_configs = [
        ("total_points", [3, 5, 10], "mean"),
        ("minutes", [3, 5], "mean"),
        ("goals_scored", [3, 5], "mean"),
        ("goals_scored", [3, 5], "sum"), 
        ("assists", [3, 5], "mean"),
        ("assists", [3, 5], "sum"),
        ("clean_sheets", [3, 5], "mean"),
        ("clean_sheets", [3, 5], "sum"),
        ("goals_conceded", [3, 5], "mean"),
        ("saves", [3], "mean"), 
        ("bonus", [3, 5], "mean"),
        ("bonus", [3, 5], "sum"),
        ("influence", [3, 5], "mean"),
        ("creativity", [3, 5], "mean"),
        ("threat", [3, 5], "mean"),
        ("bps", [3, 5], "mean"), 
    ]

    for col, windows, agg_func in rolling_features_configs:
        for window in windows:
            # Only create rolling feature if the base column exists in df
            if col in df.columns: 
                # Shift by 1 to ensure only previous game data is used
                if agg_func == "mean":
                    df[f"roll_{window}_{col}_mean"] = df.groupby("player_id")[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    ).fillna(0.0)
                elif agg_func == "sum":
                    df[f"roll_{window}_{col}_sum"] = df.groupby("player_id")[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
                    ).fillna(0.0)

    # Merge player specific static features (value and position)
    player_cols_to_use = ['id', 'element_type', 'value']
    player_data_present_cols = [col for col in player_cols_to_use if col in players.columns]
    
    # Only proceed with merge if 'id' exists in players for a valid join key
    if 'id' in players.columns:
        player_data = players[player_data_present_cols].copy() 
        player_data = player_data.rename(columns={'id': 'player_id'})
        df = df.merge(player_data, on='player_id', how='left')

    # Add player value feature, with a default if 'value' column is missing after merge
    if 'value' in df.columns:
        df['player_value'] = df['value'] / 10.0 
    else:
        df['player_value'] = 0.0 

    # Add player position feature, with a default if 'element_type' column is missing
    if 'element_type' in df.columns:
        # Fill NaNs in 'element_type' before converting to category to prevent crash
        df['player_position'] = df['element_type'].fillna(-1).astype('category').cat.codes 
    else:
        df['player_position'] = -1 

    # Merge fixture data for opponent difficulty
    fx_cols_to_use = ['id', 'team_h_difficulty', 'team_a_difficulty', 'team_h', 'team_a']
    fx_data_present_cols = [col for col in fx_cols_to_use if col in fx.columns]
    
    # Only proceed with merge if 'id' exists in fx for a valid join key
    if 'id' in fx.columns:
        fx_data = fx[fx_data_present_cols].copy()
        fx_data = fx_data.rename(columns={'id': 'fixture_id'})
        df = df.merge(fx_data, on='fixture_id', how='left')
    
    # Fill NaNs in difficulty columns that might result from merge
    df['team_h_difficulty'] = df['team_h_difficulty'].fillna(3.0) # Default neutral difficulty
    df['team_a_difficulty'] = df['team_a_difficulty'].fillna(3.0) # Default neutral difficulty

    # Calculate opponent difficulty using numpy.where for efficiency and robustness
    if 'was_home' in df.columns and 'team_a_difficulty' in df.columns and 'team_h_difficulty' in df.columns:
        df['opponent_difficulty'] = np.where(
            df['was_home'] == 1, 
            df['team_a_difficulty'], 
            df['team_h_difficulty'] 
        )
    else:
        df['opponent_difficulty'] = 3.0 
    
    # Interaction features: minutes played vs opponent difficulty
    if 'minutes' in df.columns and 'opponent_difficulty' in df.columns:
        df['minutes_x_opponent_difficulty'] = df['minutes'] * df['opponent_difficulty']
    else:
        df['minutes_x_opponent_difficulty'] = 0.0 

    # Drop temporary/redundant columns after creating features
    df = df.drop(columns=[
        'value', 'element_type', 
        'team_h_difficulty', 'team_a_difficulty', 'team_h', 'team_a', 
    ], errors='ignore')

    # Define the final list of feature columns
    # Include base features only if they exist in the dataframe
    feature_cols = []
    
    # Base features
    base_features = [
        "minutes", "was_home", "lagged_points", "lagged_minutes",
        "player_value", "player_position", "opponent_difficulty",
        "minutes_x_opponent_difficulty"
    ]
    for col in base_features:
        if col in df.columns:
            feature_cols.append(col)
                
    # Dynamically add all created rolling features
    for col, windows, agg_func in rolling_features_configs:
        for window in windows:
            feature_name = f"roll_{window}_{col}_{agg_func}"
            if feature_name in df.columns: # Check if the feature was actually created
                feature_cols.append(feature_name)
                
    # Final check for feature columns against the dataframe columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Fill any remaining NaNs in feature columns as a final safeguard
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)

    return df, feature_cols

def train_and_evaluate():
    """
    Agent can modify model hyperparameters here.
    This version updates XGBoost hyperparameters and increases boosting rounds.
    Includes additional checks to prevent crashes from empty dataframes/splits.
    """
    fx, hist, players, teams = load_backtest_data()
    df, feature_cols = engineer_features(fx, hist, players, teams)
    
    # Initial check for empty df or no features after engineering
    if df.empty or len(feature_cols) == 0:
        print("VERIFIER_SCORE: 999.0")
        return
        
    X_train, y_train, X_test, y_test = split_data(df, feature_cols)
    
    # Check if split_data returned valid (non-empty) data for training and testing
    # Also check if X_train is not empty before creating DMatrix
    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        print("VERIFIER_SCORE: 999.0")
        return

    # XGBoost Hyperparameters optimized for MAE
    params = dict(
        objective="reg:quantileerror", quantile_alpha=0.50, # Optimized for MAE (median regression)
        learning_rate=0.02,     # Reduced learning rate for more stable convergence
        max_depth=5,            # Slightly increased depth to capture more complex interactions
        subsample=0.7,          # Reduced subsample ratio to reduce variance
        colsample_bytree=0.7,   # Feature subsampling per tree
        min_child_weight=15,    # Increased min_child_weight to prevent overfitting
        gamma=0.1,              # Minimum loss reduction required to make a further partition
        nthread=-1,             # Use all available threads for parallel processing
        tree_method='hist',     # 'hist' for faster training on large datasets
        eval_metric='mae',      # Explicitly set evaluation metric to MAE for consistency
        verbosity=0,            # Suppress verbose output
        seed=42                 # For reproducibility
    )
    
    num_boost_round = 750 # Increased number of boosting rounds
    
    # Train the XGBoost model
    m = xgb.train(params, xgb.DMatrix(X_train, label=y_train), num_boost_round=num_boost_round)
    
    # Make predictions on the test set
    preds = m.predict(xgb.DMatrix(X_test))
    mae = mean_absolute_error(y_test, preds)
    
    # The agent uses RegEx to find this exact string format
    print(f"VERIFIER_SCORE: {mae:.4f}")

if __name__ == "__main__":
    train_and_evaluate()