import json
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

# Silence Optuna's verbose per-trial logging (prints summary at end)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_xgboost_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/best_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on the XGBoost model
    using exact chronological train/test splitting based on split_date.
    Includes series context features (game_number, blue_series_lead, blue_prev_win).
    Saves the optimal hyperparameters to a JSON file.
    """
    # 1. Load dataset & sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

    # 2. Identify feature subsets (Identical feature selection logic as model_trainer.py)
    elo_features = ['elo_diff', 'blue_elo_pre', 'red_elo_pre', 'blue_elo_win_prob', 'blue_firstpick']

    series_features = ['game_number', 'blue_series_lead', 'blue_prev_win']

    player_features = [
        col for col in df.columns
        if col.endswith('_player_games_pre') or
           col.endswith('_player_winrate_pre') or
           col.endswith('_champ_games_pre') or
           col.endswith('_champ_winrate_pre')
    ]

    h2h_matchup_features = [
        col for col in df.columns
        if 'h2h' in col or 'lane_matchup' in col or 'p2p' in col
    ]

    synergy_roster_features = [
        col for col in df.columns
        if 'roster' in col or 'duo' in col
    ]

    draft_champ_features = [
        col for col in df.columns
        if 'patch' in col or 'counter' in col or 'synergy' in col or 'cohesion' in col or 'comp' in col
    ]

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    feature_cols = (
        elo_features +
        series_features +
        player_features +
        h2h_matchup_features +
        synergy_roster_features +
        draft_champ_features +
        champ_features
    )
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    # 3. Apply exact chronological split date
    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    print("=" * 60)
    print("      XGBOOST HYPERPARAMETER OPTIMIZATION (OPTUNA)      ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Features: {len(feature_cols)}")
    print(f"Series Features Included: {[f for f in series_features if f in df.columns]}")
    print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")
    print(f"Running {n_trials} optimization trials... Please wait.\n")

    # 4. Objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'eval_metric': 'logloss',
            'enable_categorical': True,
            'early_stopping_rounds': 30,
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds_proba = model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds_proba)
        return loss

    # 5. Execute Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['n_estimators'] = 1000
    best_params['eval_metric'] = 'logloss'
    best_params['enable_categorical'] = True
    best_params['early_stopping_rounds'] = 30
    best_params['random_state'] = 42

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Best Test Log-Loss Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("=" * 60)

    # 6. Save parameters to JSON
    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


def optimize_lightgbm_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "best_lightgbm_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on the LightGBM model
    using exact chronological train/test splitting based on split_date.
    Includes series context features (game_number, blue_series_lead, blue_prev_win).
    Saves the optimal hyperparameters to a JSON file.
    """
    # 1. Load dataset & sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

    # 2. Identify feature subsets (Identical feature selection logic as model_trainer.py)
    elo_features = ['elo_diff', 'blue_elo_pre', 'red_elo_pre', 'blue_elo_win_prob', 'blue_firstpick']

    series_features = ['game_number', 'blue_series_lead', 'blue_prev_win']

    player_features = [
        col for col in df.columns
        if col.endswith('_player_games_pre') or
           col.endswith('_player_winrate_pre') or
           col.endswith('_champ_games_pre') or
           col.endswith('_champ_winrate_pre')
    ]

    h2h_matchup_features = [
        col for col in df.columns
        if 'h2h' in col or 'lane_matchup' in col or 'p2p' in col
    ]

    synergy_roster_features = [
        col for col in df.columns
        if 'roster' in col or 'duo' in col
    ]

    draft_champ_features = [
        col for col in df.columns
        if 'patch' in col or 'counter' in col or 'synergy' in col or 'cohesion' in col or 'comp' in col
    ]

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    feature_cols = (
        elo_features +
        series_features +
        player_features +
        h2h_matchup_features +
        synergy_roster_features +
        draft_champ_features +
        champ_features
    )
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    # 3. Apply exact chronological split date
    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    print("=" * 60)
    print("      LIGHTGBM HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Features: {len(feature_cols)}")
    print(f"Series Features Included: {[f for f in series_features if f in df.columns]}")
    print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")
    print(f"Running {n_trials} optimization trials... Please wait.\n")

    # 4. Objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'verbosity': -1
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        preds_proba = model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds_proba)
        return loss

    # 5. Execute Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['n_estimators'] = 1000
    best_params['objective'] = 'binary'
    best_params['subsample_freq'] = 1
    best_params['random_state'] = 42
    best_params['verbosity'] = -1

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Best Test Log-Loss Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("=" * 60)

    # 6. Save parameters to JSON
    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


if __name__ == "__main__":
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"

    # # Optimize XGBoost
    # optimize_xgboost_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=400,
    #     output_json_path="models/best_params.json"
    # )

    # Optimize LightGBM
    optimize_lightgbm_hyperparameters(
        filepath=dataset_path,
        split_date="2026-04-01",
        n_trials=400,
        output_json_path="models/best_lightgbm_params.json"
    )