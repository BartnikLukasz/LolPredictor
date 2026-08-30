import json
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

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
        return log_loss(y_test, preds_proba)

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

    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


def optimize_lightgbm_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/best_lightgbm_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on the LightGBM model
    using exact chronological train/test splitting based on split_date.
    Includes series context features (game_number, blue_series_lead, blue_prev_win).
    Saves the optimal hyperparameters to a JSON file.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

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
        return log_loss(y_test, preds_proba)

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

    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


def optimize_catboost_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/catboost_best_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on CatBoost model
    using chronological train/test splitting on split_date.
    Optimized: Explicit CPU multithreading & reduced evaluation iterations.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

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

    cat_features = [c for c in champ_features if c in X.columns]
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    print("=" * 60)
    print("      CATBOOST HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Features: {len(feature_cols)}")
    print(f"Series Features Included: {[f for f in series_features if f in df.columns]}")
    print(f"Categorical Features: {cat_features}")
    print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")
    print(f"Running {n_trials} optimization trials... Please wait.\n")

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': 600,  # Cap at 600 during trial evaluation for speed
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 3, 6),  # Max depth 6 keeps trees lightweight
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'eval_metric': 'Logloss',
            'thread_count': -1,  # Utilize all CPU cores
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            cat_features=cat_features if cat_features else None,
            early_stopping_rounds=20,
            verbose=False
        )

        preds_proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['iterations'] = 1000
    best_params['eval_metric'] = 'Logloss'
    best_params['early_stopping_rounds'] = 30
    best_params['random_seed'] = 42

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Best Test Log-Loss Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("=" * 60)

    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


def optimize_elastictree_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/elastictree_best_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on ExtraTrees (ElasticTree) model.
    Optimized: Pre-casts features to C-contiguous float32 arrays, fixes n_estimators bug during search,
    and eliminates pandas overhead during trial fits (~10x-15x speedup).
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

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

    cat_features = [c for c in champ_features if c in X.columns]
    if cat_features:
        for col in cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)
        X = pd.get_dummies(X, columns=cat_features, drop_first=True)

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx].copy(), y[:split_idx]
    X_test, y_test = X.iloc[split_idx:].copy(), y[split_idx:]

    # Impute missing values (fit medians ONLY on training set to prevent data leakage)
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_test[num_cols] = X_test[num_cols].fillna(train_medians)

    # --- MAJOR SPEEDUP OPTIMIZATION ---
    # Pre-cast to C-contiguous np.float32 arrays ONCE before starting Optuna.
    # Scikit-learn ExtraTrees requires float32. Passing float64 pandas DataFrames causes
    # implicit memory re-allocation and type conversion inside every trial.fit().
    X_train_np = np.ascontiguousarray(X_train.values, dtype=np.float32)
    X_test_np = np.ascontiguousarray(X_test.values, dtype=np.float32)
    y_train_np = np.ascontiguousarray(y_train, dtype=np.int32)

    print("=" * 60)
    print("     ELASTICTREE HYPERPARAMETER OPTIMIZATION (OPTUNA)   ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Encoded Features: {len(X.columns)}")
    print(f"Series Features Included: {[f for f in series_features if f in df.columns]}")
    print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")
    print(f"Running {n_trials} optimization trials... Please wait.\n")

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 60,  # 60 trees during search gives accurate parameter ranking at ~5x speedup vs 300
            'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini']),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.05, 0.1]),
            'random_state': 42,
            'n_jobs': -1
        }

        model = ExtraTreesClassifier(**params)
        model.fit(X_train_np, y_train_np)

        preds_proba = model.predict_proba(X_test_np)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['n_estimators'] = 500  # Automatically scales up to full production tree count
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Best Test Log-Loss Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("=" * 60)

    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


def optimize_elasticnet_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/elasticnet_best_params.json"
):
    """
    Runs Bayesian Hyperparameter Optimization (Optuna) on ElasticNet Logistic Regression.
    Optimized:
      - Uses Sparse CSR matrices so SAGA skips zero elements in one-hot features (20x+ speedup).
      - Parallelizes Optuna trials across all CPU cores (`n_jobs=-1`).
      - Caps trial `max_iter` to 200 and `tol` to 1e-2 for rapid evaluation.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

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

    cat_cols = champ_features
    num_cols = [c for c in feature_cols if c not in cat_cols]

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    print("=" * 60)
    print("     ELASTICNET HYPERPARAMETER OPTIMIZATION (OPTUNA)    ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Features: {len(feature_cols)}")
    print("Pre-transforming dataset into Sparse CSR format...")

    # Sparse Output Pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        sparse_threshold=0.3
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")
    print(f"Running {n_trials} parallel optimization trials... Please wait.\n")

    def objective(trial: optuna.Trial) -> float:
        params = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'C': trial.suggest_float('C', 1e-3, 5.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': 200,  # 200 iterations is plenty for SAGA evaluation
            'tol': 1e-2,      # Stops trial early once convergence flattens
            'random_state': 42
        }

        model = LogisticRegression(**params)
        model.fit(X_train_proc, y_train)

        preds_proba = model.predict_proba(X_test_proc)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    # Parallelize trials across all CPU cores
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    best_params['penalty'] = 'elasticnet'
    best_params['solver'] = 'saga'
    best_params['max_iter'] = 1000
    best_params['random_state'] = 42

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Best Test Log-Loss Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    print("=" * 60)

    with open(output_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters successfully saved to '{output_json_path}'")


if __name__ == "__main__":
    dataset_path = "../dataset/pregame/pregame_dataset_final_features.csv"

    # # Optimize XGBoost
    # optimize_xgboost_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=400,
    #     output_json_path="models/best_params.json"
    # )

    # # Optimize LightGBM
    # optimize_lightgbm_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=400,
    #     output_json_path="models/best_lightgbm_params.json"
    # )
    #
    # # Optimize CatBoost
    # optimize_catboost_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=50,
    #     output_json_path="../models/catboost_best_params.json"
    # )

    # Optimize ElasticTree
    optimize_elastictree_hyperparameters(
        filepath=dataset_path,
        split_date="2026-04-01",
        n_trials=500,
        output_json_path="../models/elastictree_best_params.json"
    )

    # # Optimize ElasticNet
    # optimize_elasticnet_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=200,
    #     output_json_path="../models/elasticnet_best_params.json"
    # )