import os
import json
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

# Silence Optuna's verbose per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def save_best_params_if_improved(best_params: dict, current_logloss: float, output_json_path: str):
    """
    Compares the current run's best log-loss against the stored log-loss in output_json_path.
    Only updates the file if the current log-loss is strictly better (lower).
    """
    previous_logloss = float('inf')

    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                existing_data = json.load(f)
                previous_logloss = existing_data.get('best_logloss', float('inf'))
        except Exception as e:
            print(f"[!] Warning: Could not read existing JSON ({e}). Overwriting file.")

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Current Run Best Log-Loss:  {current_logloss:.6f}")
    if previous_logloss != float('inf'):
        print(f"Previous Saved Log-Loss:    {previous_logloss:.6f}")
    else:
        print("Previous Saved Log-Loss:    None (New file)")

    if current_logloss < previous_logloss:
        best_params['best_logloss'] = round(float(current_logloss), 6)
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"[✓] Improvement detected! Updated hyperparameters saved to '{output_json_path}'")
    else:
        print(f"[!] Current run did not beat saved best ({previous_logloss:.6f}). Keeping existing JSON.")
    print("=" * 60 + "\n")


def optimize_xgboost_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/best_params.json"
):
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

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 500,
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
    best_params['n_estimators'] = 2000
    best_params['eval_metric'] = 'logloss'
    best_params['enable_categorical'] = True
    best_params['early_stopping_rounds'] = 30
    best_params['random_state'] = 42

    save_best_params_if_improved(best_params, study.best_value, output_json_path)


def optimize_lightgbm_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/best_lightgbm_params.json"
):
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

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 500,
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
    best_params['n_estimators'] = 2000
    best_params['objective'] = 'binary'
    best_params['subsample_freq'] = 1
    best_params['random_state'] = 42
    best_params['verbosity'] = -1

    save_best_params_if_improved(best_params, study.best_value, output_json_path)


def optimize_catboost_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/catboost_best_params.json"
):
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
    h2h_matchup_features = [col for col in df.columns if 'h2h' in col or 'lane_matchup' in col or 'p2p' in col]
    synergy_roster_features = [col for col in df.columns if 'roster' in col or 'duo' in col]
    draft_champ_features = [col for col in df.columns if 'patch' in col or 'counter' in col or 'synergy' in col or 'cohesion' in col or 'comp' in col]
    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    feature_cols = elo_features + series_features + player_features + h2h_matchup_features + synergy_roster_features + draft_champ_features + champ_features
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
    print("Pre-constructing CatBoost Pool objects for accelerated evaluation...")

    # OPTIMIZATION: Instantiate Pool objects ONCE outside the optimization loop
    train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)
    test_pool = Pool(X_test, y_test, cat_features=cat_features if cat_features else None)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': 300,  # Cap at 300 iterations during search for maximum throughput
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'eval_metric': 'Logloss',
            'thread_count': -1,
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=15,
            verbose=False
        )

        preds_proba = model.predict_proba(test_pool)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['iterations'] = 1000
    best_params['eval_metric'] = 'Logloss'
    best_params['early_stopping_rounds'] = 30
    best_params['random_seed'] = 42

    save_best_params_if_improved(best_params, study.best_value, output_json_path)


def optimize_elastictree_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/elastictree_best_params.json"
):
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
    h2h_matchup_features = [col for col in df.columns if 'h2h' in col or 'lane_matchup' in col or 'p2p' in col]
    synergy_roster_features = [col for col in df.columns if 'roster' in col or 'duo' in col]
    draft_champ_features = [col for col in df.columns if 'patch' in col or 'counter' in col or 'synergy' in col or 'cohesion' in col or 'comp' in col]
    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    feature_cols = elo_features + series_features + player_features + h2h_matchup_features + synergy_roster_features + draft_champ_features + champ_features
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

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_test[num_cols] = X_test[num_cols].fillna(train_medians)

    X_train_np = np.ascontiguousarray(X_train.values, dtype=np.float32)
    X_test_np = np.ascontiguousarray(X_test.values, dtype=np.float32)
    y_train_np = np.ascontiguousarray(y_train, dtype=np.int32)

    print("=" * 60)
    print("     ELASTICTREE HYPERPARAMETER OPTIMIZATION (OPTUNA)   ")
    print("=" * 60)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 50,  # 50 trees for swift evaluation
            'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini']),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.05, 0.1]),
            'random_state': 42,
            'n_jobs': 1  # Keep 1 thread per estimator for parallel Optuna execution
        }

        model = ExtraTreesClassifier(**params)
        model.fit(X_train_np, y_train_np)

        preds_proba = model.predict_proba(X_test_np)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    # OPTIMIZATION: Run trials in parallel across all available CPU cores
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    best_params = study.best_params
    best_params['n_estimators'] = 500
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    save_best_params_if_improved(best_params, study.best_value, output_json_path)

def optimize_elasticnet_hyperparameters(
        filepath: str,
        split_date: str = "2026-04-01",
        n_trials: int = 50,
        output_json_path: str = "models/elasticnet_best_params.json"
):
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

    def objective(trial: optuna.Trial) -> float:
        params = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'C': trial.suggest_float('C', 1e-3, 5.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': 200,
            'tol': 1e-2,
            'random_state': 42
        }

        model = LogisticRegression(**params)
        model.fit(X_train_proc, y_train)

        preds_proba = model.predict_proba(X_test_proc)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    best_params['penalty'] = 'elasticnet'
    best_params['solver'] = 'saga'
    best_params['max_iter'] = 1000
    best_params['random_state'] = 42

    save_best_params_if_improved(best_params, study.best_value, output_json_path)

if __name__ == "__main__":
    dataset_path = "../dataset/pregame/pregame_dataset_final_features.csv"

    # # Optimize XGBoost
    # optimize_xgboost_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=400,
    #     output_json_path="../models/best_params.json"
    # )
    #
    # # Optimize LightGBM
    # optimize_lightgbm_hyperparameters(
    #     filepath=dataset_path,
    #     split_date="2026-04-01",
    #     n_trials=400,
    #     output_json_path="../models/best_lightgbm_params.json"
    # )

    # Optimize CatBoost
    optimize_catboost_hyperparameters(
        filepath=dataset_path,
        split_date="2026-04-01",
        n_trials=400,
        output_json_path="../models/catboost_best_params.json"
    )

    # Optimize ElasticTree
    optimize_elastictree_hyperparameters(
        filepath=dataset_path,
        split_date="2026-04-01",
        n_trials=400,
        output_json_path="../models/elastictree_best_params.json"
    )

    # Optimize ElasticNet
    optimize_elasticnet_hyperparameters(
        filepath=dataset_path,
        split_date="2026-04-01",
        n_trials=400,
        output_json_path="../models/elasticnet_best_params.json"
    )