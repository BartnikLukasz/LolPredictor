import os
import json
from datetime import date, timedelta
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

from trainers.trainer_helpers import extract_features
from util import generate_game_id

# Silence Optuna's verbose per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def refresh_baseline_if_exists(output_json_path: str, eval_callback) -> float:
    """
    Evaluates saved parameters against the CURRENT dataset split using tuning-level
    iteration limits. Updates 'best_logloss' in the JSON file ONLY if the dataset or score changed.
    """
    if not os.path.exists(output_json_path):
        print("[i] No existing parameter file found. Proceeding with fresh optimization search.")
        return float('inf')

    try:
        with open(output_json_path, 'r') as f:
            saved_params = json.load(f)

        previous_logloss = saved_params.get('best_logloss', None)

        meta_keys = ['best_logloss']
        clean_params = {k: v for k, v in saved_params.items() if k not in meta_keys}

        print(f"[*] Re-evaluating saved hyperparameters from '{output_json_path}' using tuning limits...")
        baseline_logloss = eval_callback(clean_params)
        baseline_logloss = round(float(baseline_logloss), 6)

        # Check if logloss actually changed compared to saved JSON value
        if previous_logloss is None or round(float(previous_logloss), 6) != baseline_logloss:
            saved_params['best_logloss'] = baseline_logloss
            os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(saved_params, f, indent=4)
            print(f"[✓] Refreshed baseline Log-Loss on new dataset: {previous_logloss} -> {baseline_logloss:.6f}\n")
        else:
            print(f"[=] Baseline Log-Loss unchanged on current dataset ({baseline_logloss:.6f}). Keeping stored parameters.\n")

        return baseline_logloss

    except Exception as e:
        print(f"[!] Warning: Failed to re-evaluate existing baseline ({e}). Proceeding without baseline refresh.")
        return float('inf')


def compute_sample_logloss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Calculates individual binary log-loss for each sample."""
    p = np.clip(y_proba, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def save_validation_predictions(
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        y_probs: np.ndarray,
        output_csv_path: str
):
    """Saves match metadata, probabilities, ground truth, and unique game_id to CSV."""
    results_df = pd.DataFrame()

    # Define champion feature columns expected in test_df
    blue_champ_cols = ['blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion']
    red_champ_cols = ['red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion']

    # Filter columns that exist in the test dataframe
    b_cols = [c for c in blue_champ_cols if c in test_df.columns]
    r_cols = [c for c in red_champ_cols if c in test_df.columns]

    # Deterministic game_id generation row by row
    game_ids = []
    for idx, row in test_df.iterrows():
        b_team = row.get('blue_team', 'Unknown')
        r_team = row.get('red_team', 'Unknown')
        b_champs = [row[c] for c in b_cols]
        r_champs = [row[c] for c in r_cols]
        fp = row.get('blue_firstpick', row.get('blue_first_pick', row.get('first_pick', '')))

        g_id = generate_game_id(b_team, r_team, b_champs, r_champs, fp)
        game_ids.append(g_id)

    results_df['game_id'] = game_ids  # <-- ADDED AS FIRST COLUMN

    # Standardize date/datetime (using .values to avoid index alignment NaN drops)
    date_col = 'datetime' if 'datetime' in test_df.columns else ('date' if 'date' in test_df.columns else None)
    if date_col is not None:
        results_df['datetime'] = pd.to_datetime(test_df[date_col]).dt.strftime('%Y-%m-%d %H:%M:%S').values
    else:
        results_df['datetime'] = "N/A"

    # Match metadata
    results_df['blue_team'] = test_df['blue_team'].values if 'blue_team' in test_df.columns else "Unknown"
    results_df['red_team'] = test_df['red_team'].values if 'red_team' in test_df.columns else "Unknown"

    # Prediction metrics
    probs_blue = np.round(y_probs, 6)
    results_df['prob_blue_win'] = probs_blue
    results_df['prob_red_win'] = np.round(1.0 - probs_blue, 6)
    results_df['actual_blue_win'] = y_test.astype(int)

    # Calculate sample log-loss and accuracy flag
    results_df['sample_logloss'] = np.round(compute_sample_logloss(y_test, probs_blue), 6)
    results_df['is_correct'] = (probs_blue >= 0.5) == (y_test == 1)

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"[✓] Saved standardized validation predictions with game_id to '{output_csv_path}'")


def save_best_params_if_improved(
    best_params: dict,
    current_logloss: float,
    output_json_path: str,
    test_df: pd.DataFrame = None,
    y_test: np.ndarray = None,
    best_y_probs: np.ndarray = None
):
    """
    Compares the current run's best log-loss against the stored log-loss.
    Saves new parameters AND prediction CSV if an improvement is detected.
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

        # Save predictions CSV alongside parameter JSON
        if test_df is not None and y_test is not None and best_y_probs is not None:
            output_csv_path = output_json_path.replace('models', 'metadata/predictions').replace('.json', '_predictions.csv')
            save_validation_predictions(test_df, y_test, best_y_probs, output_csv_path)
    else:
        print(f"[!] Current run did not beat refreshed baseline ({previous_logloss:.6f}). Keeping existing JSON.")
    print("=" * 60 + "\n")


def optimize_xgboost_hyperparameters(
        filepath: str,
        dynamic_test_window: int = 60,
        n_trials: int = 50,
        output_json_path: str = "models/best_params.json"
):
    split_date = date.today() - timedelta(days=dynamic_test_window)
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]
    test_df_subset = df.iloc[split_idx:].copy()

    print("=" * 60)
    print("      XGBOOST HYPERPARAMETER OPTIMIZATION (OPTUNA)      ")
    print("=" * 60)

    def eval_xgb(params):
        p = params.copy()
        p['n_estimators'] = 200
        p['eval_metric'] = 'logloss'
        p['enable_categorical'] = True
        p['early_stopping_rounds'] = 30
        p['random_state'] = 42

        model = xgb.XGBClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds_proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, preds_proba)

    refresh_baseline_if_exists(output_json_path, eval_xgb)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 200,
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.09, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 11),
            'subsample': trial.suggest_float('subsample', 0.2, 0.6),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 7, 13),
            'gamma': trial.suggest_float('gamma', 0.2, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 7, 15.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 2, 5.0, log=True),
            'eval_metric': 'logloss',
            'enable_categorical': True,
            'early_stopping_rounds': 30,
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds_proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Re-predict using best trial parameters to capture test probabilities
    best_eval_params = study.best_params.copy()
    best_eval_params['n_estimators'] = 200
    best_eval_params['eval_metric'] = 'logloss'
    best_eval_params['enable_categorical'] = True
    best_eval_params['early_stopping_rounds'] = 30
    best_eval_params['random_state'] = 42

    best_model = xgb.XGBClassifier(**best_eval_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    best_preds_proba = best_model.predict_proba(X_test)[:, 1]

    # Save hyperparams (with 1000 estimators for full training)
    final_params = study.best_params.copy()
    final_params['n_estimators'] = 1000
    final_params['eval_metric'] = 'logloss'
    final_params['enable_categorical'] = True
    final_params['early_stopping_rounds'] = 30
    final_params['random_state'] = 42

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path,
        test_df=test_df_subset,
        y_test=y_test,
        best_y_probs=best_preds_proba
    )


def optimize_lightgbm_hyperparameters(
        filepath: str,
        dynamic_test_window: int = 60,
        n_trials: int = 50,
        output_json_path: str = "models/best_lightgbm_params.json"
):
    split_date = date.today() - timedelta(days=dynamic_test_window)
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]
    test_df_subset = df.iloc[split_idx:].copy()

    print("=" * 60)
    print("      LIGHTGBM HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)

    # Re-evaluate saved baseline using tuning n_estimators (200)
    def eval_lgbm(params):
        p = params.copy()
        p['n_estimators'] = 200  # Override final model 1000 estimators
        p['objective'] = 'binary'
        p['metric'] = 'binary_logloss'
        p['random_state'] = 42
        p['verbosity'] = -1

        model = LGBMClassifier(**p)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        preds_proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, preds_proba)

    refresh_baseline_if_exists(output_json_path, eval_lgbm)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 200,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'num_leaves': trial.suggest_int('num_leaves', 30, 73),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 60),
            'subsample': trial.suggest_float('subsample', 0.2, 0.5),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
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

    # Generate probabilities for best trial parameters
    best_eval_params = study.best_params.copy()
    best_eval_params['n_estimators'] = 200
    best_eval_params['objective'] = 'binary'
    best_eval_params['subsample_freq'] = 1
    best_eval_params['random_state'] = 42
    best_eval_params['verbosity'] = -1

    best_model = LGBMClassifier(**best_eval_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    best_preds_proba = best_model.predict_proba(X_test)[:, 1]

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['n_estimators'] = 1000
    final_params['objective'] = 'binary'
    final_params['subsample_freq'] = 1
    final_params['random_state'] = 42
    final_params['verbosity'] = -1

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path,
        test_df=test_df_subset,
        y_test=y_test,
        best_y_probs=best_preds_proba
    )


def optimize_catboost_hyperparameters(
        filepath: str,
        dynamic_test_window: int = 60,
        n_trials: int = 50,
        output_json_path: str = "models/catboost_best_params.json"
):
    split_date = date.today() - timedelta(days=dynamic_test_window)
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

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
    test_df_subset = df.iloc[split_idx:].copy()

    print("=" * 60)
    print("      CATBOOST HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)

    train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)
    test_pool = Pool(X_test, y_test, cat_features=cat_features if cat_features else None)

    # Re-evaluate saved baseline using tuning iterations (300) and early stopping (15)
    def eval_cb(params):
        p = params.copy()
        p['iterations'] = 300  # Override final model 1000 iterations
        p['eval_metric'] = 'Logloss'
        p['thread_count'] = -1
        p['random_seed'] = 42
        p['verbose'] = False

        model = CatBoostClassifier(**p)
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=15, verbose=False)
        preds_proba = model.predict_proba(test_pool)[:, 1]
        return log_loss(y_test, preds_proba)

    refresh_baseline_if_exists(output_json_path, eval_cb)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.08, log=True),
            'depth': trial.suggest_int('depth', 6, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 8, 24.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.8, 4.0),
            'eval_metric': 'Logloss',
            'thread_count': -1,
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=15, verbose=False)

        preds_proba = model.predict_proba(test_pool)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Generate probabilities for best trial parameters
    best_eval_params = study.best_params.copy()
    best_eval_params['iterations'] = 300
    best_eval_params['eval_metric'] = 'Logloss'
    best_eval_params['thread_count'] = -1
    best_eval_params['random_seed'] = 42
    best_eval_params['verbose'] = False

    best_model = CatBoostClassifier(**best_eval_params)
    best_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=15, verbose=False)
    best_preds_proba = best_model.predict_proba(test_pool)[:, 1]

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['iterations'] = 1000
    final_params['eval_metric'] = 'Logloss'
    final_params['early_stopping_rounds'] = 30
    final_params['random_seed'] = 42

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path,
        test_df=test_df_subset,
        y_test=y_test,
        best_y_probs=best_preds_proba
    )


def optimize_elastictree_hyperparameters(
        filepath: str,
        dynamic_test_window: int = 60,
        n_trials: int = 50,
        output_json_path: str = "models/elastictree_best_params.json"
):
    split_date = date.today() - timedelta(days=dynamic_test_window)
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]
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
    test_df_subset = df.iloc[split_idx:].copy()

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

    # Re-evaluate saved baseline using tuning n_estimators (50)
    def eval_et(params):
        p = params.copy()
        p['n_estimators'] = 50  # Override final model 1000 estimators
        p['criterion'] = 'gini'
        p['random_state'] = 42
        p['n_jobs'] = -1

        model = ExtraTreesClassifier(**p)
        model.fit(X_train_np, y_train_np)
        preds_proba = model.predict_proba(X_test_np)[:, 1]
        return log_loss(y_test, preds_proba)

    refresh_baseline_if_exists(output_json_path, eval_et)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': 50,
            'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini']),
            'max_depth': trial.suggest_int('max_depth', 15, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 3, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_float('max_features', 0.05, 0.3, step=0.05, log=False),
            'random_state': 42,
            'n_jobs': 1
        }

        model = ExtraTreesClassifier(**params)
        model.fit(X_train_np, y_train_np)

        preds_proba = model.predict_proba(X_test_np)[:, 1]
        return log_loss(y_test, preds_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    # Generate probabilities for best trial parameters
    best_eval_params = study.best_params.copy()
    best_eval_params['n_estimators'] = 50
    best_eval_params['random_state'] = 42
    best_eval_params['n_jobs'] = -1

    best_model = ExtraTreesClassifier(**best_eval_params)
    best_model.fit(X_train_np, y_train_np)
    best_preds_proba = best_model.predict_proba(X_test_np)[:, 1]

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['n_estimators'] = 1000
    final_params['random_state'] = 42
    final_params['n_jobs'] = -1

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path,
        test_df=test_df_subset,
        y_test=y_test,
        best_y_probs=best_preds_proba
    )


def optimize_elasticnet_hyperparameters(
        filepath: str,
        dynamic_test_window: int = 60,
        n_trials: int = 50,
        output_json_path: str = "models/elasticnet_best_params.json"
):
    split_date = date.today() - timedelta(days=dynamic_test_window)
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].values

    cat_cols = champ_features
    num_cols = [c for c in feature_cols if c not in cat_cols]

    split_dt = pd.to_datetime(split_date)
    split_mask = df['date'] >= split_dt
    split_idx = int(split_mask.idxmax())

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]
    test_df_subset = df.iloc[split_idx:].copy()

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

    # Re-evaluate saved baseline using tuning max_iter (200) and tol (1e-2)
    def eval_en(params):
        p = params.copy()
        p['penalty'] = 'elasticnet'
        p['solver'] = 'saga'
        p['max_iter'] = 200  # Override final model 2000 max_iter
        p['tol'] = 1e-2
        p['random_state'] = 42

        model = LogisticRegression(**p)
        model.fit(X_train_proc, y_train)
        preds_proba = model.predict_proba(X_test_proc)[:, 1]
        return log_loss(y_test, preds_proba)

    refresh_baseline_if_exists(output_json_path, eval_en)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'C': trial.suggest_float('C', 1e-6, 0.01, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 0.3),
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

    # Generate probabilities for best trial parameters
    best_eval_params = study.best_params.copy()
    best_eval_params['penalty'] = 'elasticnet'
    best_eval_params['solver'] = 'saga'
    best_eval_params['max_iter'] = 200
    best_eval_params['tol'] = 1e-2
    best_eval_params['random_state'] = 42

    best_model = LogisticRegression(**best_eval_params)
    best_model.fit(X_train_proc, y_train)
    best_preds_proba = best_model.predict_proba(X_test_proc)[:, 1]

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['penalty'] = 'elasticnet'
    final_params['solver'] = 'saga'
    final_params['max_iter'] = 2000
    final_params['random_state'] = 42

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path,
        test_df=test_df_subset,
        y_test=y_test,
        best_y_probs=best_preds_proba
    )


if __name__ == "__main__":
    dataset_path = "../dataset/pregame/pregame_dataset_final_features.csv"
    i = 0

    while i < 10:
        optimize_xgboost_hyperparameters(
            filepath=dataset_path,
            n_trials=200,
            output_json_path="../models/best_params.json"
        )

        optimize_lightgbm_hyperparameters(
            filepath=dataset_path,
            n_trials=200,
            output_json_path="../models/best_lightgbm_params.json"
        )

        optimize_catboost_hyperparameters(
            filepath=dataset_path,
            n_trials=60,
            output_json_path="../models/catboost_best_params.json"
        )

        optimize_elastictree_hyperparameters(
            filepath=dataset_path,
            n_trials=100,
            output_json_path="../models/elastictree_best_params.json"
        )

        optimize_elasticnet_hyperparameters(
            filepath=dataset_path,
            n_trials=200,
            output_json_path="../models/elasticnet_best_params.json"
        )
        i += 1