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
from util import refresh_baseline_if_exists, save_best_params_if_improved

# Silence Optuna's verbose per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        output_json_path=output_json_path
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

    print("=" * 60)
    print("      LIGHTGBM HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)

    def eval_lgbm(params):
        p = params.copy()
        p['n_estimators'] = 200
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
        output_json_path=output_json_path
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

    print("=" * 60)
    print("      CATBOOST HYPERPARAMETER OPTIMIZATION (OPTUNA)     ")
    print("=" * 60)

    train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)
    test_pool = Pool(X_test, y_test, cat_features=cat_features if cat_features else None)

    def eval_cb(params):
        p = params.copy()
        p['iterations'] = 300
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

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['iterations'] = 1000
    final_params['eval_metric'] = 'Logloss'
    final_params['early_stopping_rounds'] = 30
    final_params['random_seed'] = 42

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path
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

    def eval_et(params):
        p = params.copy()
        p['n_estimators'] = 50
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
            'max_depth': trial.suggest_int('max_depth', 10, 25),
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

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['n_estimators'] = 1000
    final_params['random_state'] = 42
    final_params['n_jobs'] = -1

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path
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

    def eval_en(params):
        p = params.copy()
        p['penalty'] = 'elasticnet'
        p['solver'] = 'saga'
        p['max_iter'] = 200
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

    # Final params for full training
    final_params = study.best_params.copy()
    final_params['penalty'] = 'elasticnet'
    final_params['solver'] = 'saga'
    final_params['max_iter'] = 2000
    final_params['random_state'] = 42

    save_best_params_if_improved(
        best_params=final_params,
        current_logloss=study.best_value,
        output_json_path=output_json_path
    )


if __name__ == "__main__":
    dataset_path = "../dataset/pregame/pregame_dataset_final_features.csv"
    i = 0

    while i < 2:
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