import os
import json
from datetime import date, timedelta

import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    classification_report
)

from trainers.trainer_helpers import extract_features, save_feature_importance
from util import save_validation_predictions

# --- PATH CONFIGURATION ---
DATASET_PATH = "dataset/pregame/pregame_dataset_final_features.csv"
PARAMS_PATH = "models/catboost_best_params.json"
MODEL_OUTPUT_PATH = "models/catboost_model.pkl"
TARGET_COL = "blue_win"


def load_best_params(params_path: str) -> dict:
    """Loads CatBoost hyperparameters from JSON if present, otherwise returns defaults."""
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": False
    }

    if os.path.exists(params_path):
        print(f"[CONFIG] Loading custom hyperparameters from '{params_path}'...")
        try:
            with open(params_path, "r") as f:
                user_params = json.load(f)
            default_params.update(user_params)
            default_params["verbose"] = False
        except Exception as e:
            print(f"[!] Error reading JSON parameter file ({e}). Falling back to defaults.")
    else:
        print(f"[CONFIG] Parameter file '{params_path}' not found. Using default hyperparameter values.")

    return default_params


def extract_feature_matrix(df: pd.DataFrame):
    """Extracts identical feature subsets used across XGBoost, LightGBM, and CatBoost engines."""
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    X = df[feature_cols].copy()

    # Cast champion categorical columns explicitly to string for CatBoost
    cat_features = [c for c in champ_features if c in X.columns]
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    return X, cat_features


def train_catboost(
        filepath: str = DATASET_PATH,
        dynamic_test_window: int = 60,
        only_full_train: bool = False,
        params_path: str = PARAMS_PATH,
        output_model_path: str = MODEL_OUTPUT_PATH,
        importance_output_path: str = "metadata/catboost_feature_importance.csv",
        predictions_output_path: str = "metadata/predictions/catboost_predictions.csv"
):
    """
    Trains a CatBoost model using best parameters from tuner if available.

    By default (only_full_train=False):
      Phase 1: Trains on historical split, evaluates on validation test window, prints performance, and saves validation predictions.
      Phase 2: Trains on the 100% full dataset and saves the final model artifact.
    """
    # 1. Load Data & Sort Chronologically
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    print(f"Reading dataset from '{filepath}'...")
    df = pd.read_csv(filepath, low_memory=False)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Extract Feature Matrix
    X, cat_features = extract_feature_matrix(df)
    y = df[TARGET_COL].values

    print(f"Loaded {len(df)} matches | Features selected: {len(X.columns)}")
    print(f"Categorical Champion Features ({len(cat_features)}): {cat_features}")

    # 3. Load Hyperparameters
    params = load_best_params(params_path)
    params.pop('best_logloss', None)

    # 4. Check Test Split Availability
    latest_dataset_date = df['date'].max()
    split_date = date.today() - timedelta(days=dynamic_test_window)
    split_dt = pd.to_datetime(split_date)

    if split_dt >= latest_dataset_date:
        print(f"\n[INFO] split_date '{split_date}' is on/after latest match ({latest_dataset_date.strftime('%Y-%m-%d')}). Skipping test validation phase.")
        only_full_train = True

    # -------------------------------------------------------------------------
    # PHASE 1: Validation Run & Prediction Export (Skipped if only_full_train=True)
    # -------------------------------------------------------------------------
    if not only_full_train:
        split_mask = df['date'] >= split_dt
        split_idx = int(split_mask.idxmax())

        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y[split_idx:]
        test_df = df.iloc[split_idx:].copy()

        train_dates = df['date'].iloc[:split_idx]
        test_dates = df['date'].iloc[split_idx:]

        print("\n" + "=" * 55)
        print("     PHASE 1: CATBOOST VALIDATION & EVALUATION     ")
        print("=" * 55)
        print(f"Train Period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_train)} matches)")
        print(f"Test Period:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(X_val)} matches)")
        print("-" * 55)

        val_params = params.copy()
        early_stopping_rounds = val_params.pop("early_stopping_rounds", 30)

        train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)
        val_pool = Pool(X_val, y_val, cat_features=cat_features if cat_features else None)

        val_model = CatBoostClassifier(**val_params)
        val_model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False
        )

        val_probs = val_model.predict_proba(val_pool)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        acc = accuracy_score(y_val, val_preds)
        loss = log_loss(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        brier = brier_score_loss(y_val, val_probs)

        print("\n" + "=" * 45)
        print("      CATBOOST PREDICTION EVALUATION MODEL     ")
        print("=" * 45)
        print(f"Accuracy:    {acc * 100:.2f}%")
        print(f"Log-Loss:    {loss:.4f} (Baseline ~0.693)")
        print(f"ROC-AUC:     {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print("=" * 45)

        # Save test validation predictions CSV
        save_validation_predictions(test_df, y_val, val_probs, predictions_output_path)

    # -------------------------------------------------------------------------
    # PHASE 2: Full Dataset Training & Model Export
    # -------------------------------------------------------------------------
    start_dt = df['date'].min().strftime('%Y-%m-%d')
    end_dt = df['date'].max().strftime('%Y-%m-%d')

    print("\n" + "=" * 55)
    print("      PHASE 2: CATBOOST FULL DATASET TRAINING      ")
    print(f"Training on all {len(df)} matches ({start_dt} to {end_dt})")
    print("=" * 55)

    full_params = params.copy()
    full_params.pop("early_stopping_rounds", None)

    full_pool = Pool(X, y, cat_features=cat_features if cat_features else None)
    final_model = CatBoostClassifier(**full_params)
    final_model.fit(full_pool, verbose=False)

    # Save artifact
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_names": list(X.columns),
        "cat_features": cat_features
    }

    joblib.dump(artifact, output_model_path)
    print(f"[✓] Trained CatBoost model successfully saved to '{output_model_path}'")

    # Export Feature Importance
    save_feature_importance(final_model, X, importance_output_path)