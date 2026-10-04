import os
import json
from datetime import date, timedelta

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

from trainers.trainer_helpers import save_feature_importance
from util import save_validation_predictions


def train_secondary_model(
        dataset_path: str = "dataset/pregame/pregame_dataset_final_features.csv",
        model_output_path: str = "models/lightgbm_model.pkl",
        params_path: str = "models/best_lightgbm_params.json",
        dynamic_test_window: int = 60,
        only_full_train: bool = False,
        importance_output_path: str = "metadata/lightgbm_feature_importance.csv",
        predictions_output_path: str = "metadata/predictions/lightgbm_predictions.csv"
):
    """
    Trains a LightGBM secondary model using best parameters from tuner if available.

    By default (only_full_train=False):
      Phase 1: Trains on historical split, evaluates on validation test window, prints performance, and saves validation predictions.
      Phase 2: Trains on the 100% full dataset and saves the final model artifact.
    """
    # 1. Load dataset & sort chronologically
    df = pd.read_csv(dataset_path, low_memory=False)

    if "date" not in df.columns:
        raise KeyError("Dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    target_col = 'blue_win'
    exclude_cols = [
        "gameid", "date", "blue_team", "red_team",
        "blue_teamid", "red_teamid", target_col
    ]
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    print(f"Loaded {len(df)} matches. Total features selected for LightGBM training: {len(feature_cols)}")

    # 2. Load Tuned Hyperparameters (or Fallback to Defaults)
    default_params = {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1
    }

    if os.path.exists(params_path):
        print(f"[*] Found tuned hyperparameter file at '{params_path}'. Loading...")
        try:
            with open(params_path, "r") as f:
                params = json.load(f)

            # Cast integer hyperparameter values explicitly
            int_keys = ["n_estimators", "max_depth", "num_leaves", "min_child_samples", "subsample_freq", "max_bin"]
            for k in int_keys:
                if k in params:
                    params[k] = int(params[k])

            params["random_state"] = 42
            params["verbosity"] = -1
            print(f"[✓] Applied tuned parameters successfully.")
        except Exception as e:
            print(f"[!] Error loading '{params_path}': {e}. Falling back to default parameters.")
            params = default_params
    else:
        print(f"[!] No tuned hyperparameter file found at '{params_path}'. Using default parameters.")
        params = default_params

    params.pop('best_logloss', None)

    # 3. Check Test Split Availability
    latest_dataset_date = df['date'].max()
    split_date = date.today() - timedelta(days=dynamic_test_window)
    cutoff_dt = pd.to_datetime(split_date)

    if cutoff_dt >= latest_dataset_date:
        print(f"\n[INFO] cutoff_date '{cutoff_dt.strftime('%Y-%m-%d')}' is on/after latest match ({latest_dataset_date.strftime('%Y-%m-%d')}). Skipping test validation phase.")
        only_full_train = True

    # -------------------------------------------------------------------------
    # PHASE 1: Validation Run & Prediction Export (Skipped if only_full_train=True)
    # -------------------------------------------------------------------------
    if not only_full_train:
        train_mask = df["date"] < cutoff_dt
        test_mask = df["date"] >= cutoff_dt

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col].values
        X_val = df.loc[test_mask, feature_cols]
        y_val = df.loc[test_mask, target_col].values
        test_df = df.loc[test_mask].copy()

        if len(X_train) == 0 or len(X_val) == 0:
            min_date = df['date'].min().strftime('%Y-%m-%d')
            max_date = df['date'].max().strftime('%Y-%m-%d')
            raise ValueError(
                f"Invalid split_date '{split_date}'. "
                f"Dataset date range spans from {min_date} to {max_date}."
            )

        print("\n" + "=" * 55)
        print("    PHASE 1: LIGHTGBM VALIDATION & EVALUATION    ")
        print("=" * 55)
        print(f"Train set: {len(X_train)} matches ({df.loc[train_mask, 'date'].min().strftime('%Y-%m-%d')} to {df.loc[train_mask, 'date'].max().strftime('%Y-%m-%d')})")
        print(f"Test set:  {len(X_val)} matches ({cutoff_dt.strftime('%Y-%m-%d')} to {df.loc[test_mask, 'date'].max().strftime('%Y-%m-%d')})")
        print("-" * 55)

        val_model = LGBMClassifier(**params)
        val_model.fit(X_train, y_train)

        val_probs = val_model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        acc = accuracy_score(y_val, val_preds)
        loss = log_loss(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        brier = brier_score_loss(y_val, val_probs)

        print("\n" + "=" * 45)
        print("      LIGHTGBM PREDICTION EVALUATION MODEL     ")
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
    print("      PHASE 2: LIGHTGBM FULL DATASET TRAINING     ")
    print(f"Training on all {len(df)} matches ({start_dt} to {end_dt})")
    print("=" * 55)

    X_full = df[feature_cols]
    y_full = df[target_col].values

    final_model = LGBMClassifier(**params)
    final_model.fit(X_full, y_full)

    # Save artifact (model + feature names)
    os.makedirs(os.path.dirname(model_output_path) or '.', exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_names": feature_cols
    }
    joblib.dump(artifact, model_output_path)
    print(f"[✓] Model 2 successfully saved to '{model_output_path}'")

    # Export Feature Importance
    save_feature_importance(final_model, feature_cols, importance_output_path)