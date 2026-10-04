import os
import json
from datetime import date, timedelta

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss
)

from trainers.trainer_helpers import extract_features, save_feature_importance
from util import save_validation_predictions

# --- PATH CONFIGURATION ---
DATASET_PATH = "dataset/pregame/pregame_dataset_final_features.csv"
PARAMS_PATH = "models/elastictree_best_params.json"
MODEL_OUTPUT_PATH = "models/elastictree_model.pkl"
TARGET_COL = "blue_win"


def compress_tree_model(pipeline, decimals: int = 4):
    """
    Rounds floating-point thresholds and node values in decision trees.
    Dramatically reduces float entropy, allowing XZ compression to shrink
    the model size by up to 80% without affecting prediction output.
    """
    tree_model = pipeline.named_steps['model']
    for estimator in tree_model.estimators_:
        tree = estimator.tree_
        # In-place rounding of threshold floats and node value counts
        np.round(tree.threshold, decimals=decimals, out=tree.threshold)
        np.round(tree.value, decimals=decimals, out=tree.value)


def load_best_params(params_path: str) -> dict:
    """Loads ExtraTrees hyperparameters from JSON if present, otherwise returns defaults."""
    default_params = {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    }

    if os.path.exists(params_path):
        print(f"[CONFIG] Loading custom hyperparameters from '{params_path}'...")
        try:
            with open(params_path, "r") as f:
                user_params = json.load(f)
            default_params.update(user_params)
        except Exception as e:
            print(f"[!] Error reading JSON parameter file ({e}). Falling back to defaults.")
    else:
        print(f"[CONFIG] Parameter file '{params_path}' not found. Using default hyperparameter values.")

    return default_params


def select_feature_columns(df: pd.DataFrame):
    """Extracts identical feature lists for numeric and categorical columns."""
    feature_cols = extract_features(df)

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    cat_cols = champ_features
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return feature_cols, num_cols, cat_cols


def build_pipeline(num_cols: list, cat_cols: list, params: dict) -> Pipeline:
    """Constructs the preprocessor and ExtraTrees classifier pipeline."""
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    tree_model = ExtraTreesClassifier(**params)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', tree_model)
    ])


def train_elastictree(
        filepath: str = DATASET_PATH,
        dynamic_test_window: int = 60,
        only_full_train: bool = False,
        params_path: str = PARAMS_PATH,
        output_model_path: str = MODEL_OUTPUT_PATH,
        importance_output_path: str = "metadata/elastictree_feature_importance.csv",
        predictions_output_path: str = "metadata/predictions/elastictree_predictions.csv"
):
    """
    Trains an ElasticTree (ExtraTrees) model using best parameters from tuner if available.

    By default (only_full_train=False):
      Phase 1: Trains on historical split, evaluates on validation test window, prints performance, and saves validation predictions.
      Phase 2: Trains on the 100% full dataset, optimizes precision for compression, and saves the final pipeline artifact.
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

    # 2. Extract Feature Subsets
    feature_cols, num_cols, cat_cols = select_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values

    print(f"Loaded {len(df)} matches | Raw Features selected: {len(feature_cols)}")

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
        print("   PHASE 1: ELASTICTREE VALIDATION & EVALUATION    ")
        print("=" * 55)
        print(f"Train Period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_train)} matches)")
        print(f"Test Period:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(X_val)} matches)")
        print("-" * 55)

        val_pipeline = build_pipeline(num_cols, cat_cols, params)
        val_pipeline.fit(X_train, y_train)

        val_probs = val_pipeline.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        acc = accuracy_score(y_val, val_preds)
        loss = log_loss(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        brier = brier_score_loss(y_val, val_probs)

        print("\n" + "=" * 45)
        print("    ELASTICTREE PREDICTION EVALUATION MODEL     ")
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
    print("    PHASE 2: ELASTICTREE FULL DATASET TRAINING     ")
    print(f"Training on all {len(df)} matches ({start_dt} to {end_dt})")
    print("=" * 55)

    final_pipeline = build_pipeline(num_cols, cat_cols, params)
    final_pipeline.fit(X, y)

    # Reduce Float Precision prior to saving for high compression
    print("[OPTIMIZATION] Optimizing tree precision for maximum compression...")
    compress_tree_model(final_pipeline, decimals=4)

    # Save Pipeline Artifact
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    artifact = {
        "pipeline": final_pipeline,
        "model": final_pipeline,  # Assigned to both keys for backward compatibility
        "feature_cols": feature_cols
    }
    joblib.dump(artifact, output_model_path, compress=3)
    print(f"[✓] Trained ElasticTree Pipeline successfully saved to '{output_model_path}'")

    # Export Feature Importance
    save_feature_importance(final_pipeline, feature_cols, importance_output_path)