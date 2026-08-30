import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score


def train_elasticnet_model(
        filepath: str = "dataset/pregame/pregame_dataset_final_features.csv",
        split_date: str = "2026-04-01",
        full_train: bool = False,
        params_json_path: str = "models/elasticnet_best_params.json",
        model_output_path: str = "models/elasticnet_model.joblib"
):
    """
    Trains an ElasticNet Logistic Regression model using a full scikit-learn Pipeline.

    Parameters:
    - filepath: Path to input CSV dataset.
    - split_date: Threshold date for train/test splitting (ignored if full_train=True).
    - full_train: If True, trains on the entire dataset without test evaluation.
    - params_json_path: Path to hyperparameter JSON file saved by an Optuna tuner.
    - model_output_path: Destination path to save the trained pipeline bundle.
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

    # Default parameters if JSON config is missing
    params = {
        'penalty': 'elasticnet',
        'solver': 'saga',
        'C': 0.1,
        'l1_ratio': 0.5,
        'max_iter': 2000,
        'random_state': 42,
        'n_jobs': -1
    }

    if os.path.exists(params_json_path):
        with open(params_json_path, 'r') as f:
            loaded_params = json.load(f)
            params.update(loaded_params)
        print(f"Loaded hyperparameters from '{params_json_path}'")
    else:
        print(f"Notice: Hyperparameter file '{params_json_path}' not found. Using default parameters.")

    # Enforce ElasticNet solver configuration
    params['penalty'] = 'elasticnet'
    params['solver'] = 'saga'
    params['n_jobs'] = -1

    # Preprocessing pipelines
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
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

    # Full Pipeline containing both feature transformers and model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**params))
    ])

    if full_train:
        print("=" * 60)
        print("         ELASTICNET MODEL TRAINING (FULL DATASET)        ")
        print("=" * 60)
        print(f"Training on all {len(X)} matches...")

        model_pipeline.fit(X, y)

        preds_proba = model_pipeline.predict_proba(X)[:, 1]
        loss = log_loss(y, preds_proba)
        acc = accuracy_score(y, (preds_proba >= 0.5).astype(int))
        auc = roc_auc_score(y, preds_proba)

        print(f"\nFull Dataset Metrics:")
        print(f"  -> Log-Loss: {loss:.4f}")
        print(f"  -> Accuracy: {acc:.4f}")
        print(f"  -> ROC-AUC:  {auc:.4f}")

    else:
        split_dt = pd.to_datetime(split_date)
        split_mask = df['date'] >= split_dt
        split_idx = int(split_mask.idxmax())

        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y[split_idx:]

        print("=" * 60)
        print("       ELASTICNET MODEL TRAINING (CHRONOLOGICAL SPLIT)   ")
        print("=" * 60)
        print(f"Train Set: {len(X_train)} matches | Test Set: {len(X_test)} matches (>= {split_date})")

        model_pipeline.fit(X_train, y_train)

        preds_proba = model_pipeline.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds_proba)
        acc = accuracy_score(y_test, (preds_proba >= 0.5).astype(int))
        auc = roc_auc_score(y_test, preds_proba)

        print("\nTest Set Metrics:")
        print(f"  -> Test Log-Loss: {loss:.4f}")
        print(f"  -> Test Accuracy: {acc:.4f}")
        print(f"  -> Test ROC-AUC:  {auc:.4f}")

    # Ensure output directory exists and save model pipeline
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_pipeline, model_output_path)
    print("=" * 60)
    print(f"Successfully saved full model pipeline to '{model_output_path}'")


if __name__ == "__main__":
    dataset_path = "../dataset/pregame/pregame_dataset_final_features.csv"

    # Train with chronological split
    train_elasticnet_model(
        filepath=dataset_path,
        split_date="2026-04-01",
        full_train=False,
        params_json_path="../models/elasticnet_best_params.json",
        model_output_path="../models/elasticnet_model.joblib"
    )

    # Train on full dataset (uncomment to run)
    # train_elasticnet_model(
    #     filepath=dataset_path,
    #     full_train=True,
    #     params_json_path="models/elasticnet_best_params.json",
    #     model_output_path="models/elasticnet_model_full.joblib"
    # )