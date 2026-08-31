import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report

# --- PATH CONFIGURATION ---
DATASET_PATH = "dataset/pregame/pregame_dataset_final_features.csv"
PARAMS_PATH = "models/elastictree_best_params.json"
MODEL_OUTPUT_PATH = "models/elastictree_model.pkl"
TARGET_COL = "blue_win"


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
        print(f"📥 Loading custom hyperparameters from '{params_path}'...")
        try:
            with open(params_path, "r") as f:
                user_params = json.load(f)
            default_params.update(user_params)
        except Exception as e:
            print(f"⚠️ Error reading JSON parameter file ({e}). Falling back to defaults.")
    else:
        print(f"ℹ️ Parameter file '{params_path}' not found. Using default hyperparameter values.")

    return default_params


def select_feature_columns(df: pd.DataFrame):
    """Extracts identical feature lists for numeric and categorical columns."""
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

    cat_cols = champ_features
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return feature_cols, num_cols, cat_cols


def train_elastictree(
        filepath: str = DATASET_PATH,
        split_date: str = "2026-04-01",
        full_train: bool = False,
        params_path: str = PARAMS_PATH,
        output_model_path: str = MODEL_OUTPUT_PATH
):
    # 1. Load Data & Sort Chronologically
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    print(f"📊 Reading dataset from '{filepath}'...")
    df = pd.read_csv(filepath, low_memory=False)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Extract Feature Subsets (Raw un-encoded DataFrames)
    feature_cols, num_cols, cat_cols = select_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values

    # 3. Splitting Strategy
    print("=" * 60)
    print("              ELASTICTREE MODEL TRAINING                ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Raw Features: {len(feature_cols)}")

    if full_train:
        print("🌐 Mode: FULL DATASET TRAINING (Ignoring split date)")
        X_train, y_train = X, y
        X_val, y_val = None, None
        print(f"Training Set Size: {len(X_train)} matches (100% of data)")
    else:
        print(f"📅 Mode: DATE-BASED SPLIT (Split Date: {split_date})")
        split_dt = pd.to_datetime(split_date)
        split_mask = df['date'] >= split_dt
        split_idx = int(split_mask.idxmax())

        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y[split_idx:]
        print(f"Train Set: {len(X_train)} matches (< {split_date})")
        print(f"Validation Set: {len(X_val)} matches (>= {split_date})")
    print("=" * 60)

    # 4. Construct Preprocessing & Model Pipeline
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

    params = load_best_params(params_path)
    params.pop('best_logloss', None)
    tree_model = ExtraTreesClassifier(**params)

    # Wrap preprocessor and classifier in a single Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', tree_model)
    ])

    print("\n🚀 Starting ElasticTree (Pipeline) Training...")
    full_pipeline.fit(X_train, y_train)

    # 5. Evaluate Metrics
    if full_train or X_val is None:
        eval_X, eval_y = X_train, y_train
        eval_label = "TRAINING (FULL DATASET)"
    else:
        eval_X, eval_y = X_val, y_val
        eval_label = "VALIDATION"

    eval_preds_prob = full_pipeline.predict_proba(eval_X)[:, 1]
    eval_preds_binary = (eval_preds_prob >= 0.5).astype(int)

    acc = accuracy_score(eval_y, eval_preds_binary)
    auc = roc_auc_score(eval_y, eval_preds_prob)
    loss = log_loss(eval_y, eval_preds_prob)

    print("\n" + "=" * 60)
    print(f"🎯 PERFORMANCE METRICS ({eval_label})")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Log Loss : {loss:.4f}")
    print("-" * 60)
    print(classification_report(eval_y, eval_preds_binary, digits=4))

    # 6. Save Full Pipeline Artifact
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    artifact = {
        "pipeline": full_pipeline,
        "model": full_pipeline,  # Assigned to both keys for backward compatibility with app.py
        "feature_cols": feature_cols,
        "metrics": {"accuracy": acc, "roc_auc": auc, "log_loss": loss}
    }

    joblib.dump(artifact, output_model_path, compress=3)
    print(f"\n💾 Trained ElasticTree Pipeline saved to '{output_model_path}'!")


if __name__ == "__main__":
    train_elastictree(
        filepath=DATASET_PATH,
        split_date="2026-04-01",
        full_train=False,
        params_path=PARAMS_PATH,
        output_model_path=MODEL_OUTPUT_PATH
    )