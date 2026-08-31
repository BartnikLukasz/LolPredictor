import os
import json
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report

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
        "verbose": 100
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


def extract_feature_matrix(df: pd.DataFrame):
    """Extracts identical feature subsets used across XGBoost, LightGBM, and CatBoost engines."""
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

    # Cast champion categorical columns explicitly to string for CatBoost
    cat_features = [c for c in champ_features if c in X.columns]
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    return X, cat_features


def train_catboost(
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

    # 2. Extract Feature Matrix
    X, cat_features = extract_feature_matrix(df)
    y = df[TARGET_COL].values

    # 3. Handling Split vs Full Dataset Training
    print("=" * 60)
    print("                CATBOOST MODEL TRAINING                 ")
    print("=" * 60)
    print(f"Dataset Loaded: {len(df)} matches | Features: {len(X.columns)}")
    print(f"Categorical Features ({len(cat_features)}): {cat_features}")

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

    # 4. Load Params and Train
    params = load_best_params(params_path)
    params.pop('best_logloss', None)
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)

    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)

    print("\n🚀 Starting CatBoost Training...")
    if full_train or X_val is None:
        model.fit(train_pool)
        eval_pool = train_pool
        eval_y = y_train
        eval_label = "TRAINING (FULL DATASET)"
    else:
        val_pool = Pool(X_val, y_val, cat_features=cat_features if cat_features else None)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True
        )
        eval_pool = val_pool
        eval_y = y_val
        eval_label = "VALIDATION"

    # 5. Evaluate Metrics
    eval_preds_prob = model.predict_proba(eval_pool)[:, 1]
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

    # 6. Save Artifact
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "cat_features": cat_features,
        "metrics": {"accuracy": acc, "roc_auc": auc, "log_loss": loss}
    }

    joblib.dump(artifact, output_model_path)
    print(f"\n💾 Trained CatBoost model saved to '{output_model_path}'!")


if __name__ == "__main__":
    train_catboost(
        filepath=DATASET_PATH,
        split_date="2026-04-01",
        full_train=False,  # Set to True for production model building
        params_path=PARAMS_PATH,
        output_model_path=MODEL_OUTPUT_PATH
    )