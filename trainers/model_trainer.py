import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    classification_report
)
from xgboost import XGBClassifier

from trainers.trainer_helpers import extract_features, save_feature_importance


def train_lol_prediction_model(
        filepath: str,
        test_split_ratio: float = 0.20,
        split_date: str = None,
        full_train: bool = False,
        params_filepath: str = "models/best_params.json",
        output_model_path: str = "models/xgboost_model.json",
        importance_output_path: str = "metadata/xgboost_feature_importance.csv"
) -> XGBClassifier:
    """Trains and evaluates an XGBoost model on pre-game LoL match features including momentum."""
    # 1. Load dataset, sort chronologically, and enrich with Momentum Features
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

    print(f"Loaded {len(df)} matches. Total features selected for training: {len(feature_cols)}")

    # 3. Preprocess Categorical Champion Features
    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    # 4. Determine Split Strategy
    latest_dataset_date = df['date'].max()

    if split_date:
        split_dt = pd.to_datetime(split_date)
        if split_dt >= latest_dataset_date:
            print(
                f"\n[INFO] split_date '{split_date}' is on/after latest match ({latest_dataset_date.strftime('%Y-%m-%d')}). Switching to full training mode.")
            full_train = True

    if full_train:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print("\n" + "-" * 55)
        print(f"TRAINING MODE: Full Dataset ({len(X_train)} matches up to {latest_dataset_date.strftime('%Y-%m-%d')})")
        print("-" * 55)
    else:
        if split_date:
            split_mask = df['date'] >= split_dt
            split_idx = int(split_mask.idxmax())
        else:
            split_idx = int(len(df) * (1 - test_split_ratio))

        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y[split_idx:]

        train_dates = df['date'].iloc[:split_idx]
        test_dates = df['date'].iloc[split_idx:]

        print("\n" + "-" * 55)
        print(
            f"Train Period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_train)} matches)")
        print(
            f"Test Period:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(X_test)} matches)")
        print("-" * 55)

    # 5. Load Hyperparameters dynamically
    if os.path.exists(params_filepath):
        print(f"[CONFIG] Found '{params_filepath}'. Loading optimized hyperparameters...")
        with open(params_filepath, 'r') as f:
            model_params = json.load(f)
    else:
        print(f"[CONFIG] '{params_filepath}' not found. Using default XGBoost hyperparameters...")
        model_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'enable_categorical': True,
            'random_state': 42
        }

    # 6. Train XGBoost Model
    model_params.pop('best_logloss', None)
    if full_train:
        model_params.pop('early_stopping_rounds', None)
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, verbose=False)
    else:
        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

    # 7. Model Evaluation (Only when test set exists)
    if not full_train and X_test is not None:
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_binary = (preds_proba >= 0.50).astype(int)

        acc = accuracy_score(y_test, preds_binary)
        loss = log_loss(y_test, preds_proba)
        auc = roc_auc_score(y_test, preds_proba)
        brier = brier_score_loss(y_test, preds_proba)

        print("\n" + "=" * 45)
        print("      MATCH PREDICTION EVALUATION MODEL     ")
        print("=" * 45)
        print(f"Accuracy:    {acc * 100:.2f}%")
        print(f"Log-Loss:    {loss:.4f} (Baseline ~0.693)")
        print(f"ROC-AUC:     {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print("=" * 45)

        test_df = df.iloc[len(X_train):].copy()
        test_df['pred_proba'] = preds_proba
        test_df['pred_binary'] = preds_binary

        league_stats = []
        for league_name, group in test_df.groupby('league'):
            y_sub = group[target_col].values
            p_sub = group['pred_proba'].values
            b_sub = group['pred_binary'].values

            acc_sub = accuracy_score(y_sub, b_sub)
            loss_sub = log_loss(y_sub, p_sub, labels=[0, 1])
            auc_sub = round(roc_auc_score(y_sub, p_sub), 4) if len(np.unique(y_sub)) > 1 else "N/A"

            league_stats.append({
                'League': league_name,
                'Matches': len(group),
                'Accuracy (%)': round(acc_sub * 100, 2),
                'Log-Loss': round(loss_sub, 4),
                'ROC-AUC': auc_sub
            })

        league_summary = pd.DataFrame(league_stats).sort_values('Matches', ascending=False).reset_index(drop=True)
        print("\n" + "=" * 55)
        print("          PERFORMANCE ACCURACY BY LEAGUE           ")
        print("=" * 55)
        print(league_summary.to_string(index=False))
        print("=" * 55)

    # 8. Feature Importance Analysis & Export
    save_feature_importance(model, feature_cols, importance_output_path)

    # 9. Export Trained Model Binary
    model.save_model(output_model_path)
    print(f"[SUCCESS] Model binary successfully saved to '{output_model_path}'")

    return model