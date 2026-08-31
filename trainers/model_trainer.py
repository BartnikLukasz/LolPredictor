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


def train_lol_prediction_model(
        filepath: str,
        test_split_ratio: float = 0.20,
        split_date: str = None,
        full_train: bool = False,
        params_filepath: str = "models/best_params.json",
        output_model_path: str = "models/xgboost_model.json"
) -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    """
    Trains and evaluates an XGBoost model on pre-game LoL match features.

    Parameters:
        filepath (str): Path to enriched CSV dataset.
        test_split_ratio (float): Proportion reserved for testing (if split_date is None and full_train=False).
        split_date (str, optional): Cut-off date ('YYYY-MM-DD').
        full_train (bool): If True, trains on 100% of historical data for production deployment.
        params_filepath (str): Path to best_params.json created by hyperparameter_tuner.py.
        output_model_path (str): Destination path to save the trained model binary.
    """
    # 1. Load dataset and sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'blue_win'

    # 2. Identify feature subsets
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
        # Remove early_stopping_rounds when there is no evaluation set
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

    # 8. Feature Importance Analysis
    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nTop 15 Most Influential Features:")
    print(importance_df.head(15).to_string(index=False))

    # 9. Export Trained Model Binary
    model.save_model(output_model_path)
    print(f"\n[SUCCESS] Model binary successfully saved to '{output_model_path}'")

    return model, importance_df


if __name__ == "__main__":
    dataset_path = "multi_year_pregame_dataset_final_features.csv"

    # Set full_train=True when training for live production deployment up to today
    model, feature_importance = train_lol_prediction_model(
        filepath=dataset_path,
        full_train=True
    )