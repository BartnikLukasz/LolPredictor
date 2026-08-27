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
        split_date: str = None
) -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    """
    Trains and evaluates an XGBoost model on pre-game LoL match features
    (including Elo, Player Mastery, Team H2H, Lane/P2P Matchups, Duo Synergies, and Roster Continuity)
    and computes per-league performance metrics on the test set.

    Parameters:
        filepath (str): Path to enriched CSV dataset.
        test_split_ratio (float): Proportion of recent matches reserved for testing (used if split_date is None).
        split_date (str, optional): Cut-off date ('YYYY-MM-DD'). Matches before this date become
                                    training data; matches on or after become testing data.

    Returns:
        tuple: (Trained XGBoost model, Feature Importance DataFrame)
    """
    # 1. Load dataset and sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Define target variable
    target_col = 'blue_win'

    # 3. Identify feature subsets
    elo_features = ['elo_diff', 'blue_elo_pre', 'red_elo_pre', 'blue_elo_win_prob', 'blue_firstpick']

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

    champ_features = [
        'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
        'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
    ]
    champ_features = [c for c in champ_features if c in df.columns]

    # Combine all feature groups into a unique list
    feature_cols = (
        elo_features +
        player_features +
        h2h_matchup_features +
        synergy_roster_features +
        champ_features
    )
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col in df.columns]

    print(f"Loaded {len(df)} matches. Total features selected for training: {len(feature_cols)}")
    print(f" -> Elo Features:          {len([f for f in elo_features if f in df.columns])}")
    print(f" -> Player/Mastery:        {len(player_features)}")
    print(f" -> H2H & Lane Matchups:   {len(h2h_matchup_features)}")
    print(f" -> Duo & Roster Synergy:  {len(synergy_roster_features)}")
    print(f" -> Categorical Champions: {len(champ_features)}")

    # 4. Preprocess Categorical Champion Features for XGBoost
    X = df[feature_cols].copy()
    y = df[target_col].values

    for col in champ_features:
        X[col] = X[col].astype('category')

    # 5. Chronological Train / Test Split
    if split_date:
        split_dt = pd.to_datetime(split_date)
        split_mask = df['date'] >= split_dt
        if not split_mask.any():
            raise ValueError(f"No matches found on or after split_date '{split_date}'.")
        if split_mask.all():
            raise ValueError(f"All matches are on or after split_date '{split_date}'. No training data available.")
        split_idx = int(split_mask.idxmax())
    else:
        split_idx = int(len(df) * (1 - test_split_ratio))

    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    train_dates = df['date'].iloc[:split_idx]
    test_dates = df['date'].iloc[split_idx:]

    print("\n" + "-" * 55)
    print(f"Train Period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} ({len(X_train)} matches)")
    print(f"Test Period:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} ({len(X_test)} matches)")
    print("-" * 55)

    # 6. Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        enable_categorical=True,
        early_stopping_rounds=30,
        random_state=42
    )

    # 7. Train Model with Validation Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 8. Overall Model Evaluation
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

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds_binary, target_names=['Red Win', 'Blue Win']))

    # 9. Breakdown Evaluation Per League
    test_df = df.iloc[split_idx:].copy()
    test_df['pred_proba'] = preds_proba
    test_df['pred_binary'] = preds_binary

    league_stats = []
    for league_name, group in test_df.groupby('league'):
        y_sub = group[target_col].values
        p_sub = group['pred_proba'].values
        b_sub = group['pred_binary'].values

        acc_sub = accuracy_score(y_sub, b_sub)
        loss_sub = log_loss(y_sub, p_sub, labels=[0, 1])

        if len(np.unique(y_sub)) > 1:
            auc_sub = round(roc_auc_score(y_sub, p_sub), 4)
        else:
            auc_sub = "N/A"

        league_stats.append({
            'League': league_name,
            'Matches': len(group),
            'Accuracy (%)': round(acc_sub * 100, 2),
            'Log-Loss': round(loss_sub, 4),
            'ROC-AUC': auc_sub
        })

    league_summary = (
        pd.DataFrame(league_stats)
        .sort_values('Matches', ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 55)
    print("          PERFORMANCE ACCURACY BY LEAGUE           ")
    print("=" * 55)
    print(league_summary.to_string(index=False))
    print("=" * 55)

    # 10. Feature Importance Analysis
    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nTop 20 Most Influential Pre-Game Features:")
    print(importance_df.head(20).to_string(index=False))

    return model, importance_df


if __name__ == "__main__":
    dataset_path = "multi_year_pregame_dataset_final_features.csv"

    # Example 1: Standard ratio split (20% test)
    # model, feature_importance = train_lol_prediction_model(
    #     filepath=dataset_path,
    #     test_split_ratio=0.20
    # )

    # Example 2: Train on pre-2025 data, test strictly on 2025 onwards
    model, feature_importance = train_lol_prediction_model(
        filepath=dataset_path,
        split_date="2025-01-01"
    )