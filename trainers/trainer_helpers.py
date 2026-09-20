import os

import numpy as np
import pandas as pd


def extract_features(df):
    # 2. Identify feature subsets
    elo_features = ['elo_diff', 'blue_elo_pre', 'red_elo_pre', 'blue_elo_win_prob', 'blue_firstpick']

    # Newly added Momentum Features
    momentum_features = [
        'blue_elo_delta_10', 'red_elo_delta_10', 'elo_delta_diff_10',
        'blue_overperform_10', 'red_overperform_10', 'overperformance_diff_10'
    ]

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
            momentum_features +
            series_features +
            player_features +
            h2h_matchup_features +
            synergy_roster_features +
            draft_champ_features +
            champ_features
    )
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col in df.columns]
    return feature_cols

def save_feature_importance(model, feature_cols, importance_output_path="feature_importance.csv"):
    # 1. Normalize input feature columns to a list of strings
    if isinstance(feature_cols, pd.DataFrame):
        feature_names = feature_cols.columns.tolist()
    elif isinstance(feature_cols, pd.Index):
        feature_names = feature_cols.tolist()
    else:
        feature_names = list(feature_cols)

    # 2. Handle Scikit-learn Pipeline models
    estimator = model
    if hasattr(model, "steps"):
        # Attempt to retrieve transformed feature names from preprocessing steps
        if len(model.steps) > 1:
            preprocessor = model[:-1]
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_names = list(preprocessor.get_feature_names_out(feature_names))
                except Exception:
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        pass
        estimator = model.steps[-1][1]

    # 3. Extract importance scores or coefficients
    if hasattr(estimator, "feature_importances_"):
        importance_scores = estimator.feature_importances_
    elif hasattr(estimator, "get_feature_importance"):
        importance_scores = estimator.get_feature_importance()
    elif hasattr(estimator, "coef_"):
        importance_scores = np.abs(estimator.coef_)
    else:
        raise AttributeError(
            f"Model step '{type(estimator).__name__}' does not expose feature importances or coefficients."
        )

    # 4. Flatten score array to 1D
    importance_scores = np.ravel(importance_scores)

    # 5. Length verification & safety fallback
    if len(feature_names) != len(importance_scores):
        print(f"⚠️ Notice: Pipeline transformed features ({len(importance_scores)}) "
              f"differ from raw columns ({len(feature_names)}). Adjusting names...")
        if len(feature_names) < len(importance_scores):
            # Fill missing names created by encoding steps
            extended_names = feature_names.copy()
            extended_names.extend(
                [f"transformed_feature_{i}" for i in range(len(feature_names), len(importance_scores))]
            )
            feature_names = extended_names
        else:
            feature_names = feature_names[:len(importance_scores)]

    # 6. Build and sort feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nTop 15 Most Influential Features:")
    print(importance_df.head(15).to_string(index=False))

    # 7. Save cleanly to CSV
    dir_name = os.path.dirname(importance_output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    importance_df.to_csv(importance_output_path, index=False)
    print(f"\n[SUCCESS] Feature importances saved to '{importance_output_path}'")