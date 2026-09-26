import os
import pandas as pd
import numpy as np


def compare_model_csvs(live_csv_path: str, optuna_csv_path: str, output_csv_path: str):
    """
    Matches live predictions with Optuna validation predictions from newest to oldest.
    Prioritizes 'game_id' matching if present in both datasets, falling back to
    team-pair reverse matching for older records lacking game_id.
    """
    if not os.path.exists(live_csv_path):
        print(f"[!] Live CSV missing: '{live_csv_path}'")
        return
    if not os.path.exists(optuna_csv_path):
        print(f"[!] Optuna CSV missing: '{optuna_csv_path}'")
        return

    df_live = pd.read_csv(live_csv_path)
    df_optuna = pd.read_csv(optuna_csv_path)

    # Standardize and sort by datetime ascending
    df_live['datetime'] = pd.to_datetime(df_live['datetime'])
    df_optuna['datetime'] = pd.to_datetime(df_optuna['datetime'])

    df_live = df_live.sort_values('datetime').reset_index(drop=True)
    df_optuna = df_optuna.sort_values('datetime').reset_index(drop=True)

    # Check if game_id is present in both dataframes
    has_game_id = ('game_id' in df_live.columns) and ('game_id' in df_optuna.columns)

    used_optuna_indices = set()
    comparison_records = []
    unmatched_count = 0

    # Iterate backward through live predictions (newest to oldest)
    for live_idx in reversed(range(len(df_live))):
        live_row = df_live.iloc[live_idx]
        blue = live_row['blue_team']
        red = live_row['red_team']

        live_gid = str(live_row['game_id']).strip() if (has_game_id and pd.notna(live_row.get('game_id'))) else None
        matched_optuna_idx = None

        # Strategy 1: Match by exact game_id if available
        if live_gid:
            for opt_idx in reversed(range(len(df_optuna))):
                if opt_idx in used_optuna_indices:
                    continue
                opt_gid = str(df_optuna.iloc[opt_idx].get('game_id')).strip() if pd.notna(
                    df_optuna.iloc[opt_idx].get('game_id')) else None
                if opt_gid == live_gid:
                    matched_optuna_idx = opt_idx
                    break

        # Strategy 2: Fallback to reverse team-pair matching if game_id match was not found
        if matched_optuna_idx is None:
            for opt_idx in reversed(range(len(df_optuna))):
                if opt_idx in used_optuna_indices:
                    continue
                opt_row = df_optuna.iloc[opt_idx]
                if opt_row['blue_team'] == blue and opt_row['red_team'] == red:
                    matched_optuna_idx = opt_idx
                    break

        if matched_optuna_idx is not None:
            used_optuna_indices.add(matched_optuna_idx)
            opt_row = df_optuna.iloc[matched_optuna_idx]

            live_p_blue = float(live_row['prob_blue_win'])
            opt_p_blue = float(opt_row['prob_blue_win'])

            live_ll = float(live_row['sample_logloss'])
            opt_ll = float(opt_row['sample_logloss'])

            live_correct = bool(live_row['is_correct'])
            opt_correct = bool(opt_row['is_correct'])

            # Determine predicted winner for both
            live_pred_winner = blue if live_p_blue >= 0.5 else red
            opt_pred_winner = blue if opt_p_blue >= 0.5 else red
            winner_changed = live_pred_winner != opt_pred_winner

            record = {}
            if has_game_id:
                record['game_id'] = live_gid or str(opt_row.get('game_id', ''))

            record.update({
                'datetime_live': live_row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'datetime_optuna': opt_row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'blue_team': blue,
                'red_team': red,
                'actual_blue_win': int(live_row['actual_blue_win']),
                'live_prob_blue': round(live_p_blue, 6),
                'optuna_prob_blue': round(opt_p_blue, 6),
                'prob_diff_live_minus_opt': round(live_p_blue - opt_p_blue, 6),
                'live_logloss': round(live_ll, 6),
                'optuna_logloss': round(opt_ll, 6),
                'logloss_diff_live_minus_opt': round(live_ll - opt_ll, 6),
                'live_is_correct': live_correct,
                'optuna_is_correct': opt_correct,
                'prediction_winner_changed': winner_changed,
                'live_predicted_winner': live_pred_winner,
                'optuna_predicted_winner': opt_pred_winner
            })
            comparison_records.append(record)
        else:
            unmatched_count += 1

    # Reverse records so output CSV remains in chronological order
    df_comparison = pd.DataFrame(list(reversed(comparison_records)))

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df_comparison.to_csv(output_csv_path, index=False)

    print(f"[✓] Successfully generated comparison: '{output_csv_path}'")
    print(f"    - Matched games: {len(df_comparison)}")
    if unmatched_count > 0:
        print(f"    - [!] Unmatched live games (not found in Optuna test split): {unmatched_count}")


def compare_all_models():
    """Runs comparison for all 5 models."""
    models = [
        ("xgboost", "live-data/live_xgboost.csv", "metadata/predictions/best_params_predictions.csv"),
        ("lightgbm", "live-data/live_lightgbm.csv", "metadata/predictions/best_lightgbm_params_predictions.csv"),
        ("catboost", "live-data/live_catboost.csv", "metadata/predictions/catboost_best_params_predictions.csv"),
        ("elastictree", "live-data/live_elastictree.csv", "metadata/predictions/elastictree_best_params_predictions.csv"),
        ("elasticnet", "live-data/live_elasticnet.csv", "metadata/predictions/elasticnet_best_params_predictions.csv"),
    ]

    for model_name, live_path, optuna_path in models:
        out_path = f"comparisons/{model_name}_comparison.csv"
        print(f"\n--- Comparing {model_name.upper()} ---")
        compare_model_csvs(live_path, optuna_path, out_path)


if __name__ == "__main__":
    compare_all_models()