import os
import pandas as pd
import numpy as np

from database_converter import convert_live_json_to_model_csvs
from ods_update import update_ods_with_csvs, ods_file, csv_mappings


def get_id_column(df: pd.DataFrame) -> str:
    """Helper to detect whether the dataset uses 'game_id' or 'gameid'."""
    if 'game_id' in df.columns:
        return 'game_id'
    elif 'gameid' in df.columns:
        return 'gameid'
    return None


def compare_model_csvs(live_csv_path: str, optuna_csv_path: str, output_csv_path: str):
    """
    Matches live predictions with Optuna validation predictions strictly based on game IDs.
    """
    if not os.path.exists(live_csv_path):
        print(f"[!] Live CSV missing: '{live_csv_path}'")
        return
    if not os.path.exists(optuna_csv_path):
        print(f"[!] Optuna CSV missing: '{optuna_csv_path}'")
        return

    df_live = pd.read_csv(live_csv_path)
    df_optuna = pd.read_csv(optuna_csv_path)

    # Detect ID columns
    id_col_live = get_id_column(df_live)
    id_col_optuna = get_id_column(df_optuna)

    if not id_col_live or not id_col_optuna:
        print(f"[!] Missing game ID column in CSVs. Live: '{id_col_live}', Optuna: '{id_col_optuna}'")
        return

    # Standardize and sort chronologically
    df_live['datetime'] = pd.to_datetime(df_live['datetime'])
    df_optuna['datetime'] = pd.to_datetime(df_optuna['datetime'])

    df_live = df_live.sort_values('datetime').reset_index(drop=True)
    df_optuna = df_optuna.sort_values('datetime').reset_index(drop=True)

    # Build an O(1) fast lookup dictionary for Optuna predictions by ID
    # Drops rows missing game IDs and converts IDs to standardized strings
    optuna_valid_df = df_optuna.dropna(subset=[id_col_optuna]).copy()
    optuna_valid_df['clean_id'] = optuna_valid_df[id_col_optuna].astype(str).str.strip()
    optuna_dict = {row['clean_id']: row for _, row in optuna_valid_df.iterrows()}

    comparison_records = []
    unmatched_count = 0

    # Iterate through live predictions
    for _, live_row in df_live.iterrows():
        raw_gid = live_row.get(id_col_live)
        if pd.isna(raw_gid):
            unmatched_count += 1
            continue

        live_gid = str(raw_gid).strip()
        opt_row = optuna_dict.get(live_gid)

        if opt_row is not None:
            blue = live_row['blue_team']
            red = live_row['red_team']

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

            record = {
                'game_id': live_gid,
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
            }
            comparison_records.append(record)
        else:
            unmatched_count += 1

    df_comparison = pd.DataFrame(comparison_records)

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df_comparison.to_csv(output_csv_path, index=False)

    print(f"[✓] Successfully generated comparison: '{output_csv_path}'")
    print(f"    - Matched games: {len(df_comparison)}")
    if unmatched_count > 0:
        print(f"    - [!] Unmatched live games (ID missing or not found in Optuna test split): {unmatched_count}")


def compare_all_models():
    """Runs comparison for all 5 models."""
    models = [
        ("xgboost", "live-data/live_xgboost.csv", "metadata/predictions/xgboost_predictions.csv"),
        ("lightgbm", "live-data/live_lightgbm.csv", "metadata/predictions/lightgbm_predictions.csv"),
        ("catboost", "live-data/live_catboost.csv", "metadata/predictions/catboost_predictions.csv"),
        ("elastictree", "live-data/live_elastictree.csv", "metadata/predictions/elastictree_predictions.csv"),
        ("elasticnet", "live-data/live_elasticnet.csv", "metadata/predictions/elasticnet_predictions.csv"),
    ]

    for model_name, live_path, optuna_path in models:
        out_path = f"metadata/comparisons/{model_name}_comparison.csv"
        print(f"\n--- Comparing {model_name.upper()} ---")
        compare_model_csvs(live_path, optuna_path, out_path)


if __name__ == "__main__":
    convert_live_json_to_model_csvs(output_dir="live-data/")
    compare_all_models()
    update_ods_with_csvs(ods_file, csv_mappings)