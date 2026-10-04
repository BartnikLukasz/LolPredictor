import hashlib
import json
import os

import numpy as np
import pandas as pd

def generate_game_id(blue_team, red_team, blue_champs, red_champs, first_pick=""):
    """
    Generates a deterministic 10-character MD5 match fingerprint.
    Normalizes team names, champion lists (sorted to avoid role-swap mismatches),
    and first-pick side representation.
    """
    b_team = str(blue_team).strip().lower()
    r_team = str(red_team).strip().lower()

    # Sort champions so role swaps or pick order differences don't break the match ID
    b_champs = ",".join(sorted([str(c).strip().lower() for c in blue_champs if pd.notna(c)]))
    r_champs = ",".join(sorted([str(c).strip().lower() for c in red_champs if pd.notna(c)]))

    # Standardize first pick representation (handles 1/0, True/False, "Blue"/"Red")
    fp_str = str(first_pick).strip().lower()
    if fp_str in ['1', '1.0', 'blue', 'true']:
        fp_norm = 'blue'
    elif fp_str in ['0', '0.0', 'red', 'false']:
        fp_norm = 'red'
    else:
        fp_norm = fp_str

    raw_str = f"{b_team}|{r_team}|{b_champs}|{r_champs}|{fp_norm}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()[:16]

def compute_sample_logloss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Calculates individual binary log-loss for each sample."""
    p = np.clip(y_proba, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def save_validation_predictions(
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        y_probs: np.ndarray,
        output_csv_path: str
):
    """Saves match metadata, probabilities, ground truth, and unique game_id to CSV."""
    results_df = pd.DataFrame()

    # Define champion feature columns expected in test_df
    blue_champ_cols = ['blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion']
    red_champ_cols = ['red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion']

    # Filter columns that exist in the test dataframe
    b_cols = [c for c in blue_champ_cols if c in test_df.columns]
    r_cols = [c for c in red_champ_cols if c in test_df.columns]

    # Deterministic game_id generation row by row
    game_ids = []
    for idx, row in test_df.iterrows():
        b_team = row.get('blue_team', 'Unknown')
        r_team = row.get('red_team', 'Unknown')
        b_champs = [row[c] for c in b_cols]
        r_champs = [row[c] for c in r_cols]
        fp = row.get('blue_firstpick', row.get('blue_first_pick', row.get('first_pick', '')))

        g_id = generate_game_id(b_team, r_team, b_champs, r_champs, fp)
        game_ids.append(g_id)

    results_df['game_id'] = game_ids  # <-- ADDED AS FIRST COLUMN

    # Standardize date/datetime (using .values to avoid index alignment NaN drops)
    date_col = 'datetime' if 'datetime' in test_df.columns else ('date' if 'date' in test_df.columns else None)
    if date_col is not None:
        results_df['datetime'] = pd.to_datetime(test_df[date_col]).dt.strftime('%Y-%m-%d %H:%M:%S').values
    else:
        results_df['datetime'] = "N/A"

    # Match metadata
    results_df['blue_team'] = test_df['blue_team'].values if 'blue_team' in test_df.columns else "Unknown"
    results_df['red_team'] = test_df['red_team'].values if 'red_team' in test_df.columns else "Unknown"

    # Prediction metrics
    probs_blue = np.round(y_probs, 6)
    results_df['prob_blue_win'] = probs_blue
    results_df['prob_red_win'] = np.round(1.0 - probs_blue, 6)
    results_df['actual_blue_win'] = y_test.astype(int)

    # Calculate sample log-loss and accuracy flag
    results_df['sample_logloss'] = np.round(compute_sample_logloss(y_test, probs_blue), 6)
    results_df['is_correct'] = (probs_blue >= 0.5) == (y_test == 1)

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"[✓] Saved standardized validation predictions with game_id to '{output_csv_path}'")

def save_best_params_if_improved(
    best_params: dict,
    current_logloss: float,
    output_json_path: str
):
    """
    Compares the current run's best log-loss against the stored log-loss.
    Saves new parameters if an improvement is detected.
    """
    previous_logloss = float('inf')

    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                existing_data = json.load(f)
                previous_logloss = existing_data.get('best_logloss', float('inf'))
        except Exception as e:
            print(f"[!] Warning: Could not read existing JSON ({e}). Overwriting file.")

    print("\n" + "=" * 60)
    print("                  OPTIMIZATION COMPLETE                 ")
    print("=" * 60)
    print(f"Current Run Best Log-Loss:  {current_logloss:.6f}")
    if previous_logloss != float('inf'):
        print(f"Previous Saved Log-Loss:    {previous_logloss:.6f}")
    else:
        print("Previous Saved Log-Loss:    None (New file)")

    if current_logloss < previous_logloss:
        best_params['best_logloss'] = round(float(current_logloss), 6)
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"[✓] Improvement detected! Updated hyperparameters saved to '{output_json_path}'")
    else:
        print(f"[!] Current run did not beat refreshed baseline ({previous_logloss:.6f}). Keeping existing JSON.")
    print("=" * 60 + "\n")

def refresh_baseline_if_exists(output_json_path: str, eval_callback) -> float:
    """
    Evaluates saved parameters against the CURRENT dataset split using tuning-level
    iteration limits. Updates 'best_logloss' in the JSON file ONLY if the dataset or score changed.
    """
    if not os.path.exists(output_json_path):
        print("[i] No existing parameter file found. Proceeding with fresh optimization search.")
        return float('inf')

    try:
        with open(output_json_path, 'r') as f:
            saved_params = json.load(f)

        previous_logloss = saved_params.get('best_logloss', None)

        meta_keys = ['best_logloss']
        clean_params = {k: v for k, v in saved_params.items() if k not in meta_keys}

        print(f"[*] Re-evaluating saved hyperparameters from '{output_json_path}' using tuning limits...")
        baseline_logloss = eval_callback(clean_params)
        baseline_logloss = round(float(baseline_logloss), 6)

        # Check if logloss actually changed compared to saved JSON value
        if previous_logloss is None or round(float(previous_logloss), 6) != baseline_logloss:
            saved_params['best_logloss'] = baseline_logloss
            os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(saved_params, f, indent=4)
            print(f"[✓] Refreshed baseline Log-Loss on new dataset: {previous_logloss} -> {baseline_logloss:.6f}\n")
        else:
            print(f"[=] Baseline Log-Loss unchanged on current dataset ({baseline_logloss:.6f}). Keeping stored parameters.\n")

        return baseline_logloss

    except Exception as e:
        print(f"[!] Warning: Failed to re-evaluate existing baseline ({e}). Proceeding without baseline refresh.")
        return float('inf')