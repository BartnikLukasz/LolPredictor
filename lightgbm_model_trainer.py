import os
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def train_secondary_model(
        dataset_path: str = "dataset/pregame/pregame_dataset_final_features.csv",
        model_output_path: str = "models/lightgbm_model.pkl",
        params_path: str = "models/best_lightgbm_params.json",
        test_start_date: str = "2024-01-01",
        full_train: bool = False
):
    """
    Trains a LightGBM secondary model using best parameters from tuner if available.

    Parameters:
        dataset_path (str): Path to final pregame feature dataset.
        model_output_path (str): Destination path for saved model artifact.
        params_path (str): Path to JSON file containing best tuned hyperparameters.
        test_start_date (str): Cutoff date ('YYYY-MM-DD'). Matches on or after this date
                               are placed into the test evaluation set (ignored if full_train=True).
        full_train (bool): If True, trains on 100% of the dataset without splitting or testing.
    """
    # 1. Load dataset
    df = pd.read_csv(dataset_path, low_memory=False)

    if "date" not in df.columns:
        raise KeyError("Dataset must contain a 'date' column.")

    # Convert date and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Exclude non-feature columns
    target_col = "blue_win"
    exclude_cols = [
        "gameid", "date", "blue_team", "red_team",
        "blue_teamid", "red_teamid", target_col
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    # 3. Handle Train / Test Data Selection
    if full_train:
        X_train = df[feature_cols]
        y_train = df[target_col]
        X_val, y_val = None, None

        start_dt = df['date'].min().strftime('%Y-%m-%d')
        end_dt = df['date'].max().strftime('%Y-%m-%d')
        print(
            f"[*] FULL TRAIN MODE: Training LightGBM Model 2 on ALL {len(X_train)} matches ({start_dt} to {end_dt})...")
    else:
        cutoff_dt = pd.to_datetime(test_start_date)
        train_mask = df["date"] < cutoff_dt
        test_mask = df["date"] >= cutoff_dt

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[test_mask, feature_cols]
        y_val = df.loc[test_mask, target_col]

        if len(X_train) == 0 or len(X_val) == 0:
            min_date = df['date'].min().strftime('%Y-%m-%d')
            max_date = df['date'].max().strftime('%Y-%m-%d')
            raise ValueError(
                f"Invalid test_start_date '{test_start_date}'. "
                f"Dataset date range spans from {min_date} to {max_date}."
            )

        print(f"[*] Training LightGBM Model 2 across {len(feature_cols)} features...")
        print(
            f"[*] Train set: {len(X_train)} matches ({df.loc[train_mask, 'date'].min().strftime('%Y-%m-%d')} to {df.loc[train_mask, 'date'].max().strftime('%Y-%m-%d')})")
        print(
            f"[*] Test set:  {len(X_val)} matches (from {cutoff_dt.strftime('%Y-%m-%d')} to {df.loc[test_mask, 'date'].max().strftime('%Y-%m-%d')})")

    # 4. Load Tuned Hyperparameters (or Fallback to Defaults)
    default_params = {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1
    }

    if os.path.exists(params_path):
        print(f"[*] Found tuned hyperparameter file at '{params_path}'. Loading...")
        try:
            with open(params_path, "r") as f:
                params = json.load(f)

            # Cast integer hyperparameter values explicitly to avoid type issues
            int_keys = ["n_estimators", "max_depth", "num_leaves", "min_child_samples", "subsample_freq", "max_bin"]
            for k in int_keys:
                if k in params:
                    params[k] = int(params[k])

            # Force required execution settings
            params["random_state"] = 42
            params["verbosity"] = -1

            print(f"[✓] Applied tuned parameters: {params}")
        except Exception as e:
            print(f"[!] Error loading '{params_path}': {e}. Falling back to default parameters.")
            params = default_params
    else:
        print(f"[!] No tuned hyperparameter file found at '{params_path}'. Using default parameters.")
        params = default_params

    # 5. Train LightGBM Model
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # 6. Evaluate (Only if not full_train)
    if not full_train:
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        acc = accuracy_score(y_val, val_preds)
        loss = log_loss(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)

        print("\n--- Model 2 (LightGBM) Performance ---")
        print(f"Test Cutoff Date: {test_start_date}")
        print(f"Test Accuracy:    {acc * 100:.2f}%")
        print(f"Test Log Loss:    {loss:.4f}")
        print(f"Test ROC AUC:     {auc:.4f}")
    else:
        print("\n--- Model 2 (LightGBM) Training Complete ---")
        print("Skipped validation evaluation (Full Train Mode).")

    # 7. Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": feature_cols
    }
    joblib.dump(artifact, model_output_path)
    print(f"[✓] Model 2 successfully saved to '{model_output_path}'")


if __name__ == "__main__":
    # Standard evaluation split:
    # train_secondary_model(test_start_date="2024-01-01", full_train=False)

    # Production run (Train on 100% of available data):
    train_secondary_model(full_train=True)