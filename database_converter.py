import json
import os
import streamlit as st
import numpy as np
import pandas as pd
from upstash_redis import Redis

from app import TRACKING_KEY


def get_redis_client():
    return Redis(
        url=st.secrets["UPSTASH_REDIS_REST_URL"],
        token=st.secrets["UPSTASH_REDIS_REST_TOKEN"]
    )


redis = get_redis_client()


def load_tracking_data() -> dict:
    raw_data = redis.get(TRACKING_KEY)
    if not raw_data:
        return {"total_games": 0, "correct_predictions": 0, "logs": []}
    if isinstance(raw_data, str):
        return json.loads(raw_data)
    return raw_data

def compute_sample_logloss(y_true: float, p_blue: float, eps: float = 1e-15) -> float:
    """Calculates binary log-loss for a single match."""
    p = np.clip(p_blue, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def convert_live_json_to_model_csvs(output_dir: str = "live_csvs"):
    """
    Parses live predictions JSON and creates individual CSVs per model,
    using identical columns as Optuna validation output.
    """
    data = load_tracking_data()

    # Handle both top-level list [...] or dict wrapped list {"games": [...]}
    games_list = data if isinstance(data, list) else data.get("logs", [data])

    # Storage grouped by model name
    model_records = {}

    for game in games_list:
        game_id = game.get("game_id", "")
        dt = game.get("datetime", "N/A")
        blue_team = game.get("blue_team", "Unknown")
        red_team = game.get("red_team", "Unknown")

        for m in game.get("models", []):
            model_name = m.get("model_used", "Unknown")
            prob_blue = round(float(m.get("blue_win_probability", 0.5)), 6)
            prob_red = round(float(m.get("red_win_probability", 0.5)), 6)
            actual_winner = m.get("actual_winner", "")

            # Binary ground truth: 1 if Blue won, 0 if Red won
            actual_blue_win = 1 if actual_winner == blue_team else 0

            # Log-loss calculation
            sample_ll = round(compute_sample_logloss(actual_blue_win, prob_blue), 6)
            is_correct = bool(m.get("is_correct", (prob_blue >= 0.5) == (actual_blue_win == 1)))

            record = {
                "game_id": game_id,
                "datetime": dt,
                "blue_team": blue_team,
                "red_team": red_team,
                "prob_blue_win": prob_blue,
                "prob_red_win": prob_red,
                "actual_blue_win": actual_blue_win,
                "sample_logloss": sample_ll,
                "is_correct": is_correct
            }

            if model_name not in model_records:
                model_records[model_name] = []
            model_records[model_name].append(record)

    os.makedirs(output_dir, exist_ok=True)

    # Save each model's predictions to a separate CSV
    for model_name, records in model_records.items():
        filename_slug = model_name.lower().replace(" ", "_")
        csv_path = os.path.join(output_dir, f"live_{filename_slug}.csv")
        df_model = pd.DataFrame(records)
        df_model.to_csv(csv_path, index=False)
        print(f"[✓] Saved {len(df_model)} rows for '{model_name}' -> '{csv_path}'")


if __name__ == "__main__":
    convert_live_json_to_model_csvs(
        output_dir="live-data/"
    )