# test_live_engine.py
import os
import pandas as pd
from live_feature_engine import LiveFeatureEngine


def run_validation():
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"
    model_path = "models/xgboost_model.json"

    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print(f"[!] Target files missing: '{dataset_path}' or '{model_path}'.")
        print("Ensure historical dataset CSV and model JSON exist in the directory.")
        return

    print("Initializing LiveFeatureEngine...")
    engine = LiveFeatureEngine(dataset_path=dataset_path, model_path=model_path)

    # Mock Live Draft Payload (G2 vs T1 example on Patch 14.10)
    draft_payload = {
        "blue_team": "G2 Esports",
        "red_team": "T1",
        "patch": "14.10",
        "blue_firstpick": 1,
        "game_number": 2,
        "blue_series_lead": 1,
        "blue_prev_win": 1,
        "blue_players": ["BrokenBlade", "Yike", "Caps", "Hans Sama", "Mikyx"],
        "red_players": ["Doran", "Oner", "Faker", "Gumayusi", "Keria"],
        "blue_champs": ["Aatrox", "Viego", "Ahri", "Varus", "Nautilus"],
        "red_champs": ["Rumble", "Lee Sin", "Orianna", "Kalista", "Rell"],
    }

    print("Executing match prediction with live feature extraction...")
    results = engine.predict_match(draft_payload)

    print("\n==================================================")
    print("           PREDICTION SUMMARY & METRICS          ")
    print("==================================================")

    print(f"\nFinal Win Probabilities:")
    print(f"  {draft_payload['blue_team']} (Blue): {results['blue_win_percentage']}%")
    print(f"  {draft_payload['red_team']} (Red):  {results['red_win_percentage']}%")

    print("\n--- Stage Impact Progression ---")
    print(results['progression_data'].to_string(index=False))

    print("\n--- Elo Baseline Metrics ---")
    for k, v in results['elo_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Early Game Metrics ---")
    for k, v in results['early_game_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Strategic Priority Metrics ---")
    for k, v in results['strategic_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Resource & Playstyle Profile Metrics ---")
    for k, v in results['resource_playstyle_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Vision & Map Control Metrics ---")
    for k, v in results['vision_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Patch & Meta Adaptability Metrics ---")
    for k, v in results['patch_adaptability_metrics'].items():
        print(f"  {k}: {v}")

    print("\n--- Role-by-Role Player/Champion Breakdown ---")
    roles_df = pd.DataFrame(results['role_breakdown'])
    print(roles_df.to_string(index=False))

    print("\n==================================================")
    print("Validation Test Complete!")


if __name__ == "__main__":
    run_validation()