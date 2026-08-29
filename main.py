from champ_stats_calculator import calculate_champion_and_draft_stats
from download_latest_data import download_latest_match_data
from elo_calculator import compute_team_elo_ratings
from match_data_converter import prepare_oracles_elixir_pregame
from player_stats_calculator import compute_player_and_mastery_stats
from model_trainer import train_lol_prediction_model
import pandas as pd
import numpy as np
import json

if __name__ == '__main__':

    # download_latest_match_data()
    prepare_oracles_elixir_pregame(["dataset/match/2014_match_data.csv",
                                    "dataset/match/2015_match_data.csv",
                                    "dataset/match/2016_match_data.csv",
                                    "dataset/match/2017_match_data.csv",
                                    "dataset/match/2018_match_data.csv",
                                    "dataset/match/2019_match_data.csv",
                                    "dataset/match/2020_match_data.csv",
                                    "dataset/match/2021_match_data.csv",
                                    "dataset/match/2022_match_data.csv",
                                    "dataset/match/2023_match_data.csv",
                                    "dataset/match/2024_match_data.csv",
                                    "dataset/match/2025_match_data.csv",
                                    "dataset/match/2026_match_data.csv"],
                                   "dataset/pregame/pregame.csv")

    enriched_df, team_leaderboard = compute_team_elo_ratings(
        filepath="dataset/pregame/pregame.csv",
        output_filepath="dataset/pregame/pregame_dataset_with_elo.csv",
        init_rating=1500,
        first_pick_bonus=10.0,
        season_soft_reset_factor=0.2
    )

    compute_player_and_mastery_stats(
        filepath="dataset/pregame/pregame_dataset_with_elo.csv",
        output_filepath="dataset/pregame/pregame_dataset_with_player_stats.csv",
        prior_weight=1.0,
        prior_prob=0.50
    )

    calculate_champion_and_draft_stats(
        input_filepath="dataset/pregame/pregame_dataset_with_player_stats.csv",
        output_filepath="dataset/pregame/pregame_dataset_final_features.csv"
    )

    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"

    model, feature_importance = train_lol_prediction_model(
        filepath=dataset_path,
        split_date="2026-04-01",
        full_train=True
    )

    # 2. Save the trained XGBoost model artifact
    model.save_model("lol_xgb_model.json")
    print("\n[ARTIFACT] Saved model to 'lol_xgb_model.json'")

    # 3. Extract active rosters dynamically from the dataset and save to JSON
    df = pd.read_csv(dataset_path, low_memory=False)

    roster_dict = {}

    # Get unique teams across blue and red side columns
    teams = set(df['blue_team'].dropna().unique()).union(set(df['red_team'].dropna().unique()))

    for team in teams:
        # Grab the most recent match for this team
        latest_match = df[(df['blue_team'] == team) | (df['red_team'] == team)].iloc[-1]

        if latest_match['blue_team'] == team:
            roster = [
                latest_match.get('blue_top_player', ''),
                latest_match.get('blue_jng_player', ''),
                latest_match.get('blue_mid_player', ''),
                latest_match.get('blue_bot_player', ''),
                latest_match.get('blue_sup_player', '')
            ]
        else:
            roster = [
                latest_match.get('red_top_player', ''),
                latest_match.get('red_jng_player', ''),
                latest_match.get('red_mid_player', ''),
                latest_match.get('red_bot_player', ''),
                latest_match.get('red_sup_player', '')
            ]

        roster_dict[team] = roster

    with open("team_rosters.json", "w") as f:
        json.dump(roster_dict, f, indent=4)

    print("[ARTIFACT] Saved team rosters to 'team_rosters.json'")

    print(team_leaderboard)