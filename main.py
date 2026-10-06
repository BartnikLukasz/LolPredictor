from calculators.champ_stats_calculator import calculate_champion_and_draft_stats
from calculators.download_latest_data import download_latest_match_data
from calculators.elo_calculator import compute_team_elo_ratings
from calculators.match_data_converter import prepare_oracles_elixir_pregame
from calculators.player_stats_calculator import compute_player_and_mastery_stats
from trainers.catboost_model_trainer import train_catboost
from trainers.elasticnet_model_trainer import train_elasticnet_model
from trainers.elastictree_model_trainer import train_elastictree
from trainers.lightgbm_model_trainer import train_secondary_model
from trainers.model_trainer import train_lol_prediction_model
from trainers.rosters import save_team_rosters

# Feature profile / raw-champion switches live in trainers/trainer_helpers.py
# (FEATURE_PROFILE, USE_RAW_CHAMPIONS). Run hyperparameter_tuner.py with the same settings first.

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

    # Each call: honest holdout evaluation (printed + saved to metadata/predictions/), then a full-data
    # refit saved to models/. Artifact paths and formats are unchanged, so app.py loads them as before.
    train_lol_prediction_model(filepath=dataset_path)
    train_secondary_model(dataset_path=dataset_path)
    train_catboost(filepath=dataset_path)
    train_elastictree(filepath=dataset_path)
    train_elasticnet_model(filepath=dataset_path)

    # Team rosters for the app's dropdowns (max_inactive_days=365 would hide defunct teams).
    save_team_rosters(dataset_path, "models/team_rosters.json", max_inactive_days=None)
