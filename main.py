from elo_calculator import compute_team_elo_ratings
from match_data_converter import prepare_oracles_elixir_pregame
from player_stats_calculator import compute_player_and_mastery_stats
from model_trainer import train_lol_prediction_model

if __name__ == '__main__':
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
        output_filepath="dataset/pregame/pregame_dataset_final_features.csv",
        prior_weight=2.0,
        prior_prob=0.50
    )

    train_lol_prediction_model(
        filepath="dataset/pregame/pregame_dataset_final_features.csv",
        test_split_ratio=0.03,
        split_date="2026-04-01"
    )