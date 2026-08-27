import pandas as pd
import numpy as np


def compute_team_elo_ratings(
        filepath: str,
        output_filepath: str = None,
        init_rating: float = 1500.0,
        k_factor: float = 32.0,
        blue_side_bonus: float = 20.0,
        season_soft_reset_factor: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes dynamic pre-match Elo ratings and win probabilities for LoL esports teams.

    Parameters:
        filepath (str): Path to input CSV (output of the pre-game dataset script).
        output_filepath (str, optional): Destination path for the enriched dataset.
        init_rating (float): Baseline Elo rating for new/unseen teams (default: 1500.0).
        k_factor (float): Magnitude multiplier for post-match rating updates (default: 32.0).
        blue_side_bonus (float): Elo rating advantage added to Blue side expected score (default: 20.0).
        season_soft_reset_factor (float): Reversion factor towards 1500 when crossing calendar years (default: 0.5).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (Enriched dataset, Final Team Leaderboard)
    """
    # 1. Load dataset and sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    ratings = {}
    blue_elo_pre = []
    red_elo_pre = []
    blue_expected_win_prob = []

    current_year = None

    # 2. Iterate chronologically match by match
    for idx, row in df.iterrows():
        game_year = row.get('year', None)

        # Soft-reset team ratings at the start of a new calendar year/season
        if current_year is not None and game_year != current_year and season_soft_reset_factor > 0:
            for team in ratings:
                ratings[team] = init_rating + season_soft_reset_factor * (ratings[team] - init_rating)
        current_year = game_year

        blue_team = row['blue_teamid']
        red_team = row['red_teamid']
        blue_win = row['blue_win']

        # Fetch pre-game ratings (default to init_rating if first time seeing team)
        r_blue = ratings.get(blue_team, init_rating)
        r_red = ratings.get(red_team, init_rating)

        # Store pre-game ratings to avoid target leakage
        blue_elo_pre.append(r_blue)
        red_elo_pre.append(r_red)

        # Calculate pre-match expected Blue win probability (including Blue side bonus)
        r_blue_effective = r_blue + blue_side_bonus
        exp_blue = 1.0 / (1.0 + 10.0 ** ((r_red - r_blue_effective) / 400.0))
        blue_expected_win_prob.append(exp_blue)

        # Post-match rating update (unbiased expected win probability without side bonus)
        exp_blue_raw = 1.0 / (1.0 + 10.0 ** ((r_red - r_blue) / 400.0))
        score_blue = 1.0 if blue_win == 1 else 0.0

        ratings[blue_team] = r_blue + k_factor * (score_blue - exp_blue_raw)
        ratings[red_team] = r_red + k_factor * ((1.0 - score_blue) - (1.0 - exp_blue_raw))

    # 3. Add engineered Elo features back to DataFrame
    df['blue_elo_pre'] = blue_elo_pre
    df['red_elo_pre'] = red_elo_pre
    df['elo_diff'] = df['blue_elo_pre'] - df['red_elo_pre']
    df['blue_elo_win_prob'] = blue_expected_win_prob

    # 4. Generate standalone Team Standings Leaderboard
    leaderboard = pd.DataFrame(
        list(ratings.items()),
        columns=['teamid', 'elo_rating']
    ).sort_values('elo_rating', ascending=False).reset_index(drop=True)

    # 5. Export enriched dataset if output path provided
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Successfully processed {len(df)} matches. Enriched dataset saved to {output_filepath}")

    return df, leaderboard


if __name__ == "__main__":
    # Example execution using dataset created by step 1
    input_file = "2025_pregame_dataset_major_regions.csv"
    output_file = "2025_pregame_dataset_with_elo.csv"

    enriched_df, team_leaderboard = compute_team_elo_ratings(
        filepath=input_file,
        output_filepath=output_file,
        init_rating=1500.0,
        k_factor=32.0,
        blue_side_bonus=20.0,
        season_soft_reset_factor=0.5
    )

    print("\nTop 10 Teams by Current Elo Rating:")
    print(team_leaderboard.head(10))