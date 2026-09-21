import os
import pandas as pd
import numpy as np

from calculators import team_momentum_calculator
from calculators.team_momentum_calculator import add_momentum_features_to_dataset


# 1. International tournament codes to ignore when determining home region
INTERNATIONAL_LEAGUES = {
    "WLDS", "WORLDS", "MSI", "EWC", "FST", "RR", "RIFT RIVALS", "MSC"
}

# 2. Tier to Elo mapping table
TIER_BASE_ELO = {
    1: 1600.0,
    2: 1550.0,
    3: 1500.0,
    4: 1450.0,
    5: 1400.0
}

# 3. Configurable region groupings per tier (Domestic Leagues Only)
DEFAULT_REGION_CATEGORIES = {
    1: ["LCK", "OGN"],                                      # Tier 1: 1600 Elo
    2: ["LPL"],                                             # Tier 2: 1550 Elo
    3: ["LEC", "EU LCS", "LCS", "NA LCS", "LTA", "LTA N", "LTA S"], # Tier 3: 1500 Elo
    4: ["LCP", "VCS", "PCS", "LMS"],                        # Tier 4: 1450 Elo
    5: []                                                   # Tier 5: 1400 Elo (Default / Minor)
}


def get_region_base_elo(region_code: str, region_categories: dict[int, list[str]]) -> float:
    """Determines baseline Elo according to domestic regional tier configuration."""
    if not region_code or pd.isna(region_code):
        return TIER_BASE_ELO[5]

    clean_region = str(region_code).strip().upper()

    for tier, regions in region_categories.items():
        if tier == 5:
            continue
        if any(clean_region == r.upper() for r in regions):
            return TIER_BASE_ELO[tier]

    return TIER_BASE_ELO[5]


def extract_field(row, *possible_cols, default="Unknown"):
    """Helper function to extract field values safely across varying column headers."""
    for col in possible_cols:
        if col in row and pd.notna(row[col]):
            return str(row[col]).strip()
    return default


def compute_team_elo_ratings(
        filepath: str,
        output_filepath: str = None,
        leaderboard_filepath: str = "metadata/elo_leaderboard.csv",
        region_categories: dict[int, list[str]] = None,
        k_factor: float = 32.0,
        first_pick_bonus: float = 20.0,
        season_soft_reset_factor: float = 0.5,
        filter_active_year_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes dynamic pre-match Elo ratings using region-aware base Elo initialization,
    locks team home regions, and exports active team leaderboards.
    """
    if region_categories is None:
        region_categories = DEFAULT_REGION_CATEGORIES

    # 1. Load dataset and sort chronologically
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Convert year column to int if available
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(df['date'].dt.year)
    else:
        df['year'] = df['date'].dt.year

    latest_year = int(df['year'].max())

    # 2. Pre-scan pass: Lock each team's domestic home region (ignoring international events)
    team_home_regions = {}
    team_names = {}

    for idx, row in df.iterrows():
        league = extract_field(row, 'league', 'region', 'league_name', default='UNKNOWN').upper()
        blue_id = row['blue_teamid']
        red_id = row['red_teamid']

        blue_name = extract_field(row, 'blue_team', 'blue_teamname', 'blue_team_name', default=str(blue_id))
        red_name = extract_field(row, 'red_team', 'red_teamname', 'red_team_name', default=str(red_id))

        team_names[blue_id] = blue_name
        team_names[red_id] = red_name

        # Assign home region if domestic league and not yet assigned
        if league not in INTERNATIONAL_LEAGUES and league != 'UNKNOWN':
            if blue_id not in team_home_regions:
                team_home_regions[blue_id] = league
            if red_id not in team_home_regions:
                team_home_regions[red_id] = league

    # 3. Initialize locked base Elo ratings for all identified teams
    ratings = {}
    team_init_ratings = {}
    team_last_active_year = {}

    for tid, home_region in team_home_regions.items():
        base_elo = get_region_base_elo(home_region, region_categories)
        team_init_ratings[tid] = base_elo
        ratings[tid] = base_elo

    blue_elo_pre = []
    red_elo_pre = []
    blue_expected_win_prob = []

    current_year = None

    # 4. Iterate chronologically match by match
    for idx, row in df.iterrows():
        game_year = int(row['year'])

        # Soft-reset team ratings at start of new calendar year towards team's fixed base Elo
        if current_year is not None and game_year != current_year and season_soft_reset_factor > 0:
            for team_id in ratings:
                base_elo = team_init_ratings.get(team_id, TIER_BASE_ELO[5])
                ratings[team_id] = base_elo + season_soft_reset_factor * (ratings[team_id] - base_elo)
        current_year = game_year

        blue_team = row['blue_teamid']
        red_team = row['red_teamid']
        blue_win = row['blue_win']

        # Update last active year for both teams
        team_last_active_year[blue_team] = game_year
        team_last_active_year[red_team] = game_year

        # Fallback initialization for any team missing from pre-scan
        if blue_team not in ratings:
            home_reg = team_home_regions.get(blue_team, "UNKNOWN")
            base_elo = get_region_base_elo(home_reg, region_categories)
            team_init_ratings[blue_team] = base_elo
            ratings[blue_team] = base_elo

        if red_team not in ratings:
            home_reg = team_home_regions.get(red_team, "UNKNOWN")
            base_elo = get_region_base_elo(home_reg, region_categories)
            team_init_ratings[red_team] = base_elo
            ratings[red_team] = base_elo

        # Determine First Pick advantage allocation
        blue_has_first_pick = row.get('blue_firstpick', 1) == 1

        r_blue = ratings[blue_team]
        r_red = ratings[red_team]

        blue_elo_pre.append(r_blue)
        red_elo_pre.append(r_red)

        # Calculate pre-match expected Blue win probability with dynamic draft advantage
        effective_bonus = first_pick_bonus if blue_has_first_pick else -first_pick_bonus
        r_blue_effective = r_blue + effective_bonus

        exp_blue = 1.0 / (1.0 + 10.0 ** ((r_red - r_blue_effective) / 400.0))
        blue_expected_win_prob.append(exp_blue)

        # Post-match rating update
        exp_blue_raw = 1.0 / (1.0 + 10.0 ** ((r_red - r_blue) / 400.0))
        score_blue = 1.0 if blue_win == 1 else 0.0

        ratings[blue_team] = r_blue + k_factor * (score_blue - exp_blue_raw)
        ratings[red_team] = r_red + k_factor * ((1.0 - score_blue) - (1.0 - exp_blue_raw))

    # 5. Add engineered Elo features back to DataFrame
    df['blue_elo_pre'] = blue_elo_pre
    df['red_elo_pre'] = red_elo_pre
    df['elo_diff'] = df['blue_elo_pre'] - df['red_elo_pre']
    df['blue_elo_win_prob'] = blue_expected_win_prob

    # 6. Generate Standalone Leaderboard
    leaderboard_records = [
        {
            'team_name': team_names.get(tid, str(tid)),
            'team_id': tid,
            'region': team_home_regions.get(tid, 'UNKNOWN'),
            'base_elo': team_init_ratings.get(tid, TIER_BASE_ELO[5]),
            'elo_rating': round(rating, 2),
            'last_active_year': team_last_active_year.get(tid, 0)
        }
        for tid, rating in ratings.items()
    ]

    leaderboard = pd.DataFrame(leaderboard_records).sort_values('elo_rating', ascending=False).reset_index(drop=True)

    # Filter leaderboard to only active teams in the latest year (e.g., 2026)
    if filter_active_year_only:
        leaderboard = leaderboard[leaderboard['last_active_year'] == latest_year].reset_index(drop=True)

    df = add_momentum_features_to_dataset(df, 10)

    # 7. Export enriched dataset
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Successfully processed {len(df)} matches. Enriched dataset saved to {output_filepath}")

    # 8. Save standalone Leaderboard
    if leaderboard_filepath:
        os.makedirs(os.path.dirname(leaderboard_filepath), exist_ok=True)
        leaderboard.to_csv(leaderboard_filepath, index=False)
        print(f"Team Leaderboard ({'Active ' + str(latest_year) if filter_active_year_only else 'All-Time'}) saved to {leaderboard_filepath}")

    return df, leaderboard


if __name__ == "__main__":
    input_file = "multi_year_pregame_dataset.csv"
    output_file = "multi_year_pregame_dataset_with_elo.csv"

    enriched_df, team_leaderboard = compute_team_elo_ratings(
        filepath=input_file,
        output_filepath=output_file,
        leaderboard_filepath="metadata/elo_leaderboard.csv",
        k_factor=32.0,
        first_pick_bonus=10.0,
        season_soft_reset_factor=0.2,
        filter_active_year_only=True  # Keeps only 2026 active teams
    )

    print("\nTop 10 Active Teams in Latest Season by Current Elo Rating:")
    print(team_leaderboard[['team_name', 'region', 'base_elo', 'elo_rating', 'last_active_year']].head(10))