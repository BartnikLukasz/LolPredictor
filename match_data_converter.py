import pandas as pd
import numpy as np
from typing import Union, List


def prepare_oracles_elixir_pregame(
        filepaths: Union[str, List[str]],
        output_filepath: str = None,
        target_leagues: list = None
) -> pd.DataFrame:
    """
    Transforms multi-year Oracle's Elixir raw datasets into a unified single-row per match
    pre-game dataset sorted chronologically across all years.

    Parameters:
        filepaths (str | list[str]): Path or list of paths to Oracle's Elixir CSV files.
        output_filepath (str, optional): CSV output destination path.
        target_leagues (list, optional): List of league codes to include.

    Returns:
        pd.DataFrame: Structured multi-year pre-game dataframe with 1 row per gameid.
    """
    if target_leagues is None:
        target_leagues = [
            'LCK', 'LPL', 'LEC', 'LCS', 'LTA N', 'LTA North', 'LTA', 'EWC'
            'FST', 'MSI', 'WLDs', 'WORLDS', 'LCP', 'VCS', 'LTA S', 'EU LCS', 'NA LCS', 'CBLOL', 'LMS'
        ]

    # Ensure filepaths is a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    print(f"Loading {len(filepaths)} raw data file(s)...")
    dfs = [pd.read_csv(fp, low_memory=False) for fp in filepaths]
    df = pd.concat(dfs, ignore_index=True)

    # 1. Deduplicate raw rows if overlapping files/years were passed
    df = df.drop_duplicates().copy()

    # 2. Filter for target leagues
    if target_leagues:
        df = df[df['league'].isin(target_leagues)].copy()

    # Standardize position strings
    df['position'] = df['position'].str.lower().str.strip()

    # 3. Split into team summary rows and player individual rows
    team_rows = df[df['position'] == 'team'].copy()
    player_rows = df[df['position'] != 'team'].copy()

    # De-duplicate team and player entries per gameid/side/position
    team_rows = team_rows.drop_duplicates(subset=['gameid', 'side'])
    player_rows = player_rows.drop_duplicates(subset=['gameid', 'side', 'position'])

    # Separate Blue and Red team summary data
    blue_teams = team_rows[team_rows['side'] == 'Blue'].set_index('gameid')
    red_teams = team_rows[team_rows['side'] == 'Red'].set_index('gameid')

    # Find gameids present in both sides
    valid_gameids = blue_teams.index.intersection(red_teams.index).unique()

    blue_teams = blue_teams.loc[valid_gameids]
    red_teams = red_teams.loc[valid_gameids]

    # 4. Construct base match DataFrame
    match_df = pd.DataFrame(index=valid_gameids)
    match_df['gameid'] = valid_gameids

    # Metadata columns
    meta_cols = ['date', 'league', 'year', 'split', 'patch']
    for col in meta_cols:
        if col in blue_teams.columns:
            match_df[col] = blue_teams[col].values

    # Convert date to datetime for chronological sorting
    match_df['date'] = pd.to_datetime(match_df['date'])

    # Target Variable: 1 if Blue wins, 0 if Red wins
    match_df['blue_win'] = blue_teams['result'].astype(int).values

    # Match metadata & side info
    match_df['blue_teamid'] = blue_teams['teamid'].values
    match_df['red_teamid'] = red_teams['teamid'].values

    if 'firstPick' in blue_teams.columns:
        match_df['blue_firstpick'] = blue_teams['firstPick'].fillna(0).astype(int).values
    else:
        match_df['blue_firstpick'] = 0

    # 5. Extract Draft Bans
    ban_cols = ['ban1', 'ban2', 'ban3', 'ban4', 'ban5']
    for col in ban_cols:
        if col in blue_teams.columns:
            match_df[f'blue_{col}'] = blue_teams[col].values
            match_df[f'red_{col}'] = red_teams[col].values

    # 6. Pivot 10 individual player picks and IDs by position
    roles = ['top', 'jng', 'mid', 'bot', 'sup']

    for side in ['Blue', 'Red']:
        side_prefix = side.lower()
        side_players = player_rows[player_rows['side'] == side]

        for role in roles:
            role_df = (
                side_players[side_players['position'] == role]
                .set_index('gameid')
                .reindex(valid_gameids)
            )

            match_df[f'{side_prefix}_{role}_playerid'] = role_df['playerid'].values
            match_df[f'{side_prefix}_{role}_champion'] = role_df['champion'].values

    # 7. Sort chronologically across all combined years
    match_df = match_df.sort_values('date').reset_index(drop=True)

    if output_filepath:
        match_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed {len(match_df)} matches across target leagues. Saved to {output_filepath}")

    return match_df


# Example usage combining 3 years of data:
if __name__ == "__main__":
    multi_year_files = [
        "2023_match_data.csv",
        "2024_match_data.csv",
        "2025_match_data.csv"
    ]

    df_multi_year = prepare_oracles_elixir_pregame(
        filepaths=multi_year_files,
        output_filepath="multi_year_pregame_dataset.csv"
    )