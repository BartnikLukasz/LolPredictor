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
    pre-game dataset sorted chronologically across all years, incorporating draft side/first pick info,
    team names, player names, and intra-series tracking features (BO3/BO5 momentum & leads).

    Parameters:
        filepaths (str | list[str]): Path or list of paths to Oracle's Elixir CSV files.
        output_filepath (str, optional): CSV output destination path.
        target_leagues (list, optional): List of league codes to include.

    Returns:
        pd.DataFrame: Structured multi-year pre-game dataframe with 1 row per gameid.
    """
    if target_leagues is None:
        target_leagues = [
            'LCK', 'LPL', 'LEC', 'LCS', 'LTA N', 'LTA North', 'LTA', 'EWC', "OGN",
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

    # Match metadata & team IDs
    match_df['blue_teamid'] = blue_teams['teamid'].values
    match_df['red_teamid'] = red_teams['teamid'].values

    # --- Extract Team Names (handling variations across years) ---
    team_name_col = next((col for col in ['teamname', 'team'] if col in blue_teams.columns), None)
    if team_name_col:
        match_df['blue_team'] = blue_teams[team_name_col].values
        match_df['red_team'] = red_teams[team_name_col].values

    # --- Robust Extract & Clean of First Pick Column ---
    if 'firstPick' in blue_teams.columns:
        blue_fp = pd.to_numeric(blue_teams['firstPick'], errors='coerce')
        red_fp = pd.to_numeric(red_teams['firstPick'], errors='coerce')

        # Fallback cross-inferences for missing or NaN values
        blue_fp = blue_fp.fillna(1.0 - red_fp).fillna(1.0).astype(int)
        red_fp = red_fp.fillna(1.0 - blue_fp).fillna(0.0).astype(int)

        match_df['blue_firstpick'] = blue_fp.values
        match_df['red_firstpick'] = red_fp.values
    else:
        # Default fallback: Blue side = 1 (First Pick), Red side = 0 (Counter Pick)
        match_df['blue_firstpick'] = 1
        match_df['red_firstpick'] = 0

    # 5. Extract Draft Bans
    ban_cols = ['ban1', 'ban2', 'ban3', 'ban4', 'ban5']
    for col in ban_cols:
        if col in blue_teams.columns:
            match_df[f'blue_{col}'] = blue_teams[col].values
            match_df[f'red_{col}'] = red_teams[col].values

    # 6. Pivot 10 individual player picks, IDs, and player names by position
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    player_name_col = next((col for col in ['playername', 'player'] if col in player_rows.columns), None)

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

            if player_name_col and player_name_col in role_df.columns:
                match_df[f'{side_prefix}_{role}_player'] = role_df[player_name_col].values

    # 7. Calculate Intra-Series Features (game_number, blue_series_lead, blue_prev_win)
    if 'matchid' in blue_teams.columns and blue_teams['matchid'].notna().any():
        match_df['series_id'] = blue_teams['matchid'].values
    else:
        # Fallback series identifier: YYYY-MM-DD + league + sorted team pair
        dates_str = match_df['date'].dt.strftime('%Y-%m-%d')
        leagues_str = match_df['league'].astype(str)
        teams_sorted = [
            "_vs_".join(sorted([str(b), str(r)]))
            for b, r in zip(match_df['blue_team'], match_df['red_team'])
        ]
        match_df['series_id'] = dates_str + "_" + leagues_str + "_" + pd.Series(teams_sorted, index=match_df.index)

    # Sort chronologically across all combined years before calculating running series state
    match_df = match_df.sort_values('date').reset_index(drop=True)

    game_numbers = np.zeros(len(match_df), dtype=int)
    blue_series_leads = np.zeros(len(match_df), dtype=int)
    blue_prev_wins = np.zeros(len(match_df), dtype=float)

    # Compute series metrics per match grouping
    for _, group_indices in match_df.groupby('series_id', sort=False).groups.items():
        scores = {}
        last_winner = None

        for i, idx in enumerate(group_indices):
            b_team = match_df.at[idx, 'blue_team']
            r_team = match_df.at[idx, 'red_team']

            b_score = scores.get(b_team, 0)
            r_score = scores.get(r_team, 0)

            game_numbers[idx] = i + 1
            blue_series_leads[idx] = b_score - r_score

            if i == 0:
                blue_prev_wins[idx] = 0.5  # Neutral baseline for Game 1
            else:
                blue_prev_wins[idx] = 1.0 if last_winner == b_team else 0.0

            # Update running scores for subsequent games in this series
            winner = b_team if match_df.at[idx, 'blue_win'] == 1 else r_team
            scores[winner] = scores.get(winner, 0) + 1
            last_winner = winner

    match_df['game_number'] = game_numbers
    match_df['blue_series_lead'] = blue_series_leads
    match_df['blue_prev_win'] = blue_prev_wins

    match_df = match_df.drop(columns=['series_id'])

    if output_filepath:
        match_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed {len(match_df)} matches across target leagues. Saved to {output_filepath}")

    return match_df


# Example usage combining multi-year datasets:
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