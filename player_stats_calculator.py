import pandas as pd
import numpy as np


def compute_player_and_mastery_stats(
        filepath: str,
        output_filepath: str = None,
        prior_weight: float = 2.0,
        prior_prob: float = 0.50
) -> pd.DataFrame:
    """
    Calculates rolling player win rates, champion mastery, Team H2H, Lane Champion Matchups,
    and Direct Player-vs-Player H2H stats prior to each match (no target leakage).

    Parameters:
        filepath (str): Path to processed CSV dataset.
        output_filepath (str, optional): CSV destination path for enriched dataset.
        prior_weight (float): Bayesian prior weight (default: 2.0).
        prior_prob (float): Bayesian prior expected win rate (default: 0.50).

    Returns:
        pd.DataFrame: DataFrame enriched with all player, mastery, and H2H features.
    """
    # 1. Load dataset
    df = pd.read_csv(filepath, low_memory=False)

    # Safeguard: Remove duplicate column names if previous steps were merged
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. State tracking dictionaries (accumulated historical records)
    player_history = {}  # pid -> {'games': int, 'wins': int}
    player_champ_history = {}  # (pid, champion) -> {'games': int, 'wins': int}

    # NEW State Tracking: H2H & Matchups
    team_h2h_history = {}  # (team1, team2) [sorted] -> {'games': int, 'team1_wins': int}
    lane_matchup_history = {}  # (role, champ1, champ2) [sorted] -> {'games': int, 'champ1_wins': int}
    player_h2h_history = {}  # (role, pid1, pid2) [sorted] -> {'games': int, 'pid1_wins': int}

    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    sides = ['blue', 'red']

    # Pre-allocate feature column containers
    feature_vectors = {}

    # Standard player & mastery features
    for side in sides:
        for role in roles:
            feature_vectors[f'{side}_{role}_player_games_pre'] = []
            feature_vectors[f'{side}_{role}_player_winrate_pre'] = []
            feature_vectors[f'{side}_{role}_champ_games_pre'] = []
            feature_vectors[f'{side}_{role}_champ_winrate_pre'] = []

    # NEW Feature Containers: Team H2H, Lane Matchups, and P2P H2H
    feature_vectors['h2h_team_games_pre'] = []
    feature_vectors['blue_h2h_team_winrate_pre'] = []

    for role in roles:
        feature_vectors[f'blue_{role}_lane_matchup_games_pre'] = []
        feature_vectors[f'blue_{role}_lane_matchup_winrate_pre'] = []
        feature_vectors[f'blue_{role}_p2p_games_pre'] = []
        feature_vectors[f'blue_{role}_p2p_winrate_pre'] = []

    def extract_scalar(val):
        """Safely extract a primitive hashable string/None from pandas/numpy objects."""
        if isinstance(val, (pd.Series, np.ndarray, list)):
            if len(val) > 0:
                val = val.iloc[0] if hasattr(val, 'iloc') else val[0]
            else:
                return None
        if pd.isna(val) or val is None or str(val).lower() == 'nan':
            return None
        return str(val).strip()

    def calc_smoothed_winrate(wins: int, games: int) -> float:
        """Applies Laplace/Bayesian smoothing to prevent extreme win rates on low game counts."""
        return (wins + prior_weight * prior_prob) / (games + prior_weight)

    # 3. Iterate chronologically match by match
    for idx, row in df.iterrows():
        blue_win = row['blue_win']
        blue_team = extract_scalar(row.get('blue_teamid'))
        red_team = extract_scalar(row.get('red_teamid'))

        # --- STEP A: READ PRE-MATCH HISTORICAL STATS (NO LEAKAGE) ---

        # A1. Standard Player & Champion Mastery
        for side in sides:
            for role in roles:
                raw_pid = row.get(f'{side}_{role}_playerid')
                raw_champ = row.get(f'{side}_{role}_champion')

                pid = extract_scalar(raw_pid)
                champ = extract_scalar(raw_champ)

                if pid is None:
                    feature_vectors[f'{side}_{role}_player_games_pre'].append(0)
                    feature_vectors[f'{side}_{role}_player_winrate_pre'].append(prior_prob)
                    feature_vectors[f'{side}_{role}_champ_games_pre'].append(0)
                    feature_vectors[f'{side}_{role}_champ_winrate_pre'].append(prior_prob)
                    continue

                p_stats = player_history.get(pid, {'games': 0, 'wins': 0})
                p_games = p_stats['games']
                p_winrate = calc_smoothed_winrate(p_stats['wins'], p_games)

                pc_key = (pid, champ) if champ is not None else None
                if pc_key is not None:
                    pc_stats = player_champ_history.get(pc_key, {'games': 0, 'wins': 0})
                    pc_games = pc_stats['games']
                    pc_winrate = calc_smoothed_winrate(pc_stats['wins'], pc_games)
                else:
                    pc_games = 0
                    pc_winrate = prior_prob

                feature_vectors[f'{side}_{role}_player_games_pre'].append(p_games)
                feature_vectors[f'{side}_{role}_player_winrate_pre'].append(round(p_winrate, 4))
                feature_vectors[f'{side}_{role}_champ_games_pre'].append(pc_games)
                feature_vectors[f'{side}_{role}_champ_winrate_pre'].append(round(pc_winrate, 4))

        # A2. Direct Team Head-to-Head (Team A vs Team B)
        if blue_team and red_team:
            t1, t2 = sorted([blue_team, red_team])
            t_stats = team_h2h_history.get((t1, t2), {'games': 0, 'team1_wins': 0})
            t_games = t_stats['games']
            t_blue_wins = t_stats['team1_wins'] if blue_team == t1 else (t_games - t_stats['team1_wins'])
            t_winrate = calc_smoothed_winrate(t_blue_wins, t_games)

            feature_vectors['h2h_team_games_pre'].append(t_games)
            feature_vectors['blue_h2h_team_winrate_pre'].append(round(t_winrate, 4))
        else:
            feature_vectors['h2h_team_games_pre'].append(0)
            feature_vectors['blue_h2h_team_winrate_pre'].append(prior_prob)

        # A3. Lane Champion Matchups & Direct Player H2H (Per Role)
        for role in roles:
            b_pid = extract_scalar(row.get(f'blue_{role}_playerid'))
            r_pid = extract_scalar(row.get(f'red_{role}_playerid'))
            b_champ = extract_scalar(row.get(f'blue_{role}_champion'))
            r_champ = extract_scalar(row.get(f'red_{role}_champion'))

            # Champion Matchup in Lane
            if b_champ and r_champ:
                c1, c2 = sorted([b_champ, r_champ])
                c_stats = lane_matchup_history.get((role, c1, c2), {'games': 0, 'champ1_wins': 0})
                c_games = c_stats['games']
                c_blue_wins = c_stats['champ1_wins'] if b_champ == c1 else (c_games - c_stats['champ1_wins'])
                c_winrate = calc_smoothed_winrate(c_blue_wins, c_games)

                feature_vectors[f'blue_{role}_lane_matchup_games_pre'].append(c_games)
                feature_vectors[f'blue_{role}_lane_matchup_winrate_pre'].append(round(c_winrate, 4))
            else:
                feature_vectors[f'blue_{role}_lane_matchup_games_pre'].append(0)
                feature_vectors[f'blue_{role}_lane_matchup_winrate_pre'].append(prior_prob)

            # Player vs Player H2H in Lane
            if b_pid and r_pid:
                p1, p2 = sorted([b_pid, r_pid])
                p_stats = player_h2h_history.get((role, p1, p2), {'games': 0, 'pid1_wins': 0})
                p_games = p_stats['games']
                p_blue_wins = p_stats['pid1_wins'] if b_pid == p1 else (p_games - p_stats['pid1_wins'])
                p_winrate = calc_smoothed_winrate(p_blue_wins, p_games)

                feature_vectors[f'blue_{role}_p2p_games_pre'].append(p_games)
                feature_vectors[f'blue_{role}_p2p_winrate_pre'].append(round(p_winrate, 4))
            else:
                feature_vectors[f'blue_{role}_p2p_games_pre'].append(0)
                feature_vectors[f'blue_{role}_p2p_winrate_pre'].append(prior_prob)

        # --- STEP B: UPDATE HISTORICAL RECORDS AFTER MATCH RESOLUTION ---

        # B1. Standard Player & Champion Mastery Updates
        for side in sides:
            is_blue = (side == 'blue')
            team_won = (blue_win == 1) if is_blue else (blue_win == 0)
            win_add = 1 if team_won else 0

            for role in roles:
                pid = extract_scalar(row.get(f'{side}_{role}_playerid'))
                champ = extract_scalar(row.get(f'{side}_{role}_champion'))

                if pid is None:
                    continue

                if pid not in player_history:
                    player_history[pid] = {'games': 0, 'wins': 0}
                player_history[pid]['games'] += 1
                player_history[pid]['wins'] += win_add

                if champ is not None:
                    pc_key = (pid, champ)
                    if pc_key not in player_champ_history:
                        player_champ_history[pc_key] = {'games': 0, 'wins': 0}
                    player_champ_history[pc_key]['games'] += 1
                    player_champ_history[pc_key]['wins'] += win_add

        # B2. Team H2H Update
        if blue_team and red_team:
            t1, t2 = sorted([blue_team, red_team])
            if (t1, t2) not in team_h2h_history:
                team_h2h_history[(t1, t2)] = {'games': 0, 'team1_wins': 0}
            team_h2h_history[(t1, t2)]['games'] += 1
            if (blue_win == 1 and blue_team == t1) or (blue_win == 0 and red_team == t1):
                team_h2h_history[(t1, t2)]['team1_wins'] += 1

        # B3. Lane Matchups & Player H2H Updates
        for role in roles:
            b_pid = extract_scalar(row.get(f'blue_{role}_playerid'))
            r_pid = extract_scalar(row.get(f'red_{role}_playerid'))
            b_champ = extract_scalar(row.get(f'blue_{role}_champion'))
            r_champ = extract_scalar(row.get(f'red_{role}_champion'))

            # Update Lane Champion Matchups
            if b_champ and r_champ:
                c1, c2 = sorted([b_champ, r_champ])
                key = (role, c1, c2)
                if key not in lane_matchup_history:
                    lane_matchup_history[key] = {'games': 0, 'champ1_wins': 0}
                lane_matchup_history[key]['games'] += 1
                if (blue_win == 1 and b_champ == c1) or (blue_win == 0 and r_champ == c1):
                    lane_matchup_history[key]['champ1_wins'] += 1

            # Update Player vs Player Lane H2H
            if b_pid and r_pid:
                p1, p2 = sorted([b_pid, r_pid])
                key = (role, p1, p2)
                if key not in player_h2h_history:
                    player_h2h_history[key] = {'games': 0, 'pid1_wins': 0}
                player_h2h_history[key]['games'] += 1
                if (blue_win == 1 and b_pid == p1) or (blue_win == 0 and r_pid == p1):
                    player_h2h_history[key]['pid1_wins'] += 1

    # 4. Attach feature vectors to DataFrame
    for col_name, values in feature_vectors.items():
        df[col_name] = values

    # 5. Export dataset if path is provided
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Successfully processed {len(df)} matches. Saved to {output_filepath}")

    return df


if __name__ == "__main__":
    input_file = "multi_year_pregame_dataset_with_elo.csv"
    output_file = "multi_year_pregame_dataset_final_features.csv"

    enriched_df = compute_player_and_mastery_stats(
        filepath=input_file,
        output_filepath=output_file,
        prior_weight=2.0,
        prior_prob=0.50
    )

    print("\nDataset ready! Sample new H2H columns:")
    sample_cols = [
        'gameid', 'h2h_team_games_pre', 'blue_h2h_team_winrate_pre',
        'blue_mid_lane_matchup_winrate_pre', 'blue_mid_p2p_winrate_pre'
    ]
    print(enriched_df[sample_cols].head())