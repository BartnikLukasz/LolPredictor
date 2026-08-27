import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Bayesian smoothing parameters for low-sample combinations
PRIOR_WINRATE = 0.50
SMOOTHING_WEIGHT = 5.0  # Equivalent to 5 pseudo-games at 50% win rate


def get_smoothed_winrate(wins: int, games: int, weight: float = SMOOTHING_WEIGHT, prior: float = PRIOR_WINRATE) -> float:
    """Computes Bayesian smoothed win rate to handle small sample sizes smoothly."""
    return (wins + weight * prior) / (games + weight)


def calculate_champion_and_draft_stats(
        input_filepath: str,
        output_filepath: str
) -> pd.DataFrame:
    """
    Chronologically engineers Category 3 draft composition and champion synergy features:
    1. Patch-Specific Champion Meta (Win rate & sample size per patch)
    2. Role-Specific Lane Counter-Picks (Smoothed H2H between lane opponents)
    3. Essential Role-Pair Synergies (Mid-Jng, Bot-Sup, Jng-Sup)
    4. Team Composition Cohesion (Average pairwise synergy across all 5 champions)

    Parameters:
        input_filepath (str): Path to CSV from player_stats_calculator.py.
        output_filepath (str): Path where enriched dataset will be saved.

    Returns:
        pd.DataFrame: Enriched dataset containing final feature matrix.
    """
    df = pd.read_csv(input_filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    roles = ['top', 'jng', 'mid', 'bot', 'sup']

    # Historical state tracking (Strictly pre-game, updated POST-match)
    # 1. Patch Champ Stats: (patch, champion) -> {'games': int, 'wins': int}
    patch_champ_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    # 2. Lane Matchups: (role, champ_A, champ_B) -> {'games': int, 'wins_A': int}
    lane_counter_stats: Dict[Tuple[str, str, str], Dict[str, int]] = {}

    # 3. Pairwise Champion Synergies: (champ_A, champ_B) [alphabetical] -> {'games': int, 'wins': int}
    champ_synergy_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    # Data containers for new features
    feature_records = []

    print(f"Engineers Category 3 Draft & Champion Features across {len(df)} matches...")

    for idx, row in df.iterrows():
        blue_win = int(row['blue_win'])
        patch = str(row.get('patch', 'unknown'))

        # Extract champion vectors
        blue_champs = {r: str(row[f'blue_{r}_champion']) for r in roles}
        red_champs = {r: str(row[f'red_{r}_champion']) for r in roles}

        rec = {}

        # -------------------------------------------------------------
        # STEP 1: PRE-MATCH FEATURE COMPUTATION (Zero Target Leakage)
        # -------------------------------------------------------------

        # A. Patch-Specific Meta Performance
        blue_patch_wrs = []
        red_patch_wrs = []

        for r in roles:
            b_c = blue_champs[r]
            r_c = red_champs[r]

            b_stat = patch_champ_stats.get((patch, b_c), {'games': 0, 'wins': 0})
            r_stat = patch_champ_stats.get((patch, r_c), {'games': 0, 'wins': 0})

            b_wr = get_smoothed_winrate(b_stat['wins'], b_stat['games'])
            r_wr = get_smoothed_winrate(r_stat['wins'], r_stat['games'])

            rec[f'blue_{r}_patch_wr_pre'] = round(b_wr, 4)
            rec[f'red_{r}_patch_wr_pre'] = round(r_wr, 4)

            blue_patch_wrs.append(b_wr)
            red_patch_wrs.append(r_wr)

        rec['blue_team_patch_wr_avg_pre'] = round(float(np.mean(blue_patch_wrs)), 4)
        rec['red_team_patch_wr_avg_pre'] = round(float(np.mean(red_patch_wrs)), 4)
        rec['patch_winrate_diff'] = round(rec['blue_team_patch_wr_avg_pre'] - rec['red_team_patch_wr_avg_pre'], 4)

        # B. Role-Specific Lane Counter-Picks (H2H)
        lane_counter_diffs = []

        for r in roles:
            b_c = blue_champs[r]
            r_c = red_champs[r]

            # Direct matchup lookup: Blue Champ vs Red Champ in Role r
            h2h = lane_counter_stats.get((r, b_c, r_c), {'games': 0, 'wins_A': 0})
            b_counter_wr = get_smoothed_winrate(h2h['wins_A'], h2h['games'])

            rec[f'{r}_lane_counter_wr_pre'] = round(b_counter_wr, 4)
            lane_counter_diffs.append(b_counter_wr - 0.50)  # Variance from 50% neutral baseline

        rec['lane_counter_diff_sum'] = round(float(np.sum(lane_counter_diffs)), 4)

        # C. Key Role-Pair Champion Synergies
        key_pairs = [('mid', 'jng'), ('bot', 'sup'), ('jng', 'sup')]

        for r1, r2 in key_pairs:
            # Blue Pair
            b_pair = tuple(sorted([blue_champs[r1], blue_champs[r2]]))
            b_syn_stat = champ_synergy_stats.get(b_pair, {'games': 0, 'wins': 0})
            b_syn_wr = get_smoothed_winrate(b_syn_stat['wins'], b_syn_stat['games'])

            # Red Pair
            r_pair = tuple(sorted([red_champs[r1], red_champs[r2]]))
            r_syn_stat = champ_synergy_stats.get(r_pair, {'games': 0, 'wins': 0})
            r_syn_wr = get_smoothed_winrate(r_syn_stat['wins'], r_syn_stat['games'])

            rec[f'blue_{r1}_{r2}_synergy_wr_pre'] = round(b_syn_wr, 4)
            rec[f'red_{r1}_{r2}_synergy_wr_pre'] = round(r_syn_wr, 4)
            rec[f'{r1}_{r2}_synergy_diff'] = round(b_syn_wr - r_syn_wr, 4)

        # D. Team-Wide Composition Cohesion Score (Average of all 10 pairwise combinations)
        blue_all_champs = list(blue_champs.values())
        red_all_champs = list(red_champs.values())

        blue_pair_wrs = []
        red_pair_wrs = []

        for i in range(5):
            for j in range(i + 1, 5):
                # Blue pairwise
                b_p = tuple(sorted([blue_all_champs[i], blue_all_champs[j]]))
                b_st = champ_synergy_stats.get(b_p, {'games': 0, 'wins': 0})
                blue_pair_wrs.append(get_smoothed_winrate(b_st['wins'], b_st['games']))

                # Red pairwise
                r_p = tuple(sorted([red_all_champs[i], red_all_champs[j]]))
                r_st = champ_synergy_stats.get(r_p, {'games': 0, 'wins': 0})
                red_pair_wrs.append(get_smoothed_winrate(r_st['wins'], r_st['games']))

        rec['blue_comp_cohesion_score'] = round(float(np.mean(blue_pair_wrs)), 4)
        rec['red_comp_cohesion_score'] = round(float(np.mean(red_pair_wrs)), 4)
        rec['comp_cohesion_diff'] = round(rec['blue_comp_cohesion_score'] - rec['red_comp_cohesion_score'], 4)

        feature_records.append(rec)

        # -------------------------------------------------------------
        # STEP 2: POST-MATCH HISTORICAL STATE UPDATE
        # -------------------------------------------------------------

        # 1. Update Patch Champ Stats
        for r in roles:
            b_c = blue_champs[r]
            r_c = red_champs[r]

            b_key = (patch, b_c)
            r_key = (patch, r_c)

            if b_key not in patch_champ_stats:
                patch_champ_stats[b_key] = {'games': 0, 'wins': 0}
            patch_champ_stats[b_key]['games'] += 1
            patch_champ_stats[b_key]['wins'] += blue_win

            if r_key not in patch_champ_stats:
                patch_champ_stats[r_key] = {'games': 0, 'wins': 0}
            patch_champ_stats[r_key]['games'] += 1
            patch_champ_stats[r_key]['wins'] += (1 - blue_win)

        # 2. Update Lane Counter-Pick Stats
        for r in roles:
            b_c = blue_champs[r]
            r_c = red_champs[r]

            lane_key = (r, b_c, r_c)
            if lane_key not in lane_counter_stats:
                lane_counter_stats[lane_key] = {'games': 0, 'wins_A': 0}
            lane_counter_stats[lane_key]['games'] += 1
            lane_counter_stats[lane_key]['wins_A'] += blue_win

            # Mirror record for Red perspective
            mirror_key = (r, r_c, b_c)
            if mirror_key not in lane_counter_stats:
                lane_counter_stats[mirror_key] = {'games': 0, 'wins_A': 0}
            lane_counter_stats[mirror_key]['games'] += 1
            lane_counter_stats[mirror_key]['wins_A'] += (1 - blue_win)

        # 3. Update Champion Pairwise Synergies (All 10 pairs per team)
        for i in range(5):
            for j in range(i + 1, 5):
                # Blue Team Pairs
                b_p = tuple(sorted([blue_all_champs[i], blue_all_champs[j]]))
                if b_p not in champ_synergy_stats:
                    champ_synergy_stats[b_p] = {'games': 0, 'wins': 0}
                champ_synergy_stats[b_p]['games'] += 1
                champ_synergy_stats[b_p]['wins'] += blue_win

                # Red Team Pairs
                r_p = tuple(sorted([red_all_champs[i], red_all_champs[j]]))
                if r_p not in champ_synergy_stats:
                    champ_synergy_stats[r_p] = {'games': 0, 'wins': 0}
                champ_synergy_stats[r_p]['games'] += 1
                champ_synergy_stats[r_p]['wins'] += (1 - blue_win)

    # Merge features back into original dataframe
    new_features_df = pd.DataFrame(feature_records)
    output_df = pd.concat([df, new_features_df], axis=1)

    output_df.to_csv(output_filepath, index=False)
    print(f"Successfully processed Category 3 draft features!")
    print(f"Output saved to: '{output_filepath}' (Total columns: {output_df.shape[1]})")

    return output_df


if __name__ == "__main__":
    input_csv = "multi_year_dataset_with_player_stats.csv"
    output_csv = "multi_year_pregame_dataset_final_features.csv"

    calculate_champion_and_draft_stats(
        input_filepath=input_csv,
        output_filepath=output_csv
    )