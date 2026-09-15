import pandas as pd
import numpy as np


def _get_series(df: pd.DataFrame, col_name: str, default_val: float = 0.0) -> pd.Series:
    """Helper to ensure dataframe column extractions always return a pd.Series."""
    if col_name in df.columns:
        return df[col_name].fillna(default_val)
    return pd.Series(default_val, index=df.index)


def generate_strategic_priority_features(
        filepath: str,
        output_filepath: str = None,
        window: int = 10
) -> pd.DataFrame:
    """
    Computes rolling historical strategic map priority and jungle aggression metrics per team:
      - Topside Map Focus (Grubs/Herald focus vs. Dragons)
      - Topside Objective Control Rate (% of Grubs/Heralds secured vs opponent)
      - Dragon Control Rate (% of Dragons secured vs opponent)
      - Jungle Aggression Index (Composite early objective contest rate)

    Parameters:
        filepath (str): Input CSV dataset path.
        output_filepath (str, optional): Destination path for enriched dataset.
        window (int): Rolling match history window size (default: 10 matches).

    Returns:
        pd.DataFrame: Match dataset enriched with map priority features and deltas.
    """
    print(f"Loading input dataset: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 1. Safely extract raw objective counts for Blue and Red
    b_dragons = _get_series(df, 'blue_dragons', 0.0)
    r_dragons = _get_series(df, 'red_dragons', 0.0)
    b_heralds = _get_series(df, 'blue_heralds', 0.0)
    r_heralds = _get_series(df, 'red_heralds', 0.0)
    b_grubs = _get_series(df, 'blue_void_grubs', 0.0)
    r_grubs = _get_series(df, 'red_void_grubs', 0.0)

    b_fb = _get_series(df, 'blue_firstblood', 0.0)
    r_fb = _get_series(df, 'red_firstblood', 0.0)
    b_fd = _get_series(df, 'blue_firstdragon', 0.0)
    r_fd = _get_series(df, 'red_firstdragon', 0.0)
    b_fh = _get_series(df, 'blue_firstherald', 0.0)
    r_fh = _get_series(df, 'red_firstherald', 0.0)
    b_ft = _get_series(df, 'blue_firsttower', 0.0)
    r_ft = _get_series(df, 'red_firsttower', 0.0)

    # Calculate Topside Weighted Points (1 Herald = 1.0, 3 Void Grubs = 1.0)
    b_topside = b_heralds + (b_grubs / 3.0)
    r_topside = r_heralds + (r_grubs / 3.0)

    # 2. Build Team History Records
    blue_records = pd.DataFrame({
        'gameid': df['gameid'],
        'date': df['date'],
        'teamid': df['blue_teamid'],
        'team_name': df['blue_team'],
        'topside_objs': b_topside,
        'opp_topside_objs': r_topside,
        'dragons': b_dragons,
        'opp_dragons': r_dragons,
        'topside_share': b_topside / (b_topside + b_dragons + 1e-5),
        'jungle_aggression': (b_fb + b_fd + b_fh + b_ft) / 4.0
    })

    red_records = pd.DataFrame({
        'gameid': df['gameid'],
        'date': df['date'],
        'teamid': df['red_teamid'],
        'team_name': df['red_team'],
        'topside_objs': r_topside,
        'opp_topside_objs': b_topside,
        'dragons': r_dragons,
        'opp_dragons': b_dragons,
        'topside_share': r_topside / (r_topside + r_dragons + 1e-5),
        'jungle_aggression': (r_fb + r_fd + r_fh + r_ft) / 4.0
    })

    team_history = pd.concat([blue_records, red_records], ignore_index=True)
    team_history = team_history.sort_values(['date', 'gameid']).reset_index(drop=True)

    # 3. Compute Leakage-Free Rolling Metrics grouped by teamid (.shift(1))
    grouped = team_history.groupby('teamid')

    team_history['roll_topside_objs'] = grouped['topside_objs'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_opp_topside_objs'] = grouped['opp_topside_objs'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_dragons'] = grouped['dragons'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_opp_dragons'] = grouped['opp_dragons'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_topside_share'] = grouped['topside_share'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_jungle_aggression'] = grouped['jungle_aggression'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )

    # Calculated ratios from rolling averages
    team_history['roll_topside_control_rate'] = team_history['roll_topside_objs'] / (
        team_history['roll_topside_objs'] + team_history['roll_opp_topside_objs'] + 1e-5
    )
    team_history['roll_dragon_control_rate'] = team_history['roll_dragons'] / (
        team_history['roll_dragons'] + team_history['roll_opp_dragons'] + 1e-5
    )

    # Fill NaNs for unseen teams
    default_fill = {
        'roll_topside_share': 0.5,
        'roll_topside_control_rate': 0.5,
        'roll_dragon_control_rate': 0.5,
        'roll_jungle_aggression': 0.5
    }
    team_history = team_history.fillna(value=default_fill)

    # 4. Merge back to Blue/Red match rows
    feature_keys = [
        'gameid', 'teamid',
        'roll_topside_share', 'roll_topside_control_rate',
        'roll_dragon_control_rate', 'roll_jungle_aggression'
    ]

    blue_features = team_history[feature_keys].copy()
    blue_features.columns = ['gameid', 'blue_teamid'] + [f'blue_{c}' for c in feature_keys[2:]]

    red_features = team_history[feature_keys].copy()
    red_features.columns = ['gameid', 'red_teamid'] + [f'red_{c}' for c in feature_keys[2:]]

    df = df.merge(blue_features, on=['gameid', 'blue_teamid'], how='left')
    df = df.merge(red_features, on=['gameid', 'red_teamid'], how='left')

    for col, val in default_fill.items():
        df[f'blue_{col}'] = df[f'blue_{col}'].fillna(val)
        df[f'red_{col}'] = df[f'red_{col}'].fillna(val)

    # 5. Compute Match Differentials
    metrics_to_diff = [
        'roll_topside_share', 'roll_topside_control_rate',
        'roll_dragon_control_rate', 'roll_jungle_aggression'
    ]

    for m in metrics_to_diff:
        df[f'diff_{m}'] = df[f'blue_{m}'] - df[f'red_{m}']

    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Successfully generated strategic features for {len(df)} matches. Saved to: {output_filepath}")

    return df


if __name__ == "__main__":
    input_file = "multi_year_pregame_dataset_with_early_game_features.csv"
    output_file = "multi_year_pregame_dataset_with_strategic_features.csv"

    enriched_df = generate_strategic_priority_features(
        filepath=input_file,
        output_filepath=output_file,
        window=10
    )

    diff_cols = [c for c in enriched_df.columns if 'diff_roll_' in c]
    print("\nEngineered Strategic Match Differentials (Sample):")
    print(enriched_df[['gameid', 'blue_team', 'red_team'] + diff_cols].head())