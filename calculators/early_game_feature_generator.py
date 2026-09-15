import pandas as pd
import numpy as np


def _get_series(df: pd.DataFrame, col_name: str, default_val: float = 0.0) -> pd.Series:
    """Helper to ensure dataframe column extractions always return a pd.Series."""
    if col_name in df.columns:
        return df[col_name]
    return pd.Series(default_val, index=df.index)


def generate_early_game_features(
        filepath: str,
        output_filepath: str = None,
        window: int = 10
) -> pd.DataFrame:
    """
    Computes rolling 10-game historical early game & lane dominance metrics per team
    without data leakage (using shift(1)), and calculates pre-game match differentials.
    """
    print(f"Loading input dataset: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Pre-extract gold diff Series to guarantee pd.Series type for comparison ops
    blue_gd15 = _get_series(df, 'blue_golddiffat15', 0.0)
    red_gd15 = _get_series(df, 'red_golddiffat15', 0.0)

    # 1. Build team-level longitudinal history table
    blue_records = pd.DataFrame({
        'gameid': df['gameid'],
        'date': df['date'],
        'teamid': df['blue_teamid'],
        'team_name': df['blue_team'],
        'golddiff10': _get_series(df, 'blue_golddiffat10', 0.0),
        'golddiff15': blue_gd15,
        'xpdiff15': _get_series(df, 'blue_xpdiffat15', 0.0),
        'csdiff15': _get_series(df, 'blue_csdiffat15', 0.0),
        'plates_taken': _get_series(df, 'blue_turretplates', 0.0),
        'plates_conceded': _get_series(df, 'blue_opp_turretplates', 0.0),
        'firstblood': _get_series(df, 'blue_firstblood', 0.0),
        'firsttower': _get_series(df, 'blue_firsttower', 0.0),
        'firstdragon': _get_series(df, 'blue_firstdragon', 0.0),
        'win': df['blue_win'],
        'had_lead15': (blue_gd15 > 0).astype(float),
        'won_with_lead15': ((blue_gd15 > 0) & (df['blue_win'] == 1)).astype(float)
    })

    red_records = pd.DataFrame({
        'gameid': df['gameid'],
        'date': df['date'],
        'teamid': df['red_teamid'],
        'team_name': df['red_team'],
        'golddiff10': _get_series(df, 'red_golddiffat10', 0.0),
        'golddiff15': red_gd15,
        'xpdiff15': _get_series(df, 'red_xpdiffat15', 0.0),
        'csdiff15': _get_series(df, 'red_csdiffat15', 0.0),
        'plates_taken': _get_series(df, 'red_turretplates', 0.0),
        'plates_conceded': _get_series(df, 'red_opp_turretplates', 0.0),
        'firstblood': _get_series(df, 'red_firstblood', 0.0),
        'firsttower': _get_series(df, 'red_firsttower', 0.0),
        'firstdragon': _get_series(df, 'red_firstdragon', 0.0),
        'win': (1 - df['blue_win']),
        'had_lead15': (red_gd15 > 0).astype(float),
        'won_with_lead15': ((red_gd15 > 0) & (df['blue_win'] == 0)).astype(float)
    })

    team_history = pd.concat([blue_records, red_records], ignore_index=True)
    team_history = team_history.sort_values(['date', 'gameid']).reset_index(drop=True)

    # 2. Compute Leakage-Free Rolling Metrics grouped by teamid
    grouped = team_history.groupby('teamid')

    roll_cols = ['golddiff10', 'golddiff15', 'xpdiff15', 'csdiff15', 'firstblood', 'firsttower', 'firstdragon']
    for col in roll_cols:
        team_history[f'roll_{col}'] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Rolling plate statistics
    team_history['roll_plates_taken'] = grouped['plates_taken'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_plates_conceded'] = grouped['plates_conceded'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    team_history['roll_plate_ratio'] = team_history['roll_plates_taken'] / (
        team_history['roll_plates_taken'] + team_history['roll_plates_conceded'] + 1e-5
    )

    # Rolling early lead conversion rate
    team_history['roll_had_lead15'] = grouped['had_lead15'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    )
    team_history['roll_won_with_lead15'] = grouped['won_with_lead15'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    )
    team_history['roll_early_lead_conv'] = team_history['roll_won_with_lead15'] / (
        team_history['roll_had_lead15'] + 1e-5
    )

    # Replace NaNs for unseen/first-game teams with baseline defaults
    default_fill = {
        'roll_golddiff10': 0.0,
        'roll_golddiff15': 0.0,
        'roll_xpdiff15': 0.0,
        'roll_csdiff15': 0.0,
        'roll_firstblood': 0.5,
        'roll_firsttower': 0.5,
        'roll_firstdragon': 0.5,
        'roll_plate_ratio': 0.5,
        'roll_early_lead_conv': 0.5
    }
    team_history = team_history.fillna(value=default_fill)

    # 3. Join rolling metrics back to Blue and Red match rows
    feature_keys = [
        'gameid', 'teamid',
        'roll_golddiff10', 'roll_golddiff15', 'roll_xpdiff15', 'roll_csdiff15',
        'roll_firstblood', 'roll_firsttower', 'roll_firstdragon',
        'roll_plate_ratio', 'roll_early_lead_conv'
    ]

    blue_features = team_history[feature_keys].copy()
    blue_features.columns = ['gameid', 'blue_teamid'] + [f'blue_{c}' for c in feature_keys[2:]]

    red_features = team_history[feature_keys].copy()
    red_features.columns = ['gameid', 'red_teamid'] + [f'red_{c}' for c in feature_keys[2:]]

    df = df.merge(blue_features, on=['gameid', 'blue_teamid'], how='left')
    df = df.merge(red_features, on=['gameid', 'red_teamid'], how='left')

    # Fill remaining un-matched initial NAs with defaults
    for col, val in default_fill.items():
        df[f'blue_{col}'] = df[f'blue_{col}'].fillna(val)
        df[f'red_{col}'] = df[f'red_{col}'].fillna(val)

    # 4. Compute Match Differentials (Delta = Blue Metric - Red Metric)
    metrics_to_diff = [
        'roll_golddiff10', 'roll_golddiff15', 'roll_xpdiff15', 'roll_csdiff15',
        'roll_firstblood', 'roll_firsttower', 'roll_firstdragon',
        'roll_plate_ratio', 'roll_early_lead_conv'
    ]

    for m in metrics_to_diff:
        df[f'diff_{m}'] = df[f'blue_{m}'] - df[f'red_{m}']

    # 5. Export enriched match dataset
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Successfully generated early game features for {len(df)} matches. Saved to: {output_filepath}")

    return df


if __name__ == "__main__":
    input_file = "multi_year_pregame_dataset_with_elo.csv"
    output_file = "multi_year_pregame_dataset_with_early_game_features.csv"

    enriched_df = generate_early_game_features(
        filepath=input_file,
        output_filepath=output_file,
        window=10
    )

    diff_cols = [c for c in enriched_df.columns if c.startswith('diff_roll_')]
    print("\nEngineered Early Game Match Differentials (Sample):")
    print(enriched_df[['gameid', 'blue_team', 'red_team'] + diff_cols].head())