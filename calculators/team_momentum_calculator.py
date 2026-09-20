from collections import defaultdict
import numpy as np
import pandas as pd


def add_momentum_features_to_dataset(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Chronologically pre-computes momentum and overperformance features for each match in df.

    Ensures zero data leakage by only using match history prior to each row.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # Store chronological match history per team: list of (elo_pre, outcome, exp_prob)
    history = defaultdict(list)

    b_elo_deltas, r_elo_deltas = [], []
    b_overperforms, r_overperforms = [], []

    for _, row in df.iterrows():
        b_team = str(row.get('blue_team', '')).strip().lower()
        r_team = str(row.get('red_team', '')).strip().lower()

        b_elo = float(row.get('blue_elo_pre', 1500.0))
        r_elo = float(row.get('red_elo_pre', 1500.0))
        b_win = float(row.get('blue_win', 0.5))

        # 1. Compute Blue Momentum Features from prior window
        b_hist = history[b_team][-window:]
        if len(b_hist) >= 2:
            b_delta = b_elo - b_hist[0]['elo_pre']
            b_over = float(np.mean([h['actual'] - h['exp'] for h in b_hist]))
        else:
            b_delta = 0.0
            b_over = 0.0

        # 2. Compute Red Momentum Features from prior window
        r_hist = history[r_team][-window:]
        if len(r_hist) >= 2:
            r_delta = r_elo - r_hist[0]['elo_pre']
            r_over = float(np.mean([h['actual'] - h['exp'] for h in r_hist]))
        else:
            r_delta = 0.0
            r_over = 0.0

        b_elo_deltas.append(b_delta)
        r_elo_deltas.append(r_delta)
        b_overperforms.append(b_over)
        r_overperforms.append(r_over)

        # 3. Update team history after calculating features (prevents data leakage)
        exp_b = 1.0 / (1.0 + 10.0 ** ((r_elo - b_elo) / 400.0))

        history[b_team].append({'elo_pre': b_elo, 'actual': b_win, 'exp': exp_b})
        history[r_team].append({'elo_pre': r_elo, 'actual': 1.0 - b_win, 'exp': 1.0 - exp_b})

    # Attach engineered columns
    df[f'blue_elo_delta_{window}'] = b_elo_deltas
    df[f'red_elo_delta_{window}'] = r_elo_deltas
    df[f'elo_delta_diff_{window}'] = np.array(b_elo_deltas) - np.array(r_elo_deltas)

    df[f'blue_overperform_{window}'] = b_overperforms
    df[f'red_overperform_{window}'] = r_overperforms
    df[f'overperformance_diff_{window}'] = np.array(b_overperforms) - np.array(r_overperforms)

    return df