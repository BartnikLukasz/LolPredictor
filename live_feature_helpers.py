# live_feature_helpers.py
import pandas as pd
import numpy as np

ROLES = ['top', 'jng', 'mid', 'bot', 'sup']

META_COLUMNS = [
    'date', 'blue_win', 'match_id', 'game_id',
    'blue_team', 'red_team', 'league', 'patch',
    'split', 'tournament', 'year', 'season'
]

EARLY_GAME_METRICS = [
    'golddiff10', 'golddiff15', 'xpdiff15', 'csdiff15',
    'firstblood', 'firsttower', 'firstdragon',
    'plate_ratio', 'early_lead_conv'
]

EARLY_GAME_DEFAULTS = {
    'golddiff10': 0.0,
    'golddiff15': 0.0,
    'xpdiff15': 0.0,
    'csdiff15': 0.0,
    'firstblood': 0.5,
    'firsttower': 0.5,
    'firstdragon': 0.5,
    'plate_ratio': 0.5,
    'early_lead_conv': 0.5
}

STRATEGIC_METRICS = [
    'topside_share',
    'topside_control_rate',
    'dragon_control_rate',
    'jungle_aggression'
]

STRATEGIC_DEFAULTS = {
    'topside_share': 0.5,
    'topside_control_rate': 0.5,
    'dragon_control_rate': 0.5,
    'jungle_aggression': 0.5
}

RESOURCE_PLAYSTYLE_METRICS = [
    'gold_share',
    'damage_share',
    'team_gold_hhi',
    'aggression_index',
    'early_game_orientation',
    'objective_priority_score'
]

RESOURCE_PLAYSTYLE_DEFAULTS = {
    'gold_share': 0.20,
    'damage_share': 0.20,
    'team_gold_hhi': 0.20,
    'aggression_index': 0.0,
    'early_game_orientation': 0.0,
    'objective_priority_score': 1.0
}

VISION_METRICS = [
    'vspm',
    'wpm',
    'wcpm',
    'ward_clear_ratio',
    'cwpm',
    'map_control_score'
]

VISION_DEFAULTS = {
    'vspm': 1.5,
    'wpm': 1.0,
    'wcpm': 0.4,
    'ward_clear_ratio': 0.4,
    'cwpm': 0.2,
    'map_control_score': 1.0
}

PATCH_ADAPTABILITY_METRICS = [
    'patch_winrate',
    'patch_wr_delta',
    'champ_pool_depth',
    'meta_alignment_score'
]

PATCH_ADAPTABILITY_DEFAULTS = {
    'patch_winrate': 0.50,
    'patch_wr_delta': 0.0,
    'champ_pool_depth': 10,
    'meta_alignment_score': 0.50
}

DEFAULT_PLAYER_CHAMP_STATS = {
    'player_games': 10,
    'player_winrate': 0.50,
    'champ_games': 10,
    'champ_winrate': 0.50,
}


def extract_model_feature_names(model) -> list:
    """Extracts the exact ordered list of feature names expected by the model."""
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        return list(model.feature_names_in_)

    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        fn = model.booster_.feature_name()
        if fn and len(fn) > 0 and not fn[0].startswith("Column_"):
            return list(fn)

    if hasattr(model, "get_booster"):
        try:
            fn = model.get_booster().feature_names
            if fn and len(fn) > 0:
                return list(fn)
        except Exception:
            pass

    return []


def build_elo_lookup(df_hist: pd.DataFrame) -> dict:
    """Builds dictionary of latest team Elo ratings."""
    latest_elo = {}
    for _, row in df_hist.iterrows():
        if pd.notna(row.get('blue_team')) and pd.notna(row.get('blue_elo_pre')):
            latest_elo[str(row['blue_team'])] = float(row['blue_elo_pre'])
        if pd.notna(row.get('red_team')) and pd.notna(row.get('red_elo_pre')):
            latest_elo[str(row['red_team'])] = float(row['red_elo_pre'])
    return latest_elo


def build_early_game_lookups(df_hist: pd.DataFrame) -> dict:
    """Builds lookup dictionary of latest rolling early game metrics per team."""
    team_early_game = {}
    for _, row in df_hist.iterrows():
        b_team = row.get('blue_team')
        r_team = row.get('red_team')

        if pd.notna(b_team):
            b_dict = {}
            for m in EARLY_GAME_METRICS:
                col = f'blue_roll_{m}'
                b_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else EARLY_GAME_DEFAULTS[m]
            team_early_game[str(b_team)] = b_dict

        if pd.notna(r_team):
            r_dict = {}
            for m in EARLY_GAME_METRICS:
                col = f'red_roll_{m}'
                r_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else EARLY_GAME_DEFAULTS[m]
            team_early_game[str(r_team)] = r_dict

    return team_early_game


def build_strategic_lookups(df_hist: pd.DataFrame) -> dict:
    """Builds lookup dictionary of latest rolling strategic map priority metrics per team."""
    team_strategic = {}
    for _, row in df_hist.iterrows():
        b_team = row.get('blue_team')
        r_team = row.get('red_team')

        if pd.notna(b_team):
            b_dict = {}
            for m in STRATEGIC_METRICS:
                col = f'blue_roll_{m}'
                b_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else STRATEGIC_DEFAULTS[m]
            team_strategic[str(b_team)] = b_dict

        if pd.notna(r_team):
            r_dict = {}
            for m in STRATEGIC_METRICS:
                col = f'red_roll_{m}'
                r_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else STRATEGIC_DEFAULTS[m]
            team_strategic[str(r_team)] = r_dict

    return team_strategic


def build_resource_playstyle_lookups(df_hist: pd.DataFrame) -> dict:
    """Builds lookup dictionary of latest rolling resource allocation and playstyle metrics per team."""
    team_rp = {}
    for _, row in df_hist.iterrows():
        b_team = row.get('blue_team')
        r_team = row.get('red_team')

        if pd.notna(b_team):
            b_dict = {}
            for m in RESOURCE_PLAYSTYLE_METRICS:
                possible_cols = [
                    f'blue_roll_{m}',
                    f'blue_hist_{m}_avg_last10',
                    f'blue_{m}'
                ]
                val = None
                for col in possible_cols:
                    if col in row and pd.notna(row[col]):
                        val = float(row[col])
                        break
                b_dict[m] = val if val is not None else RESOURCE_PLAYSTYLE_DEFAULTS[m]
            team_rp[str(b_team)] = b_dict

        if pd.notna(r_team):
            r_dict = {}
            for m in RESOURCE_PLAYSTYLE_METRICS:
                possible_cols = [
                    f'red_roll_{m}',
                    f'red_hist_{m}_avg_last10',
                    f'red_{m}'
                ]
                val = None
                for col in possible_cols:
                    if col in row and pd.notna(row[col]):
                        val = float(row[col])
                        break
                r_dict[m] = val if val is not None else RESOURCE_PLAYSTYLE_DEFAULTS[m]
            team_rp[str(r_team)] = r_dict

    return team_rp


def build_vision_lookups(df_hist: pd.DataFrame) -> dict:
    """Builds lookup dictionary of latest rolling Vision & Map Control metrics per team."""
    team_vision = {}
    for _, row in df_hist.iterrows():
        b_team = row.get('blue_team')
        r_team = row.get('red_team')

        if pd.notna(b_team):
            b_dict = {}
            for m in VISION_METRICS:
                possible_cols = [
                    f'blue_hist_{m}_avg_last10',
                    f'blue_roll_{m}',
                    f'blue_{m}',
                    f'blue_calc_{m}'
                ]
                val = None
                for col in possible_cols:
                    if col in row and pd.notna(row[col]):
                        val = float(row[col])
                        break
                b_dict[m] = val if val is not None else VISION_DEFAULTS[m]
            team_vision[str(b_team)] = b_dict

        if pd.notna(r_team):
            r_dict = {}
            for m in VISION_METRICS:
                possible_cols = [
                    f'red_hist_{m}_avg_last10',
                    f'red_roll_{m}',
                    f'red_{m}',
                    f'red_calc_{m}'
                ]
                val = None
                for col in possible_cols:
                    if col in row and pd.notna(row[col]):
                        val = float(row[col])
                        break
                r_dict[m] = val if val is not None else VISION_DEFAULTS[m]
            team_vision[str(r_team)] = r_dict

    return team_vision


def build_patch_meta_lookup(df_hist: pd.DataFrame) -> dict:
    """Builds dictionary of top meta champions per patch (top 25% pick rate)."""
    champ_cols = [f"{side}_{role}_champion" for side in ["blue", "red"] for role in ["top", "jng", "mid", "bot", "sup"]]
    existing_champ_cols = [c for c in champ_cols if c in df_hist.columns]

    if existing_champ_cols and "patch" in df_hist.columns:
        melted = df_hist.melt(id_vars=["patch"], value_vars=existing_champ_cols, value_name="champion").dropna()
        patch_meta = (
            melted.groupby(["patch", "champion"])
            .size()
            .groupby(level=0, group_keys=False)
            .apply(lambda x: set(x[x >= x.quantile(0.75)].index.tolist()))
            .to_dict()
        )
        return patch_meta
    return {}


def build_patch_adaptability_lookups(df_hist: pd.DataFrame) -> dict:
    """Builds lookup dictionary of latest team-level patch adaptability metrics."""
    team_pa = {}
    for _, row in df_hist.iterrows():
        b_team = row.get('blue_team')
        r_team = row.get('red_team')

        if pd.notna(b_team):
            b_dict = {}
            for m in PATCH_ADAPTABILITY_METRICS:
                col = f'blue_hist_{m}'
                b_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else PATCH_ADAPTABILITY_DEFAULTS[m]
            team_pa[str(b_team)] = b_dict

        if pd.notna(r_team):
            r_dict = {}
            for m in PATCH_ADAPTABILITY_METRICS:
                col = f'red_hist_{m}'
                r_dict[m] = float(row[col]) if col in row and pd.notna(row[col]) else PATCH_ADAPTABILITY_DEFAULTS[m]
            team_pa[str(r_team)] = r_dict

    return team_pa


def build_player_and_champ_lookups(df_hist: pd.DataFrame) -> tuple[dict, dict]:
    """Builds player and champion historical performance lookup dictionaries."""
    player_stats = {}
    champ_stats = {}

    for role in ROLES:
        for side in ['blue', 'red']:
            p_col = f'{side}_{role}_player'
            g_col = f'{side}_{role}_player_games_pre'
            w_col = f'{side}_{role}_player_winrate_pre'
            c_col = f'{side}_{role}_champion'
            cg_col = f'{side}_{role}_champ_games_pre'
            cw_col = f'{side}_{role}_champ_winrate_pre'

            if p_col in df_hist.columns and w_col in df_hist.columns:
                for _, row in df_hist[[p_col, g_col, w_col]].dropna().iterrows():
                    player_stats[str(row[p_col])] = {
                        'games': int(row[g_col]),
                        'winrate': float(row[w_col])
                    }

            if c_col in df_hist.columns and cw_col in df_hist.columns:
                for _, row in df_hist[[c_col, cg_col, cw_col]].dropna().iterrows():
                    champ_stats[str(row[c_col])] = {
                        'games': int(row[cg_col]),
                        'winrate': float(row[cw_col])
                    }

    return player_stats, champ_stats


def align_dtypes_and_shape(df: pd.DataFrame, model, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Aligns DataFrame column types and strictly forces model feature count."""
    df = df.copy()

    model_type_str = str(type(model)).lower()
    is_catboost = "catboost" in model_type_str
    is_lgb = "lightgbm" in model_type_str

    saved_cats = getattr(model, "pandas_categorical_", None)

    for col in df.columns:
        if col.endswith('_champion') or col.endswith('_player') or col.endswith('_team'):
            if is_catboost:
                if col in df_hist.columns:
                    known_cats = set(str(x) for x in df_hist[col].dropna().unique())
                    vals = df[col].astype(str)
                    df[col] = vals.apply(lambda x: x if x in known_cats else 'missing')
                else:
                    df[col] = df[col].fillna('missing').astype(str)

                df[col] = df[col].replace({'nan': 'missing', 'NaN': 'missing', 'None': 'missing', '': 'missing'})

            elif saved_cats and col in saved_cats:
                df[col] = pd.Categorical(df[col].astype(str), categories=saved_cats[col])

            elif is_lgb and not saved_cats:
                df[col] = df[col].astype('category').cat.codes.astype('float64')

            else:
                if col in df_hist.columns:
                    known_cats = [str(x) for x in df_hist[col].dropna().unique().tolist()]
                    df[col] = pd.Categorical(df[col].astype(str), categories=known_cats)
                else:
                    df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    target_n_features = None
    if hasattr(model, "n_features_in_"):
        target_n_features = model.n_features_in_
    elif hasattr(model, "booster_") and hasattr(model.booster_, "num_feature"):
        target_n_features = model.booster_.num_feature()

    if target_n_features and df.shape[1] != target_n_features:
        if df.shape[1] > target_n_features:
            clean_cols = [c for c in df.columns if c not in META_COLUMNS]
            if len(clean_cols) == target_n_features:
                df = df[clean_cols]
            else:
                df = df.iloc[:, :target_n_features]
        elif df.shape[1] < target_n_features:
            for i in range(df.shape[1], target_n_features):
                df[f"missing_feature_{i}"] = 0.0

    return df


def build_role_breakdown(
    draft_payload: dict,
    player_stats: dict,
    champ_stats: dict,
    defaults: dict
) -> list:
    """Builds per-role metrics breakdown for display/analysis."""
    role_breakdown = []
    blue_players = draft_payload.get('blue_players', [])
    red_players = draft_payload.get('red_players', [])
    blue_champs = draft_payload.get('blue_champs', [])
    red_champs = draft_payload.get('red_champs', [])

    for i, role in enumerate(ROLES):
        bp = blue_players[i] if i < len(blue_players) else ''
        rp = red_players[i] if i < len(red_players) else ''
        bc = blue_champs[i] if i < len(blue_champs) else ''
        rc = red_champs[i] if i < len(red_champs) else ''

        bp_s = player_stats.get(str(bp), {'games': defaults['player_games'], 'winrate': defaults['player_winrate']})
        rp_s = player_stats.get(str(rp), {'games': defaults['player_games'], 'winrate': defaults['player_winrate']})
        bc_s = champ_stats.get(str(bc), {'games': defaults['champ_games'], 'winrate': defaults['champ_winrate']})
        rc_s = champ_stats.get(str(rc), {'games': defaults['champ_games'], 'winrate': defaults['champ_winrate']})

        role_breakdown.append({
            'role': role.upper(),
            'blue_player': bp,
            'blue_p_wr': bp_s['winrate'],
            'blue_p_games': bp_s['games'],
            'red_player': rp,
            'red_p_wr': rp_s['winrate'],
            'red_p_games': rp_s['games'],
            'blue_champ': bc,
            'blue_c_wr': bc_s['winrate'],
            'red_champ': rc,
            'red_c_wr': rc_s['winrate'],
        })

    return role_breakdown