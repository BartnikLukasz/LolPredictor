import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

ROLES = ['top', 'jng', 'mid', 'bot', 'sup']
META_COLUMNS = [
    'date', 'blue_win', 'match_id', 'game_id',
    'blue_team', 'red_team', 'league', 'patch',
    'split', 'tournament', 'year', 'season'
]


class LiveFeatureEngine:
    def __init__(self, dataset_path: str, model_path: str = "models/xgboost_model.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Run model_trainer.py first.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Historical feature dataset '{dataset_path}' not found.")

        # 1. Load Trained Model (.json or .pkl / .joblib)
        self.model_path = model_path
        if model_path.endswith(".json"):
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            artifact = joblib.load(model_path)
            self.model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact

        # 2. Load Historical Data
        print("Loading reference lookup data from historical dataset...")
        self.df_hist = pd.read_csv(dataset_path, low_memory=False)
        if 'date' in self.df_hist.columns:
            self.df_hist['date'] = pd.to_datetime(self.df_hist['date'])
            self.df_hist = self.df_hist.sort_values('date').reset_index(drop=True)

        # 3. Extract Exact Expected Features from Loaded Model
        self.expected_features = self._extract_model_feature_names()

        # Fallback if model has no saved feature schema or generic names
        valid_hist_cols = [c for c in self.df_hist.columns if c not in META_COLUMNS]
        if not self.expected_features or not all(f in self.df_hist.columns for f in self.expected_features):
            self.expected_features = valid_hist_cols

        # 4. Build Lookup Maps
        self._build_elo_lookup()
        self._build_momentum_lookup()
        self._build_player_and_champ_lookups()

    def _extract_model_feature_names(self) -> list:
        """Extracts the exact ordered list of feature names expected by the model."""
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            return list(self.model.feature_names_in_)

        if hasattr(self.model, "booster_") and hasattr(self.model.booster_, "feature_name"):
            fn = self.model.booster_.feature_name()
            if fn and len(fn) > 0 and not fn[0].startswith("Column_"):
                return list(fn)

        if hasattr(self.model, "get_booster"):
            try:
                fn = self.model.get_booster().feature_names
                if fn and len(fn) > 0:
                    return list(fn)
            except Exception:
                pass

        return []

    def _build_elo_lookup(self):
        """Builds dictionary of latest team Elo ratings."""
        self.latest_elo = {}
        for _, row in self.df_hist.iterrows():
            if pd.notna(row.get('blue_team')) and pd.notna(row.get('blue_elo_pre')):
                self.latest_elo[row['blue_team']] = float(row['blue_elo_pre'])
            if pd.notna(row.get('red_team')) and pd.notna(row.get('red_elo_pre')):
                self.latest_elo[row['red_team']] = float(row['red_elo_pre'])

    def _build_momentum_lookup(self):
        """Builds dictionary of latest team momentum / streak / overperformance metrics."""
        self.latest_momentum = {}
        for _, row in self.df_hist.iterrows():
            b_team = row.get('blue_team')
            r_team = row.get('red_team')

            def _clean_val(val, default=0.0):
                return float(val) if pd.notna(val) else default

            if pd.notna(b_team) and str(b_team).strip() != '':
                self.latest_momentum[str(b_team)] = {
                    'momentum': _clean_val(row.get('blue_momentum_pre', row.get('blue_recent_winrate_pre')), 0.50),
                    'streak': _clean_val(row.get('blue_streak_pre', row.get('blue_win_streak_pre')), 0.0),
                    'elo_delta_10': _clean_val(row.get('blue_elo_delta_10'), 0.0),
                    'overperform_10': _clean_val(row.get('blue_overperform_10'), 0.0)
                }

            if pd.notna(r_team) and str(r_team).strip() != '':
                self.latest_momentum[str(r_team)] = {
                    'momentum': _clean_val(row.get('red_momentum_pre', row.get('red_recent_winrate_pre')), 0.50),
                    'streak': _clean_val(row.get('red_streak_pre', row.get('red_win_streak_pre')), 0.0),
                    'elo_delta_10': _clean_val(row.get('red_elo_delta_10'), 0.0),
                    'overperform_10': _clean_val(row.get('red_overperform_10'), 0.0)
                }

    def _build_player_and_champ_lookups(self):
        """Builds player and champion historical performance lookup dictionaries."""
        self.player_stats = {}
        self.champ_stats = {}

        self.defaults = {
            'player_games': 10,
            'player_winrate': 0.50,
            'champ_games': 10,
            'champ_winrate': 0.50,
        }

        for role in ROLES:
            for side in ['blue', 'red']:
                p_col = f'{side}_{role}_player'
                g_col = f'{side}_{role}_player_games_pre'
                w_col = f'{side}_{role}_player_winrate_pre'
                c_col = f'{side}_{role}_champion'
                cg_col = f'{side}_{role}_champ_games_pre'
                cw_col = f'{side}_{role}_champ_winrate_pre'

                if p_col in self.df_hist.columns and w_col in self.df_hist.columns:
                    for _, row in self.df_hist[[p_col, g_col, w_col]].dropna().iterrows():
                        self.player_stats[str(row[p_col])] = {
                            'games': int(row[g_col]),
                            'winrate': float(row[w_col])
                        }

                if c_col in self.df_hist.columns and cw_col in self.df_hist.columns:
                    for _, row in self.df_hist[[c_col, cg_col, cw_col]].dropna().iterrows():
                        self.champ_stats[str(row[c_col])] = {
                            'games': int(row[cg_col]),
                            'winrate': float(row[cw_col])
                        }

    def get_player_stat(self, player_name: str) -> dict:
        return self.player_stats.get(player_name, {
            'games': self.defaults['player_games'],
            'winrate': self.defaults['player_winrate']
        })

    def get_champ_stat(self, champ_name: str) -> dict:
        return self.champ_stats.get(champ_name, {
            'games': self.defaults['champ_games'],
            'winrate': self.defaults['champ_winrate']
        })

    def build_feature_vector(self, draft_payload: dict) -> pd.DataFrame:
        row = {}

        # 1. Elo Features
        blue_team = draft_payload.get('blue_team', '')
        red_team = draft_payload.get('red_team', '')
        blue_fp = draft_payload.get('blue_firstpick', 1)

        custom_elo = draft_payload.get('custom_elo_metrics')
        if custom_elo:
            b_elo = custom_elo.get('blue_elo_pre', self.latest_elo.get(blue_team, 1500.0))
            r_elo = custom_elo.get('red_elo_pre', self.latest_elo.get(red_team, 1500.0))
            elo_diff = custom_elo.get('elo_diff', b_elo - r_elo)
            blue_win_prob = custom_elo.get('blue_elo_win_prob', 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)))
        else:
            b_elo = self.latest_elo.get(blue_team, 1500.0)
            r_elo = self.latest_elo.get(red_team, 1500.0)
            first_pick_bonus = 10.0 if blue_fp == 1 else -10.0
            elo_diff = (b_elo + first_pick_bonus) - r_elo
            blue_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        row['blue_elo_pre'] = b_elo
        row['red_elo_pre'] = r_elo
        row['elo_diff'] = elo_diff
        row['blue_elo_win_prob'] = blue_win_prob
        row['blue_firstpick'] = blue_fp

        # 2. Team Momentum Features
        b_mom_data = self.latest_momentum.get(blue_team, {})
        r_mom_data = self.latest_momentum.get(red_team, {})

        b_mom = b_mom_data.get('momentum', 0.50)
        r_mom = r_mom_data.get('momentum', 0.50)
        b_streak = b_mom_data.get('streak', 0.0)
        r_streak = r_mom_data.get('streak', 0.0)

        b_delta_10 = b_mom_data.get('elo_delta_10', 0.0)
        r_delta_10 = r_mom_data.get('elo_delta_10', 0.0)
        b_overperform_10 = b_mom_data.get('overperform_10', 0.0)
        r_overperform_10 = r_mom_data.get('overperform_10', 0.0)

        row['blue_momentum_pre'] = b_mom
        row['red_momentum_pre'] = r_mom
        row['momentum_diff'] = b_mom - r_mom
        row['blue_streak_pre'] = b_streak
        row['red_streak_pre'] = r_streak

        row['blue_elo_delta_10'] = b_delta_10
        row['red_elo_delta_10'] = r_delta_10
        row['elo_delta_diff_10'] = b_delta_10 - r_delta_10
        row['blue_overperform_10'] = b_overperform_10
        row['red_overperform_10'] = r_overperform_10
        row['overperformance_diff_10'] = b_overperform_10 - r_overperform_10

        # 3. Series Context Features
        row['game_number'] = draft_payload.get('game_number', 1)
        row['blue_series_lead'] = draft_payload.get('blue_series_lead', 0)
        row['blue_prev_win'] = draft_payload.get('blue_prev_win', 0)

        # 4. Champion Picks & Players
        blue_champs = draft_payload.get('blue_champs', ['', '', '', '', ''])
        red_champs = draft_payload.get('red_champs', ['', '', '', '', ''])
        blue_players = draft_payload.get('blue_players', ['', '', '', '', ''])
        red_players = draft_payload.get('red_players', ['', '', '', '', ''])

        for idx, role in enumerate(ROLES):
            b_player = blue_players[idx] if idx < len(blue_players) else ''
            r_player = red_players[idx] if idx < len(red_players) else ''
            b_champ = blue_champs[idx] if idx < len(blue_champs) else ''
            r_champ = red_champs[idx] if idx < len(red_champs) else ''

            row[f'blue_{role}_champion'] = b_champ
            row[f'red_{role}_champion'] = r_champ
            row[f'blue_{role}_player'] = b_player
            row[f'red_{role}_player'] = r_player

            bp_stat = self.get_player_stat(b_player)
            rp_stat = self.get_player_stat(r_player)
            bc_stat = self.get_champ_stat(b_champ)
            rc_stat = self.get_champ_stat(r_champ)

            row[f'blue_{role}_player_games_pre'] = bp_stat['games']
            row[f'blue_{role}_player_winrate_pre'] = bp_stat['winrate']
            row[f'blue_{role}_champ_games_pre'] = bc_stat['games']
            row[f'blue_{role}_champ_winrate_pre'] = bc_stat['winrate']

            row[f'red_{role}_player_games_pre'] = rp_stat['games']
            row[f'red_{role}_player_winrate_pre'] = rp_stat['winrate']
            row[f'red_{role}_champ_games_pre'] = rc_stat['games']
            row[f'red_{role}_champ_winrate_pre'] = rc_stat['winrate']

        live_df = pd.DataFrame([row])

        # Fill any missing columns from historical schema with defaults
        for col in self.expected_features:
            if col not in live_df.columns:
                live_df[col] = 0.0

        # Strictly filter and align features to match target training count
        live_df = live_df[[c for c in self.expected_features if c in live_df.columns]].copy()

        return live_df

    def _categorize_features(self, columns: list) -> dict:
        """Groups DataFrame columns into the 8 target feature categories."""
        categories = {
            'elo': [],
            'momentum': [],
            'series': [],
            'player': [],
            'h2h': [],
            'synergy': [],
            'draft_champ': [],
            'champ': []
        }

        for col in columns:
            if col in ['elo_diff', 'blue_elo_pre', 'red_elo_pre', 'blue_elo_win_prob', 'blue_firstpick']:
                categories['elo'].append(col)
            elif col in [
                'blue_elo_delta_10', 'red_elo_delta_10', 'elo_delta_diff_10',
                'blue_overperform_10', 'red_overperform_10', 'overperformance_diff_10',
                'blue_momentum_pre', 'red_momentum_pre', 'momentum_diff',
                'blue_streak_pre', 'red_streak_pre'
            ] or 'momentum' in col or 'streak' in col or 'overperform' in col or 'delta' in col:
                categories['momentum'].append(col)
            elif col in ['game_number', 'blue_series_lead', 'blue_prev_win'] or 'series' in col:
                categories['series'].append(col)
            elif col.endswith('_player_games_pre') or col.endswith('_player_winrate_pre') or \
                    col.endswith('_champ_games_pre') or col.endswith('_champ_winrate_pre') or \
                    col.endswith('_player'):
                categories['player'].append(col)
            elif 'h2h' in col or 'lane_matchup' in col or 'p2p' in col:
                categories['h2h'].append(col)
            elif 'roster' in col or 'duo' in col:
                categories['synergy'].append(col)
            elif 'patch' in col or 'counter' in col or 'synergy' in col or 'cohesion' in col or 'comp' in col:
                categories['draft_champ'].append(col)
            elif col in [
                'blue_top_champion', 'blue_jng_champion', 'blue_mid_champion', 'blue_bot_champion', 'blue_sup_champion',
                'red_top_champion', 'red_jng_champion', 'red_mid_champion', 'red_bot_champion', 'red_sup_champion'
            ] or col.endswith('_champion'):
                categories['champ'].append(col)
            else:
                categories['draft_champ'].append(col)

        return categories

    def _create_neutral_df(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """Creates a baseline DataFrame with completely neutralized features."""
        neutral_df = aligned_df.copy()
        is_catboost = "catboost" in str(type(self.model)).lower()

        for col in neutral_df.columns:
            if col.endswith('_champion') or col.endswith('_player') or col.endswith('_team'):
                neutral_df[col] = 'missing' if is_catboost else None
            elif col.endswith('_winrate_pre') or col == 'blue_elo_win_prob':
                neutral_df[col] = 0.50
            elif col.endswith('_games_pre'):
                neutral_df[col] = 10
            elif 'elo' in col:
                if 'diff' in col or 'delta' in col:
                    neutral_df[col] = 0.0
                else:
                    neutral_df[col] = 1500.0
            else:
                neutral_df[col] = 0.0

        return neutral_df

    def _align_dtypes_and_shape(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns DataFrame column types and strictly forces model feature count."""
        df = df.copy()

        model_type_str = str(type(self.model)).lower()
        is_catboost = "catboost" in model_type_str
        is_lgb = "lightgbm" in model_type_str

        saved_cats = getattr(self.model, "pandas_categorical_", None)

        for col in df.columns:
            if col.endswith('_champion') or col.endswith('_player') or col.endswith('_team'):
                if is_catboost:
                    if col in self.df_hist.columns:
                        known_cats = set(str(x) for x in self.df_hist[col].dropna().unique())
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
                    if col in self.df_hist.columns:
                        known_cats = [str(x) for x in self.df_hist[col].dropna().unique().tolist()]
                        df[col] = pd.Categorical(df[col].astype(str), categories=known_cats)
                    else:
                        df[col] = df[col].astype('category')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        target_n_features = None
        if hasattr(self.model, "n_features_in_"):
            target_n_features = self.model.n_features_in_
        elif hasattr(self.model, "booster_") and hasattr(self.model.booster_, "num_feature"):
            target_n_features = self.model.booster_.num_feature()

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

    def _eval_model_prob(self, df: pd.DataFrame) -> float:
        """Utility runner for probability extraction."""
        aligned = self._align_dtypes_and_shape(df)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(aligned)[0][1])
        else:
            dmatrix = xgb.DMatrix(aligned, enable_categorical=True)
            return float(self.model.predict(dmatrix)[0])

    def predict_match(self, draft_payload: dict) -> dict:
        feature_df = self.build_feature_vector(draft_payload)
        aligned_df = self._align_dtypes_and_shape(feature_df)

        blue_team = draft_payload.get('blue_team', 'Blue Team')
        red_team = draft_payload.get('red_team', 'Red Team')

        # 1. Group columns into categories
        cat_map = self._categorize_features(aligned_df.columns.tolist())

        # 2. Build cumulative 8-stage DataFrame predictions
        current_df = self._create_neutral_df(aligned_df)

        # Base 50% state
        base_pct = 50.0

        stage_results = {}
        category_order = [
            ('elo', '1. Elo Rating'),
            ('momentum', '2. Team Momentum'),
            ('series', '3. Series Context'),
            ('player', '4. Player Mastery'),
            ('h2h', '5. Head-to-Head'),
            ('synergy', '6. Roster Synergy'),
            ('draft_champ', '7. Draft Synergy & Counters'),
            ('champ', '8. Champion Picks')
        ]

        prev_prob = 0.50

        for cat_key, cat_label in category_order:
            cols = cat_map[cat_key]
            if cols:
                # Inject actual feature values for this category
                for col in cols:
                    current_df[col] = aligned_df[col].values

            try:
                prob = self._eval_model_prob(current_df)
            except Exception:
                prob = prev_prob

            stage_results[cat_key] = {
                'label': cat_label,
                'pct': round(prob * 100, 2),
                'swing': round((prob - prev_prob) * 100, 2)
            }
            prev_prob = prob

        final_pct = stage_results['champ']['pct']
        proba_blue = final_pct / 100.0
        proba_red = 1.0 - proba_blue

        # Build Progression Map DataFrame for Plotly / Dashboard Waterfall
        prog_stages = ["0. Baseline (50%)"] + [item['label'] for item in stage_results.values()] + ["Final Prediction"]
        prog_vals = [base_pct] + [item['pct'] for item in stage_results.values()] + [final_pct]
        prog_deltas = [0.0] + [item['swing'] for item in stage_results.values()] + [0.0]

        progression_data = pd.DataFrame({
            "Stage": prog_stages,
            f"{blue_team} Win %": prog_vals,
            "Impact Delta": prog_deltas
        })

        # Role Breakdown
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

            bp_s = self.get_player_stat(bp)
            rp_s = self.get_player_stat(rp)
            bc_s = self.get_champ_stat(bc)
            rc_s = self.get_champ_stat(rc)

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

        avg_blue_p_wr = np.mean([r['blue_p_wr'] for r in role_breakdown])
        avg_red_p_wr = np.mean([r['red_p_wr'] for r in role_breakdown])
        avg_blue_c_wr = np.mean([r['blue_c_wr'] for r in role_breakdown])
        avg_red_c_wr = np.mean([r['red_c_wr'] for r in role_breakdown])

        b_elo = self.latest_elo.get(blue_team, 1500.0)
        r_elo = self.latest_elo.get(red_team, 1500.0)

        return {
            'blue_win_probability': proba_blue,
            'red_win_probability': proba_red,
            'blue_win_percentage': final_pct,
            'red_win_percentage': round(proba_red * 100, 2),
            'progression_data': progression_data,
            'draft_swings': {
                'elo_swing': stage_results['elo']['swing'],
                'momentum_swing': stage_results['momentum']['swing'],
                'series_swing': stage_results['series']['swing'],
                'player_swing': stage_results['player']['swing'],
                'h2h_swing': stage_results['h2h']['swing'],
                'synergy_swing': stage_results['synergy']['swing'],
                'draft_champ_swing': stage_results['draft_champ']['swing'],
                'champ_swing': stage_results['champ']['swing'],
                'total_swing': round(final_pct - base_pct, 2)
            },
            'series_metrics': {
                'game_number': draft_payload.get('game_number', 1),
                'blue_series_lead': draft_payload.get('blue_series_lead', 0),
                'blue_prev_win': draft_payload.get('blue_prev_win', 0)
            },
            'elo_metrics': {
                'blue_elo': round(b_elo, 1),
                'red_elo': round(r_elo, 1),
                'elo_diff': round(b_elo - r_elo, 1),
                'elo_implied_blue_winrate': stage_results['elo']['pct']
            },
            'player_metrics': {
                'avg_blue_p_wr': round(avg_blue_p_wr * 100, 2),
                'avg_red_p_wr': round(avg_red_p_wr * 100, 2),
                'p_wr_diff': round((avg_blue_p_wr - avg_red_p_wr) * 100, 2)
            },
            'draft_metrics': {
                'avg_blue_c_wr': round(avg_blue_c_wr * 100, 2),
                'avg_red_c_wr': round(avg_red_c_wr * 100, 2),
                'c_wr_diff': round((avg_blue_c_wr - avg_red_c_wr) * 100, 2)
            },
            'role_breakdown': role_breakdown
        }