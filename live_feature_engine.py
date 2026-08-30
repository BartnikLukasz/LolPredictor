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
        self._build_player_and_champ_lookups()

    def _extract_model_feature_names(self) -> list:
        """Extracts the exact ordered list of feature names expected by the model."""
        # Check Scikit-Learn wrapper feature names attribute
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            return list(self.model.feature_names_in_)

        # Check LightGBM Booster feature names
        if hasattr(self.model, "booster_") and hasattr(self.model.booster_, "feature_name"):
            fn = self.model.booster_.feature_name()
            if fn and len(fn) > 0 and not fn[0].startswith("Column_"):
                return list(fn)

        # Check XGBoost Booster feature names
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

        # 2. Series Context Features
        row['game_number'] = draft_payload.get('game_number', 1)
        row['blue_series_lead'] = draft_payload.get('blue_series_lead', 0)
        row['blue_prev_win'] = draft_payload.get('blue_prev_win', 0)

        # 3. Champion Picks & Players
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

    def _align_dtypes_and_shape(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns DataFrame column types and strictly forces model feature count."""
        df = df.copy()

        # Determine target feature count expected by LightGBM
        target_n_features = None
        if hasattr(self.model, "n_features_in_"):
            target_n_features = self.model.n_features_in_
        elif hasattr(self.model, "booster_") and hasattr(self.model.booster_, "num_feature"):
            target_n_features = self.model.booster_.num_feature()

        # Handle LightGBM categorical variables
        is_lgb = "lightgbm" in str(type(self.model)).lower()
        saved_cats = getattr(self.model, "pandas_categorical_", None)

        for col in df.columns:
            if saved_cats and col in saved_cats:
                df[col] = pd.Categorical(df[col].astype(str), categories=saved_cats[col])
            elif col.endswith('_champion') or col.endswith('_player') or col.endswith('_team'):
                if is_lgb and not saved_cats:
                    df[col] = df[col].astype('category').cat.codes.astype('float64')
                else:
                    df[col] = df[col].astype('category')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Truncate or pad columns if feature count still mismatches exact target shape
        if target_n_features and df.shape[1] != target_n_features:
            if df.shape[1] > target_n_features:
                # Exclude metadata columns first
                clean_cols = [c for c in df.columns if c not in META_COLUMNS]
                if len(clean_cols) == target_n_features:
                    df = df[clean_cols]
                else:
                    df = df.iloc[:, :target_n_features]
            elif df.shape[1] < target_n_features:
                for i in range(df.shape[1], target_n_features):
                    df[f"missing_feature_{i}"] = 0.0

        return df

    def predict_match(self, draft_payload: dict) -> dict:
        feature_df = self.build_feature_vector(draft_payload)
        feature_df = self._align_dtypes_and_shape(feature_df)

        # Model Inference
        if hasattr(self.model, "predict_proba"):
            proba_blue = float(self.model.predict_proba(feature_df)[0][1])
        else:
            dmatrix = xgb.DMatrix(feature_df, enable_categorical=True)
            proba_blue = float(self.model.predict(dmatrix)[0])

        proba_red = 1.0 - proba_blue

        blue_team = draft_payload.get('blue_team', 'Blue Team')
        red_team = draft_payload.get('red_team', 'Red Team')
        b_elo = self.latest_elo.get(blue_team, 1500.0)
        r_elo = self.latest_elo.get(red_team, 1500.0)
        elo_diff = (b_elo + (10.0 if draft_payload.get('blue_firstpick', 1) == 1 else -10.0)) - r_elo
        elo_implied_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

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

        return {
            'blue_win_probability': proba_blue,
            'red_win_probability': proba_red,
            'blue_win_percentage': round(proba_blue * 100, 2),
            'red_win_percentage': round(proba_red * 100, 2),
            'series_metrics': {
                'game_number': draft_payload.get('game_number', 1),
                'blue_series_lead': draft_payload.get('blue_series_lead', 0),
                'blue_prev_win': draft_payload.get('blue_prev_win', 0)
            },
            'elo_metrics': {
                'blue_elo': round(b_elo, 1),
                'red_elo': round(r_elo, 1),
                'elo_diff': round(elo_diff, 1),
                'elo_implied_blue_winrate': round(elo_implied_prob * 100, 2)
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


if __name__ == "__main__":
    dataset_file = "dataset/pregame/pregame_dataset_final_features.csv"

    if os.path.exists(dataset_file) and os.path.exists("models/xgboost_model.json"):
        engine = LiveFeatureEngine(dataset_path=dataset_file)

        sample_draft = {
            "blue_team": "T1",
            "red_team": "Gen.G",
            "blue_players": ["Doran", "Oner", "Faker", "Gumayusi", "Keria"],
            "red_players": ["Kiin", "Canyon", "Chovy", "Ruler", "Duro"],
            "blue_champs": ["Aatrox", "Sejuani", "Ahri", "Jinx", "Nautilus"],
            "red_champs": ["K'Sante", "Vi", "Azir", "Varus", "Rakan"],
            "blue_firstpick": 1,
            "game_number": 3,
            "blue_series_lead": 1,
            "blue_prev_win": 1
        }

        result = engine.predict_match(sample_draft)
        print("\n" + "=" * 45)
        print("          LIVE MATCH PREDICTION RESULT          ")
        print("=" * 45)
        print(f"Blue Side ({sample_draft['blue_team']}): {result['blue_win_percentage']}%")
        print(f"Red Side  ({sample_draft['red_team']}): {result['red_win_percentage']}%")
        print("=" * 45)