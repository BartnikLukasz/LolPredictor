# live_feature_engine.py
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from live_feature_helpers import (
    ROLES,
    META_COLUMNS,
    EARLY_GAME_METRICS,
    EARLY_GAME_DEFAULTS,
    STRATEGIC_METRICS,
    STRATEGIC_DEFAULTS,
    RESOURCE_PLAYSTYLE_METRICS,
    RESOURCE_PLAYSTYLE_DEFAULTS,
    VISION_METRICS,
    VISION_DEFAULTS,
    PATCH_ADAPTABILITY_METRICS,
    PATCH_ADAPTABILITY_DEFAULTS,
    DEFAULT_PLAYER_CHAMP_STATS,
    extract_model_feature_names,
    build_elo_lookup,
    build_early_game_lookups,
    build_strategic_lookups,
    build_resource_playstyle_lookups,
    build_vision_lookups,
    build_patch_meta_lookup,
    build_patch_adaptability_lookups,
    build_player_and_champ_lookups,
    align_dtypes_and_shape,
    build_role_breakdown,
)


class LiveFeatureEngine:
    def __init__(self, dataset_path: str, model_path: str = "models/xgboost_model.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Run model_trainer.py first.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Historical feature dataset '{dataset_path}' not found.")

        # 1. Load Trained Model
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

        # 3. Extract Expected Features
        self.expected_features = extract_model_feature_names(self.model)
        valid_hist_cols = [c for c in self.df_hist.columns if c not in META_COLUMNS]
        if not self.expected_features or not all(f in self.df_hist.columns for f in self.expected_features):
            self.expected_features = valid_hist_cols

        # 4. Build Lookups via Helpers
        self.latest_elo = build_elo_lookup(self.df_hist)
        self.team_early_game = build_early_game_lookups(self.df_hist)
        self.team_strategic = build_strategic_lookups(self.df_hist)
        self.team_resource_playstyle = build_resource_playstyle_lookups(self.df_hist)
        self.team_vision = build_vision_lookups(self.df_hist)
        self.patch_meta = build_patch_meta_lookup(self.df_hist)
        self.team_patch_adaptability = build_patch_adaptability_lookups(self.df_hist)
        self.player_stats, self.champ_stats = build_player_and_champ_lookups(self.df_hist)
        self.defaults = DEFAULT_PLAYER_CHAMP_STATS.copy()

    def get_player_stat(self, player_name: str) -> dict:
        return self.player_stats.get(str(player_name), {
            'games': self.defaults['player_games'],
            'winrate': self.defaults['player_winrate']
        })

    def get_champ_stat(self, champ_name: str) -> dict:
        return self.champ_stats.get(str(champ_name), {
            'games': self.defaults['champ_games'],
            'winrate': self.defaults['champ_winrate']
        })

    def get_team_early_game(self, team_name: str) -> dict:
        return self.team_early_game.get(str(team_name), EARLY_GAME_DEFAULTS.copy())

    def get_team_strategic(self, team_name: str) -> dict:
        return self.team_strategic.get(str(team_name), STRATEGIC_DEFAULTS.copy())

    def get_team_resource_playstyle(self, team_name: str) -> dict:
        return self.team_resource_playstyle.get(str(team_name), RESOURCE_PLAYSTYLE_DEFAULTS.copy())

    def get_team_vision(self, team_name: str) -> dict:
        return self.team_vision.get(str(team_name), VISION_DEFAULTS.copy())

    def get_team_patch_adaptability(self, team_name: str) -> dict:
        return self.team_patch_adaptability.get(str(team_name), PATCH_ADAPTABILITY_DEFAULTS.copy())

    def build_feature_vector(self, draft_payload: dict) -> pd.DataFrame:
        row = {}

        # 1. Elo Features
        blue_team = str(draft_payload.get('blue_team', ''))
        red_team = str(draft_payload.get('red_team', ''))
        blue_fp = draft_payload.get('blue_firstpick', 1)

        b_elo = self.latest_elo.get(blue_team, 1500.0)
        r_elo = self.latest_elo.get(red_team, 1500.0)
        first_pick_bonus = 10.0 if blue_fp == 1 else -10.0

        elo_diff = (b_elo + first_pick_bonus) - r_elo
        row['blue_elo_pre'] = b_elo
        row['red_elo_pre'] = r_elo
        row['elo_diff'] = elo_diff
        row['blue_elo_win_prob'] = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        row['blue_firstpick'] = blue_fp

        # 2. Rolling Early Game Features
        b_eg = self.get_team_early_game(blue_team)
        r_eg = self.get_team_early_game(red_team)
        for m in EARLY_GAME_METRICS:
            b_val = b_eg.get(m, EARLY_GAME_DEFAULTS[m])
            r_val = r_eg.get(m, EARLY_GAME_DEFAULTS[m])
            row[f'blue_roll_{m}'] = b_val
            row[f'red_roll_{m}'] = r_val
            row[f'diff_roll_{m}'] = b_val - r_val

        # 3. Rolling Strategic Priority Features
        b_strat = self.get_team_strategic(blue_team)
        r_strat = self.get_team_strategic(red_team)
        for m in STRATEGIC_METRICS:
            b_val = b_strat.get(m, STRATEGIC_DEFAULTS[m])
            r_val = r_strat.get(m, STRATEGIC_DEFAULTS[m])
            row[f'blue_roll_{m}'] = b_val
            row[f'red_roll_{m}'] = r_val
            row[f'diff_roll_{m}'] = b_val - r_val

        # 4. Step 3: Resource Allocation & Playstyle Profile Features
        b_rp = self.get_team_resource_playstyle(blue_team)
        r_rp = self.get_team_resource_playstyle(red_team)
        for m in RESOURCE_PLAYSTYLE_METRICS:
            b_val = b_rp.get(m, RESOURCE_PLAYSTYLE_DEFAULTS[m])
            r_val = r_rp.get(m, RESOURCE_PLAYSTYLE_DEFAULTS[m])

            row[f'blue_roll_{m}'] = b_val
            row[f'red_roll_{m}'] = r_val
            row[f'diff_roll_{m}'] = b_val - r_val

            row[f'blue_hist_{m}_avg_last10'] = b_val
            row[f'red_hist_{m}_avg_last10'] = r_val
            row[f'diff_hist_{m}_avg_last10'] = b_val - r_val

        # 5. Step 4: Vision & Map Control Features
        b_vis = self.get_team_vision(blue_team)
        r_vis = self.get_team_vision(red_team)
        for m in VISION_METRICS:
            b_val = b_vis.get(m, VISION_DEFAULTS[m])
            r_val = r_vis.get(m, VISION_DEFAULTS[m])

            row[f'blue_roll_{m}'] = b_val
            row[f'red_roll_{m}'] = r_val
            row[f'diff_roll_{m}'] = b_val - r_val

            row[f'blue_hist_{m}_avg_last10'] = b_val
            row[f'red_hist_{m}_avg_last10'] = r_val
            row[f'diff_hist_{m}_avg_last10'] = b_val - r_val

        # 6. Step 5: Patch & Meta Adaptability Features
        current_patch = str(draft_payload.get('patch', ''))
        if not current_patch and 'patch' in self.df_hist.columns:
            current_patch = str(self.df_hist['patch'].dropna().iloc[-1])

        patch_meta_champs = set(self.patch_meta.get(current_patch, []))

        blue_champs = draft_payload.get('blue_champs', ['', '', '', '', ''])
        red_champs = draft_payload.get('red_champs', ['', '', '', '', ''])

        b_valid_champs = [c for c in blue_champs if c and str(c) != '']
        r_valid_champs = [c for c in red_champs if c and str(c) != '']

        b_meta_score = (sum(1 for c in b_valid_champs if c in patch_meta_champs) / len(b_valid_champs)) if b_valid_champs else 0.50
        r_meta_score = (sum(1 for c in r_valid_champs if c in patch_meta_champs) / len(r_valid_champs)) if r_valid_champs else 0.50

        b_pa = self.get_team_patch_adaptability(blue_team)
        r_pa = self.get_team_patch_adaptability(red_team)

        row['blue_hist_patch_winrate'] = b_pa.get('patch_winrate', 0.50)
        row['red_hist_patch_winrate'] = r_pa.get('patch_winrate', 0.50)
        row['diff_hist_patch_winrate'] = row['blue_hist_patch_winrate'] - row['red_hist_patch_winrate']

        row['blue_hist_patch_wr_delta'] = b_pa.get('patch_wr_delta', 0.0)
        row['red_hist_patch_wr_delta'] = r_pa.get('patch_wr_delta', 0.0)
        row['diff_hist_patch_wr_delta'] = row['blue_hist_patch_wr_delta'] - row['red_hist_patch_wr_delta']

        row['blue_hist_champ_pool_depth'] = b_pa.get('champ_pool_depth', 10)
        row['red_hist_champ_pool_depth'] = r_pa.get('champ_pool_depth', 10)
        row['diff_hist_champ_pool_depth'] = row['blue_hist_champ_pool_depth'] - row['red_hist_champ_pool_depth']

        row['blue_hist_meta_alignment_score'] = b_meta_score
        row['red_hist_meta_alignment_score'] = r_meta_score
        row['diff_hist_meta_alignment_score'] = b_meta_score - r_meta_score

        # 7. Series Context Features
        row['game_number'] = draft_payload.get('game_number', 1)
        row['blue_series_lead'] = draft_payload.get('blue_series_lead', 0)
        row['blue_prev_win'] = draft_payload.get('blue_prev_win', 0)

        # 8. Champion Picks & Players
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

        for col in self.expected_features:
            if col not in live_df.columns:
                live_df[col] = 0.0

        return live_df[[c for c in self.expected_features if c in live_df.columns]].copy()

    def predict_match(self, draft_payload: dict) -> dict:
        feature_df = self.build_feature_vector(draft_payload)
        aligned_df = align_dtypes_and_shape(feature_df, self.model, self.df_hist)

        # 1. Final Prediction
        if hasattr(self.model, "predict_proba"):
            proba_blue = float(self.model.predict_proba(aligned_df)[0][1])
        else:
            dmatrix = xgb.DMatrix(aligned_df, enable_categorical=True)
            proba_blue = float(self.model.predict(dmatrix)[0])

        proba_red = 1.0 - proba_blue

        # 2. Stage 1: Elo Baseline
        blue_team = str(draft_payload.get('blue_team', 'Blue Team'))
        red_team = str(draft_payload.get('red_team', 'Red Team'))
        b_elo = self.latest_elo.get(blue_team, 1500.0)
        r_elo = self.latest_elo.get(red_team, 1500.0)
        elo_diff = (b_elo + (10.0 if draft_payload.get('blue_firstpick', 1) == 1 else -10.0)) - r_elo
        elo_base_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # 3. Stage 2: Elo + Player Mastery
        player_stage_df = aligned_df.copy()
        is_cb = 'catboost' in str(type(self.model)).lower()

        for col in player_stage_df.columns:
            if col.endswith('_champion'):
                player_stage_df[col] = 'missing' if is_cb else None
            elif col.endswith('_champ_winrate_pre'):
                player_stage_df[col] = 0.50
            elif col.endswith('_champ_games_pre'):
                player_stage_df[col] = 10
            elif col.startswith('diff_roll_') or col.startswith('diff_hist_'):
                player_stage_df[col] = 0.0

        try:
            if hasattr(self.model, "predict_proba"):
                player_stage_prob = float(self.model.predict_proba(player_stage_df)[0][1])
            else:
                dmatrix_p = xgb.DMatrix(player_stage_df, enable_categorical=True)
                player_stage_prob = float(self.model.predict(dmatrix_p)[0])
        except Exception:
            player_stage_prob = elo_base_prob

        # 4. Stage 3: Elo + Player Mastery + Team Macro, Vision & Meta Adaptability
        early_game_stage_df = aligned_df.copy()
        for col in early_game_stage_df.columns:
            if col.endswith('_champion'):
                early_game_stage_df[col] = 'missing' if is_cb else None
            elif col.endswith('_champ_winrate_pre'):
                early_game_stage_df[col] = 0.50
            elif col.endswith('_champ_games_pre'):
                early_game_stage_df[col] = 10

        try:
            if hasattr(self.model, "predict_proba"):
                early_game_stage_prob = float(self.model.predict_proba(early_game_stage_df)[0][1])
            else:
                dmatrix_eg = xgb.DMatrix(early_game_stage_df, enable_categorical=True)
                early_game_stage_prob = float(self.model.predict(dmatrix_eg)[0])
        except Exception:
            early_game_stage_prob = player_stage_prob

        # Calculations & Percentages
        elo_pct = round(elo_base_prob * 100, 2)
        player_pct = round(player_stage_prob * 100, 2)
        early_game_pct = round(early_game_stage_prob * 100, 2)
        final_pct = round(proba_blue * 100, 2)

        player_swing = round(player_pct - elo_pct, 2)
        early_game_swing = round(early_game_pct - player_pct, 2)
        draft_swing = round(final_pct - early_game_pct, 2)

        progression_data = pd.DataFrame({
            "Stage": [
                "1. Elo Baseline",
                "2. Player Mastery Impact",
                "3. Team Macro & Vision Impact",
                "4. Champion Draft Impact",
                "5. Final Prediction"
            ],
            f"{blue_team} Win %": [elo_pct, player_pct, early_game_pct, final_pct, final_pct],
            "Impact Delta": [0.0, player_swing, early_game_swing, draft_swing, 0.0]
        })

        role_breakdown = build_role_breakdown(draft_payload, self.player_stats, self.champ_stats, self.defaults)

        avg_blue_p_wr = np.mean([r['blue_p_wr'] for r in role_breakdown])
        avg_red_p_wr = np.mean([r['red_p_wr'] for r in role_breakdown])
        avg_blue_c_wr = np.mean([r['blue_c_wr'] for r in role_breakdown])
        avg_red_c_wr = np.mean([r['red_c_wr'] for r in role_breakdown])

        b_eg_stats = self.get_team_early_game(blue_team)
        r_eg_stats = self.get_team_early_game(red_team)
        b_strat_stats = self.get_team_strategic(blue_team)
        r_strat_stats = self.get_team_strategic(red_team)
        b_rp_stats = self.get_team_resource_playstyle(blue_team)
        r_rp_stats = self.get_team_resource_playstyle(red_team)
        b_vis_stats = self.get_team_vision(blue_team)
        r_vis_stats = self.get_team_vision(red_team)
        b_pa_stats = self.get_team_patch_adaptability(blue_team)
        r_pa_stats = self.get_team_patch_adaptability(red_team)

        current_patch = str(draft_payload.get('patch', ''))
        if not current_patch and 'patch' in self.df_hist.columns:
            current_patch = str(self.df_hist['patch'].dropna().iloc[-1])

        return {
            'blue_win_probability': proba_blue,
            'red_win_probability': proba_red,
            'blue_win_percentage': final_pct,
            'red_win_percentage': round(proba_red * 100, 2),
            'progression_data': progression_data,
            'draft_swings': {
                'player_swing': player_swing,
                'early_game_swing': early_game_swing,
                'draft_swing': draft_swing,
                'total_swing': round(final_pct - elo_pct, 2)
            },
            'series_metrics': {
                'game_number': draft_payload.get('game_number', 1),
                'blue_series_lead': draft_payload.get('blue_series_lead', 0),
                'blue_prev_win': draft_payload.get('blue_prev_win', 0)
            },
            'elo_metrics': {
                'blue_elo': round(b_elo, 1),
                'red_elo': round(r_elo, 1),
                'elo_diff': round(elo_diff, 1),
                'elo_implied_blue_winrate': elo_pct
            },
            'early_game_metrics': {
                'blue_golddiff15': round(b_eg_stats['golddiff15'], 1),
                'red_golddiff15': round(r_eg_stats['golddiff15'], 1),
                'golddiff15_diff': round(b_eg_stats['golddiff15'] - r_eg_stats['golddiff15'], 1),
                'blue_plate_ratio': round(b_eg_stats['plate_ratio'] * 100, 1),
                'red_plate_ratio': round(r_eg_stats['plate_ratio'] * 100, 1),
            },
            'strategic_metrics': {
                'blue_topside_share': round(b_strat_stats['topside_share'] * 100, 1),
                'red_topside_share': round(r_strat_stats['topside_share'] * 100, 1),
                'blue_topside_control': round(b_strat_stats['topside_control_rate'] * 100, 1),
                'red_topside_control': round(r_strat_stats['topside_control_rate'] * 100, 1),
                'blue_dragon_control': round(b_strat_stats['dragon_control_rate'] * 100, 1),
                'red_dragon_control': round(r_strat_stats['dragon_control_rate'] * 100, 1),
                'blue_jungle_aggression': round(b_strat_stats['jungle_aggression'] * 100, 1),
                'red_jungle_aggression': round(r_strat_stats['jungle_aggression'] * 100, 1),
            },
            'resource_playstyle_metrics': {
                'blue_gold_hhi': round(b_rp_stats['team_gold_hhi'], 4),
                'red_gold_hhi': round(r_rp_stats['team_gold_hhi'], 4),
                'blue_aggression_index': round(b_rp_stats['aggression_index'], 2),
                'red_aggression_index': round(r_rp_stats['aggression_index'], 2),
                'blue_early_orientation': round(b_rp_stats['early_game_orientation'], 2),
                'red_early_orientation': round(r_rp_stats['early_game_orientation'], 2),
                'blue_objective_priority': round(b_rp_stats['objective_priority_score'], 2),
                'red_objective_priority': round(r_rp_stats['objective_priority_score'], 2),
            },
            'vision_metrics': {
                'blue_vspm': round(b_vis_stats['vspm'], 2),
                'red_vspm': round(r_vis_stats['vspm'], 2),
                'vspm_diff': round(b_vis_stats['vspm'] - r_vis_stats['vspm'], 2),
                'blue_ward_clear_ratio': round(b_vis_stats['ward_clear_ratio'], 3),
                'red_ward_clear_ratio': round(r_vis_stats['ward_clear_ratio'], 3),
                'blue_cwpm': round(b_vis_stats['cwpm'], 2),
                'red_cwpm': round(r_vis_stats['cwpm'], 2),
                'blue_map_control_score': round(b_vis_stats['map_control_score'], 2),
                'red_map_control_score': round(r_vis_stats['map_control_score'], 2),
            },
            'patch_adaptability_metrics': {
                'patch': current_patch,
                'blue_patch_winrate': round(b_pa_stats['patch_winrate'] * 100, 1),
                'red_patch_winrate': round(r_pa_stats['patch_winrate'] * 100, 1),
                'blue_patch_wr_delta': round(b_pa_stats['patch_wr_delta'] * 100, 1),
                'red_patch_wr_delta': round(r_pa_stats['patch_wr_delta'] * 100, 1),
                'blue_champ_pool_depth': int(b_pa_stats['champ_pool_depth']),
                'red_champ_pool_depth': int(r_pa_stats['champ_pool_depth']),
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