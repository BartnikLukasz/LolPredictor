import copy
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

from bs4 import BeautifulSoup
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from live_feature_engine import LiveFeatureEngine
import xgboost as xgb
from upstash_redis import Redis

TRACKING_KEY = "live_accuracy_tracking"
ODDS_ENDPOINT_URL = "http://127.0.0.1:5000/odds"

MODEL_REGISTRY = {
    "XGBoost": "models/xgboost_model.json",
    "LightGBM": "models/lightgbm_model.pkl",
    "CatBoost": "models/catboost_model.pkl",
    "ElasticTree": "models/elastictree_model.pkl",
    "ElasticNet": "models/elasticnet_model.joblib",
}


def check_is_admin() -> bool:
    """Verifies owner/admin authorization via query params or session state."""
    try:
        admin_secret = st.secrets.get("ADMIN_KEY", "")
    except Exception:
        admin_secret = ""
    if not admin_secret:
        return False
    if st.query_params.get("admin") == admin_secret or st.session_state.get("is_admin", False):
        return True
    return False


@st.cache_resource
def get_redis_client():
    """Returns persistent Redis client instance."""
    return Redis(
        url=st.secrets["UPSTASH_REDIS_REST_URL"],
        token=st.secrets["UPSTASH_REDIS_REST_TOKEN"]
    )


def load_tracking_data() -> dict:
    """Fetches accuracy tracking metrics from Redis."""
    redis = get_redis_client()
    raw_data = redis.get(TRACKING_KEY)
    if not raw_data:
        return {"total_games": 0, "correct_predictions": 0, "logs": []}
    if isinstance(raw_data, str):
        return json.loads(raw_data)
    return raw_data


def save_tracking_data(data: dict):
    """Saves updated accuracy tracking metrics to Redis."""
    redis = get_redis_client()
    redis.set(TRACKING_KEY, json.dumps(data))


@st.cache_resource
def load_predictor_assets():
    """Loads feature engines, roster data, champion list, and historical dataset."""
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"
    base_engine = LiveFeatureEngine(dataset_path=dataset_path)
    engines = {}

    for model_name, model_path in MODEL_REGISTRY.items():
        if os.path.exists(model_path):
            eng = copy.deepcopy(base_engine)
            if model_path.endswith(".json"):
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                eng.model = model
            else:
                artifact = joblib.load(model_path)
                if isinstance(artifact, dict):
                    eng.model = artifact.get("pipeline", artifact.get("model", artifact))
                else:
                    eng.model = artifact
            engines[model_name] = eng
        else:
            engines[model_name] = base_engine

    with open("models/team_rosters.json", "r") as f:
        roster_data = json.load(f)

    champ_cols = [c for c in base_engine.df_hist.columns if 'champion' in c]
    champions_set = set()
    for col in champ_cols:
        champions_set.update(base_engine.df_hist[col].dropna().unique().tolist())

    champions_list = sorted(list(champions_set)) if champions_set else ["Ahri", "Aatrox", "Azir"]
    return engines, roster_data, champions_list, base_engine.df_hist


def send_odds_to_endpoint(blue_team: str, red_team: str, p_blue: float, p_red: float):
    """Posts calculated fair odds to local monitoring endpoint."""
    payload = {
        "odds": {
            blue_team: round(1.0 / p_blue, 2) if p_blue > 0 else 0,
            red_team: round(1.0 / p_red, 2) if p_red > 0 else 0
        },
        "model_probs": {
            blue_team: round(p_blue, 4),
            red_team: round(p_red, 4)
        }
    }
    try:
        requests.post(ODDS_ENDPOINT_URL, json=payload, timeout=2)
        st.toast("Dispatched odds to prediction monitor!", icon="📡")
    except Exception:
        st.toast(f"Could not reach endpoint ({ODDS_ENDPOINT_URL})", icon="⚠️")


# --- TEAM ROSTER CALLBACKS ---
def update_blue_roster_callback(team_rosters: dict):
    selected_team = st.session_state.get("blue_team_select")
    roster = team_rosters.get(selected_team, ["", "", "", "", ""])
    for i in range(5):
        st.session_state[f"bp_{i}"] = roster[i] if i < len(roster) else ""


def update_red_roster_callback(team_rosters: dict):
    selected_team = st.session_state.get("red_team_select")
    roster = team_rosters.get(selected_team, ["", "", "", "", ""])
    for i in range(5):
        st.session_state[f"rp_{i}"] = roster[i] if i < len(roster) else ""


def swap_sides_callback():
    temp_blue = st.session_state.get("blue_team_select")
    temp_red = st.session_state.get("red_team_select")
    st.session_state["blue_team_select"] = temp_red
    st.session_state["red_team_select"] = temp_blue

    for i in range(5):
        bc_key, rc_key = f"bc_{i}", f"rc_{i}"
        bp_key, rp_key = f"bp_{i}", f"rp_{i}"
        if bc_key in st.session_state and rc_key in st.session_state:
            st.session_state[bc_key], st.session_state[rc_key] = st.session_state[rc_key], st.session_state[bc_key]
        if bp_key in st.session_state and rp_key in st.session_state:
            st.session_state[bp_key], st.session_state[rp_key] = st.session_state[rp_key], st.session_state[bp_key]


def fetch_golgg_draft(url: str) -> dict:
    """Scrapes match draft, teams, and player details directly from a gol.gg game URL."""
    logs = []
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        logs.append(f"HTTP Status: {response.status_code}")
    except Exception as e:
        raise ConnectionError(f"Connection failed: {e}\nLogs:\n" + "\n".join(logs))

    if response.status_code != 200:
        raise ConnectionError(f"HTTP {response.status_code}: Page unavailable.\nLogs:\n" + "\n".join(logs))

    soup = BeautifulSoup(response.text, 'html.parser')

    blue_team_elem = soup.select_one('.blue-line-header a')
    red_team_elem = soup.select_one('.red-line-header a')

    blue_team = blue_team_elem.get_text(strip=True) if blue_team_elem else ""
    red_team = red_team_elem.get_text(strip=True) if red_team_elem else ""

    logs.append(f"Teams Extracted -> Blue: '{blue_team}', Red: '{red_team}'")

    first_pick = "Blue"
    first_pick_img = (
        soup.find('img', src=re.compile(r'first\.png', re.IGNORECASE)) or
        soup.find('img', alt=re.compile(r'first pick', re.IGNORECASE))
    )

    if first_pick_img:
        curr = first_pick_img.parent
        found_side = None
        while curr and curr.name != '[document]':
            classes = " ".join(curr.get('class', [])).lower()
            if 'red' in classes:
                found_side = "Red"
                break
            elif 'blue' in classes:
                found_side = "Blue"
                break
            curr = curr.parent

        if found_side:
            first_pick = found_side
        else:
            raw_html = str(soup)
            img_pos = raw_html.find('first.png')
            red_hdr_pos = raw_html.find('red-line-header')
            if img_pos != -1 and red_hdr_pos != -1 and img_pos > red_hdr_pos:
                first_pick = "Red"

    logs.append(f"First Pick: {first_pick}")

    blue_champs, red_champs = [], []
    blue_players, red_players = [], []

    tables = soup.select('table.playersInfosLine')
    logs.append(f"Player Info Tables Found: {len(tables)}")

    for idx, tbl in enumerate(tables):
        is_blue = bool(tbl.select_one('.blue-line-header')) or (idx == 0)
        is_red = bool(tbl.select_one('.red-line-header')) or (idx == 1 and not is_blue)

        champ_links = tbl.select('a[href*="/champion/"]')
        for champ_link in champ_links:
            champ_img = champ_link.find('img')
            champ_name = ""
            if champ_img and champ_img.get('alt'):
                champ_name = champ_img['alt'].strip()
            elif champ_link.get('title'):
                champ_name = champ_link['title'].replace(' stats', '').strip()

            parent_td = champ_link.find_parent('td')
            player_link = parent_td.select_one('a.link-blanc') if parent_td else None
            player_name = player_link.get_text(strip=True) if player_link else ""

            if champ_name:
                if is_blue:
                    blue_champs.append(champ_name)
                    if player_name:
                        blue_players.append(player_name)
                elif is_red:
                    red_champs.append(champ_name)
                    if player_name:
                        red_players.append(player_name)

    logs.append(f"Blue Side -> Champs: {blue_champs} | Players: {blue_players}")
    logs.append(f"Red Side  -> Champs: {red_champs} | Players: {red_players}")

    if len(blue_champs) < 5 or len(red_champs) < 5:
        raise ValueError(
            f"Draft extraction incomplete (Found Blue: {len(blue_champs)}, Red: {len(red_champs)}).\n"
            f"--- DEBUG LOGS ---\n" + "\n".join(logs)
        )

    return {
        "blue_team": blue_team,
        "red_team": red_team,
        "first_pick": first_pick,
        "blue_champs": blue_champs[:5],
        "red_champs": red_champs[:5],
        "blue_players": blue_players[:5],
        "red_players": red_players[:5],
        "debug_logs": logs
    }


def match_team_name(
    scraped_name: str,
    valid_teams: list[str],
    fetched_players: list[str] = None,
    team_rosters: dict = None,
    df_hist: pd.DataFrame = None
) -> str:
    """Matches scraped team string to valid dataset team entities using roster overlap and fuzzy metrics."""
    if not scraped_name or not valid_teams:
        return valid_teams[0] if valid_teams else ""

    scraped_clean = scraped_name.strip().lower()

    for team in valid_teams:
        if scraped_clean == team.strip().lower():
            return team

    if fetched_players and team_rosters:
        scraped_players_set = {p.strip().lower() for p in fetched_players if p and p.strip()}
        if scraped_players_set:
            best_roster_match = None
            max_overlap = 0

            for team_name in valid_teams:
                known_roster = {p.strip().lower() for p in team_rosters.get(team_name, []) if p}
                overlap = len(scraped_players_set.intersection(known_roster))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_roster_match = team_name

            if best_roster_match and max_overlap >= 2:
                return best_roster_match

    candidate_scores = []
    for team in valid_teams:
        sim_score = SequenceMatcher(None, scraped_clean, team.lower()).ratio()

        latest_date = pd.Timestamp.min
        if df_hist is not None and ('date' in df_hist.columns or 'date_utc' in df_hist.columns):
            date_col = 'date' if 'date' in df_hist.columns else 'date_utc'
            team_matches = df_hist[(df_hist['team_blue'] == team) | (df_hist['team_red'] == team)]
            if not team_matches.empty:
                latest_date = pd.to_datetime(team_matches[date_col]).max()

        candidate_scores.append((team, sim_score, latest_date))

    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidate_scores[0][0]


def match_champion_name(scraped_name: str, valid_champions: list) -> str:
    """Fuzzy matches champion names against valid dataset list."""
    if not scraped_name or not valid_champions:
        return valid_champions[0] if valid_champions else ""
    scraped_clean = re.sub(r'[^a-zA-Z0-9]', '', scraped_name).lower()
    for champ in valid_champions:
        champ_clean = re.sub(r'[^a-zA-Z0-9]', '', champ).lower()
        if scraped_clean == champ_clean:
            return champ
    return valid_champions[0]


def compute_model_accuracies(tracking_data: dict, min_confidence_pct: float = 50.0) -> pd.DataFrame:
    """Calculates accuracy breakdown per model using live tracking log history."""
    model_stats = {}
    threshold = min_confidence_pct / 100.0

    for log in tracking_data.get("logs", []):
        if "models" in log and isinstance(log["models"], list):
            for m in log["models"]:
                name = m.get("model_used")
                if not name:
                    continue
                p_blue = m.get("blue_win_probability", 0.5)
                p_red = m.get("red_win_probability", 0.5)
                if max(p_blue, p_red) < threshold:
                    continue
                if name not in model_stats:
                    model_stats[name] = {"correct": 0, "total": 0}
                model_stats[name]["total"] += 1
                if m.get("is_correct"):
                    model_stats[name]["correct"] += 1

    rows = [
        {
            "Model": k,
            "Accuracy (%)": round((v["correct"] / v["total"] * 100), 1) if v["total"] > 0 else 0.0,
            "Correct": v["correct"],
            "Total": v["total"]
        }
        for k, v in model_stats.items()
    ]
    df_acc = pd.DataFrame(rows)
    if not df_acc.empty:
        df_acc = df_acc.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
    return df_acc


def get_historical_team_metrics(df_hist: pd.DataFrame, blue_team: str, red_team: str) -> dict:
    """Queries head-to-head records and recent team form metrics."""
    h2h_matches = df_hist[
        ((df_hist['blue_team'] == blue_team) & (df_hist['red_team'] == red_team)) |
        ((df_hist['blue_team'] == red_team) & (df_hist['red_team'] == blue_team))
    ].sort_values('date', ascending=False)

    total_h2h = len(h2h_matches)
    blue_h2h_wins = 0
    if total_h2h > 0:
        for _, row in h2h_matches.iterrows():
            if (row['blue_team'] == blue_team and row['blue_win'] == 1) or (row['red_team'] == blue_team and row['blue_win'] == 0):
                blue_h2h_wins += 1

    blue_matches = df_hist[(df_hist['blue_team'] == blue_team) | (df_hist['red_team'] == blue_team)].sort_values('date', ascending=False).head(10)
    red_matches = df_hist[(df_hist['blue_team'] == red_team) | (df_hist['red_team'] == red_team)].sort_values('date', ascending=False).head(10)

    blue_recent_wins = sum((row['blue_win'] == 1 if row['blue_team'] == blue_team else row['blue_win'] == 0) for _, row in blue_matches.iterrows())
    red_recent_wins = sum((row['blue_win'] == 1 if row['red_team'] == red_team else row['blue_win'] == 0) for _, row in red_matches.iterrows())

    return {
        'total_h2h': total_h2h,
        'blue_h2h_wins': blue_h2h_wins,
        'red_h2h_wins': total_h2h - blue_h2h_wins,
        'blue_h2h_wr': round((blue_h2h_wins / total_h2h * 100), 1) if total_h2h > 0 else 50.0,
        'blue_recent_wr': round((blue_recent_wins / max(len(blue_matches), 1) * 100), 1),
        'red_recent_wr': round((red_recent_wins / max(len(red_matches), 1) * 100), 1)
    }


def prob_to_american_odds(prob: float) -> str:
    """Converts win probability into standard American Odds format string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    return f"{int(round(-100 * prob / (1 - prob)))}" if prob >= 0.5 else f"+{int(round(100 * (1 - prob) / prob))}"


def create_ensemble_result(model_results_dict: dict) -> dict:
    """Calculates an unweighted average ensemble result."""
    single_models = [res for key, res in model_results_dict.items() if key != "Even Split"]
    avg_blue_prob = sum(res['blue_win_probability'] for res in single_models) / len(single_models)
    ensemble_res = copy.deepcopy(single_models[0])
    ensemble_res['blue_win_probability'] = round(avg_blue_prob, 4)
    ensemble_res['red_win_probability'] = round(1.0 - avg_blue_prob, 4)
    ensemble_res['blue_win_percentage'] = round(avg_blue_prob * 100, 1)
    ensemble_res['red_win_percentage'] = round((1.0 - avg_blue_prob) * 100, 1)
    return ensemble_res


def compute_db_model_weights(tracking_data: dict, model_names: list) -> tuple[dict, dict]:
    """Calculates model weights dynamically based on recorded database accuracy."""
    stats = {m: {"total": 0, "correct": 0} for m in model_names}
    logs = tracking_data.get("logs", [])

    for entry in logs:
        for m_log in entry.get("models", []):
            m_name = m_log.get("model_used")
            if m_name in stats:
                stats[m_name]["total"] += 1
                if m_log.get("is_correct", False):
                    stats[m_name]["correct"] += 1

    accuracies = {}
    for m in model_names:
        tot = stats[m]["total"]
        accuracies[m] = (stats[m]["correct"] / tot) if tot > 0 else 0.50

    T = 0.1
    exp_acc = {m: np.exp(acc / T) for m, acc in accuracies.items()}
    tot_exp = sum(exp_acc.values())
    weights = {m: exp_val / tot_exp for m, exp_val in exp_acc.items()}

    return weights, accuracies


def create_weighted_ensemble_result(all_model_results: dict, model_weights: dict) -> dict:
    """Combines model predictions using normalized historical accuracy weights."""
    base_results = {k: v for k, v in all_model_results.items() if k in model_weights}
    if not base_results:
        return next(iter(all_model_results.values()))

    w_sum = sum(model_weights[k] for k in base_results.keys())
    norm_weights = {k: (model_weights[k] / w_sum if w_sum > 0 else 1.0 / len(base_results)) for k in base_results.keys()}

    w_p_blue = sum(norm_weights[m] * base_results[m]['blue_win_probability'] for m in base_results)
    w_p_red = 1.0 - w_p_blue

    w_player_swing = sum(norm_weights[m] * base_results[m].get('draft_swings', {}).get('player_swing', 0.0) for m in base_results)
    w_eg_swing = sum(norm_weights[m] * base_results[m].get('draft_swings', {}).get('early_game_swing', 0.0) for m in base_results)
    w_draft_swing = sum(norm_weights[m] * base_results[m].get('draft_swings', {}).get('draft_swing', 0.0) for m in base_results)

    first_res = next(iter(base_results.values()))
    elo_base = first_res.get('elo_metrics', {}).get('elo_implied_blue_winrate', 50.0)

    final_pct = round(w_p_blue * 100, 2)
    player_pct = round(elo_base + w_player_swing, 2)

    progression_data = pd.DataFrame({
        "Stage": ["1. Elo Baseline", "2. Player Mastery Impact", "3. Early Macro Impact", "4. Champion Draft Impact", "5. Final Prediction"],
        "Win %": [elo_base, player_pct, round(player_pct + w_eg_swing, 2), final_pct, final_pct],
        "Impact Delta": [0.0, round(w_player_swing, 2), round(w_eg_swing, 2), round(w_draft_swing, 2), 0.0]
    })

    res = copy.deepcopy(first_res)
    res['blue_win_probability'] = w_p_blue
    res['red_win_probability'] = w_p_red
    res['blue_win_percentage'] = final_pct
    res['red_win_percentage'] = round(w_p_red * 100, 2)
    res['progression_data'] = progression_data
    res['draft_swings'] = {
        'player_swing': round(w_player_swing, 2),
        'early_game_swing': round(w_eg_swing, 2),
        'draft_swing': round(w_draft_swing, 2),
        'total_swing': round(final_pct - elo_base, 2)
    }
    res['weights_used'] = norm_weights
    return res

def _sync_shared_state(source_key: str, target_key: str):
    """Synchronizes widget selection across model tabs."""
    if source_key in st.session_state:
        st.session_state[target_key] = st.session_state[source_key]


def render_model_dashboard(model_name: str, results: dict, active_pred: dict, h2h_data: dict, is_admin: bool):
    """Renders win metrics, model logger, step-by-step feature breakdowns, and role analytics."""
    b_team, r_team = active_pred['blue_team'], active_pred['red_team']
    predicted_winner = b_team if results['blue_win_probability'] >= 0.5 else r_team

    analysis_options = [
        "⚡ Step 1: Elo & Series",
        "⏳ Step 1: Early Game & Gold",
        "🗺️ Step 2: Strategic Priority",
        "💰 Step 3: Resource & Playstyle",
        "👁️ Step 4: Vision & Map Control",
        "🔄 Step 5: Patch Adaptability",
        "👤 Player Mastery",
        "⚔️ Draft Impact",
        "🛡️ Team H2H",
        "🎲 Value Odds"
    ]

    # Initialize shared states if not present or if teams changed
    if "shared_actual_winner" not in st.session_state or st.session_state["shared_actual_winner"] not in [b_team, r_team]:
        st.session_state["shared_actual_winner"] = b_team

    if "shared_analysis_tab" not in st.session_state or st.session_state["shared_analysis_tab"] not in analysis_options:
        st.session_state["shared_analysis_tab"] = analysis_options[0]

    res_b, res_r = st.columns(2)
    res_b.metric(f"{b_team} Win Probability", f"{results['blue_win_percentage']}%")
    res_r.metric(f"{r_team} Win Probability", f"{results['red_win_percentage']}%")
    st.progress(results['blue_win_probability'])

    if model_name == "Weighted Split" and "weights_used" in results:
        with st.expander("⚖️ DB-Weighted Model Breakdown", expanded=False):
            w_cols = st.columns(len(results["weights_used"]))
            for idx, (m_k, w_v) in enumerate(results["weights_used"].items()):
                acc_v = active_pred.get("model_accuracies", {}).get(m_k, 0.5) * 100
                w_cols[idx].metric(m_k, f"{w_v * 100:.1f}% Weight", help=f"Historical DB Accuracy: {acc_v:.1f}%")

    with st.container(border=True):
        st.subheader("📝 Record Live Game Result")
        st.write(f"Model ({model_name}) Predicted Winner: **{predicted_winner}**")

        act_col1, act_col2 = st.columns([3, 1])
        with act_col1:
            radio_key = f"selected_actual_winner_{model_name}"
            st.session_state[radio_key] = st.session_state["shared_actual_winner"]

            actual_winner = st.radio(
                "Select Actual Game Winner:",
                options=[b_team, r_team],
                horizontal=True,
                key=radio_key,
                on_change=_sync_shared_state,
                args=(radio_key, "shared_actual_winner")
            )

        with act_col2:
            st.write("")
            if st.button("Save & Log Result", type="primary" if is_admin else "secondary", disabled=not is_admin, key=f"save_btn_{model_name}"):
                current_track_data = load_tracking_data()
                models_log_list = []
                xgb_is_correct = False

                for m_name, m_res in active_pred['model_results'].items():
                    m_pred = b_team if m_res['blue_win_probability'] >= 0.5 else r_team
                    m_corr = (actual_winner == m_pred)
                    if m_name == "XGBoost":
                        xgb_is_correct = m_corr

                    models_log_list.append({
                        "model_used": m_name,
                        "blue_win_probability": m_res['blue_win_probability'],
                        "red_win_probability": m_res['red_win_probability'],
                        "predicted_winner": m_pred,
                        "actual_winner": actual_winner,
                        "is_correct": m_corr
                    })

                current_track_data["total_games"] = current_track_data.get("total_games", 0) + 1
                if xgb_is_correct:
                    current_track_data["correct_predictions"] = current_track_data.get("correct_predictions", 0) + 1

                current_track_data.setdefault("logs", []).append({
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blue_team": b_team,
                    "red_team": r_team,
                    "models": models_log_list
                })
                save_tracking_data(current_track_data)
                st.toast("Result Logged!", icon="🎯")
                del st.session_state["active_prediction"]
                st.rerun()

    st.markdown(f"### 📊 Comprehensive Match Analysis ({model_name})")

    nav_key = f"active_analysis_tab_{model_name}"
    st.session_state[nav_key] = st.session_state["shared_analysis_tab"]

    selected_analysis_tab = st.radio(
        "Analysis View Navigation",
        options=analysis_options,
        horizontal=True,
        key=nav_key,
        label_visibility="collapsed",
        on_change=_sync_shared_state,
        args=(nav_key, "shared_analysis_tab")
    )

    # --- STEP 1: ELO & SERIES CONTEXT ---
    if selected_analysis_tab == "⚡ Step 1: Elo & Series":
        e1, e2, e3, e4 = st.columns(4)
        elo_m = results.get('elo_metrics', {})
        s_m = results.get('series_metrics', {})
        e1.metric(f"{b_team} Elo", f"{elo_m.get('blue_elo', 1500)}")
        e2.metric(f"{r_team} Elo", f"{elo_m.get('red_elo', 1500)}")
        e3.metric("Elo Implied Winrate", f"{elo_m.get('elo_implied_blue_winrate', 50.0)}%")
        e4.metric("Series Status", f"Game {s_m.get('game_number', 1)} (Lead: {s_m.get('blue_series_lead', 0):+d})")

    # --- STEP 1: EARLY GAME MACRO ---
    elif selected_analysis_tab == "⏳ Step 1: Early Game & Gold":
        eg_m = results.get('early_game_metrics', {})

        b_gd15 = int(eg_m.get('blue_golddiff15', 0))
        r_gd15 = int(eg_m.get('red_golddiff15', 0))
        gd15_diff = int(eg_m.get('golddiff15_diff', 0))

        col1, col2, col3 = st.columns(3)
        col1.metric(f"{b_team} GD@15 Avg", f"{b_gd15:+d}")
        col2.metric(f"{r_team} GD@15 Avg", f"{r_gd15:+d}")
        col3.metric("GD@15 Advantage Delta", f"{gd15_diff:+d}")

        col4, col5 = st.columns(2)
        col4.metric(f"{b_team} Turret Plate Ratio", f"{eg_m.get('blue_plate_ratio', 0.5) * 100:.1f}%")
        col5.metric(f"{r_team} Turret Plate Ratio", f"{eg_m.get('red_plate_ratio', 0.5) * 100:.1f}%")

    # --- STEP 2: STRATEGIC OBJECTIVES ---
    elif selected_analysis_tab == "🗺️ Step 2: Strategic Priority":
        st_m = results.get('strategic_metrics', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{b_team} Topside Share", f"{st_m.get('blue_topside_share', 0.5)*100:.1f}%")
        c2.metric(f"{r_team} Topside Share", f"{st_m.get('red_topside_share', 0.5)*100:.1f}%")
        c3.metric(f"{b_team} Dragon Control", f"{st_m.get('blue_dragon_control', 0.5)*100:.1f}%")
        c4.metric(f"{r_team} Dragon Control", f"{st_m.get('red_dragon_control', 0.5)*100:.1f}%")

        c5, c6 = st.columns(2)
        c5.metric(f"{b_team} Jungle Aggression Rate", f"{st_m.get('blue_jungle_aggression', 0.0):.2f}")
        c6.metric(f"{r_team} Jungle Aggression Rate", f"{st_m.get('red_jungle_aggression', 0.0):.2f}")

    # --- STEP 3: RESOURCE ALLOCATION & PLAYSTYLE ---
    elif selected_analysis_tab == "💰 Step 3: Resource & Playstyle":
        rp_m = results.get('resource_playstyle_metrics', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{b_team} Gold HHI", f"{rp_m.get('blue_gold_hhi', 0.2):.3f}", help="Gold Concentration Index")
        c2.metric(f"{r_team} Gold HHI", f"{rp_m.get('red_gold_hhi', 0.2):.3f}", help="Gold Concentration Index")
        c3.metric(f"{b_team} Aggression Index", f"{rp_m.get('blue_aggression_index', 1.0):.2f}")
        c4.metric(f"{r_team} Aggression Index", f"{rp_m.get('red_aggression_index', 1.0):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric(f"{b_team} Early Orientation", f"{rp_m.get('blue_early_orientation', 0.5)*100:.1f}%")
        c6.metric(f"{r_team} Early Orientation", f"{rp_m.get('red_early_orientation', 0.5)*100:.1f}%")
        c7.metric(f"{b_team} Obj Priority Score", f"{rp_m.get('blue_objective_priority', 0.5)*100:.1f}%")
        c8.metric(f"{r_team} Obj Priority Score", f"{rp_m.get('red_objective_priority', 0.5)*100:.1f}%")

    # --- STEP 4: VISION & MAP CONTROL ---
    elif selected_analysis_tab == "👁️ Step 4: Vision & Map Control":
        vis_m = results.get('vision_metrics', {})
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{b_team} VSPM", f"{vis_m.get('blue_vspm', 0.0):.2f}")
        c2.metric(f"{r_team} VSPM", f"{vis_m.get('red_vspm', 0.0):.2f}")
        c3.metric("VSPM Difference", f"{vis_m.get('vspm_diff', 0.0):+.2f}")

        c4, c5, c6, c7 = st.columns(4)
        c4.metric(f"{b_team} Ward Clear Ratio", f"{vis_m.get('blue_ward_clear_ratio', 0.4)*100:.1f}%")
        c5.metric(f"{r_team} Ward Clear Ratio", f"{vis_m.get('red_ward_clear_ratio', 0.4)*100:.1f}%")
        c6.metric(f"{b_team} Map Control Score", f"{vis_m.get('blue_map_control_score', 50.0):.1f}")
        c7.metric(f"{r_team} Map Control Score", f"{vis_m.get('red_map_control_score', 50.0):.1f}")

    # --- STEP 5: PATCH & META ADAPTABILITY ---
    elif selected_analysis_tab == "🔄 Step 5: Patch Adaptability":
        pa_m = results.get('patch_adaptability_metrics', {})
        st.write(f"**Tournament Patch Version:** `{pa_m.get('patch', '14.1')}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{b_team} Patch WR", f"{pa_m.get('blue_patch_winrate', 0.5)*100:.1f}%")
        c2.metric(f"{r_team} Patch WR", f"{pa_m.get('red_patch_winrate', 0.5)*100:.1f}%")
        c3.metric(f"{b_team} Patch WR Delta", f"{pa_m.get('blue_patch_wr_delta', 0.0):+.1f}%")
        c4.metric(f"{r_team} Patch WR Delta", f"{pa_m.get('red_patch_wr_delta', 0.0):+.1f}%")

        c5, c6 = st.columns(2)
        c5.metric(f"{b_team} Champion Pool Depth", f"{pa_m.get('blue_champ_pool_depth', 0)} champs")
        c6.metric(f"{r_team} Champion Pool Depth", f"{pa_m.get('red_champ_pool_depth', 0)} champs")

    # --- PLAYER MASTERY ---
    elif selected_analysis_tab == "👤 Player Mastery":
        player_rows = []
        for r in results.get('role_breakdown', []):
            b_p_wr = r.get('blue_p_wr', 0.5) * 100
            r_p_wr = r.get('red_p_wr', 0.5) * 100
            player_rows.append({
                "Role": r['role'],
                f"{b_team} Player": r['blue_player'],
                "Blue WR": f"{b_p_wr:.1f}%",
                f"{r_team} Player": r['red_player'],
                "Red WR": f"{r_p_wr:.1f}%",
                "Mastery Swing": f"{(b_p_wr - r_p_wr):+.1f}%"
            })
        st.dataframe(pd.DataFrame(player_rows), use_container_width=True, hide_index=True)

    # --- DRAFT IMPACT & WATERFALL ---
    elif selected_analysis_tab == "⚔️ Draft Impact":
        swings = results.get("draft_swings", {})
        role_data = results.get("role_breakdown", [])

        # Extract baseline Elo
        elo_base = results.get('elo_metrics', {}).get('elo_implied_blue_winrate', 50.0)

        # Step 1 to Step 5 Impact Swings
        s1_early = swings.get('early_game_swing', 0.0)  # Step 1: Early Gold/XP Delta
        s2_strat = swings.get('strategic_swing', 0.0)  # Step 2: Objectives/Grubs/Dragons
        s3_resource = swings.get('resource_swing', 0.0)  # Step 3: Aggression & Gold HHI
        s4_vision = swings.get('vision_swing', 0.0)  # Step 4: Map Control & VSPM
        s5_patch = swings.get('patch_swing', 0.0)  # Step 5: Patch/Meta Adaptability
        player_swing = swings.get('player_swing', 0.0)  # Player Mastery
        draft_swing = swings.get('draft_swing', 0.0)  # Champion Pick/Draft Impact

        final_pct = results.get('blue_win_percentage', round(
            elo_base + s1_early + s2_strat + s3_resource + s4_vision + s5_patch + player_swing + draft_swing, 2
        ))

        st.markdown(f"#### 📈 7-Stage Prediction Waterfall ({b_team})")

        fig = go.Figure(go.Waterfall(
            name="Winrate Swing",
            orientation="v",
            measure=[
                "absolute",  # Baseline
                "relative",  # Step 1
                "relative",  # Step 2
                "relative",  # Step 3
                "relative",  # Step 4
                "relative",  # Step 5
                "relative",  # Player
                "relative",  # Draft
                "total"  # Final Prediction
            ],
            x=[
                "Elo Baseline",
                "1. Early Macro",
                "2. Strategic Priority",
                "3. Playstyle/Resource",
                "4. Vision Control",
                "5. Patch Adaptability",
                "Player Mastery",
                "Draft Impact",
                "Final Prediction"
            ],
            textposition="outside",
            text=[
                f"{elo_base:.1f}%",
                f"{s1_early:+.2f}%",
                f"{s2_strat:+.2f}%",
                f"{s3_resource:+.2f}%",
                f"{s4_vision:+.2f}%",
                f"{s5_patch:+.2f}%",
                f"{player_swing:+.2f}%",
                f"{draft_swing:+.2f}%",
                f"{final_pct:.1f}%"
            ],
            y=[elo_base, s1_early, s2_strat, s3_resource, s4_vision, s5_patch, player_swing, draft_swing, 0],
            connector={"line": {"color": "#888", "width": 1.5}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}}
        ))

        fig.update_layout(
            yaxis_title=f"{b_team} Win Probability (%)",
            yaxis=dict(range=[0, max(100, final_pct + 15)]),
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🎯 Role-by-Role Draft Swing Breakdown")
        champ_rows = []
        for r in role_data:
            b_c_wr = r.get('blue_c_wr', 0.5) * 100
            r_c_wr = r.get('red_c_wr', 0.5) * 100
            champ_rows.append({
                "Role": r['role'],
                f"{b_team} Pick": r['blue_champ'],
                "Blue Champ WR": f"{b_c_wr:.1f}%",
                f"{r_team} Pick": r['red_champ'],
                "Red Champ WR": f"{r_c_wr:.1f}%",
                "Role Impact Swing": f"{(b_c_wr - r_c_wr):+.1f}%"
            })
        st.dataframe(pd.DataFrame(champ_rows), use_container_width=True, hide_index=True)

    # --- TEAM H2H ---
    elif selected_analysis_tab == "🛡️ Team H2H":
        st.info(f"Historical Matchups: {h2h_data['total_h2h']} | {b_team} H2H Winrate: {h2h_data['blue_h2h_wr']}%")

    # --- VALUE ODDS ---
    elif selected_analysis_tab == "🎲 Value Odds":
        p_b, p_r = results['blue_win_probability'], results['red_win_probability']
        st.write(f"**{b_team} Fair Decimal:** {round(1.0/p_b, 2) if p_b > 0 else 0} ({prob_to_american_odds(p_b)})")
        st.write(f"**{r_team} Fair Decimal:** {round(1.0/p_r, 2) if p_r > 0 else 0} ({prob_to_american_odds(p_r)})")