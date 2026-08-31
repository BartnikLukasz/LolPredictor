import copy
import json
import os
import re
from datetime import datetime
import xgboost as xgb
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
from live_feature_engine import LiveFeatureEngine
from upstash_redis import Redis

st.set_page_config(page_title="LoL Match Predictor", layout="wide")

ODDS_ENDPOINT_URL = "http://127.0.0.1:5000/odds"
TRACKING_KEY = "live_accuracy_tracking"

MODEL_REGISTRY = {
    "XGBoost": "models/xgboost_model.json",
    "LightGBM": "models/lightgbm_model.pkl",
    "CatBoost": "models/catboost_model.pkl",
    "ElasticTree": "models/elastictree_model.pkl",
    "ElasticNet": "models/elasticnet_model.joblib",
}


# --- GOL.GG WEB SCRAPER ENGINE ---
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

    # 1. Extract Team Names
    blue_team_elem = soup.select_one('.blue-line-header a')
    red_team_elem = soup.select_one('.red-line-header a')

    blue_team = blue_team_elem.get_text(strip=True) if blue_team_elem else ""
    red_team = red_team_elem.get_text(strip=True) if red_team_elem else ""

    logs.append(f"Teams Extracted -> Blue: '{blue_team}', Red: '{red_team}'")

    # 2. Extract First Pick Side
    first_pick = "Blue"  # Default fallback
    first_pick_img = (
            soup.find('img', src=re.compile(r'first\.png', re.IGNORECASE)) or
            soup.find('img', alt=re.compile(r'first pick', re.IGNORECASE))
    )

    if first_pick_img:
        # Check parent container tree for side keywords
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
            # Fallback: check DOM position relative to Red Header tag
            raw_html = str(soup)
            img_pos = raw_html.find('first.png')
            red_hdr_pos = raw_html.find('red-line-header')
            if img_pos != -1 and red_hdr_pos != -1 and img_pos > red_hdr_pos:
                first_pick = "Red"

    logs.append(f"First Pick: {first_pick}")

    # 3. Extract Player Names & Champion Picks
    blue_champs, red_champs = [], []
    blue_players, red_players = [], []

    tables = soup.select('table.playersInfosLine')
    logs.append(f"Player Info Tables Found: {len(tables)}")

    for idx, tbl in enumerate(tables):
        # Determine team side by header class or table index position (0=Blue, 1=Red)
        is_blue = bool(tbl.select_one('.blue-line-header')) or (idx == 0)
        is_red = bool(tbl.select_one('.red-line-header')) or (idx == 1 and not is_blue)

        # Target champion links directly inside the table cells
        champ_links = tbl.select('a[href*="/champion/"]')
        for champ_link in champ_links:
            # Extract Champion Name
            champ_img = champ_link.find('img')
            champ_name = ""
            if champ_img and champ_img.get('alt'):
                champ_name = champ_img['alt'].strip()
            elif champ_link.get('title'):
                champ_name = champ_link['title'].replace(' stats', '').strip()

            # Extract Player Name from the same cell
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

    # Validation Guard
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

def match_team_name(scraped_name: str, valid_teams: list) -> str:
    if not scraped_name or not valid_teams:
        return valid_teams[0] if valid_teams else ""
    scraped_clean = scraped_name.lower().strip()
    for team in valid_teams:
        if team.lower().strip() == scraped_clean or scraped_clean in team.lower() or team.lower() in scraped_clean:
            return team
    return valid_teams[0]


def match_champion_name(scraped_name: str, valid_champions: list) -> str:
    if not scraped_name or not valid_champions:
        return valid_champions[0] if valid_champions else ""
    scraped_clean = re.sub(r'[^a-zA-Z0-9]', '', scraped_name).lower()
    for champ in valid_champions:
        champ_clean = re.sub(r'[^a-zA-Z0-9]', '', champ).lower()
        if scraped_clean == champ_clean:
            return champ
    return valid_champions[0]


def check_is_admin() -> bool:
    try:
        admin_secret = st.secrets.get("ADMIN_KEY", "")
    except Exception:
        admin_secret = ""
    if not admin_secret:
        return False
    if st.query_params.get("admin") == admin_secret or st.session_state.get("is_admin", False):
        return True
    return False


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


@st.cache_resource
def get_redis_client():
    return Redis(
        url=st.secrets["UPSTASH_REDIS_REST_URL"],
        token=st.secrets["UPSTASH_REDIS_REST_TOKEN"]
    )


redis = get_redis_client()


def load_tracking_data() -> dict:
    raw_data = redis.get(TRACKING_KEY)
    if not raw_data:
        return {"total_games": 0, "correct_predictions": 0, "logs": []}
    if isinstance(raw_data, str):
        return json.loads(raw_data)
    return raw_data


def save_tracking_data(data: dict):
    redis.set(TRACKING_KEY, json.dumps(data))


def compute_model_accuracies(tracking_data: dict, min_confidence_pct: float = 50.0) -> pd.DataFrame:
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


@st.cache_resource
def load_predictor_assets():
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


def get_historical_team_metrics(df_hist, blue_team, red_team):
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
    if prob <= 0 or prob >= 1:
        return "N/A"
    return f"{int(round(-100 * prob / (1 - prob)))}" if prob >= 0.5 else f"+{int(round(100 * (1 - prob) / prob))}"


def send_odds_to_endpoint(blue_team: str, red_team: str, p_blue: float, p_red: float):
    payload = {
        "odds": {blue_team: round(1.0 / p_blue, 2) if p_blue > 0 else 0, red_team: round(1.0 / p_red, 2) if p_red > 0 else 0},
        "model_probs": {blue_team: round(p_blue, 4), red_team: round(p_red, 4)}
    }
    try:
        requests.post(ODDS_ENDPOINT_URL, json=payload, timeout=2)
        st.toast("Dispatched odds to prediction monitor!", icon="📡")
    except Exception:
        st.toast(f"Could not reach endpoint ({ODDS_ENDPOINT_URL})", icon="⚠️")


def create_ensemble_result(model_results_dict: dict) -> dict:
    single_models = [res for key, res in model_results_dict.items() if key != "Even Split"]
    avg_blue_prob = sum(res['blue_win_probability'] for res in single_models) / len(single_models)
    ensemble_res = copy.deepcopy(single_models[0])
    ensemble_res['blue_win_probability'] = round(avg_blue_prob, 4)
    ensemble_res['red_win_probability'] = round(1.0 - avg_blue_prob, 4)
    ensemble_res['blue_win_percentage'] = round(avg_blue_prob * 100, 1)
    ensemble_res['red_win_percentage'] = round((1.0 - avg_blue_prob) * 100, 1)
    return ensemble_res


# Load Predictor Assets
engines, team_rosters, champion_list, df_hist = load_predictor_assets()
is_admin = check_is_admin()

# --- SIDEBAR ---
st.sidebar.title("🎯 Live Accuracy Tracker")
tracking_data = load_tracking_data()
total_g = tracking_data.get("total_games", 0)
correct_p = tracking_data.get("correct_predictions", 0)
acc_rate = (correct_p / total_g * 100) if total_g > 0 else 0.0

st.sidebar.metric("Live Accuracy Rate", f"{acc_rate:.1f}%")
st.sidebar.metric("Record", f"{correct_p} Correct / {total_g} Total")

if st.sidebar.button("📊 View Model Accuracy Chart", use_container_width=True):
    st.session_state["show_accuracy_chart"] = not st.session_state.get("show_accuracy_chart", False)

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Manual Count Override"):
    if is_admin:
        manual_total = st.number_input("Total Live Games", min_value=0, value=int(total_g), step=1)
        manual_correct = st.number_input("Correct Predictions", min_value=0, value=int(correct_p), step=1)
        if st.button("Save Manual Counts", use_container_width=True):
            tracking_data["total_games"] = int(manual_total)
            tracking_data["correct_predictions"] = int(manual_correct)
            save_tracking_data(tracking_data)
            st.toast("Tracking counts updated!", icon="💾")
            st.rerun()
    else:
        st.info("🔒 Owner access required.")

with st.sidebar.expander("🔐 Owner Login"):
    if is_admin:
        st.success("Admin Access Unlocked")
        if st.button("Logout Admin", use_container_width=True):
            st.session_state["is_admin"] = False
            st.query_params.clear()
            st.rerun()
    else:
        admin_input = st.text_input("Admin Key", type="password")
        if st.button("Unlock Admin Features", use_container_width=True):
            if admin_input == st.secrets.get("ADMIN_KEY", ""):
                st.session_state["is_admin"] = True
                st.toast("Unlocked Owner Mode!", icon="🔓")
                st.rerun()
            else:
                st.error("Incorrect Admin Key")

st.title("League of Legends Pre-Game Match Predictor")

if st.session_state.get("show_accuracy_chart", False):
    with st.container(border=True):
        st.subheader("📊 Live Accuracy by Model")
        min_conf = st.slider("Minimum Model Win Probability Confidence (%)", 50, 100, 50, 5)
        df_accuracy = compute_model_accuracies(tracking_data, min_confidence_pct=min_conf)
        if not df_accuracy.empty:
            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.bar_chart(df_accuracy, x="Model", y="Accuracy (%)", height=300)
            with table_col:
                st.dataframe(df_accuracy, use_container_width=True, hide_index=True)

# --- GOL.GG AUTO-IMPORT SECTION ---
with st.expander("🌐 Import Match Draft from gol.gg", expanded=True):
    col_url, col_btn = st.columns([4, 1])
    with col_url:
        gol_url = st.text_input("gol.gg Game URL", placeholder="https://gol.gg/game/stats/82174/page-game/", key="gol_url_input")
    with col_btn:
        st.write("")
        if st.button("⚡ Fetch Draft", type="primary", use_container_width=True):
            if gol_url.strip():
                with st.status("Scraping draft from gol.gg...", expanded=True) as status:
                    try:
                        draft = fetch_golgg_draft(gol_url.strip())

                        # Render diagnostic trace in real-time UI
                        for log_entry in draft.get("debug_logs", []):
                            st.text(f"🔍 {log_entry}")

                        valid_teams = list(team_rosters.keys())

                        # Write directly to session state
                        st.session_state["blue_team_select"] = match_team_name(draft["blue_team"], valid_teams)
                        st.session_state["red_team_select"] = match_team_name(draft["red_team"], valid_teams)
                        st.session_state["first_pick_radio"] = draft["first_pick"]

                        for i in range(5):
                            if i < len(draft["blue_champs"]):
                                st.session_state[f"bc_{i}"] = match_champion_name(draft["blue_champs"][i], champion_list)
                            if i < len(draft["red_champs"]):
                                st.session_state[f"rc_{i}"] = match_champion_name(draft["red_champs"][i], champion_list)

                            if i < len(draft["blue_players"]):
                                st.session_state[f"bp_{i}"] = draft["blue_players"][i]
                            if i < len(draft["red_players"]):
                                st.session_state[f"rp_{i}"] = draft["red_players"][i]

                        status.update(label="Draft Loaded Successfully!", state="complete", expanded=False)
                        st.toast("Draft successfully loaded into GUI!", icon="🚀")
                        st.rerun()

                    except Exception as e:
                        status.update(label="Draft Import Failed", state="error", expanded=True)
                        st.error(f"**Error Details:**\n```text\n{e}\n```")
            else:
                st.warning("Please enter a valid gol.gg match URL.")

# --- TEAM & SERIES CONTEXT ---
col_blue_header, col_swap_btn, col_red_header = st.columns([4, 2, 4])

with col_blue_header:
    st.subheader("Blue Side")
    blue_team = st.selectbox("Select Blue Team", options=list(team_rosters.keys()), key="blue_team_select")

with col_swap_btn:
    st.write("")
    st.write("")
    st.button("🔄 Swap Sides", on_click=swap_sides_callback, use_container_width=True)

with col_red_header:
    st.subheader("Red Side")
    red_team = st.selectbox("Select Red Team", options=list(team_rosters.keys()), index=1 if len(team_rosters) > 1 else 0, key="red_team_select")

st.markdown("##### 🎮 Match & Series Context")
s1, s2, s3, s4 = st.columns(4)
with s1:
    first_pick_side = st.radio("First Pick Side", options=["Blue", "Red"], horizontal=True, key="first_pick_radio")
with s2:
    game_number = st.number_input("Game Number in Series", 1, 7, 1)
with s3:
    blue_series_lead = st.number_input(f"{blue_team} Series Lead", -3, 3, 0)
with s4:
    blue_prev_win_raw = st.selectbox(f"Did {blue_team} Win Previous Game?", options=["N/A (Game 1)", "Yes", "No"])
    blue_prev_win = 1 if blue_prev_win_raw == "Yes" else 0

st.markdown("---")

# --- STABLE PLAYER ROSTERS & DRAFT GRID ---
roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
blue_default = team_rosters.get(blue_team, ["", "", "", "", ""])
red_default = team_rosters.get(red_team, ["", "", "", "", ""])

# Ensure session state exists prior to rendering without conflicting `value=` arguments
for i in range(5):
    if f"bp_{i}" not in st.session_state:
        st.session_state[f"bp_{i}"] = blue_default[i] if i < len(blue_default) else ""
    if f"rp_{i}" not in st.session_state:
        st.session_state[f"rp_{i}"] = red_default[i] if i < len(red_default) else ""

c1, c2, c3, c4 = st.columns([2, 3, 2, 3])
blue_players, blue_champs = [], []
red_players, red_champs = [], []

with c1:
    st.markdown("**Blue Players**")
    for i, role in enumerate(roles):
        p = st.text_input(f"Blue {role} Player", key=f"bp_{i}")
        blue_players.append(p)

with c2:
    st.markdown("**Blue Champions**")
    for i, role in enumerate(roles):
        c = st.selectbox(f"Blue {role} Pick", options=champion_list, key=f"bc_{i}")
        blue_champs.append(c)

with c3:
    st.markdown("**Red Players**")
    for i, role in enumerate(roles):
        p = st.text_input(f"Red {role} Player", key=f"rp_{i}")
        red_players.append(p)

with c4:
    st.markdown("**Red Champions**")
    for i, role in enumerate(roles):
        c = st.selectbox(f"Red {role} Pick", options=champion_list, key=f"rc_{i}")
        red_champs.append(c)

st.markdown("---")

# --- CALCULATE PREDICTIONS ---
if st.button("Calculate Match Probabilities", type="primary", use_container_width=True):
    draft_payload = {
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_players": blue_players,
        "red_players": red_players,
        "blue_champs": blue_champs,
        "red_champs": red_champs,
        "blue_firstpick": 1 if first_pick_side == "Blue" else 0,
        "game_number": game_number,
        "blue_series_lead": blue_series_lead,
        "blue_prev_win": blue_prev_win
    }

    all_model_results = {m_name: eng.predict_match(draft_payload) for m_name, eng in engines.items()}
    all_model_results["Even Split"] = create_ensemble_result(all_model_results)
    h2h_data = get_historical_team_metrics(df_hist, blue_team, red_team)

    primary_res = all_model_results.get("XGBoost", next(iter(all_model_results.values())))
    send_odds_to_endpoint(blue_team, red_team, primary_res['blue_win_probability'], primary_res['red_win_probability'])

    st.session_state["active_prediction"] = {
        "blue_team": blue_team,
        "red_team": red_team,
        "model_results": all_model_results,
        "h2h_data": h2h_data
    }


def render_model_dashboard(model_name: str, results: dict, active_pred: dict, h2h_data: dict):
    b_team, r_team = active_pred['blue_team'], active_pred['red_team']
    predicted_winner = b_team if results['blue_win_probability'] >= 0.5 else r_team

    res_b, res_r = st.columns(2)
    res_b.metric(f"{b_team} Win Probability", f"{results['blue_win_percentage']}%")
    res_r.metric(f"{r_team} Win Probability", f"{results['red_win_percentage']}%")
    st.progress(results['blue_win_probability'])

    with st.container(border=True):
        st.subheader("📝 Record Live Game Result")
        st.write(f"Model ({model_name}) Predicted Winner: **{predicted_winner}**")

        act_col1, act_col2 = st.columns([3, 1])
        with act_col1:
            actual_winner = st.radio("Select Actual Game Winner:", options=[b_team, r_team], horizontal=True, key=f"actual_winner_{model_name}")

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

    st.markdown(f"### 📊 Feature & Match Analysis ({model_name})")
    tab_elo, tab_players, tab_draft, tab_h2h, tab_odds = st.tabs([
        "⚡ Elo & Series Context",
        "👤 Player Mastery",
        "⚔️ Draft Impact",
        "🛡️ Team H2H",
        "🎲 Value Odds"
    ])

    with tab_elo:
        e1, e2, e3 = st.columns(3)
        e1.metric(f"{b_team} Elo", f"{results['elo_metrics']['blue_elo']}")
        e2.metric(f"{r_team} Elo", f"{results['elo_metrics']['red_elo']}")
        e3.metric("Elo Implied Winrate", f"{results['elo_metrics']['elo_implied_blue_winrate']}%")

    with tab_players:
        player_rows = [{
            "Role": r['role'],
            f"{b_team} Player": r['blue_player'],
            "Blue WR": f"{round(r['blue_p_wr']*100, 1)}%",
            f"{r_team} Player": r['red_player'],
            "Red WR": f"{round(r['red_p_wr']*100, 1)}%"
        } for r in results['role_breakdown']]
        st.table(pd.DataFrame(player_rows))

    with tab_draft:
        champ_rows = [{
            "Role": r['role'],
            f"{b_team} Pick": r['blue_champ'],
            "Blue Champ WR": f"{round(r['blue_c_wr']*100, 1)}%",
            f"{r_team} Pick": r['red_champ'],
            "Red Champ WR": f"{round(r['red_c_wr']*100, 1)}%"
        } for r in results['role_breakdown']]
        st.table(pd.DataFrame(champ_rows))

    with tab_h2h:
        st.info(f"Historical Matchups: {h2h_data['total_h2h']} | {b_team} H2H Winrate: {h2h_data['blue_h2h_wr']}%")

    with tab_odds:
        p_b, p_r = results['blue_win_probability'], results['red_win_probability']
        st.write(f"**{b_team} Fair Decimal:** {round(1.0/p_b, 2) if p_b > 0 else 0} ({prob_to_american_odds(p_b)})")
        st.write(f"**{r_team} Fair Decimal:** {round(1.0/p_r, 2) if p_r > 0 else 0} ({prob_to_american_odds(p_r)})")


if "active_prediction" in st.session_state:
    active_pred = st.session_state["active_prediction"]
    model_results = active_pred["model_results"]
    st.markdown("## 🤖 Prediction Engine Selector")
    model_names = list(model_results.keys())
    model_tabs = st.tabs([f"📌 {m}" if m != "Even Split" else "⚖️ Even Split" for m in model_names])

    for i, model_name in enumerate(model_names):
        with model_tabs[i]:
            render_model_dashboard(model_name, model_results[model_name], active_pred, active_pred["h2h_data"])