import copy
import json
import os
from datetime import datetime
import xgboost as xgb
import joblib
import pandas as pd
import requests
import streamlit as st

from app_helpers import compute_model_accuracies, match_team_name, match_champion_name, fetch_golgg_draft
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

# --- TEAM ROSTER CALLBACKS ---
def update_blue_roster_callback():
    selected_team = st.session_state.get("blue_team_select")
    roster = team_rosters.get(selected_team, ["", "", "", "", ""])
    for i in range(5):
        st.session_state[f"bp_{i}"] = roster[i] if i < len(roster) else ""

def update_red_roster_callback():
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

# --- INITIALIZE DEFAULT ROSTERS IN SESSION STATE IF UNSET ---
if "bp_0" not in st.session_state:
    initial_blue = st.session_state.get("blue_team_select", list(team_rosters.keys())[0] if team_rosters else "")
    blue_def = team_rosters.get(initial_blue, ["", "", "", "", ""])
    for i in range(5):
        st.session_state[f"bp_{i}"] = blue_def[i] if i < len(blue_def) else ""

if "rp_0" not in st.session_state:
    initial_red_idx = 1 if len(team_rosters) > 1 else 0
    initial_red = st.session_state.get("red_team_select", list(team_rosters.keys())[initial_red_idx] if team_rosters else "")
    red_def = team_rosters.get(initial_red, ["", "", "", "", ""])
    for i in range(5):
        st.session_state[f"rp_{i}"] = red_def[i] if i < len(red_def) else ""

# --- TEAM & SERIES CONTEXT ---
col_blue_header, col_swap_btn, col_red_header = st.columns([4, 2, 4])

with col_blue_header:
    st.subheader("Blue Side")
    blue_team = st.selectbox(
        "Select Blue Team",
        options=list(team_rosters.keys()),
        key="blue_team_select",
        on_change=update_blue_roster_callback
    )

with col_swap_btn:
    st.write("")
    st.write("")
    st.button("🔄 Swap Sides", on_click=swap_sides_callback, use_container_width=True)

with col_red_header:
    st.subheader("Red Side")
    red_team = st.selectbox(
        "Select Red Team",
        options=list(team_rosters.keys()),
        index=1 if len(team_rosters) > 1 else 0,
        key="red_team_select",
        on_change=update_red_roster_callback
    )

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
        elo_base = results.get('elo_metrics', {}).get('elo_implied_blue_winrate', 50.0)
        e1.metric(f"{b_team} Elo", f"{results['elo_metrics']['blue_elo']}")
        e2.metric(f"{r_team} Elo", f"{results['elo_metrics']['red_elo']}")
        e3.metric("Elo Implied Winrate", f"{elo_base}%")

    with tab_players:
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

    with tab_draft:
        # 1. High-level Percentage Swing Impact Metrics
        elo_base_pct = results.get('elo_metrics', {}).get('elo_implied_blue_winrate', 50.0)
        final_blue_pct = results.get('blue_win_percentage', 50.0)
        total_draft_swing = round(final_blue_pct - elo_base_pct, 1)

        role_data = results.get('role_breakdown', [])
        if role_data:
            avg_b_c_wr = sum(r.get('blue_c_wr', 0.5) for r in role_data) / len(role_data) * 100
            avg_r_c_wr = sum(r.get('red_c_wr', 0.5) for r in role_data) / len(role_data) * 100
            draft_comp_delta = round(avg_b_c_wr - avg_r_c_wr, 1)
        else:
            avg_b_c_wr, avg_r_c_wr, draft_comp_delta = 50.0, 50.0, 0.0

        # Render Summary Cards
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Draft Swing (vs Elo)", f"{total_draft_swing:+.1f}%", help="Win probability change added/lost from draft relative to Elo expectation.")
        d2.metric(f"{b_team} Draft Winrate", f"{avg_b_c_wr:.1f}%")
        d3.metric("Champ Composition Delta", f"{draft_comp_delta:+.1f}%", help="Direct winrate differential between Blue and Red champion picks.")

        # 2. Check for explicit model feature impacts if available in model outputs
        explicit_impacts = results.get("draft_impact", results.get("impact_breakdown", None))
        if explicit_impacts and isinstance(explicit_impacts, dict):
            st.markdown("#### 🔍 Model Feature Impact Breakdown")
            imp_cols = st.columns(len(explicit_impacts))
            for idx, (imp_name, imp_val) in enumerate(explicit_impacts.items()):
                imp_cols[idx].metric(imp_name, f"{imp_val:+.1f}%")

        # 3. Role-by-Role Draft Swing Breakdown
        st.markdown("#### 🎯 Role-by-Role Draft Swing Breakdown")
        champ_rows = []
        for r in role_data:
            b_c_wr = r.get('blue_c_wr', 0.5) * 100
            r_c_wr = r.get('red_c_wr', 0.5) * 100
            role_swing = round(b_c_wr - r_c_wr, 1)
            champ_rows.append({
                "Role": r['role'],
                f"{b_team} Pick": r['blue_champ'],
                "Blue Champ WR": f"{b_c_wr:.1f}%",
                f"{r_team} Pick": r['red_champ'],
                "Red Champ WR": f"{r_c_wr:.1f}%",
                "Role Impact Swing": f"{role_swing:+.1f}%"
            })
        st.dataframe(pd.DataFrame(champ_rows), use_container_width=True, hide_index=True)

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