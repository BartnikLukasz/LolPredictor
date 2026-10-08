import copy
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

from app_helpers import (
    apply_live_series_elo_adjustment,
    compute_db_model_weights,
    compute_model_accuracies,
    create_weighted_ensemble_result,
    fetch_golgg_draft,
    get_historical_team_metrics,
    match_champion_name,
    match_team_name,
    prob_to_american_odds,
    send_odds_to_endpoint,
)
from live_feature_engine import LiveFeatureEngine
from util import generate_game_id

st.set_page_config(page_title="LoL Match Predictor", layout="wide")

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
    return st.query_params.get("admin") == admin_secret or st.session_state.get("is_admin", False)


@st.cache_resource
def get_redis_client():
    """Initializes Upstash Redis safely with fallback handling."""
    try:
        url = st.secrets.get("UPSTASH_REDIS_REST_URL")
        token = st.secrets.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            return Redis(url=url, token=token)
    except Exception:
        pass
    return None


redis = get_redis_client()


def load_tracking_data() -> dict:
    """Loads prediction tracking stats from Redis or local session state."""
    if redis is not None:
        try:
            raw_data = redis.get(TRACKING_KEY)
            if raw_data:
                return json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except Exception:
            pass
    return st.session_state.get("local_tracking_data", {"total_games": 0, "correct_predictions": 0, "logs": []})


def save_tracking_data(data: dict):
    """Saves tracking stats to Redis or local session state."""
    if redis is not None:
        try:
            redis.set(TRACKING_KEY, json.dumps(data))
            return
        except Exception:
            pass
    st.session_state["local_tracking_data"] = data


@st.cache_resource
def load_predictor_assets():
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv.gz"
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path '{dataset_path}' not found.")
        st.stop()

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

    roster_path = "models/team_rosters.json"
    if os.path.exists(roster_path):
        with open(roster_path, "r") as f:
            roster_data = json.load(f)
    else:
        roster_data = {"T1": ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]}

    champ_cols = [c for c in base_engine.df_hist.columns if 'champion' in c]
    champions_set = set()
    for col in champ_cols:
        champions_set.update(base_engine.df_hist[col].dropna().unique().tolist())

    champions_list = sorted(list(champions_set)) if champions_set else ["Ahri", "Aatrox", "Azir"]
    return engines, roster_data, champions_list, base_engine.df_hist


# Load Predictor Assets
engines, team_rosters, champion_list, df_hist = load_predictor_assets()
is_admin = check_is_admin()


# --- ROSTER CALLBACKS ---
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

                        for log_entry in draft.get("debug_logs", []):
                            st.text(f"🔍 {log_entry}")

                        valid_teams = list(team_rosters.keys())

                        st.session_state["blue_team_select"] = match_team_name(
                            draft["blue_team"],
                            valid_teams,
                            fetched_players=draft.get("blue_players", []),
                            team_rosters=team_rosters,
                            df_hist=df_hist
                        )
                        st.session_state["red_team_select"] = match_team_name(
                            draft["red_team"],
                            valid_teams,
                            fetched_players=draft.get("red_players", []),
                            team_rosters=team_rosters,
                            df_hist=df_hist
                        )
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

# --- INITIALIZE DEFAULT ROSTERS IN SESSION STATE ---
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
    game_number = st.number_input("Game Number in Series", 1, 5, 1)
with s3:
    blue_series_lead = st.number_input(f"{blue_team} Series Lead", -2, 2, 0)
with s4:
    blue_prev_win_raw = st.selectbox(f"Did {blue_team} Win Previous Game?", options=["N/A (Game 1)", "Yes", "No"])
    if blue_prev_win_raw == "N/A (Game 1)":
        blue_prev_win = 0.5
    else:
        blue_prev_win = 1.0 if blue_prev_win_raw == "Yes" else 0.0

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
    total_past_games = max(0, int(game_number) - 1)

    raw_blue_wins = (total_past_games + int(blue_series_lead)) // 2
    raw_red_wins = (total_past_games - int(blue_series_lead)) // 2

    blue_series_wins = max(0, min(total_past_games, raw_blue_wins))
    red_series_wins = max(0, min(total_past_games, raw_red_wins))

    game_id = generate_game_id(
        blue_team=blue_team,
        red_team=red_team,
        blue_champs=blue_champs,
        red_champs=red_champs,
        first_pick=first_pick_side
    )

    custom_elo_metrics = apply_live_series_elo_adjustment(
        df_hist=df_hist,
        blue_team=blue_team,
        red_team=red_team,
        blue_series_wins=blue_series_wins,
        red_series_wins=red_series_wins,
        blue_has_first_pick=(first_pick_side == "Blue"),
        k_series=60.0
    )

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
        "blue_prev_win": blue_prev_win,
        "blue_series_wins": blue_series_wins,
        "red_series_wins": red_series_wins,
        "custom_elo_metrics": custom_elo_metrics
    }

    base_results = {m_name: eng.predict_match(draft_payload) for m_name, eng in engines.items()}

    model_weights, model_accuracies = compute_db_model_weights(tracking_data, list(base_results.keys()))
    weighted_res = create_weighted_ensemble_result(base_results, model_weights)

    all_model_results = base_results.copy()
    all_model_results["Weighted Split"] = weighted_res

    h2h_data = get_historical_team_metrics(df_hist, blue_team, red_team)

    primary_res = all_model_results.get("XGBoost", next(iter(all_model_results.values())))
    send_odds_to_endpoint(blue_team, red_team, primary_res['blue_win_probability'], primary_res['red_win_probability'])

    st.session_state["selected_actual_winner"] = blue_team

    st.session_state["active_prediction"] = {
        "game_id": game_id,
        "blue_team": blue_team,
        "red_team": red_team,
        "model_results": all_model_results,
        "model_weights": model_weights,
        "model_accuracies": model_accuracies,
        "h2h_data": h2h_data
    }


def render_model_dashboard(model_name: str, results: dict, active_pred: dict, h2h_data: dict):
    b_team, r_team = active_pred['blue_team'], active_pred['red_team']
    predicted_winner = b_team if results['blue_win_probability'] >= 0.5 else r_team

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
            if "selected_actual_winner" not in st.session_state:
                st.session_state["selected_actual_winner"] = b_team

            actual_winner = st.radio(
                "Select Actual Game Winner:",
                options=[b_team, r_team],
                horizontal=True,
                key="selected_actual_winner"
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
                    "game_id": active_pred.get("game_id", ""),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blue_team": b_team,
                    "red_team": r_team,
                    "models": models_log_list
                })
                save_tracking_data(current_track_data)
                st.toast("Result Logged!", icon="🎯")
                del st.session_state["active_prediction"]
                st.rerun()

    # --- REPLACED FEATURE IMPORTANCE & PREDICTION VALUES SECTION ---
    st.markdown("---")
    st.markdown(f"### 🔍 Model Feature Importance & Prediction Values ({model_name})")
    st.caption(
        "Below is the complete list of input features for the selected model, "
        "their relative importance to the model's prediction, and their exact computed values for this match."
    )

    features_df = results.get("features_df")

    if features_df is not None and not features_df.empty:
        # Filtering & Sorting Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

        with ctrl_col1:
            search_query = st.text_input(
                "🔎 Search Features",
                placeholder="e.g., elo, winrate, streak, champ...",
                key=f"search_feat_{model_name}"
            )

        with ctrl_col2:
            categories = ["All"] + sorted(list(features_df["Category"].unique()))
            selected_cat = st.selectbox(
                "📁 Filter by Category",
                options=categories,
                key=f"cat_feat_{model_name}"
            )

        with ctrl_col3:
            sort_order = st.selectbox(
                "⬆️ Sort Importance",
                options=["Highest First", "Lowest First"],
                key=f"sort_feat_{model_name}"
            )

        # Apply filtering
        display_df = features_df.copy()

        if selected_cat != "All":
            display_df = display_df[display_df["Category"] == selected_cat]

        if search_query.strip():
            query = search_query.strip().lower()
            display_df = display_df[
                display_df["Feature"].str.lower().str.contains(query) |
                display_df["Category"].str.lower().str.contains(query) |
                display_df["Value for Prediction"].astype(str).str.lower().str.contains(query)
            ]

        ascending_sort = (sort_order == "Lowest First")
        display_df = display_df.sort_values(by="Importance (%)", ascending=ascending_sort).reset_index(drop=True)

        # Highlight Top Decision Drivers
        top_3 = features_df.head(3)
        st.markdown("##### 💡 Top Decision Drivers for this Model")
        top_cols = st.columns(min(3, len(top_3)))
        for idx, (_, row) in enumerate(top_3.iterrows()):
            if idx < len(top_cols):
                top_cols[idx].metric(
                    label=f"#{idx+1}: {row['Feature']}",
                    value=f"Value: {row['Value for Prediction']}",
                    delta=f"{row['Importance (%)']}% Importance"
                )

        st.markdown("##### 📋 Complete Feature Vector & Importance Table")

        st.dataframe(
            display_df[["Feature", "Category", "Importance (%)", "Value for Prediction"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature Name", help="Input feature name used by the model"),
                "Category": st.column_config.TextColumn("Category", help="Feature category group"),
                "Importance (%)": st.column_config.NumberColumn(
                    "Importance (%)",
                    format="%.2f%%",
                    help="Global relative importance weight of this feature in the trained model"
                ),
                "Value for Prediction": st.column_config.TextColumn(
                    "Value for Prediction",
                    help="Exact input value generated for this specific match prediction"
                )
            }
        )
        st.caption(f"Showing {len(display_df)} of {len(features_df)} total features for model **{model_name}**.")

    else:
        st.info("Feature importance data is not available for this model.")


if "active_prediction" in st.session_state:
    active_pred = st.session_state["active_prediction"]
    model_results = active_pred["model_results"]
    st.markdown("## 🤖 Prediction Engine Selector")

    model_names = list(model_results.keys())

    if "active_model_tab" not in st.session_state or st.session_state["active_model_tab"] not in model_names:
        st.session_state["active_model_tab"] = "Weighted Split" if "Weighted Split" in model_names else model_names[0]

    selected_model_name = st.radio(
        "Select Active Prediction Model Engine",
        options=model_names,
        horizontal=True,
        key="active_model_tab"
    )

    if selected_model_name in model_results:
        render_model_dashboard(
            selected_model_name,
            model_results[selected_model_name],
            active_pred,
            active_pred["h2h_data"]
        )