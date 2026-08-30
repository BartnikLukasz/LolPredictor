import copy
import json
import os
from datetime import datetime
import xgboost as xgb
import joblib
import pandas as pd
import requests
import streamlit as st
from live_feature_engine import LiveFeatureEngine
from upstash_redis import Redis

st.set_page_config(page_title="LoL Match Predictor", layout="wide")

ODDS_ENDPOINT_URL = "http://127.0.0.1:5000/odds"
TRACKING_KEY = "live_accuracy_tracking"

# --- EXTENSIBLE MODEL REGISTRY ---
# To add a new engine, train it, place the .pkl in models/, and add an entry here:
MODEL_REGISTRY = {
    "XGBoost": "models/xgboost_model.json",
    "LightGBM": "models/lightgbm_model.pkl",
    "CatBoost": "models/catboost_model.pkl",
    "ElasticTree": "models/elastictree_model.pkl",
    "ElasticNet": "models/elasticnet_model.joblib",
}


# Initialize Redis connection via Streamlit Secrets
@st.cache_resource
def get_redis_client():
    return Redis(
        url=st.secrets["UPSTASH_REDIS_REST_URL"],
        token=st.secrets["UPSTASH_REDIS_REST_TOKEN"]
    )


redis = get_redis_client()


def load_tracking_data() -> dict:
    """Fetches tracking JSON from Upstash Redis."""
    raw_data = redis.get(TRACKING_KEY)
    if not raw_data:
        return {
            "total_games": 0,
            "correct_predictions": 0,
            "logs": []
        }
    if isinstance(raw_data, str):
        return json.loads(raw_data)
    return raw_data


def save_tracking_data(data: dict):
    """Saves updated tracking dict back to Upstash Redis."""
    redis.set(TRACKING_KEY, json.dumps(data))


@st.cache_resource
def load_predictor_assets():
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"
    base_engine = LiveFeatureEngine(dataset_path=dataset_path)

    engines = {}

    for model_name, model_path in MODEL_REGISTRY.items():
        if os.path.exists(model_path):
            eng = copy.deepcopy(base_engine)

            # Load Native XGBoost JSON
            if model_path.endswith(".json"):
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                eng.model = model

            # Load standard Pickle/Joblib
            else:
                artifact = joblib.load(model_path)

                if isinstance(artifact, dict):
                    # Priority 1: Full pipeline stored under 'pipeline' key
                    if "pipeline" in artifact:
                        eng.model = artifact["pipeline"]
                    # Priority 2: Model + Preprocessor stored separately in dict
                    elif "model" in artifact and "preprocessor" in artifact:
                        from sklearn.pipeline import Pipeline
                        eng.model = Pipeline([
                            ('preprocessor', artifact["preprocessor"]),
                            ('model', artifact["model"])
                        ])
                    else:
                        eng.model = artifact.get("model", artifact)
                else:
                    # Raw object (full Pipeline or model)
                    eng.model = artifact

            engines[model_name] = eng
        else:
            engines[model_name] = base_engine

    with open("team_rosters.json", "r") as f:
        roster_data = json.load(f)

    champ_cols = [c for c in base_engine.df_hist.columns if 'champion' in c]
    champions_set = set()
    for col in champ_cols:
        champions_set.update(base_engine.df_hist[col].dropna().unique().tolist())

    champions_list = sorted(list(champions_set)) if champions_set else ["Ahri", "Aatrox", "Azir"]

    return engines, roster_data, champions_list, base_engine.df_hist


def get_historical_team_metrics(df_hist, blue_team, red_team):
    """Computes direct H2H history and recent form (last 10 matches) from df_hist."""
    h2h_matches = df_hist[
        ((df_hist['blue_team'] == blue_team) & (df_hist['red_team'] == red_team)) |
        ((df_hist['blue_team'] == red_team) & (df_hist['red_team'] == blue_team))
        ].sort_values('date', ascending=False)

    total_h2h = len(h2h_matches)
    blue_h2h_wins = 0
    if total_h2h > 0:
        for _, row in h2h_matches.iterrows():
            if row['blue_team'] == blue_team and row['blue_win'] == 1:
                blue_h2h_wins += 1
            elif row['red_team'] == blue_team and row['blue_win'] == 0:
                blue_h2h_wins += 1

    blue_matches = df_hist[(df_hist['blue_team'] == blue_team) | (df_hist['red_team'] == blue_team)].sort_values('date',
                                                                                                                 ascending=False).head(
        10)
    red_matches = df_hist[(df_hist['blue_team'] == red_team) | (df_hist['red_team'] == red_team)].sort_values('date',
                                                                                                              ascending=False).head(
        10)

    blue_recent_wins = sum(
        (row['blue_win'] == 1 if row['blue_team'] == blue_team else row['blue_win'] == 0)
        for _, row in blue_matches.iterrows()
    )
    red_recent_wins = sum(
        (row['blue_win'] == 1 if row['red_team'] == red_team else row['blue_win'] == 0)
        for _, row in red_matches.iterrows()
    )

    return {
        'total_h2h': total_h2h,
        'blue_h2h_wins': blue_h2h_wins,
        'red_h2h_wins': total_h2h - blue_h2h_wins,
        'blue_h2h_wr': round((blue_h2h_wins / total_h2h * 100), 1) if total_h2h > 0 else 50.0,
        'blue_recent_wr': round((blue_recent_wins / max(len(blue_matches), 1) * 100), 1),
        'red_recent_wr': round((red_recent_wins / max(len(red_matches), 1) * 100), 1),
        'blue_recent_games': len(blue_matches),
        'red_recent_games': len(red_matches)
    }


def prob_to_american_odds(prob: float) -> str:
    """Converts a probability (0.0 - 1.0) into American odds format."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        odds = int(round(-100 * prob / (1 - prob)))
        return f"{odds}"
    else:
        odds = int(round(100 * (1 - prob) / prob))
        return f"+{odds}"


def send_odds_to_endpoint(blue_team: str, red_team: str, p_blue: float, p_red: float):
    """Sends calculated fair odds and model probabilities to external endpoint."""
    fair_dec_blue = round(1.0 / p_blue, 2) if p_blue > 0 else 0.0
    fair_dec_red = round(1.0 / p_red, 2) if p_red > 0 else 0.0

    payload = {
        "odds": {
            blue_team: fair_dec_blue,
            red_team: fair_dec_red
        },
        "model_probs": {
            blue_team: round(p_blue, 4),
            red_team: round(p_red, 4)
        }
    }

    try:
        response = requests.post(ODDS_ENDPOINT_URL, json=payload, timeout=2)
        if response.status_code in [200, 201]:
            st.toast("Dispatched odds to prediction monitor!", icon="📡")
        else:
            st.toast(f"Odds endpoint returned status code {response.status_code}", icon="⚠️")
    except requests.exceptions.RequestException:
        st.toast(f"Could not reach endpoint ({ODDS_ENDPOINT_URL})", icon="⚠️")


def create_ensemble_result(model_results_dict: dict) -> dict:
    """Computes an Even Split (Ensemble) prediction by averaging single model probabilities."""
    single_models = [res for key, res in model_results_dict.items() if key != "Even Split"]

    avg_blue_prob = sum(res['blue_win_probability'] for res in single_models) / len(single_models)
    avg_red_prob = 1.0 - avg_blue_prob

    # Clone base structure from primary model
    ensemble_res = copy.deepcopy(single_models[0])
    ensemble_res['blue_win_probability'] = round(avg_blue_prob, 4)
    ensemble_res['red_win_probability'] = round(avg_red_prob, 4)
    ensemble_res['blue_win_percentage'] = round(avg_blue_prob * 100, 1)
    ensemble_res['red_win_percentage'] = round(avg_red_prob * 100, 1)

    return ensemble_res


# Load App Assets
engines, team_rosters, champion_list, df_hist = load_predictor_assets()

# --- SIDEBAR: LIVE ACCURACY MONITOR & MANUAL COUNTER ADJUSTER ---
st.sidebar.title("🎯 Live Accuracy Tracker")

tracking_data = load_tracking_data()
total_g = tracking_data.get("total_games", 0)
correct_p = tracking_data.get("correct_predictions", 0)
acc_rate = (correct_p / total_g * 100) if total_g > 0 else 0.0

st.sidebar.metric("Live Accuracy Rate", f"{acc_rate:.1f}%")
st.sidebar.metric("Record", f"{correct_p} Correct / {total_g} Total")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Manual Count Override"):
    st.caption("Adjust counters directly if you missed logging a game live.")
    manual_total = st.number_input("Total Live Games", min_value=0, value=int(total_g), step=1)
    manual_correct = st.number_input("Correct Predictions", min_value=0, value=int(correct_p), step=1)

    if st.button("Save Manual Counts", use_container_width=True):
        tracking_data["total_games"] = int(manual_total)
        tracking_data["correct_predictions"] = int(manual_correct)
        save_tracking_data(tracking_data)
        st.toast("Tracking counts updated successfully!", icon="💾")
        st.rerun()

st.title("League of Legends Pre-Game Match Predictor")

# --- 1. Team & Series Context Selection ---
col_blue_header, col_red_header = st.columns(2)

with col_blue_header:
    st.subheader("Blue Side")
    blue_team = st.selectbox(
        "Select Blue Team",
        options=list(team_rosters.keys()),
        index=0 if team_rosters else 0,
        key="blue_team_select"
    )

with col_red_header:
    st.subheader("Red Side")
    red_team = st.selectbox(
        "Select Red Team",
        options=list(team_rosters.keys()),
        index=1 if len(team_rosters) > 1 else 0,
        key="red_team_select"
    )

st.markdown("##### 🎮 Match & Series Context")
s1, s2, s3, s4 = st.columns(4)

with s1:
    first_pick_side = st.radio(
        "First Pick Side",
        options=["Blue", "Red"],
        index=0,
        horizontal=True
    )

with s2:
    game_number = st.number_input(
        "Game Number in Series",
        min_value=1,
        max_value=7,
        value=1,
        step=1
    )

with s3:
    blue_series_lead = st.number_input(
        f"{blue_team} Series Lead",
        min_value=-3,
        max_value=3,
        value=0,
        step=1,
        help="Positive = Blue leading, Negative = Red leading"
    )

with s4:
    blue_prev_win_raw = st.selectbox(
        f"Did {blue_team} Win Previous Game?",
        options=["N/A (Game 1)", "Yes", "No"],
        index=0
    )
    blue_prev_win = 1 if blue_prev_win_raw == "Yes" else 0

st.markdown("---")

# --- 2. Player Rosters & Draft Grid ---
roles = ["Top", "Jungle", "Mid", "ADC", "Support"]

blue_default = team_rosters.get(blue_team, ["", "", "", "", ""])
red_default = team_rosters.get(red_team, ["", "", "", "", ""])

c1, c2, c3, c4 = st.columns([2, 3, 2, 3])

blue_players, blue_champs = [], []
red_players, red_champs = [], []

with c1:
    st.markdown("**Blue Players**")
    for i, role in enumerate(roles):
        val = blue_default[i] if i < len(blue_default) else ""
        p = st.text_input(f"Blue {role} Player", value=val, key=f"bp_{blue_team}_{i}")
        blue_players.append(p)

with c2:
    st.markdown("**Blue Champions**")
    for i, role in enumerate(roles):
        c = st.selectbox(f"Blue {role} Pick", options=champion_list, key=f"bc_{i}")
        blue_champs.append(c)

with c3:
    st.markdown("**Red Players**")
    for i, role in enumerate(roles):
        val = red_default[i] if i < len(red_default) else ""
        p = st.text_input(f"Red {role} Player", value=val, key=f"rp_{red_team}_{i}")
        red_players.append(p)

with c4:
    st.markdown("**Red Champions**")
    for i, role in enumerate(roles):
        c = st.selectbox(f"Red {role} Pick", options=champion_list, key=f"rc_{i}")
        red_champs.append(c)

st.markdown("---")

# --- 3. Calculation Logic ---
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

    # Evaluate predictions across all registered engines
    all_model_results = {}
    for model_name, eng in engines.items():
        all_model_results[model_name] = eng.predict_match(draft_payload)

    # Calculate "Even Split" Ensemble result
    all_model_results["Even Split"] = create_ensemble_result(all_model_results)

    h2h_data = get_historical_team_metrics(df_hist, blue_team, red_team)

    # Automatically post default XGBoost / baseline odds to endpoint
    primary_res = all_model_results.get("XGBoost", next(iter(all_model_results.values())))
    send_odds_to_endpoint(
        blue_team=blue_team,
        red_team=red_team,
        p_blue=primary_res['blue_win_probability'],
        p_red=primary_res['red_win_probability']
    )

    # Save calculated active predictions into session state
    st.session_state["active_prediction"] = {
        "blue_team": blue_team,
        "red_team": red_team,
        "model_results": all_model_results,
        "h2h_data": h2h_data
    }


# --- Helper Function: Render Dashboard for a Selected Model ---
def render_model_dashboard(model_name: str, results: dict, active_pred: dict, h2h_data: dict):
    b_team = active_pred['blue_team']
    r_team = active_pred['red_team']
    all_model_results = active_pred['model_results']
    predicted_winner = b_team if results['blue_win_probability'] >= 0.5 else r_team

    # Top Level Win Probability Header
    res_b, res_r = st.columns(2)
    res_b.metric(f"{b_team} Win Probability", f"{results['blue_win_percentage']}%")
    res_r.metric(f"{r_team} Win Probability", f"{results['red_win_percentage']}%")
    st.progress(results['blue_win_probability'])

    # --- LIVE GAME OUTCOME LOGGING CONTAINER ---
    with st.container(border=True):
        st.subheader("📝 Record Live Game Result")
        st.write(f"Model ({model_name}) Predicted Winner: **{predicted_winner}**")

        act_col1, act_col2 = st.columns([3, 1])

        with act_col1:
            actual_winner = st.radio(
                "Select Actual Game Winner:",
                options=[b_team, r_team],
                horizontal=True,
                key=f"actual_winner_radio_{model_name}"
            )

        with act_col2:
            st.write("")
            if st.button("Save & Log Result", type="secondary", use_container_width=True, key=f"save_btn_{model_name}"):
                current_track_data = load_tracking_data()

                # Build model prediction details across all active models
                models_log_list = []
                xgb_is_correct = False

                for m_name, m_res in all_model_results.items():
                    m_pred_winner = b_team if m_res['blue_win_probability'] >= 0.5 else r_team
                    m_is_correct = (actual_winner == m_pred_winner)

                    # Always evaluate XGBoost correctness for global accuracy tracking
                    if m_name == "XGBoost":
                        xgb_is_correct = m_is_correct

                    models_log_list.append({
                        "model_used": m_name,
                        "blue_win_probability": m_res['blue_win_probability'],
                        "red_win_probability": m_res['red_win_probability'],
                        "predicted_winner": m_pred_winner,
                        "actual_winner": actual_winner,
                        "is_correct": m_is_correct
                    })

                # Fallback to current model if XGBoost isn't present in model results
                if "XGBoost" not in all_model_results:
                    curr_pred = b_team if results['blue_win_probability'] >= 0.5 else r_team
                    xgb_is_correct = (actual_winner == curr_pred)

                # Update global counts strictly based on XGBoost output
                current_track_data["total_games"] = current_track_data.get("total_games", 0) + 1
                if xgb_is_correct:
                    current_track_data["correct_predictions"] = current_track_data.get("correct_predictions", 0) + 1

                # Top-level log record containing nested model list
                log_entry = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blue_team": b_team,
                    "red_team": r_team,
                    "models": models_log_list
                }

                current_track_data.setdefault("logs", []).append(log_entry)
                save_tracking_data(current_track_data)

                st.toast(
                    f"Result Logged! XGBoost Prediction: ({'Correct ✅' if xgb_is_correct else 'Incorrect ❌'})",
                    icon="🎯"
                )

                del st.session_state["active_prediction"]
                st.rerun()

    st.markdown(f"### 📊 Feature & Match Analysis ({model_name})")

    tab_elo, tab_players, tab_draft, tab_h2h, tab_odds = st.tabs([
        "⚡ Elo & Series Context",
        "👤 Player Mastery & Experience",
        "⚔️ Draft & Meta Impact",
        "🛡️ Team H2H & Recent Form",
        "🎲 Implied Odds & Value Calculator"
    ])

    # --- TAB 1: Elo Rating & Series Context Impact ---
    with tab_elo:
        elo_data = results['elo_metrics']
        s_data = results['series_metrics']

        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"{b_team} Elo", f"{elo_data['blue_elo']}")
        e2.metric(f"{r_team} Elo", f"{elo_data['red_elo']}")
        e3.metric(
            "Effective Elo Diff (inc. First Pick)",
            f"{elo_data['elo_diff']}",
            delta=f"{elo_data['elo_diff']} pts"
        )
        e4.metric(
            "Elo-Implied Blue Winrate",
            f"{elo_data['elo_implied_blue_winrate']}%"
        )

        st.markdown("**Series Context Details**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Game Number", f"Game {s_data['game_number']}")
        sc2.metric(f"{b_team} Series Lead", f"{s_data['blue_series_lead']} games")
        sc3.metric(f"{b_team} Previous Game Win",
                   "Yes" if s_data['blue_prev_win'] == 1 else ("No" if s_data['game_number'] > 1 else "N/A"))

    # --- TAB 2: Player Win Rate Head-to-Head ---
    with tab_players:
        p_data = results['player_metrics']
        pm1, pm2, pm3 = st.columns(3)

        pm1.metric(f"{b_team} Avg Player Winrate", f"{p_data['avg_blue_p_wr']}%")
        pm2.metric(f"{r_team} Avg Player Winrate", f"{p_data['avg_red_p_wr']}%")
        pm3.metric("Player Winrate Advantage", f"{p_data['p_wr_diff']}%", delta=f"{p_data['p_wr_diff']}%")

        st.markdown("**Role-by-Role Player Comparison**")
        player_rows = []
        for r in results['role_breakdown']:
            b_wr = round(r['blue_p_wr'] * 100, 1)
            r_wr = round(r['red_p_wr'] * 100, 1)
            diff = round(b_wr - r_wr, 1)
            adv = f"Blue (+{diff}%)" if diff > 0 else (f"Red ({diff}%)" if diff < 0 else "Even")

            player_rows.append({
                "Role": r['role'],
                f"{b_team} Player": r['blue_player'],
                "Blue WR": f"{b_wr}% ({r['blue_p_games']}g)",
                f"{r_team} Player": r['red_player'],
                "Red WR": f"{r_wr}% ({r['red_p_games']}g)",
                "Advantage": adv
            })

        st.table(pd.DataFrame(player_rows))

    # --- TAB 3: Draft & Meta Impact ---
    with tab_draft:
        d_data = results['draft_metrics']
        elo_prob = results['elo_metrics']['elo_implied_blue_winrate']
        final_prob = results['blue_win_percentage']
        draft_swing = round(final_prob - elo_prob, 2)

        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric(f"{b_team} Avg Champ WR", f"{d_data['avg_blue_c_wr']}%")
        dm2.metric(f"{r_team} Avg Champ WR", f"{d_data['avg_red_c_wr']}%")
        dm3.metric("Draft Champ Edge", f"{d_data['c_wr_diff']}%", delta=f"{d_data['c_wr_diff']}%")
        dm4.metric(
            "Draft Winrate Swing",
            f"{draft_swing}%",
            delta=f"{draft_swing}%",
            help="Difference between post-draft model prediction and pre-draft Elo implied odds."
        )

        st.markdown("**Role Pick Comparison**")
        champ_rows = []
        for r in results['role_breakdown']:
            b_cwr = round(r['blue_c_wr'] * 100, 1)
            r_cwr = round(r['red_c_wr'] * 100, 1)
            cdiff = round(b_cwr - r_cwr, 1)
            cadv = f"Blue (+{cdiff}%)" if cdiff > 0 else (f"Red ({cdiff}%)" if cdiff < 0 else "Even")

            champ_rows.append({
                "Role": r['role'],
                f"{b_team} Pick": r['blue_champ'],
                "Blue Champ WR": f"{b_cwr}%",
                f"{r_team} Pick": r['red_champ'],
                "Red Champ WR": f"{r_cwr}%",
                "Draft Edge": cadv
            })

        st.table(pd.DataFrame(champ_rows))

    # --- TAB 4: Team H2H & Recent Form ---
    with tab_h2h:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Historical H2H Matches", f"{h2h_data['total_h2h']}")
        h2.metric(f"{b_team} H2H Record", f"{h2h_data['blue_h2h_wins']}W - {h2h_data['red_h2h_wins']}L")
        h3.metric(f"{b_team} Recent Form (Last 10)", f"{h2h_data['blue_recent_wr']}%")
        h4.metric(f"{r_team} Recent Form (Last 10)", f"{h2h_data['red_recent_wr']}%")

        if h2h_data['total_h2h'] > 0:
            st.markdown(f"**Direct Head-to-Head Breakdown ({b_team} vs {r_team})**")
            st.info(
                f"{b_team} holds a **{h2h_data['blue_h2h_wr']}%** winrate across {h2h_data['total_h2h']} historical match(es) against {r_team}.")
        else:
            st.warning("No previous head-to-head matches found in the historical dataset for this exact pairing.")

    # --- TAB 5: Implied Odds & Value Calculator ---
    with tab_odds:
        p_blue = results['blue_win_probability']
        p_red = results['red_win_probability']

        fair_dec_blue = round(1.0 / p_blue, 2) if p_blue > 0 else 0
        fair_dec_red = round(1.0 / p_red, 2) if p_red > 0 else 0
        fair_ame_blue = prob_to_american_odds(p_blue)
        fair_ame_red = prob_to_american_odds(p_red)

        st.markdown("#### 🎯 Fair Model Odds")
        o1, o2 = st.columns(2)

        with o1:
            st.markdown(f"**{b_team} Fair Odds**")
            st.write(f"- Decimal Odds: **{fair_dec_blue}**")
            st.write(f"- American Odds: **{fair_ame_blue}**")

        with o2:
            st.markdown(f"**{r_team} Fair Odds**")
            st.write(f"- Decimal Odds: **{fair_dec_red}**")
            st.write(f"- American Odds: **{fair_ame_red}**")

        st.markdown("---")
        st.markdown("#### 💰 Value / Edge Calculator vs Bookmaker")

        bk1, bk2 = st.columns(2)
        with bk1:
            bm_blue_odds = st.number_input(
                f"{b_team} Bookmaker Odds (Decimal)",
                min_value=1.01,
                max_value=20.0,
                value=float(fair_dec_blue) if fair_dec_blue > 0 else 2.0,
                step=0.05,
                key=f"bm_b_{model_name}"
            )
        with bk2:
            bm_red_odds = st.number_input(
                f"{r_team} Bookmaker Odds (Decimal)",
                min_value=1.01,
                max_value=20.0,
                value=float(fair_dec_red) if fair_dec_red > 0 else 2.0,
                step=0.05,
                key=f"bm_r_{model_name}"
            )

        ev_blue = round(((p_blue * bm_blue_odds) - 1.0) * 100, 2)
        ev_red = round(((p_red * bm_red_odds) - 1.0) * 100, 2)

        val1, val2 = st.columns(2)
        val1.metric(f"{b_team} Expected Value (EV)", f"{ev_blue}%", delta=f"{ev_blue}%")
        val2.metric(f"{r_team} Expected Value (EV)", f"{ev_red}%", delta=f"{ev_red}%")


# --- 4. Render Dynamic Model Selection Tabs & Dashboards ---
if "active_prediction" in st.session_state:
    active_pred = st.session_state["active_prediction"]
    model_results = active_pred["model_results"]
    h2h_data = active_pred["h2h_data"]

    st.markdown("## 🤖 Prediction Engine Selector")

    # Build Top-Level Model Selector Tabs dynamically
    model_names = list(model_results.keys())
    model_tabs = st.tabs([f"📌 {m}" if m != "Even Split" else "⚖️ Even Split" for m in model_names])

    for i, model_name in enumerate(model_names):
        with model_tabs[i]:
            render_model_dashboard(
                model_name=model_name,
                results=model_results[model_name],
                active_pred=active_pred,
                h2h_data=h2h_data
            )