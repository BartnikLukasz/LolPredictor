import json
import os
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
from live_feature_engine import LiveFeatureEngine

st.set_page_config(page_title="LoL Match Predictor", layout="wide")

ODDS_ENDPOINT_URL = "http://127.0.0.1:5000/odds"
TRACKING_FILE = "live_accuracy_tracking.json"


# --- TRACKING FILE HELPER FUNCTIONS ---
def load_tracking_data(filepath: str = TRACKING_FILE) -> dict:
    """Loads tracking data from JSON, or initializes structure if file doesn't exist."""
    if not os.path.exists(filepath):
        return {
            "total_games": 0,
            "correct_predictions": 0,
            "logs": []
        }
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "total_games": 0,
            "correct_predictions": 0,
            "logs": []
        }


def save_tracking_data(data: dict, filepath: str = TRACKING_FILE):
    """Saves updated tracking dictionary back to the single JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


@st.cache_resource
def load_predictor_assets():
    dataset_path = "dataset/pregame/pregame_dataset_final_features.csv"
    engine_obj = LiveFeatureEngine(dataset_path=dataset_path)

    with open("team_rosters.json", "r") as f:
        roster_data = json.load(f)

    champ_cols = [c for c in engine_obj.df_hist.columns if 'champion' in c]
    champions_set = set()
    for col in champ_cols:
        champions_set.update(engine_obj.df_hist[col].dropna().unique().tolist())

    champions_list = sorted(list(champions_set)) if champions_set else ["Ahri", "Aatrox", "Azir"]

    return engine_obj, roster_data, champions_list


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

    blue_matches = df_hist[(df_hist['blue_team'] == blue_team) | (df_hist['red_team'] == blue_team)].sort_values('date', ascending=False).head(10)
    red_matches = df_hist[(df_hist['blue_team'] == red_team) | (df_hist['red_team'] == red_team)].sort_values('date', ascending=False).head(10)

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


engine, team_rosters, champion_list = load_predictor_assets()

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

    results = engine.predict_match(draft_payload)
    h2h_data = get_historical_team_metrics(engine.df_hist, blue_team, red_team)

    # Automatically post odds to endpoint
    send_odds_to_endpoint(
        blue_team=blue_team,
        red_team=red_team,
        p_blue=results['blue_win_probability'],
        p_red=results['red_win_probability']
    )

    # Store calculation session state so outcome can be logged without losing prediction UI
    st.session_state["active_prediction"] = {
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_win_probability": results['blue_win_probability'],
        "red_win_probability": results['red_win_probability'],
        "predicted_winner": blue_team if results['blue_win_probability'] >= 0.5 else red_team,
        "results": results,
        "h2h_data": h2h_data
    }

# --- 4. Render Prediction Results & Live Match Result Logger ---
if "active_prediction" in st.session_state:
    active_pred = st.session_state["active_prediction"]
    results = active_pred["results"]
    h2h_data = active_pred["h2h_data"]

    # Top Level Win Probability Header
    res_b, res_r = st.columns(2)
    res_b.metric(f"{active_pred['blue_team']} Win Probability", f"{results['blue_win_percentage']}%")
    res_r.metric(f"{active_pred['red_team']} Win Probability", f"{results['red_win_percentage']}%")
    st.progress(results['blue_win_probability'])

    # --- LIVE GAME OUTCOME LOGGING CONTAINER ---
    with st.container(border=True):
        st.subheader("📝 Record Live Game Result")
        st.write(f"Model Predicted Winner: **{active_pred['predicted_winner']}**")

        act_col1, act_col2 = st.columns([3, 1])

        with act_col1:
            actual_winner = st.radio(
                "Select Actual Game Winner:",
                options=[active_pred['blue_team'], active_pred['red_team']],
                horizontal=True,
                key="actual_winner_radio"
            )

        with act_col2:
            st.write("")  # Spacing alignment
            if st.button("Save & Log Result", type="secondary", use_container_width=True):
                is_correct = (actual_winner == active_pred['predicted_winner'])

                current_track_data = load_tracking_data()

                # 1. Update counters
                current_track_data["total_games"] = current_track_data.get("total_games", 0) + 1
                if is_correct:
                    current_track_data["correct_predictions"] = current_track_data.get("correct_predictions", 0) + 1

                # 2. Append match log entry
                log_entry = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blue_team": active_pred['blue_team'],
                    "red_team": active_pred['red_team'],
                    "blue_win_probability": active_pred['blue_win_probability'],
                    "red_win_probability": active_pred['red_win_probability'],
                    "predicted_winner": active_pred['predicted_winner'],
                    "actual_winner": actual_winner,
                    "is_correct": is_correct
                }
                current_track_data.setdefault("logs", []).append(log_entry)

                # 3. Save to single file
                save_tracking_data(current_track_data)

                st.toast(
                    f"Result Logged! Winner: {actual_winner} ({'Correct ✅' if is_correct else 'Incorrect ❌'})",
                    icon="🎯"
                )

                # Clear prediction state and reload sidebar metrics
                del st.session_state["active_prediction"]
                st.rerun()

    st.markdown("### 📊 Model Feature & Match Analysis")

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
        e1.metric(f"{active_pred['blue_team']} Elo", f"{elo_data['blue_elo']}")
        e2.metric(f"{active_pred['red_team']} Elo", f"{elo_data['red_elo']}")
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
        sc2.metric(f"{active_pred['blue_team']} Series Lead", f"{s_data['blue_series_lead']} games")
        sc3.metric(f"{active_pred['blue_team']} Previous Game Win", "Yes" if s_data['blue_prev_win'] == 1 else ("No" if s_data['game_number'] > 1 else "N/A"))

        st.caption("The XGBoost model incorporates raw team Elo, first-pick advantage, and series state (momentum and match number) to calculate the baseline probability before draft features.")

    # --- TAB 2: Player Win Rate Head-to-Head ---
    with tab_players:
        p_data = results['player_metrics']
        pm1, pm2, pm3 = st.columns(3)

        pm1.metric(f"{active_pred['blue_team']} Avg Player Winrate", f"{p_data['avg_blue_p_wr']}%")
        pm2.metric(f"{active_pred['red_team']} Avg Player Winrate", f"{p_data['avg_red_p_wr']}%")
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
                f"{active_pred['blue_team']} Player": r['blue_player'],
                "Blue WR": f"{b_wr}% ({r['blue_p_games']}g)",
                f"{active_pred['red_team']} Player": r['red_player'],
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
        dm1.metric(f"{active_pred['blue_team']} Avg Champ WR", f"{d_data['avg_blue_c_wr']}%")
        dm2.metric(f"{active_pred['red_team']} Avg Champ WR", f"{d_data['avg_red_c_wr']}%")
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
                f"{active_pred['blue_team']} Pick": r['blue_champ'],
                "Blue Champ WR": f"{b_cwr}%",
                f"{active_pred['red_team']} Pick": r['red_champ'],
                "Red Champ WR": f"{r_cwr}%",
                "Draft Edge": cadv
            })

        st.table(pd.DataFrame(champ_rows))

    # --- TAB 4: Team H2H & Recent Form ---
    with tab_h2h:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Historical H2H Matches", f"{h2h_data['total_h2h']}")
        h2.metric(f"{active_pred['blue_team']} H2H Record", f"{h2h_data['blue_h2h_wins']}W - {h2h_data['red_h2h_wins']}L")
        h3.metric(f"{active_pred['blue_team']} Recent Form (Last 10)", f"{h2h_data['blue_recent_wr']}%")
        h4.metric(f"{active_pred['red_team']} Recent Form (Last 10)", f"{h2h_data['red_recent_wr']}%")

        if h2h_data['total_h2h'] > 0:
            st.markdown(f"**Direct Head-to-Head Breakdown ({active_pred['blue_team']} vs {active_pred['red_team']})**")
            st.info(f"{active_pred['blue_team']} holds a **{h2h_data['blue_h2h_wr']}%** winrate across {h2h_data['total_h2h']} historical match(es) against {active_pred['red_team']}.")
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
            st.markdown(f"**{active_pred['blue_team']} Fair Odds**")
            st.write(f"- Decimal Odds: **{fair_dec_blue}**")
            st.write(f"- American Odds: **{fair_ame_blue}**")

        with o2:
            st.markdown(f"**{active_pred['red_team']} Fair Odds**")
            st.write(f"- Decimal Odds: **{fair_dec_red}**")
            st.write(f"- American Odds: **{fair_ame_red}**")

        st.markdown("---")
        st.markdown("#### 💰 Value / Edge Calculator vs Bookmaker")

        bk1, bk2 = st.columns(2)
        with bk1:
            bm_blue_odds = st.number_input(
                f"{active_pred['blue_team']} Bookmaker Odds (Decimal)",
                min_value=1.01,
                max_value=20.0,
                value=float(fair_dec_blue) if fair_dec_blue > 0 else 2.0,
                step=0.05
            )
        with bk2:
            bm_red_odds = st.number_input(
                f"{active_pred['red_team']} Bookmaker Odds (Decimal)",
                min_value=1.01,
                max_value=20.0,
                value=float(fair_dec_red) if fair_dec_red > 0 else 2.0,
                step=0.05
            )

        ev_blue = round(((p_blue * bm_blue_odds) - 1.0) * 100, 2)
        ev_red = round(((p_red * bm_red_odds) - 1.0) * 100, 2)

        val1, val2 = st.columns(2)
        val1.metric(f"{active_pred['blue_team']} Expected Value (EV)", f"{ev_blue}%", delta=f"{ev_blue}%")
        val2.metric(f"{active_pred['red_team']} Expected Value (EV)", f"{ev_red}%", delta=f"{ev_red}%")