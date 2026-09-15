import streamlit as st

from app_helpers import (
    check_is_admin,
    load_tracking_data,
    save_tracking_data,
    load_predictor_assets,
    send_odds_to_endpoint,
    update_blue_roster_callback,
    update_red_roster_callback,
    swap_sides_callback,
    fetch_golgg_draft,
    match_team_name,
    match_champion_name,
    compute_model_accuracies,
    get_historical_team_metrics,
    compute_db_model_weights,
    create_weighted_ensemble_result,
    render_model_dashboard,
)

st.set_page_config(page_title="LoL Match Predictor", layout="wide")

# Asset Loading & Auth State
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

st.markdown("---")

# --- TEAM & MATCH SETUP ---
st.subheader("⚔️ Team & Match Setup")
col_blue, col_swap, col_red = st.columns([4, 1, 4])

valid_teams = list(team_rosters.keys())

with col_blue:
    blue_team = st.selectbox(
        "🔵 Blue Side Team",
        options=valid_teams,
        key="blue_team_select",
        on_change=update_blue_roster_callback,
        args=(team_rosters,)
    )

with col_swap:
    st.write("")
    st.write("")
    st.button("🔄 Swap", on_click=swap_sides_callback, use_container_width=True)

with col_red:
    red_team = st.selectbox(
        "🔴 Red Side Team",
        options=valid_teams,
        key="red_team_select",
        on_change=update_red_roster_callback,
        args=(team_rosters,)
    )

col_fp, col_patch, col_gnum, col_lead = st.columns(4)
with col_fp:
    first_pick = st.radio("First Pick", ["Blue", "Red"], horizontal=True, key="first_pick_radio")
with col_patch:
    patch_ver = st.text_input("Patch Version", value="14.10", key="patch_input")
with col_gnum:
    game_number = st.number_input("Game # in Series", min_value=1, max_value=7, value=1)
with col_lead:
    blue_series_lead = st.number_input("Blue Series Lead (+/-)", value=0)

st.markdown("---")

# --- DRAFT GRID (5 ROLES) ---
st.subheader("🛡️ Draft & Player Selection")

roles = ["TOP", "JGL", "MID", "BOT", "SUP"]
blue_champs, red_champs = [], []
blue_players, red_players = [], []

default_blue_roster = team_rosters.get(blue_team, ["", "", "", "", ""])
default_red_roster = team_rosters.get(red_team, ["", "", "", "", ""])

for i, role in enumerate(roles):
    r_col1, r_col2, r_label, r_col3, r_col4 = st.columns([3, 3, 1, 3, 3])

    with r_label:
        st.markdown(f"<h5 style='text-align: center; margin-top: 28px;'>{role}</h5>", unsafe_allow_html=True)

    with r_col1:
        bc = st.selectbox(f"Blue {role} Champ", champion_list, key=f"bc_{i}", label_visibility="collapsed")
        blue_champs.append(bc)

    with r_col2:
        bp = st.text_input(f"Blue {role} Player", value=default_blue_roster[i] if i < len(default_blue_roster) else "", key=f"bp_{i}", label_visibility="collapsed")
        blue_players.append(bp)

    with r_col3:
        rc = st.selectbox(f"Red {role} Champ", champion_list, key=f"rc_{i}", label_visibility="collapsed")
        red_champs.append(rc)

    with r_col4:
        rp = st.text_input(f"Red {role} Player", value=default_red_roster[i] if i < len(default_red_roster) else "", key=f"rp_{i}", label_visibility="collapsed")
        red_players.append(rp)

st.markdown("---")

# --- PREDICTION TRIGGER ---
if st.button("🚀 Predict Match Outcome", type="primary", use_container_width=True):
    draft_payload = {
        "blue_team": blue_team,
        "red_team": red_team,
        "patch": patch_ver,
        "blue_firstpick": 1 if first_pick == "Blue" else 0,
        "game_number": game_number,
        "blue_series_lead": blue_series_lead,
        "blue_prev_win": 1 if blue_series_lead > 0 else 0,
        "blue_players": blue_players,
        "red_players": red_players,
        "blue_champs": blue_champs,
        "red_champs": red_champs,
    }

    model_results = {}
    for m_name, eng in engines.items():
        model_results[m_name] = eng.predict_match(draft_payload)

    weights, accuracies = compute_db_model_weights(tracking_data, list(engines.keys()))
    weighted_res = create_weighted_ensemble_result(model_results, weights)
    model_results["Weighted Split"] = weighted_res

    h2h_data = get_historical_team_metrics(df_hist, blue_team, red_team)

    st.session_state["active_prediction"] = {
        "blue_team": blue_team,
        "red_team": red_team,
        "model_results": model_results,
        "model_accuracies": accuracies,
        "h2h_data": h2h_data,
    }

    p_blue = weighted_res["blue_win_probability"]
    p_red = weighted_res["red_win_probability"]
    send_odds_to_endpoint(blue_team, red_team, p_blue, p_red)

# --- DASHBOARD TAB RENDERING ---
if "active_prediction" in st.session_state:
    active_pred = st.session_state["active_prediction"]
    model_names = list(active_pred["model_results"].keys())

    st.markdown("---")
    tabs = st.tabs(model_names)

    for idx, m_name in enumerate(model_names):
        with tabs[idx]:
            render_model_dashboard(
                model_name=m_name,
                results=active_pred["model_results"][m_name],
                active_pred=active_pred,
                h2h_data=active_pred["h2h_data"],
                is_admin=is_admin,
            )