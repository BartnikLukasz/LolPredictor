import json
import os
import pandas as pd
import streamlit as st
from live_feature_engine import LiveFeatureEngine

st.set_page_config(page_title="LoL Match Predictor", layout="wide")


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


engine, team_rosters, champion_list = load_predictor_assets()

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

# --- 3. Calculation & Visual Breakdown Metrics ---
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

    # Top Level Win Probability
    res_b, res_r = st.columns(2)
    res_b.metric(f"{blue_team} Win Probability", f"{results['blue_win_percentage']}%")
    res_r.metric(f"{red_team} Win Probability", f"{results['red_win_percentage']}%")
    st.progress(results['blue_win_probability'])

    st.markdown("### 📊 Model Feature Breakdown")

    tab_elo, tab_players, tab_draft = st.tabs([
        "⚡ Elo & Series Context",
        "👤 Player Win Rate Head-to-Head",
        "⚔️ Draft & Champion Factors"
    ])

    # --- TAB 1: Elo Rating & Series Context Impact ---
    with tab_elo:
        elo_data = results['elo_metrics']
        s_data = results['series_metrics']

        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"{blue_team} Elo", f"{elo_data['blue_elo']}")
        e2.metric(f"{red_team} Elo", f"{elo_data['red_elo']}")
        e3.metric(
            "Effective Elo Diff (inc. First Pick)",
            f"{elo_data['elo_diff']}",
            delta=f"{elo_data['elo_diff']} pts"
        )
        e4.metric(
            "Elo-Implied Blue Winrate",
            f"{elo_data['elo_implied_blue_winrate']}%"
        )

        st.markdown("**Series Context Inputs**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Game Number", f"Game {s_data['game_number']}")
        sc2.metric(f"{blue_team} Series Lead", f"{s_data['blue_series_lead']} games")
        sc3.metric(f"{blue_team} Previous Game Win", "Yes" if s_data['blue_prev_win'] == 1 else ("No" if s_data['game_number'] > 1 else "N/A"))

        st.caption("The XGBoost model incorporates raw team Elo, first-pick advantage, and series state (momentum and match number) to calculate the baseline probability before draft features.")

    # --- TAB 2: Player Win Rate Head-to-Head ---
    with tab_players:
        p_data = results['player_metrics']
        pm1, pm2, pm3 = st.columns(3)

        pm1.metric(f"{blue_team} Avg Player Winrate", f"{p_data['avg_blue_p_wr']}%")
        pm2.metric(f"{red_team} Avg Player Winrate", f"{p_data['avg_red_p_wr']}%")
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
                f"{blue_team} Player": r['blue_player'],
                "Blue WR": f"{b_wr}% ({r['blue_p_games']}g)",
                f"{red_team} Player": r['red_player'],
                "Red WR": f"{r_wr}% ({r['red_p_games']}g)",
                "Advantage": adv
            })

        st.table(pd.DataFrame(player_rows))

    # --- TAB 3: Draft & Champion Factors ---
    with tab_draft:
        d_data = results['draft_metrics']
        dm1, dm2, dm3 = st.columns(3)

        dm1.metric(f"{blue_team} Draft Avg Champ WR", f"{d_data['avg_blue_c_wr']}%")
        dm2.metric(f"{red_team} Draft Avg Champ WR", f"{d_data['avg_red_c_wr']}%")
        dm3.metric("Draft Champ WR Advantage", f"{d_data['c_wr_diff']}%", delta=f"{d_data['c_wr_diff']}%")

        st.markdown("**Champion Pick Breakdown**")

        champ_rows = []
        for r in results['role_breakdown']:
            b_cwr = round(r['blue_c_wr'] * 100, 1)
            r_cwr = round(r['red_c_wr'] * 100, 1)
            cdiff = round(b_cwr - r_cwr, 1)
            cadv = f"Blue (+{cdiff}%)" if cdiff > 0 else (f"Red ({cdiff}%)" if cdiff < 0 else "Even")

            champ_rows.append({
                "Role": r['role'],
                f"{blue_team} Champion": r['blue_champ'],
                "Blue Champ WR": f"{b_cwr}%",
                f"{red_team} Champion": r['red_champ'],
                "Red Champ WR": f"{r_cwr}%",
                "Draft Edge": cadv
            })

        st.table(pd.DataFrame(champ_rows))