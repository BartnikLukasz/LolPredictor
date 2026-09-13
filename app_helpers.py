import re
from difflib import SequenceMatcher

from bs4 import BeautifulSoup
import requests
import pandas as pd
import copy
import numpy as np

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


def match_team_name(
        scraped_name: str,
        valid_teams: list[str],
        fetched_players: list[str] = None,
        team_rosters: dict = None,
        df_hist: pd.DataFrame = None
) -> str:
    if not scraped_name or not valid_teams:
        return valid_teams[0] if valid_teams else ""

    scraped_clean = scraped_name.strip().lower()

    # 1. Exact or case-insensitive match
    for team in valid_teams:
        if scraped_clean == team.strip().lower():
            return team

    # 2. Player Roster Overlap (Best for rebrands like SKT -> T1)
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

            # If 2 or more roster players match, prioritize this team entity
            if best_roster_match and max_overlap >= 2:
                return best_roster_match

    # 3. Recency-weighted fuzzy matching via df_hist
    candidate_scores = []
    for team in valid_teams:
        sim_score = SequenceMatcher(None, scraped_clean, team.lower()).ratio()

        # Determine latest match date in df_hist if column exists
        latest_date = pd.Timestamp.min
        if df_hist is not None and ('date' in df_hist.columns or 'date_utc' in df_hist.columns):
            date_col = 'date' if 'date' in df_hist.columns else 'date_utc'
            team_matches = df_hist[(df_hist['team_blue'] == team) | (df_hist['team_red'] == team)]
            if not team_matches.empty:
                latest_date = pd.to_datetime(team_matches[date_col]).max()

        candidate_scores.append((team, sim_score, latest_date))

    # Sort primarily by similarity score, secondarily by most recent match date
    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidate_scores[0][0]


def match_champion_name(scraped_name: str, valid_champions: list) -> str:
    if not scraped_name or not valid_champions:
        return valid_champions[0] if valid_champions else ""
    scraped_clean = re.sub(r'[^a-zA-Z0-9]', '', scraped_name).lower()
    for champ in valid_champions:
        champ_clean = re.sub(r'[^a-zA-Z0-9]', '', champ).lower()
        if scraped_clean == champ_clean:
            return champ
    return valid_champions[0]

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


def create_ensemble_result(model_results_dict: dict) -> dict:
    single_models = [res for key, res in model_results_dict.items() if key != "Even Split"]
    avg_blue_prob = sum(res['blue_win_probability'] for res in single_models) / len(single_models)
    ensemble_res = copy.deepcopy(single_models[0])
    ensemble_res['blue_win_probability'] = round(avg_blue_prob, 4)
    ensemble_res['red_win_probability'] = round(1.0 - avg_blue_prob, 4)
    ensemble_res['blue_win_percentage'] = round(avg_blue_prob * 100, 1)
    ensemble_res['red_win_percentage'] = round((1.0 - avg_blue_prob) * 100, 1)
    return ensemble_res

def compute_db_model_weights(tracking_data: dict, model_names: list) -> tuple[dict, dict]:
    """Calculates model weights dynamically on page load based on historical recorded accuracy in Redis DB."""
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
    """Combines predictions using normalized accuracy weights from DB."""
    base_results = {k: v for k, v in all_model_results.items() if k in model_weights}
    if not base_results:
        return next(iter(all_model_results.values()))

    w_sum = sum(model_weights[k] for k in base_results.keys())
    norm_weights = {k: (model_weights[k] / w_sum if w_sum > 0 else 1.0 / len(base_results)) for k in base_results.keys()}

    w_p_blue = sum(norm_weights[m] * base_results[m]['blue_win_probability'] for m in base_results)
    w_p_red = 1.0 - w_p_blue

    w_player_swing = sum(norm_weights[m] * base_results[m].get('draft_swings', {}).get('player_swing', 0.0) for m in base_results)
    w_draft_swing = sum(norm_weights[m] * base_results[m].get('draft_swings', {}).get('draft_swing', 0.0) for m in base_results)

    first_res = next(iter(base_results.values()))
    elo_base = first_res.get('elo_metrics', {}).get('elo_implied_blue_winrate', 50.0)

    final_pct = round(w_p_blue * 100, 2)
    player_pct = round(elo_base + w_player_swing, 2)

    progression_data = pd.DataFrame({
        "Stage": ["1. Elo Baseline", "2. Player Mastery Impact", "3. Champion Draft Impact", "4. Final Prediction"],
        "Win %": [elo_base, player_pct, final_pct, final_pct],
        "Impact Delta": [0.0, round(w_player_swing, 2), round(w_draft_swing, 2), 0.0]
    })

    res = copy.deepcopy(first_res)
    res['blue_win_probability'] = w_p_blue
    res['red_win_probability'] = w_p_red
    res['blue_win_percentage'] = final_pct
    res['red_win_percentage'] = round(w_p_red * 100, 2)
    res['progression_data'] = progression_data
    res['draft_swings'] = {
        'player_swing': round(w_player_swing, 2),
        'draft_swing': round(w_draft_swing, 2),
        'total_swing': round(final_pct - elo_base, 2)
    }
    res['weights_used'] = norm_weights
    return res