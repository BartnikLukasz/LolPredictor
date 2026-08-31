import re
from bs4 import BeautifulSoup
import requests
import pandas as pd


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