import os
import re
import shutil
import tempfile
from datetime import datetime, date
import requests
import gdown

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
TARGET_PATH = "../dataset/match/2026_match_data.csv"


def is_updated_today(path: str) -> bool:
    """Check if the local file exists and was last modified today."""
    if not os.path.exists(path):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(path)).date()
    return file_mtime == date.today()


def get_drive_file_id_by_prefix(folder_id: str, prefix: str):
    """Scrapes the public GDrive folder HTML to quickly locate the single file ID matching the target prefix."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(f"https://drive.google.com/drive/folders/{folder_id}", headers=headers, timeout=10)
        # Search for embedded GDrive JSON pattern [ "FILE_ID", "2026...csv" ]
        pattern = rf'\["([a-zA-Z0-9_-]{{25,}})","({prefix}[^"]*\.csv)"'
        matches = re.findall(pattern, res.text)
        if matches:
            return matches[0][0], matches[0][1]  # Returns (file_id, file_name)
    except Exception as e:
        print(f"[!] Fast lookup failed ({e}), falling back to full folder download...")
    return None, None


def download_latest_match_data(target_path: str = TARGET_PATH):
    current_year = datetime.now().year
    year_prefix = str(current_year)

    # 1. Daily check: skip download if updated today
    if is_updated_today(target_path):
        mtime_str = datetime.fromtimestamp(os.path.getmtime(target_path)).strftime("%Y-%m-%d %H:%M")
        print(f"[+] Dataset '{target_path}' is up to date (last updated today at {mtime_str}). Skipping download.")
        return

    print(f"[*] Local dataset missing or outdated. Syncing {current_year} match data...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # 2. Fast Path: Download only the matching 2026 CSV directly
    file_id, file_name = get_drive_file_id_by_prefix(FOLDER_ID, year_prefix)
    if file_id:
        print(f"[*] Found remote file: '{file_name}' (ID: {file_id})")
        gdown.download(id=file_id, output=target_path, quiet=False)
        print(f"[✓] Successfully downloaded and updated '{target_path}'.")
        return

    # 3. Fallback Path: Fetch folder contents via temp dir
    print("[*] Running fallback folder download...")
    with tempfile.TemporaryDirectory() as temp_dir:
        gdown.download_folder(url=FOLDER_URL, output=temp_dir, quiet=False)
        matching_files = [
            f for f in os.listdir(temp_dir)
            if f.startswith(year_prefix) and f.endswith(".csv")
        ]
        if not matching_files:
            raise FileNotFoundError(f"No CSV file starting with '{year_prefix}' found in folder.")

        matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)
        newest_file = matching_files[0]
        shutil.copy2(os.path.join(temp_dir, newest_file), target_path)
        print(f"[✓] Successfully updated '{target_path}' with '{newest_file}'.")


if __name__ == "__main__":
    download_latest_match_data()