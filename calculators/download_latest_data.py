import io
import os
from datetime import date, datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
TARGET_PATH = "../dataset/match/2026_match_data.csv"

# Full "drive" scope is needed so we can copy a file we don't own (quota fallback)
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "credentials.json"  # OAuth client (Desktop app) from Google Cloud
TOKEN_FILE = "token.json"              # created automatically on first run


def is_updated_today(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return datetime.fromtimestamp(os.path.getmtime(path)).date() == date.today()


def is_valid_csv(filepath: str) -> bool:
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return False
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(500).lower()
    return "<!doctype html" not in head and "<html" not in head and "quota exceeded" not in head


def get_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # opens browser once
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_latest_file(service, folder_id: str, year_prefix: str) -> dict:
    query = (
        f"'{folder_id}' in parents and trashed = false "
        f"and name contains '{year_prefix}'"
    )
    files, page_token = [], None
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, modifiedTime, size)",
            pageSize=100,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # "name contains" matches anywhere, so enforce prefix + .csv ourselves
    candidates = [
        f for f in files
        if f["name"].startswith(year_prefix) and f["name"].lower().endswith(".csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"No .csv starting with '{year_prefix}' in the folder.")
    return max(candidates, key=lambda f: f["modifiedTime"])


def download_file(service, file_id: str, dest: str):
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def is_quota_error(err: HttpError) -> bool:
    return err.resp.status == 403 and (
        b"ownloadQuotaExceeded" in err.content or b"uota" in err.content
    )


def download_latest_match_data(target_path: str = TARGET_PATH):
    if is_updated_today(target_path) and is_valid_csv(target_path):
        print(f"[+] '{target_path}' already updated today. Skipping.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + ".part"
    year_prefix = str(datetime.now().year)

    service = get_service()
    target = find_latest_file(service, FOLDER_ID, year_prefix)
    print(f"[*] Found: {target['name']} (modified {target['modifiedTime']})")

    copy_id = None
    try:
        try:
            print("[*] Downloading directly...")
            download_file(service, target["id"], tmp_path)
        except HttpError as e:
            if not is_quota_error(e):
                raise
            print("[!] Download quota exceeded. Copying to My Drive instead...")
            copy = service.files().copy(
                fileId=target["id"],
                body={"name": f"tmp_{target['name']}", "parents": ["root"]},
                supportsAllDrives=True,
            ).execute()
            copy_id = copy["id"]
            download_file(service, copy_id, tmp_path)

        if not is_valid_csv(tmp_path):
            raise RuntimeError("Downloaded file is not a valid CSV.")
        os.replace(tmp_path, target_path)  # atomic: never leaves a half-written target
        print(f"[✓] Saved to '{target_path}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if copy_id:
            try:
                service.files().delete(fileId=copy_id, supportsAllDrives=True).execute()
                print("[*] Temporary copy deleted.")
            except HttpError as e:
                print(f"[!] Could not delete temporary copy ({copy_id}): {e}")


if __name__ == "__main__":
    download_latest_match_data()