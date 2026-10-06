import gzip
import io
import os
import shutil
from datetime import date, datetime
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"

# This file lives in <project root>/calculators/, so paths no longer depend on the working directory.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MATCH_DIR = PROJECT_ROOT / "dataset" / "match"

# Full "drive" scope is needed so we can copy a file we don't own (quota fallback)
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "credentials.json"  # OAuth client (Desktop app) from Google Cloud; only needed to sign in
TOKEN_FILE = "token.json"              # created automatically on first sign-in


def _find_file(name: str) -> Path:
    """Looks in the working directory, next to this script, then in the project root."""
    for base in (Path.cwd(), HERE, PROJECT_ROOT):
        candidate = base / name
        if candidate.exists():
            return candidate
    return Path.cwd() / name  # where a new file will be created


def _default_target(year: int) -> str:
    return str(MATCH_DIR / f"{year}_match_data.csv.gz")


def is_updated_today(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return datetime.fromtimestamp(os.path.getmtime(path)).date() == date.today()


def is_valid_csv(filepath: str) -> bool:
    """True for a plausible CSV. For '.gz' paths the file must also be genuinely gzip-compressed."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return False
    try:
        if str(filepath).lower().endswith(".gz"):
            with open(filepath, "rb") as fb:
                if fb.read(2) != b"\x1f\x8b":      # gzip magic bytes; a renamed plain CSV fails here
                    return False
            f = gzip.open(filepath, "rt", encoding="utf-8", errors="ignore")
        else:
            f = open(filepath, "r", encoding="utf-8", errors="ignore")
        with f:
            head = f.read(500).lower()
    except (OSError, EOFError):
        return False
    return "<!doctype html" not in head and "<html" not in head and "quota exceeded" not in head


def _gzip_file(src: str, dst: str):
    """Compresses src into dst (gzip)."""
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)


def _interactive_login() -> Credentials:
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        raise RuntimeError(
            "No usable Google token on this runner (token.json missing, invalid or expired). "
            "Sign in locally to create a fresh token.json and update the GOOGLE_TOKEN_JSON secret."
        )
    creds_path = _find_file(CREDENTIALS_FILE)
    if not creds_path.exists():
        raise FileNotFoundError(
            f"'{CREDENTIALS_FILE}' not found (looked in {Path.cwd()}, {HERE} and {PROJECT_ROOT}).\n"
            f"It is needed only to sign in when there is no valid {TOKEN_FILE}. To create it: Google Cloud Console "
            f"-> Google Auth Platform -> Clients -> Create client -> Application type 'Desktop app' -> "
            f"download the JSON and save it as {CREDENTIALS_FILE}."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    return flow.run_local_server(port=0)  # opens browser once


def get_service():
    token_path = _find_file(TOKEN_FILE)
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                raise RuntimeError(
                    f"Google rejected the saved token ({e}). It was revoked or expired (apps left in 'Testing' "
                    f"expire after 7 days). Delete {token_path}, sign in again locally and update the secret."
                ) from e
        else:
            creds = _interactive_login()
        token_path.write_text(creds.to_json())
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


def download_latest_match_data(target_path: str = None):
    """
    Downloads the newest '<year>*.csv' from the Drive folder.
    By default it is saved as <project root>/dataset/match/<year>_match_data.csv.gz (compressed after download), where <year> is the year of
    the file that was found (so a new year can never overwrite the previous year's data).
    """
    this_year = datetime.now().year
    skip_path = target_path or _default_target(this_year)
    if is_updated_today(skip_path) and is_valid_csv(skip_path):
        print(f"[+] '{skip_path}' already updated today. Skipping.")
        return

    service = get_service()

    target, found_year = None, None
    for year in (this_year, this_year - 1):  # early in January the new year's file may not exist yet
        try:
            target = find_latest_file(service, FOLDER_ID, str(year))
            found_year = year
            break
        except FileNotFoundError:
            continue
    if target is None:
        raise FileNotFoundError(f"No .csv starting with '{this_year}' or '{this_year - 1}' in the Drive folder.")

    target_path = target_path or _default_target(found_year)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + ".raw.part"   # plain CSV exactly as downloaded from Drive
    gz_tmp_path = target_path + ".part"    # compressed result, moved into place atomically
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
        if target_path.lower().endswith(".gz"):
            _gzip_file(tmp_path, gz_tmp_path)
            os.replace(gz_tmp_path, target_path)   # atomic: never leaves a half-written target
        else:
            os.replace(tmp_path, target_path)
        print(f"[✓] Saved to '{target_path}'")
    finally:
        for leftover in (tmp_path, gz_tmp_path):
            if os.path.exists(leftover):
                os.remove(leftover)
        if copy_id:
            try:
                service.files().delete(fileId=copy_id, supportsAllDrives=True).execute()
                print("[*] Temporary copy deleted.")
            except HttpError as e:
                print(f"[!] Could not delete temporary copy ({copy_id}): {e}")


if __name__ == "__main__":
    download_latest_match_data()