import os
import re
import glob
from datetime import datetime, date
from playwright.sync_api import sync_playwright

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
TARGET_PATH = "../dataset/match/2026_match_data.csv"
PROFILE_DIR = os.path.abspath("./drive_profile")

HEADLESS_MODE = True  # Set to False only if you need to re-login to Google


def is_updated_today(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return datetime.fromtimestamp(os.path.getmtime(path)).date() == date.today()


def is_valid_csv(filepath: str) -> bool:
    """Verifies that the file contains actual CSV data and not an HTML error page."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return False
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(500)
        if "<!DOCTYPE html>" in head or "<html" in head or "Quota exceeded" in head:
            return False
    return True


def cleanup_chrome_locks(profile_dir: str):
    """Removes stale Chromium lock files if a previous run crashed or timed out."""
    if not os.path.exists(profile_dir):
        return
    for lock_name in ["SingletonLock", "SingletonCookie", "SingletonSocket", "LOCK"]:
        for file_path in glob.glob(os.path.join(profile_dir, "**", lock_name), recursive=True):
            try:
                os.remove(file_path)
            except Exception:
                pass


def find_file_with_scroll(page, year_prefix: str, max_scrolls: int = 30):
    """
    Scrolls down Google Drive's file list and checks for target CSV file across inner text & aria labels.
    """
    # Target divs with data-id (skips hidden <script> elements)
    selector = "div[data-id]"

    for _ in range(max_scrolls):
        elements = page.locator(selector).all()
        for el in elements:
            text = el.inner_text()
            aria = el.get_attribute("aria-label") or ""
            combined_text = f"{text} {aria}"

            # Flexible match: year and .csv anywhere in the element's text/label
            if year_prefix in combined_text and ".csv" in combined_text:
                file_id = el.get_attribute("data-id")
                if not file_id or len(file_id) < 10:
                    continue  # Ignore invalid or short internal IDs

                # Parse clean file name
                lines = [line.strip() for line in text.splitlines() if ".csv" in line]
                file_name = lines[0] if lines else "2026_match_data.csv"
                return el, file_id, file_name

        # Scroll down to trigger Google Drive lazy-loading
        page.mouse.wheel(0, 2000)
        page.keyboard.press("PageDown")
        page.wait_for_timeout(800)

    return None, None, None


def download_latest_match_data(target_path: str = TARGET_PATH):
    if is_updated_today(target_path) and is_valid_csv(target_path):
        print(f"[+] Dataset '{target_path}' is already updated today. Skipping.")
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)
    cleanup_chrome_locks(PROFILE_DIR)

    year_prefix = str(datetime.now().year)

    with sync_playwright() as p:
        print("[*] Launching persistent browser session...")
        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=HEADLESS_MODE,
            accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = context.pages[0] if context.pages else context.new_page()

        try:
            print(f"[*] Navigating to folder: {FOLDER_URL}")
            page.goto(FOLDER_URL, wait_until="domcontentloaded")
            page.wait_for_selector("div[data-id]", timeout=30000)

            # Handle login prompt if required
            if not HEADLESS_MODE and ("accounts.google.com" in page.url or page.locator("a:has-text('Sign in'), a:has-text('Zaloguj się')").is_visible()):
                print("[!] Please log into your Google Account in the browser window...")
                page.wait_for_url("https://drive.google.com/**", timeout=120000)
                page.wait_for_selector("div[data-id]", timeout=30000)

            # 1. Locate target file element
            print(f"[*] Searching for '{year_prefix}' dataset CSV...")
            target_element, target_file_id, target_file_name = find_file_with_scroll(page, year_prefix)

            if not target_element:
                raise FileNotFoundError(f"Could not find a .csv file starting with '{year_prefix}' in the folder.")

            print(f"[*] Found target file: '{target_file_name}' (ID: {target_file_id})")

            target_element.scroll_into_view_if_needed()
            page.wait_for_timeout(500)

            # 2. Attempt Direct UI Download
            download_success = False
            print("[*] Attempting direct download from Google Drive...")

            target_element.click(button="right")
            page.wait_for_timeout(1000)

            download_option = page.locator("[role='menuitem']").filter(
                has_text=re.compile(r"^Download$|^Pobierz$", re.IGNORECASE)
            ).first

            if download_option.is_visible():
                try:
                    with page.expect_download(timeout=30000) as download_info:
                        download_option.click()
                    download = download_info.value
                    download.save_as(target_path)
                    download_success = is_valid_csv(target_path)
                except Exception as e:
                    print(f"[!] Direct download attempt failed/timed out: {e}")

            # 3. Quota Bypass Fallback: "Make a copy" to My Drive & Download
            if not download_success:
                print("[*] Triggering Quota Bypass: Creating a personal copy in 'My Drive'...")
                target_element.click(button="right")
                page.wait_for_timeout(1000)

                copy_option = page.locator("[role='menuitem']").filter(
                    has_text=re.compile(r"Make a copy|Utwórz kopię", re.IGNORECASE)
                ).first

                if copy_option.is_visible():
                    copy_option.click()
                    print("[*] Waiting for copy creation...")
                    page.wait_for_timeout(6000)

                    # Navigate to My Drive
                    print("[*] Navigating to My Drive...")
                    page.goto("https://drive.google.com/drive/my-drive", wait_until="domcontentloaded")
                    page.wait_for_selector("div[data-id]", timeout=30000)

                    copy_element, copy_id, copy_name = find_file_with_scroll(page, year_prefix)

                    if copy_element:
                        print(f"[*] Downloading personal copy '{copy_name}' from My Drive...")
                        copy_element.scroll_into_view_if_needed()
                        copy_element.click(button="right")
                        page.wait_for_timeout(1000)

                        dl_opt = page.locator("[role='menuitem']").filter(
                            has_text=re.compile(r"^Download$|^Pobierz$", re.IGNORECASE)
                        ).first

                        with page.expect_download(timeout=60000) as download_info:
                            dl_opt.click()

                        download = download_info.value
                        download.save_as(target_path)
                    else:
                        raise RuntimeError("Created copy was not found in My Drive.")
                else:
                    raise RuntimeError("Could not find 'Make a copy' option in Google Drive menu.")

        finally:
            context.close()

    # 4. Final CSV Verification
    if is_valid_csv(target_path):
        print(f"[✓] Download completed and verified successfully: '{target_path}'")
    else:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError("[!] Downloaded file validation failed.")


if __name__ == "__main__":
    download_latest_match_data()