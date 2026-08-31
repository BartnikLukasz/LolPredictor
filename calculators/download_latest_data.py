import os
from datetime import datetime, date
from playwright.sync_api import sync_playwright

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
TARGET_PATH = "../dataset/match/2026_match_data.csv"


def is_updated_today(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return datetime.fromtimestamp(os.path.getmtime(path)).date() == date.today()


def download_latest_match_data(target_path: str = TARGET_PATH):
    if is_updated_today(target_path):
        print(f"[+] Dataset '{target_path}' is already updated today. Skipping.")
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    year_prefix = str(datetime.now().year)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        print(f"[*] Navigating to folder: {FOLDER_URL}")
        page.goto(FOLDER_URL, wait_until="networkidle")

        # 1. Extract target file ID directly from rendered DOM
        target_file_id = None
        target_file_name = None

        elements = page.locator("[data-id]").all()
        for el in elements:
            text = el.inner_text()
            if year_prefix in text and text.endswith(".csv"):
                target_file_id = el.get_attribute("data-id")
                target_file_name = text.strip()
                break

        if not target_file_id:
            raise FileNotFoundError(
                f"Could not find a .csv file starting with '{year_prefix}' in the folder."
            )

        print(f"[*] Found target file: '{target_file_name}' (ID: {target_file_id})")

        # 2. Trigger Direct Endpoint Download
        download_url = f"https://drive.google.com/uc?export=download&id={target_file_id}"

        try:
            with page.expect_download(timeout=30000) as download_info:
                try:
                    page.goto(download_url)
                except Exception as goto_err:
                    # Catch and suppress Playwright's expected download navigation interrupt
                    if "Download is starting" not in str(goto_err):
                        raise goto_err

            download = download_info.value
            download.save_as(target_path)
            print(f"[✓] Download completed successfully: '{target_path}'")

        except Exception as err:
            print(f"[!] Direct download fallback triggered: {err}")
            # Fallback: Click the row's native download action button shown in screenshot
            row = page.locator(f"[data-id='{target_file_id}']").first
            row.hover()

            with page.expect_download(timeout=30000) as download_info:
                page.locator("div[role='button']:has-text('Download'), [aria-label*='Download']").last.click()

            download = download_info.value
            download.save_as(target_path)
            print(f"[✓] Download completed via row UI: '{target_path}'")

        browser.close()


if __name__ == "__main__":
    download_latest_match_data()