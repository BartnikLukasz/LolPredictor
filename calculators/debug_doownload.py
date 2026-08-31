from playwright.sync_api import sync_playwright

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


def debug_gdrive_folder():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"[*] Loading folder: {FOLDER_URL}")
        page.goto(FOLDER_URL, wait_until="networkidle")

        # Take a screenshot to inspect visual layout
        page.screenshot(path="folder_view.png")
        print("[✓] Saved visual state to 'folder_view.png'")

        # Extract file names and IDs directly from DOM elements
        items = page.locator("[data-id]").all()
        print(f"\n--- Found {len(items)} items in DOM ---")

        files_found = []
        for item in items:
            file_id = item.get_attribute("data-id")
            text = item.inner_text().replace("\n", " ").strip()
            if file_id and text:
                files_found.append((file_id, text))
                print(f"• ID: {file_id} | Text/Name: {text}")

        if not files_found:
            # Fallback text extraction if data-id attribute names changed
            print("\n--- Fallback: Raw Page Text Items ---")
            raw_text = page.locator("body").inner_text()
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            for line in lines:
                if ".csv" in line.lower() or "2026" in line:
                    print(f"• Found text match: '{line}'")

        browser.close()


if __name__ == "__main__":
    debug_gdrive_folder()