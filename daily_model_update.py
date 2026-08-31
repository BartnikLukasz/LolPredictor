import os
import subprocess
import sys
from datetime import datetime

# Configure the paths you want to track and stage in Git
# Update these paths to match your model output directory or specific dataset files
PATHS_TO_STAGE = ["models/"]


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Helper to run system commands and capture output cleanly."""
    print(f"[*] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def run_pipeline_and_push():
    # 1. Run main execution pipeline
    print("[*] Starting main.py pipeline...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("[✓] main.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[!] main.py failed with exit code {e.returncode}. Aborting Git push.")
        sys.exit(1)

    # 2. Stage updated files
    print("[*] Staging updated model and data files...")
    for path in PATHS_TO_STAGE:
        if os.path.exists(path):
            run_cmd(["git", "add", path], check=False)
        else:
            print(f"[!] Warning: Path '{path}' does not exist. Skipping stage.")

    # 3. Check if there are actual changes staged to commit
    status_res = run_cmd(["git", "status", "--porcelain"], check=False)
    if not status_res.stdout.strip():
        print("[+] No model or dataset changes detected. Skipping Git commit and push.")
        return

    # 4. Commit changes with a timestamped message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"auto: update trained models and dataset ({timestamp})"

    print(f"[*] Committing changes: '{commit_msg}'")
    run_cmd(["git", "commit", "-m", commit_msg])

    # 5. Push to remote repository
    print("[*] Pushing to Git remote...")
    push_res = run_cmd(["git", "push"], check=False)

    if push_res.returncode == 0:
        print("[✓] Successfully pushed updated models to remote repository!")
    else:
        print(f"[!] Git push failed:\n{push_res.stderr}")


if __name__ == "__main__":
    run_pipeline_and_push()