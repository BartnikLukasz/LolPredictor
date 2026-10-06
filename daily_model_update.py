import os
import subprocess
import sys
from datetime import datetime

# Configure the paths you want to track and stage in Git
PATHS_TO_STAGE = ["models/", "dataset/pregame/pregame_dataset_final_features.csv.gz"]

PUSH_ATTEMPTS = 3


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Runs a command, prints its output (so it shows up in CI logs), optionally raises on failure."""
    print(f"[*] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res


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

    # 3. Anything actually staged? (`git status --porcelain` also lists unrelated untracked files such as
    #    token.json or catboost_info/, which would make the commit below fail on a clean CI checkout.)
    if run_cmd(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("[+] No model or dataset changes detected. Skipping Git commit and push.")
        return

    # 4. Commit changes with a timestamped message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"auto: update trained models and dataset ({timestamp})"
    print(f"[*] Committing changes: '{commit_msg}'")
    run_cmd(["git", "commit", "-m", commit_msg])

    # 5. Push, rebasing first in case the remote moved (e.g. you pushed code from your PC meanwhile)
    print("[*] Pushing to Git remote...")
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        pull = run_cmd(["git", "pull", "--rebase"], check=False)
        if pull.returncode != 0:
            git_dir = run_cmd(["git", "rev-parse", "--git-dir"], check=False).stdout.strip() or ".git"
            if any(os.path.isdir(os.path.join(git_dir, d)) for d in ("rebase-merge", "rebase-apply")):
                run_cmd(["git", "rebase", "--abort"], check=False)
                print("[!] Rebase conflict with the remote branch. Resolve manually.")
            else:
                print("[!] 'git pull --rebase' failed (network or permissions problem?).")
            sys.exit(1)
        push = run_cmd(["git", "push"], check=False)
        if push.returncode == 0:
            print("[✓] Successfully pushed updated models to remote repository!")
            return
        print(f"[!] Git push failed (attempt {attempt}/{PUSH_ATTEMPTS}).")

    # Exit non-zero so a scheduled run shows up as FAILED instead of silently green.
    print("[!] Giving up: could not push updated models.")
    sys.exit(1)


if __name__ == "__main__":
    run_pipeline_and_push()
