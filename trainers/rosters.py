import json
import os

import pandas as pd

from trainers.trainer_helpers import ROLES


def save_team_rosters(dataset, output_path: str = "models/team_rosters.json", max_inactive_days: int = None) -> dict:
    """
    Writes {team name: [top, jng, mid, bot, sup]} using each team's most recent match (either side).
    One vectorised pass instead of scanning the dataframe once per team.

    max_inactive_days: if set, teams whose last match is older than this many days (relative to the
    dataset's latest match) are left out, which keeps defunct teams out of the app's dropdowns.
    """
    df = pd.read_csv(dataset, low_memory=False) if isinstance(dataset, str) else dataset.copy()
    df["date"] = pd.to_datetime(df["date"])

    parts = []
    for side in ("blue", "red"):
        cols = [f"{side}_{r}_player" for r in ROLES]
        missing = [c for c in [f"{side}_team"] + cols if c not in df.columns]
        if missing:
            raise KeyError(f"Cannot build rosters, missing columns: {missing}")
        part = df[["date", f"{side}_team"] + cols].copy()
        part.columns = ["date", "team"] + ROLES
        parts.append(part)

    long = pd.concat(parts, ignore_index=True).dropna(subset=["team"])
    long = long.sort_values("date", kind="stable")
    latest = long.drop_duplicates("team", keep="last")

    if max_inactive_days is not None:
        cutoff = long["date"].max() - pd.Timedelta(days=max_inactive_days)
        latest = latest[latest["date"] >= cutoff]

    roster_dict = {
        row["team"]: [str(row[r]) if pd.notna(row[r]) else "" for r in ROLES]
        for _, row in latest.iterrows()
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(roster_dict, f, indent=4)
    print(f"[ARTIFACT] Saved {len(roster_dict)} team rosters to '{output_path}'")
    return roster_dict
