import hashlib
import pandas as pd

def generate_game_id(blue_team, red_team, blue_champs, red_champs, first_pick=""):
    """
    Generates a deterministic 10-character MD5 match fingerprint.
    Normalizes team names, champion lists (sorted to avoid role-swap mismatches),
    and first-pick side representation.
    """
    b_team = str(blue_team).strip().lower()
    r_team = str(red_team).strip().lower()

    # Sort champions so role swaps or pick order differences don't break the match ID
    b_champs = ",".join(sorted([str(c).strip().lower() for c in blue_champs if pd.notna(c)]))
    r_champs = ",".join(sorted([str(c).strip().lower() for c in red_champs if pd.notna(c)]))

    # Standardize first pick representation (handles 1/0, True/False, "Blue"/"Red")
    fp_str = str(first_pick).strip().lower()
    if fp_str in ['1', '1.0', 'blue', 'true']:
        fp_norm = 'blue'
    elif fp_str in ['0', '0.0', 'red', 'false']:
        fp_norm = 'red'
    else:
        fp_norm = fp_str

    raw_str = f"{b_team}|{r_team}|{b_champs}|{r_champs}|{fp_norm}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()[:16]