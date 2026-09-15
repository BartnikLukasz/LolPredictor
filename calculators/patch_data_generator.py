import os
import numpy as np
import pandas as pd


def compute_patch_meta_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Computes team-level patch adaptability and meta alignment metrics."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Determine Patch Meta Champions (Top 25% pick rate per patch)
    champ_cols = [f"{side}_{role}_champion" for side in ["blue", "red"] for role in ["top", "jng", "mid", "bot", "sup"]]
    existing_champ_cols = [c for c in champ_cols if c in df.columns]

    # Reshape champion picks per patch to derive patch meta prevalence
    if existing_champ_cols and "patch" in df.columns:
        melted = df.melt(id_vars=["patch"], value_vars=existing_champ_cols, value_name="champion").dropna()
        patch_meta = (
            melted.groupby(["patch", "champion"])
            .size()
            .groupby(level=0, group_keys=False)
            .apply(lambda x: x[x >= x.quantile(0.75)].index.tolist())
            .to_dict()
        )
    else:
        patch_meta = {}

    # Helper function to compute team level patch & meta metrics sequentially
    team_history = {}

    def extract_patch_stats(row):
        blue_team = str(row.get("blue_team", ""))
        red_team = str(row.get("red_team", ""))
        patch = str(row.get("patch", "default"))
        blue_win = row.get("blue_win", 0)

        meta_champs = set(patch_meta.get(patch, []))

        out = {}
        for side, team, won in [("blue", blue_team, blue_win), ("red", red_team, 1 - blue_win)]:
            if team not in team_history:
                team_history[team] = []

            history = team_history[team]

            # Overall rolling winrate (last 20 games)
            past_wins = [g["win"] for g in history[-20:]]
            overall_wr = np.mean(past_wins) if past_wins else 0.50

            # Patch-specific winrate
            patch_games = [g for g in history if g["patch"] == patch]
            patch_wins = [g["win"] for g in patch_games]
            patch_wr = np.mean(patch_wins) if patch_wins else overall_wr

            # Unique champion pool depth (last N games)
            recent_champs = set()
            for g in history[-window:]:
                recent_champs.update(g.get("champs", []))
            pool_depth = len(recent_champs)

            # Meta Alignment Score (Percentage of picks in patch meta)
            side_champs = [row.get(f"{side}_{role}_champion", "") for role in ["top", "jng", "mid", "bot", "sup"]]
            valid_champs = [c for c in side_champs if c and str(c) != "nan"]
            meta_count = sum(1 for c in valid_champs if c in meta_champs)
            meta_alignment = meta_count / len(valid_champs) if valid_champs else 0.50

            # Set metrics for current game (before updating history)
            out[f"{side}_hist_patch_winrate"] = patch_wr
            out[f"{side}_hist_patch_wr_delta"] = patch_wr - overall_wr
            out[f"{side}_hist_champ_pool_depth"] = pool_depth
            out[f"{side}_hist_meta_alignment_score"] = meta_alignment

            # Append current game result to history (Post-game state)
            team_history[team].append({
                "patch": patch,
                "win": won,
                "champs": valid_champs
            })

        return pd.Series(out)

    patch_stats_df = df.apply(extract_patch_stats, axis=1)
    df = pd.concat([df, patch_stats_df], axis=1)

    # Calculate Blue vs Red Differentials
    metrics = ["patch_winrate", "patch_wr_delta", "champ_pool_depth", "meta_alignment_score"]
    for m in metrics:
        df[f"diff_hist_{m}"] = df[f"blue_hist_{m}"] - df[f"red_hist_{m}"]

    return df


def generate_patch_data(input_path: str, output_path: str, window: int = 10):
    """Pipeline executor for Step 5: Patch & Meta Adaptability features."""
    print("Computing Patch & Meta Adaptability features...")
    df = pd.read_csv(input_path, low_memory=False)

    df_step5 = compute_patch_meta_features(df, window=window)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_step5.to_csv(output_path, index=False)
    print(f"Step 5 Patch & Meta features successfully generated -> {output_path}")


if __name__ == "__main__":
    generate_patch_data(
        input_path="dataset/pregame/pregame_step4_output.csv",
        output_path="dataset/pregame/pregame_step5_output.csv",
        window=10
    )