import os
import pandas as pd


def compute_team_playstyle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates team playstyle metrics using available team-level match stats."""
    df = df.copy()

    # Normalize game duration to minutes
    game_min = df["game_length_sec"] / 60.0 if "game_length_sec" in df.columns else df.get("gamelength", 30.0)

    # 1. Aggression Index: Kills per minute + First Blood + Early Gold Advantage
    kills_pm = df.get("team_kills", df.get("kills", 10.0)) / game_min
    fb_rate = df.get("first_blood_rate", df.get("firstblood", 0.5))
    gd15 = df.get("gold_diff_at_15", df.get("golddiff15", 0.0))

    df["aggression_index"] = (kills_pm * 0.4) + (fb_rate * 0.35) + ((gd15 / 1000.0) * 0.25)

    # 2. Early vs. Late Orientation Index
    first_drag = df.get("first_dragon_rate", df.get("firstdragon", 0.5))
    df["early_game_orientation"] = (gd15 / 1500.0) + (first_drag * 0.5) - (game_min / 30.0)

    # 3. Objective Priority Score
    drag_rate = df.get("dragon_rate", 0.5)
    baron_rate = df.get("baron_rate", 0.5)
    herald_rate = df.get("herald_rate", 0.5)

    obj_score = (drag_rate + baron_rate + herald_rate) / 3.0
    df["objective_priority_score"] = obj_score / (kills_pm + 1e-5)

    # Default Herfindahl-Hirschman Index (HHI) for balanced resource distribution
    df["team_gold_hhi"] = 0.20  # Perfect 20% split across 5 players = sum(0.2^2 * 5) = 0.20

    return df


def generate_resource_allocation(input_path: str, output_path: str, window: int = 10):
    """Generates rolling Step 3 metrics using team history only."""
    print("Processing team-level playstyle profiles...")
    df = pd.read_csv(input_path)

    df_styled = compute_team_playstyle_features(df)

    playstyle_cols = [
        "aggression_index",
        "early_game_orientation",
        "objective_priority_score",
        "team_gold_hhi"
    ]

    # Calculate rolling historical averages per team
    df_styled = df_styled.sort_values("date").reset_index(drop=True)

    for side in ["blue", "red"]:
        team_col = f"team_{side}" if f"team_{side}" in df_styled.columns else f"{side}_team"

        for col in playstyle_cols:
            roll_col_name = f"{side}_hist_{col}_avg_last{window}"
            df_styled[roll_col_name] = (
                df_styled.groupby(team_col)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
                .fillna(0.0)
            )

    # Generate differentials
    for col in playstyle_cols:
        blue_col = f"blue_hist_{col}_avg_last{window}"
        red_col = f"red_hist_{col}_avg_last{window}"
        df_styled[f"diff_hist_{col}_avg_last{window}"] = df_styled[blue_col] - df_styled[red_col]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_styled.to_csv(output_path, index=False)
    print(f"Step 3 team-level features generated successfully -> {output_path}")


if __name__ == "__main__":
    generate_resource_allocation(
        input_path="dataset/pregame/pregame_step2_output.csv",
        output_path="dataset/pregame/pregame_step3_output.csv"
    )