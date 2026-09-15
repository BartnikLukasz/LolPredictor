import os
import numpy as np
import pandas as pd


def compute_vision_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates game-level vision and map control rates per team."""
    df = df.copy()

    # Game duration in minutes
    if "game_length_sec" in df.columns:
        game_min = df["game_length_sec"] / 60.0
    elif "gamelength" in df.columns:
        game_min = df["gamelength"] / 60.0 if df["gamelength"].max() > 100 else df["gamelength"]
    else:
        game_min = 30.0

    game_min = np.maximum(game_min, 1.0)

    # 1. Vision Score Per Minute (VSPM)
    if "vspm" in df.columns:
        df["calc_vspm"] = df["vspm"]
    else:
        df["calc_vspm"] = df.get("visionscore", df.get("vision_score", 0.0)) / game_min

    # 2. Wards Placed & Cleared Per Minute (WPM & WCPM)
    if "wpm" in df.columns and "wcpm" in df.columns:
        df["calc_wpm"] = df["wpm"]
        df["calc_wcpm"] = df["wcpm"]
    else:
        df["calc_wpm"] = df.get("wardspaced", df.get("wards_placed", 0.0)) / game_min
        df["calc_wcpm"] = df.get("wardscleared", df.get("wards_killed", 0.0)) / game_min

    # 3. Ward Clearance Efficiency Ratio (Wards Cleared / Wards Placed)
    df["ward_clear_ratio"] = df["calc_wcpm"] / (df["calc_wpm"] + 1e-5)

    # 4. Control Ward Investment Per Minute (CWPM)
    if "cwpm" in df.columns:
        df["calc_cwpm"] = df["cwpm"]
    else:
        df["calc_cwpm"] = df.get("controlwardsbought", df.get("vision_wards_bought", 0.0)) / game_min

    # 5. Vision Map Control Rating (Composite Score)
    # Higher rating indicates superior dark-zone control and objective vision Denial
    df["map_control_score"] = (
        (df["calc_vspm"] * 0.4) +
        (df["ward_clear_ratio"] * 10.0 * 0.35) +
        (df["calc_cwpm"] * 2.0 * 0.25)
    )

    return df


def process_vision_map_control_features(
    input_path: str, output_path: str, window: int = 10
):
    """Pipeline to process vision features and merge rolling stats onto dataset."""
    print("Computing Vision & Map Control features...")
    df = pd.read_csv(input_path, low_memory=False)

    df_vision = compute_vision_metrics(df)

    metrics_to_roll = [
        "calc_vspm",
        "calc_wpm",
        "calc_wcpm",
        "ward_clear_ratio",
        "calc_cwpm",
        "map_control_score",
    ]

    # Clean metric column names for feature dataset
    clean_name_map = {
        "calc_vspm": "vspm",
        "calc_wpm": "wpm",
        "calc_wcpm": "wcpm",
        "calc_cwpm": "cwpm",
        "ward_clear_ratio": "ward_clear_ratio",
        "map_control_score": "map_control_score",
    }

    df_vision = df_vision.sort_values("date").reset_index(drop=True)

    # Calculate rolling metrics per team side
    for side in ["blue", "red"]:
        team_col = f"team_{side}" if f"team_{side}" in df_vision.columns else f"{side}_team"

        for metric in metrics_to_roll:
            clean_name = clean_name_map[metric]
            roll_col_name = f"{side}_hist_{clean_name}_avg_last{window}"

            # Shift by 1 prevents data leakage from the target game
            df_vision[roll_col_name] = (
                df_vision.groupby(team_col)[metric]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
                .fillna(0.0)
            )

    # Calculate Blue vs Red Differentials
    for metric in metrics_to_roll:
        clean_name = clean_name_map[metric]
        blue_col = f"blue_hist_{clean_name}_avg_last{window}"
        red_col = f"red_hist_{clean_name}_avg_last{window}"
        diff_col = f"diff_hist_{clean_name}_avg_last{window}"

        df_vision[diff_col] = df_vision[blue_col] - df_vision[red_col]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_vision.to_csv(output_path, index=False)
    print(f"Vision & Map Control features saved successfully to: {output_path}")


if __name__ == "__main__":
    process_vision_map_control_features(
        input_path="dataset/pregame/pregame_step3_output.csv",
        output_path="dataset/pregame/pregame_step4_output.csv",
        window=10,
    )