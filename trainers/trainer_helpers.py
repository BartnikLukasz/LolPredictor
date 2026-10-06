"""
Shared helpers for all model trainers and the hyperparameter tuner.

Contents
  1. Project-wide training configuration (edit the constants below)
  2. Explicit feature groups + feature profiles (no substring matching)
  3. Dataset loading, chronological holdout and walk-forward folds
  4. Honest evaluation: bootstrap CIs, Elo / base-rate baselines, calibration
  5. Feature-importance export (unchanged public API)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TARGET_COL = "blue_win"
DATASET_PATH = "dataset/pregame/pregame_dataset_final_features.csv"

HOLDOUT_DAYS = 60          # final untouched test window, anchored to the dataset's latest match date
CV_FOLDS = 4               # walk-forward folds used for tuning (all strictly BEFORE the holdout)
MIN_TRAIN_ROWS = 1000      # smallest training slice allowed in any fold / split
MIN_TRAIN_DATE = None      # e.g. "2018-01-01" to drop the oldest eras; None = use everything
RANDOM_SEED = 42

# "live_supported": only features the Streamlit LiveFeatureEngine can actually reproduce at
#                   prediction time (see PROFILES below). Holdout numbers then predict live behaviour.
# "full":           everything the offline pipeline builds. Holdout numbers will be optimistic about
#                   the live app, because the live engine zero-fills the extra columns.
FEATURE_PROFILE = "live_supported"

# Raw champion identity as categorical input. Off by default: ~170 levels on a few tens of thousands
# of games mostly adds noise next to the engineered champion win rates, and categorical alignment is
# the most fragile part of serving. Turn on and compare on the holdout before trusting it.
USE_RAW_CHAMPIONS = False

ROLES = ["top", "jng", "mid", "bot", "sup"]
SIDES = ["blue", "red"]
CHAMPION_COLS = [f"{s}_{r}_champion" for s in SIDES for r in ROLES]


# =============================================================================
# 2. FEATURE GROUPS
# =============================================================================
def _per_side_role(template: str) -> list:
    return [template.format(side=s, role=r) for s in SIDES for r in ROLES]


_SYNERGY_PAIRS = [("mid", "jng"), ("bot", "sup"), ("jng", "sup")]

FEATURE_GROUPS = {
    # Elo (elo_calculator.py)
    "elo": ["elo_diff", "blue_elo_pre", "red_elo_pre", "blue_elo_win_prob", "blue_firstpick"],
    # Momentum (team_momentum_calculator.py, window=10)
    "momentum": [
        "blue_elo_delta_10", "red_elo_delta_10", "elo_delta_diff_10",
        "blue_overperform_10", "red_overperform_10", "overperformance_diff_10",
    ],
    # Intra-series state (match_data_converter.py)
    "series": ["game_number", "blue_series_lead", "blue_prev_win"],
    # Player form: keyed by player id in training (player_stats_calculator.py)
    "player": _per_side_role("{side}_{role}_player_games_pre") + _per_side_role("{side}_{role}_player_winrate_pre"),
    # Player-on-champion mastery (pid, champion) pairs in training
    "player_champ": _per_side_role("{side}_{role}_champ_games_pre") + _per_side_role("{side}_{role}_champ_winrate_pre"),
    # Team / lane / player head-to-head (player_stats_calculator.py)
    "h2h": (
        ["h2h_team_games_pre", "blue_h2h_team_winrate_pre"]
        + [f"blue_{r}_{k}_pre" for r in ROLES
           for k in ("lane_matchup_games", "lane_matchup_winrate", "p2p_games", "p2p_winrate")]
    ),
    # Patch meta, lane counters, synergies, comp cohesion (champ_stats_calculator.py)
    "draft": (
        _per_side_role("{side}_{role}_patch_wr_pre")
        + ["blue_team_patch_wr_avg_pre", "red_team_patch_wr_avg_pre", "patch_winrate_diff"]
        + [f"{r}_lane_counter_wr_pre" for r in ROLES] + ["lane_counter_diff_sum"]
        + [f"{s}_{a}_{b}_synergy_wr_pre" for (a, b) in _SYNERGY_PAIRS for s in SIDES]
        + [f"{a}_{b}_synergy_diff" for (a, b) in _SYNERGY_PAIRS]
        + ["blue_comp_cohesion_score", "red_comp_cohesion_score", "comp_cohesion_diff"]
    ),
}

# Groups the live app reproduces faithfully enough to be trusted (elo / momentum / series are built
# in LiveFeatureEngine.build_feature_vector; player stats are looked up by player name).
LIVE_SUPPORTED_GROUPS = ("elo", "momentum", "series", "player")

PROFILES = {
    "live_supported": LIVE_SUPPORTED_GROUPS,
    "full": tuple(FEATURE_GROUPS.keys()),
}


def select_features(df: pd.DataFrame,
                    profile: str = FEATURE_PROFILE,
                    use_raw_champions: bool = USE_RAW_CHAMPIONS,
                    verbose: bool = True):
    """Returns (feature_cols, cat_cols) using explicit column names only."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown feature profile '{profile}'. Choose from {list(PROFILES)}.")

    feature_cols, seen = [], set()
    for group in PROFILES[profile]:
        found = [c for c in FEATURE_GROUPS[group] if c in df.columns and c not in seen]
        seen.update(found)
        feature_cols.extend(found)
        if verbose:
            print(f"  [features] {group:<13} {len(found):>3} / {len(FEATURE_GROUPS[group])} columns present")

    cat_cols = [c for c in CHAMPION_COLS if c in df.columns] if use_raw_champions else []
    feature_cols = feature_cols + cat_cols

    if not feature_cols:
        raise ValueError("No feature columns found. Is this the final features CSV?")

    if verbose and profile == "full":
        zero_filled = [g for g in PROFILES["full"] if g not in LIVE_SUPPORTED_GROUPS and g != "player_champ"]
        print("  [WARN] Profile 'full': the live app zero-fills groups "
              f"{zero_filled} and builds 'player_champ' from a champion-only lookup (different meaning "
              "than in training). Holdout metrics will overstate live performance.")
    return feature_cols, cat_cols


def extract_features(df: pd.DataFrame):
    """Backward-compatible wrapper (old API returned numeric + champion columns in one list)."""
    return select_features(df, verbose=False)[0]


# =============================================================================
# 3. DATA, SPLITS
# =============================================================================
def load_dataset(filepath: str = DATASET_PATH, min_train_date=MIN_TRAIN_DATE) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    for col in ("date", TARGET_COL):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataset.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    # Stable sort: ties keep file order, which is the order the feature scripts processed them in.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    if min_train_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_train_date)].reset_index(drop=True)
    return df


def build_matrix(df: pd.DataFrame, feature_cols: list, cat_cols: list, cat_mode: str) -> pd.DataFrame:
    """
    Builds the model input once for the whole dataset so categorical levels are identical in every split.
      cat_mode "category": pandas Categorical (XGBoost / LightGBM native categoricals)
      cat_mode "str":      plain strings (CatBoost, sklearn one-hot)
    """
    X = df[feature_cols].copy()
    for col in feature_cols:
        if col in cat_cols:
            s = X[col].where(X[col].notna(), "Unknown").astype(str).str.strip()
            s = s.where(s != "", "Unknown")
            if cat_mode == "category":
                X[col] = pd.Categorical(s, categories=sorted(s.unique()))
            else:
                X[col] = s
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X


def holdout_index(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    """First row of the final holdout window; the window is anchored to the latest match in the data."""
    cutoff = df["date"].max().normalize() - pd.Timedelta(days=holdout_days)
    return int(df["date"].searchsorted(cutoff, side="left")), cutoff


def walk_forward_folds(n_available: int, fold_size: int, n_folds: int, min_train: int = MIN_TRAIN_ROWS):
    """
    Expanding-window folds over rows [0, n_available). Fold k trains on everything before its
    validation block, so every validation block is strictly later than its training data.
    Returns a list of (train_end, val_end) index pairs; validation = rows[train_end:val_end].
    """
    folds = []
    for k in range(n_folds, 0, -1):
        val_end = n_available - (k - 1) * fold_size
        train_end = val_end - fold_size
        if train_end >= min_train:
            folds.append((train_end, val_end))
    return folds


# =============================================================================
# 4. EVALUATION
# =============================================================================
def per_game_logloss(y, p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ids = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate_predictions(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(y, p),
    }


def elo_baseline_probs(df: pd.DataFrame):
    """The Elo-only win probability already stored in the dataset, or None."""
    if "blue_elo_win_prob" not in df.columns:
        return None
    p = pd.to_numeric(df["blue_elo_win_prob"], errors="coerce").values
    return None if np.isnan(p).any() else p


def report_holdout(name: str, test_df: pd.DataFrame, y, p, base_rate: float) -> dict:
    """Prints holdout metrics with 95% bootstrap CIs and paired comparisons against simple baselines."""
    y = np.asarray(y)
    n = len(y)
    m = evaluate_predictions(y, p)
    ll_model = per_game_logloss(y, p)
    ll_base = per_game_logloss(y, np.full(n, base_rate))
    elo_p = elo_baseline_probs(test_df)
    ll_elo = per_game_logloss(y, elo_p) if elo_p is not None else None

    d0, d1 = test_df["date"].min().strftime("%Y-%m-%d"), test_df["date"].max().strftime("%Y-%m-%d")
    acc_se = 1.96 * np.sqrt(max(m["accuracy"] * (1 - m["accuracy"]), 1e-9) / n)

    def line(label, ll):
        mean, lo, hi = bootstrap_ci(ll)
        return f"{label:<14}{mean:>9.4f}   [{lo:.4f}, {hi:.4f}]"

    print("\n" + "=" * 66)
    print(f"  HOLDOUT RESULTS: {name}   (n={n}, {d0} .. {d1})")
    print("=" * 66)
    print(f"{'':<14}{'log-loss':>9}   95% CI")
    print(line("Base rate", ll_base))
    if ll_elo is not None:
        print(line("Elo only", ll_elo))
    print(line(name, ll_model))
    out = dict(m)
    out["n"] = n
    out["accuracy_ci95"] = float(acc_se)
    for label, ll_ref, key in (("Elo only", ll_elo, "vs_elo"), ("base rate", ll_base, "vs_base_rate")):
        if ll_ref is None:
            continue
        mean, lo, hi = bootstrap_ci(ll_model - ll_ref, seed=1)
        out[key] = {"mean": mean, "lo": lo, "hi": hi}
        verdict = "model better" if hi < 0 else ("model worse" if lo > 0 else "not distinguishable")
        print(f"Paired diff vs {label}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print(f"Accuracy {m['accuracy'] * 100:.1f}% (+/- {acc_se * 100:.1f})  |  AUC {m['auc']:.4f}  |  "
          f"Brier {m['brier']:.4f}  |  ECE {m['ece']:.4f}")
    print("=" * 66)

    if "league" in test_df.columns:
        rows = []
        tmp = test_df.assign(_y=y, _p=np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6))
        for league, g in tmp.groupby("league"):
            if len(g) >= 10:
                rows.append({"League": league, "Matches": len(g),
                             "Accuracy (%)": round(accuracy_score(g["_y"], (g["_p"] >= 0.5).astype(int)) * 100, 1),
                             "Log-Loss": round(log_loss(g["_y"], g["_p"], labels=[0, 1]), 4)})
        if rows:
            print("\nBy league (leagues with >= 10 holdout games):")
            print(pd.DataFrame(rows).sort_values("Matches", ascending=False).to_string(index=False))
    return out


# =============================================================================
# 5. FEATURE IMPORTANCE (public API unchanged)
# =============================================================================
def save_feature_importance(model, feature_cols, importance_output_path="feature_importance.csv"):
    if isinstance(feature_cols, pd.DataFrame):
        feature_names = feature_cols.columns.tolist()
    elif isinstance(feature_cols, pd.Index):
        feature_names = feature_cols.tolist()
    else:
        feature_names = list(feature_cols)

    estimator = model
    if hasattr(model, "steps"):
        if len(model.steps) > 1:
            preprocessor = model[:-1]
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_names = list(preprocessor.get_feature_names_out(feature_names))
                except Exception:
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        pass
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        importance_scores = estimator.feature_importances_
    elif hasattr(estimator, "get_feature_importance"):
        importance_scores = estimator.get_feature_importance()
    elif hasattr(estimator, "coef_"):
        importance_scores = np.abs(estimator.coef_)
    else:
        raise AttributeError(f"'{type(estimator).__name__}' exposes neither feature importances nor coefficients.")

    importance_scores = np.ravel(importance_scores)
    if len(feature_names) != len(importance_scores):
        print(f"[notice] transformed features ({len(importance_scores)}) differ from raw columns "
              f"({len(feature_names)}); adjusting names.")
        if len(feature_names) < len(importance_scores):
            feature_names = feature_names + [f"transformed_feature_{i}"
                                             for i in range(len(feature_names), len(importance_scores))]
        else:
            feature_names = feature_names[:len(importance_scores)]

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_scores}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nTop 15 Most Influential Features:")
    print(importance_df.head(15).to_string(index=False))

    dir_name = os.path.dirname(importance_output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    importance_df.to_csv(importance_output_path, index=False)
    print(f"[SUCCESS] Feature importances saved to '{importance_output_path}'")
