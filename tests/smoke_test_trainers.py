"""
Smoke test for the new training stack. Uses a SYNTHETIC dataset, so it checks plumbing, not accuracy.

    python -m tests.smoke_test_trainers          (run from the project root)

For every model whose library is installed it checks that
  1. Phase 1 (holdout) and Phase 2 (full refit) run, and the validation predictions CSV is written,
  2. the saved artifact loads exactly the way app.py loads it and predicts a valid probability,
  3. a short walk-forward tuning run writes a params file the trainer then accepts (needs optuna).
Models whose library is missing are reported as SKIPPED.
"""
import importlib
import json
import os
import sys
import tempfile
import traceback

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from trainers.engine import train_model, tune_model                               # noqa: E402
from trainers.rosters import save_team_rosters                                    # noqa: E402
from trainers.trainer_helpers import CHAMPION_COLS, FEATURE_GROUPS, ROLES, SIDES  # noqa: E402

SPEC_MODULES = [
    ("XGBoost", "trainers.model_trainer"),
    ("LightGBM", "trainers.lightgbm_model_trainer"),
    ("CatBoost", "trainers.catboost_model_trainer"),
    ("ElasticTree", "trainers.elastictree_model_trainer"),
    ("ElasticNet", "trainers.elasticnet_model_trainer"),
]


def make_synthetic_dataset(path: str, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.sort(pd.date_range("2024-01-01", "2026-09-30", periods=n).values
                    + pd.to_timedelta(rng.integers(0, 3600 * 12, n), unit="s").values)
    n_teams = 24
    teams = np.array([f"Team{i}" for i in range(n_teams)])
    champs = np.array([f"Champ{i}" for i in range(60)])
    b = rng.integers(0, n_teams, n)
    r = (b + rng.integers(1, n_teams, n)) % n_teams
    strength = rng.normal(0, 80, n_teams)

    df = pd.DataFrame({"gameid": [f"G{i}" for i in range(n)], "date": dates})
    df["league"] = rng.choice(["LCK", "LPL", "LEC"], n)
    df["patch"] = "14.1"
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["blue_team"], df["red_team"] = teams[b], teams[r]

    blue_elo = 1500 + strength[b] + rng.normal(0, 15, n)
    red_elo = 1500 + strength[r] + rng.normal(0, 15, n)
    df["blue_elo_pre"], df["red_elo_pre"] = blue_elo, red_elo
    df["elo_diff"] = blue_elo - red_elo
    df["blue_firstpick"] = 1
    df["blue_elo_win_prob"] = 1 / (1 + 10 ** (-(df["elo_diff"] + 10) / 400))

    for col in FEATURE_GROUPS["momentum"] + FEATURE_GROUPS["player_champ"] + FEATURE_GROUPS["h2h"] + FEATURE_GROUPS["draft"]:
        df[col] = rng.normal(0.5, 0.05, n)
    df["game_number"] = rng.integers(1, 6, n)
    df["blue_series_lead"] = rng.integers(-2, 3, n)
    df["blue_prev_win"] = rng.choice([0.0, 0.5, 1.0], n)
    for col in FEATURE_GROUPS["player"]:
        df[col] = rng.integers(0, 200, n).astype(float) if "games" in col else rng.normal(0.5, 0.05, n)

    pw_diff = (df[[f"blue_{x}_player_winrate_pre" for x in ROLES]].mean(axis=1)
               - df[[f"red_{x}_player_winrate_pre" for x in ROLES]].mean(axis=1))
    p = 1 / (1 + np.exp(-(0.004 * df["elo_diff"] + 4.0 * pw_diff)))
    df["blue_win"] = (rng.random(n) < p).astype(int)

    for side in SIDES:
        for role in ROLES:
            df[f"{side}_{role}_champion"] = rng.choice(champs, n)
            df[f"{side}_{role}_player"] = [f"{t}_{role}" for t in df[f"{side}_team"]]
    df.to_csv(path, index=False)
    return df


def load_like_app(path: str):
    """Mirrors app.load_predictor_assets()."""
    if path.endswith(".json"):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(path)
        return m
    artifact = joblib.load(path)
    return artifact.get("pipeline", artifact.get("model", artifact)) if isinstance(artifact, dict) else artifact


def check_one(label, module_name, csv, tmp, df, **extra):
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"[SKIPPED] {label}: {e}")
        return None
    spec = mod.SPEC
    tag = f"{spec.name}_{extra.get('feature_profile', 'default')}_{int(extra.get('use_raw_champions', False))}"
    model_path = os.path.join(tmp, f"{tag}{os.path.splitext(spec.model_path)[1]}")
    pred_path = os.path.join(tmp, f"{tag}_pred.csv")
    params_path = os.path.join(tmp, f"{tag}_params.json")

    out = train_model(spec, filepath=csv, model_output_path=model_path, params_path=params_path,
                      importance_output_path=os.path.join(tmp, f"{tag}_imp.csv"),
                      predictions_output_path=pred_path, **extra)
    assert out["metrics"] is not None, "holdout metrics missing"
    preds = pd.read_csv(pred_path)
    assert {"game_id", "prob_blue_win", "actual_blue_win"} <= set(preds.columns) and len(preds) > 30

    model = load_like_app(model_path)
    meta = json.load(open(os.path.splitext(model_path)[0] + ".meta.json"))
    row = df[meta["feature_cols"]].iloc[[-1]].copy()
    for c in meta["cat_cols"]:
        row[c] = (pd.Categorical(row[c].astype(str), categories=meta["categories"][c])
                  if meta["categories"] else row[c].astype(str))
    p = float(model.predict_proba(row)[0][1])
    assert 0.0 < p < 1.0, p
    print(f"[OK] {label} ({tag}): artifact loads like app.py, live-style prediction p_blue={p:.3f}")
    return spec, params_path


def check_tuning(label, spec, csv, tmp, **extra):
    try:
        import optuna  # noqa: F401
    except ImportError:
        print(f"[SKIPPED] {label} tuning: optuna not installed")
        return
    params_path = os.path.join(tmp, f"{spec.name}_tuned.json")
    tune_model(spec, filepath=csv, n_trials=3, n_folds=2, params_path=params_path, **extra)
    saved = json.load(open(params_path))
    assert "_meta" in saved and "best_logloss" in saved
    train_model(spec, filepath=csv, params_path=params_path, model_output_path=os.path.join(tmp, f"{spec.name}_t.bin"),
                importance_output_path=os.path.join(tmp, "imp_t.csv"),
                predictions_output_path=os.path.join(tmp, "pred_t.csv"), **extra)
    print(f"[OK] {label}: tuning wrote a params file and the trainer accepted it")


if __name__ == "__main__":
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "synthetic.csv")
        df = make_synthetic_dataset(csv)
        for label, module_name in SPEC_MODULES:
            for extra in ({}, {"feature_profile": "full", "use_raw_champions": True}):
                try:
                    res = check_one(label, module_name, csv, tmp, df, **extra)
                    if res is not None and not extra:
                        check_tuning(label, res[0], csv, tmp)
                except Exception:
                    failures += 1
                    print(f"[FAIL] {label} {extra}")
                    traceback.print_exc()
        try:
            rosters = save_team_rosters(csv, os.path.join(tmp, "rosters.json"))
            assert len(rosters) == 24 and all(len(v) == 5 for v in rosters.values())
            print("[OK] rosters")
        except Exception:
            failures += 1
            traceback.print_exc()
    print("\nALL CHECKS PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)
