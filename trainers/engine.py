"""
Generic training and tuning engine shared by all five models.

Protocol (this is what fixes the old evaluation problems)
  * TUNING   uses walk-forward folds taken strictly BEFORE the final holdout window. Early stopping happens
             inside those folds only, and the mean best iteration becomes the tuned n_estimators/iterations.
  * TRAINING Phase 1 fits on everything before the holdout with FIXED hyperparameters and a FIXED number of
             trees (no early stopping, no peeking), then evaluates once on the untouched holdout.
             Phase 2 refits on all data with the very same settings and saves the artifact.
  * The holdout window is anchored to the dataset's latest match, not date.today(), so runs are reproducible.
"""
import json
import os
import platform
import warnings
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from trainers.trainer_helpers import (
    CV_FOLDS, DATASET_PATH, FEATURE_PROFILE, HOLDOUT_DAYS, MIN_TRAIN_DATE, MIN_TRAIN_ROWS,
    RANDOM_SEED, TARGET_COL, USE_RAW_CHAMPIONS,
    build_matrix, elo_baseline_probs, holdout_index, load_dataset, per_game_logloss,
    report_holdout, save_feature_importance, select_features, walk_forward_folds,
)
from util import save_validation_predictions

ITERATION_SCALE = 1.10      # final models see ~10% more data than a CV fold's training slice
_META_KEY = "_meta"
_DROP_ON_LOAD = {"best_logloss", "early_stopping_rounds"}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            pass
    return out


# =============================================================================
# MODEL SPEC BASE CLASS
# =============================================================================
class ModelSpec:
    """Everything model-specific lives in a subclass; the engine is model-agnostic."""
    name = "base"                 # short slug used in file names and logs
    label = "Base"                # pretty name for printouts
    cat_mode = "category"         # "category" (native categoricals) or "str" (strings)
    iteration_key = None          # "n_estimators" / "iterations" when the model supports early stopping
    params_path = ""
    model_path = ""
    importance_path = ""
    predictions_path = ""

    def defaults(self) -> dict:
        raise NotImplementedError

    def suggest(self, trial) -> dict:
        """Optuna search space; returns only the tuned keys."""
        raise NotImplementedError

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        """Used only while tuning. Returns (validation probabilities, best_iteration or None)."""
        raise NotImplementedError

    def fit(self, X, y, params, cat_cols):
        """Fixed-size fit with NO validation data. Returns a fitted model."""
        raise NotImplementedError

    def predict(self, model, X) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        raise NotImplementedError

    def importance_target(self, model):
        return model

    def finalize(self, params: dict, mean_best_iteration) -> dict:
        """Turns tuned params + mean early-stopping iteration into the final training params."""
        out = dict(params)
        if self.iteration_key and mean_best_iteration:
            out[self.iteration_key] = int(max(30, round(mean_best_iteration * ITERATION_SCALE)))
        return out


# =============================================================================
# PARAMETER FILES
# =============================================================================
def _read_params_file(path: str):
    """Returns (params, meta) or (None, None). Meta is None for legacy files from the old tuner."""
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Could not read '{path}' ({e}).")
        return None, None
    meta = data.pop(_META_KEY, None)
    for k in _DROP_ON_LOAD:
        data.pop(k, None)
    return data, meta


def load_params(spec, path, feature_profile, use_raw_champions, allow_legacy=False) -> dict:
    params = spec.defaults()
    saved, meta = _read_params_file(path)
    if saved is None:
        print(f"[CONFIG] No parameter file at '{path}'. Using conservative defaults (run the tuner).")
        return params
    if meta is None and not allow_legacy:
        print(f"[CONFIG] '{path}' comes from the OLD tuner (tuned on the test window, 1000-tree final fit). "
              f"Ignoring it and using defaults. Re-run the tuner, or pass allow_legacy_params=True.")
        return params
    if meta is not None and (meta.get("feature_profile") != feature_profile
                             or bool(meta.get("use_raw_champions")) != bool(use_raw_champions)):
        print(f"[WARN] '{path}' was tuned for profile={meta.get('feature_profile')}, "
              f"raw_champions={meta.get('use_raw_champions')}; current run uses profile={feature_profile}, "
              f"raw_champions={use_raw_champions}. Consider re-tuning.")
    params.update(saved)
    print(f"[CONFIG] Loaded tuned hyperparameters from '{path}'.")
    return params


# =============================================================================
# TRAINING
# =============================================================================
def train_model(spec: ModelSpec,
                filepath: str = DATASET_PATH,
                dynamic_test_window: int = HOLDOUT_DAYS,
                only_full_train: bool = False,
                params_path: str = None,
                model_output_path: str = None,
                importance_output_path: str = None,
                predictions_output_path: str = None,
                feature_profile: str = FEATURE_PROFILE,
                use_raw_champions: bool = USE_RAW_CHAMPIONS,
                min_train_date=MIN_TRAIN_DATE,
                allow_legacy_params: bool = False) -> dict:
    """Phase 1: honest holdout evaluation. Phase 2: refit on all data and save the artifact."""
    params_path = params_path or spec.params_path
    model_output_path = model_output_path or spec.model_path
    importance_output_path = importance_output_path or spec.importance_path
    predictions_output_path = predictions_output_path or spec.predictions_path

    print("\n" + "#" * 66)
    print(f"  {spec.label.upper()}  |  profile={feature_profile}  raw_champions={use_raw_champions}")
    print("#" * 66)

    df = load_dataset(filepath, min_train_date)
    feature_cols, cat_cols = select_features(df, feature_profile, use_raw_champions)
    X = build_matrix(df, feature_cols, cat_cols, spec.cat_mode)
    y = df[TARGET_COL].values
    print(f"Loaded {len(df)} matches ({df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}) | "
          f"{len(feature_cols)} features ({len(cat_cols)} categorical)")

    params = load_params(spec, params_path, feature_profile, use_raw_champions, allow_legacy_params)

    metrics = None
    split_idx, cutoff = holdout_index(df, dynamic_test_window)
    n_test = len(df) - split_idx

    if not only_full_train:
        if split_idx < MIN_TRAIN_ROWS or n_test < 30:
            print(f"[INFO] Holdout skipped (train rows={split_idx}, holdout rows={n_test}).")
        else:
            X_tr, y_tr = X.iloc[:split_idx], y[:split_idx]
            X_te, y_te = X.iloc[split_idx:], y[split_idx:]
            test_df = df.iloc[split_idx:].copy()
            print(f"\nPHASE 1 | train {len(X_tr)} games (< {cutoff:%Y-%m-%d}) | holdout {len(X_te)} games")

            val_model = spec.fit(X_tr, y_tr, params, cat_cols)
            p_te = spec.predict(val_model, X_te)
            metrics = report_holdout(spec.label, test_df, y_te, p_te, base_rate=float(y_tr.mean()))
            save_validation_predictions(test_df, y_te, p_te, predictions_output_path)

    print(f"\nPHASE 2 | refit on all {len(df)} games with identical settings")
    final_model = spec.fit(X, y, params, cat_cols)

    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    spec.save(final_model, model_output_path, feature_cols, cat_cols, X_check=X.tail(500))
    print(f"[✓] Saved {spec.label} model to '{model_output_path}'")

    meta = {
        "model": spec.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_profile": feature_profile,
        "use_raw_champions": bool(use_raw_champions),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        # Exact category order the model was trained with (needed to encode live inputs correctly).
        "categories": ({c: [str(v) for v in X[c].cat.categories] for c in cat_cols}
                       if spec.cat_mode == "category" else {}),
        "training_rows": int(len(df)),
        "date_range": [f"{df['date'].min():%Y-%m-%d}", f"{df['date'].max():%Y-%m-%d}"],
        "holdout_days": dynamic_test_window,
        "params": params,
        "holdout_metrics": metrics,
        "versions": _lib_versions(),
    }
    meta_path = os.path.splitext(model_output_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    save_feature_importance(spec.importance_target(final_model), feature_cols, importance_output_path)
    return {"model": final_model, "metrics": metrics, "feature_cols": feature_cols, "cat_cols": cat_cols}


# =============================================================================
# TUNING
# =============================================================================
def tune_model(spec: ModelSpec,
               filepath: str = DATASET_PATH,
               dynamic_test_window: int = HOLDOUT_DAYS,
               n_trials: int = 100,
               params_path: str = None,
               n_folds: int = CV_FOLDS,
               fold_size: int = None,
               feature_profile: str = FEATURE_PROFILE,
               use_raw_champions: bool = USE_RAW_CHAMPIONS,
               min_train_date=MIN_TRAIN_DATE,
               seed: int = RANDOM_SEED,
               min_improvement: float = 0.0005,
               timeout: float = None) -> dict:
    """
    Walk-forward Optuna search. The final holdout window is never touched.
    New params replace the saved ones only if they beat the saved params, re-scored on the SAME folds,
    by at least `min_improvement` log-loss (the min over many noisy trials is optimistic, so a margin is needed).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

    params_path = params_path or spec.params_path

    df = load_dataset(filepath, min_train_date)
    feature_cols, cat_cols = select_features(df, feature_profile, use_raw_champions, verbose=False)
    X = build_matrix(df, feature_cols, cat_cols, spec.cat_mode)
    y = df[TARGET_COL].values

    split_idx, cutoff = holdout_index(df, dynamic_test_window)
    fold_size = fold_size or max(150, len(df) - split_idx)
    folds = walk_forward_folds(split_idx, fold_size, n_folds, MIN_TRAIN_ROWS)
    if len(folds) < 2:
        raise ValueError(f"Only {len(folds)} usable walk-forward fold(s); need at least 2. "
                         f"Reduce fold_size/n_folds or holdout window.")

    print("=" * 66)
    print(f"  {spec.label.upper()} TUNING (Optuna, walk-forward)  profile={feature_profile}")
    print("=" * 66)
    print(f"{len(feature_cols)} features | {len(folds)} folds x {fold_size} games | "
          f"holdout from {cutoff:%Y-%m-%d} is NOT used")

    def cv_score(params):
        full = {**spec.defaults(), **params}
        losses, iters = [], []
        for tr_end, va_end in folds:
            p, it = spec.fit_eval(X.iloc[:tr_end], y[:tr_end], X.iloc[tr_end:va_end], y[tr_end:va_end], full, cat_cols)
            p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
            losses.append(float(log_loss(y[tr_end:va_end], p, labels=[0, 1])))
            if it is not None:
                iters.append(it)
        return float(np.mean(losses)), (float(np.mean(iters)) if iters else None), losses

    # Reference scores on the same folds
    elo_p = elo_baseline_probs(df)
    base_ll, elo_ll = [], []
    for tr_end, va_end in folds:
        yy = y[tr_end:va_end]
        base_ll.append(float(per_game_logloss(yy, np.full(len(yy), y[:tr_end].mean())).mean()))
        if elo_p is not None:
            elo_ll.append(float(per_game_logloss(yy, elo_p[tr_end:va_end]).mean()))

    # Re-score the incumbent on these exact folds (replaces the old "refresh baseline" logic)
    saved, meta = _read_params_file(params_path)
    incumbent_ll = None
    if saved is not None and meta is not None and meta.get("feature_profile") == feature_profile \
            and bool(meta.get("use_raw_champions")) == bool(use_raw_champions):
        try:
            incumbent_ll, _, _ = cv_score(saved)
            print(f"Incumbent params re-scored on current folds: {incumbent_ll:.5f}")
        except Exception as e:
            print(f"[!] Could not re-score saved params ({e}); treating as no incumbent.")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        mean_ll, mean_it, losses = cv_score(spec.suggest(trial))
        trial.set_user_attr("best_iteration", mean_it)
        trial.set_user_attr("fold_losses", losses)
        return mean_ll

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    best_ll = float(best.value)
    best_params = spec.finalize({**spec.defaults(), **best.params}, best.user_attrs.get("best_iteration"))

    print("\n" + "=" * 66)
    print(f"  Best CV log-loss:   {best_ll:.5f}   (folds: {', '.join(f'{x:.4f}' for x in best.user_attrs['fold_losses'])})")
    print(f"  Base rate (CV):     {np.mean(base_ll):.5f}")
    if elo_ll:
        print(f"  Elo only (CV):      {np.mean(elo_ll):.5f}")
    if incumbent_ll is not None:
        print(f"  Saved params (CV):  {incumbent_ll:.5f}")
    print("  NOTE: differences below ~0.005 are inside the noise of walk-forward CV on this much data.")

    improved = incumbent_ll is None or best_ll < incumbent_ll - min_improvement
    if improved:
        payload = dict(best_params)
        payload["best_logloss"] = round(best_ll, 6)      # kept for compatibility with util.py helpers
        payload[_META_KEY] = {
            "cv_logloss": round(best_ll, 6),
            "cv_fold_losses": [round(x, 6) for x in best.user_attrs["fold_losses"]],
            "n_folds": len(folds), "fold_size": fold_size, "n_trials": len(study.trials),
            "feature_profile": feature_profile, "use_raw_champions": bool(use_raw_champions),
            "tuned_at": datetime.now().isoformat(timespec="seconds"),
            "data_through": f"{df['date'].max():%Y-%m-%d}", "holdout_days": dynamic_test_window,
        }
        os.makedirs(os.path.dirname(params_path) or ".", exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(payload, f, indent=4, default=_json_default)
        print(f"[✓] Saved tuned params to '{params_path}'")
    else:
        print(f"[=] Did not beat saved params by >= {min_improvement}. Keeping existing file.")
    print("=" * 66 + "\n")
    return {"best_params": best_params, "best_cv_logloss": best_ll, "saved": improved}
