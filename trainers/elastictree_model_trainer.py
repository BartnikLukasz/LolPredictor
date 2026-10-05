import copy

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from trainers.engine import ModelSpec, train_model
from trainers.trainer_helpers import DATASET_PATH, HOLDOUT_DAYS, RANDOM_SEED

# --- PATH CONFIGURATION (unchanged; models/elastictree_model.pkl is what app.py loads) ---
PARAMS_PATH = "models/elastictree_best_params.json"
MODEL_OUTPUT_PATH = "models/elastictree_model.pkl"

TUNING_TREES = 200          # more trees only reduce variance, so tuning uses fewer for speed
COMPRESS_MODEL = True       # round tree thresholds/values to shrink the file (verified, see below)
COMPRESS_TOLERANCE = 2e-3   # max allowed change in any predicted probability
JOBLIB_COMPRESS = 3         # zlib level 3; use ("xz", 3) for smaller files and slower load


def shrink_extra_trees(pipeline, X_check, decimals: int = 4, tol: float = COMPRESS_TOLERANCE):
    """
    Rounds thresholds and node values on a COPY of the forest and keeps the copy only if predictions on
    X_check move by less than `tol`. Rounding is NOT guaranteed to be output-neutral, so it is verified.
    """
    candidate = copy.deepcopy(pipeline)
    for est in candidate.named_steps["model"].estimators_:
        tree = est.tree_
        np.round(tree.threshold, decimals=decimals, out=tree.threshold)
        np.round(tree.value, decimals=decimals, out=tree.value)
    drift = float(np.max(np.abs(pipeline.predict_proba(X_check)[:, 1] - candidate.predict_proba(X_check)[:, 1])))
    if drift <= tol:
        print(f"[OPTIMIZATION] Tree rounding kept (max probability change {drift:.2e}).")
        return candidate
    print(f"[OPTIMIZATION] Tree rounding rejected (max probability change {drift:.2e} > {tol}). Saving unrounded.")
    return pipeline


class ElasticTreeSpec(ModelSpec):
    name = "elastictree"
    label = "ElasticTree"
    cat_mode = "str"
    params_path = PARAMS_PATH
    model_path = MODEL_OUTPUT_PATH
    importance_path = "metadata/elastictree_feature_importance.csv"
    predictions_path = "metadata/predictions/elastictree_predictions.csv"

    def defaults(self):
        # Leaf size is the main calibration lever for ExtraTrees: tiny leaves give overconfident probabilities.
        return {"n_estimators": 500, "criterion": "gini", "max_depth": 14, "min_samples_leaf": 15,
                "min_samples_split": 30, "max_features": 0.3, "random_state": RANDOM_SEED, "n_jobs": -1}

    def suggest(self, trial):
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 200, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }

    def _build(self, X, params, cat_cols):
        num_cols = [c for c in X.columns if c not in cat_cols]
        transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)]
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                          sparse_output=True)),
            ]), cat_cols))
        pre = ColumnTransformer(transformers, verbose_feature_names_out=False)
        return Pipeline([("preprocessor", pre), ("model", ExtraTreesClassifier(**params))])

    def fit(self, X, y, params, cat_cols):
        return self._build(X, params, cat_cols).fit(X, y)

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        model = self.fit(X_tr, y_tr, {**params, "n_estimators": TUNING_TREES}, cat_cols)
        return self.predict(model, X_va), None

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        final = shrink_extra_trees(model, X_check) if (COMPRESS_MODEL and X_check is not None) else model
        # app.py reads artifact.get("pipeline", artifact.get("model", artifact))
        artifact = {"pipeline": final, "model": final, "feature_cols": feature_cols}
        joblib.dump(artifact, path, compress=JOBLIB_COMPRESS)


SPEC = ElasticTreeSpec()


def train_elastictree(
        filepath: str = DATASET_PATH,
        dynamic_test_window: int = HOLDOUT_DAYS,
        only_full_train: bool = False,
        params_path: str = PARAMS_PATH,
        output_model_path: str = MODEL_OUTPUT_PATH,
        importance_output_path: str = SPEC.importance_path,
        predictions_output_path: str = SPEC.predictions_path,
        **kwargs
):
    """Honest holdout evaluation, then a full-data refit saved for the live app. See engine.train_model."""
    return train_model(
        SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, only_full_train=only_full_train,
        params_path=params_path, model_output_path=output_model_path,
        importance_output_path=importance_output_path, predictions_output_path=predictions_output_path,
        **kwargs
    )["model"]
