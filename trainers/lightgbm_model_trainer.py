import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier

from trainers.engine import ModelSpec, train_model
from trainers.trainer_helpers import DATASET_PATH, HOLDOUT_DAYS, RANDOM_SEED

# --- PATH CONFIGURATION (unchanged; models/lightgbm_model.pkl is what app.py loads) ---
PARAMS_PATH = "models/best_lightgbm_params.json"
MODEL_OUTPUT_PATH = "models/lightgbm_model.pkl"

MAX_TREES = 1500            # cap during tuning only; the real count comes from early stopping inside CV folds
EARLY_STOPPING_ROUNDS = 50


class LightGBMSpec(ModelSpec):
    name = "lightgbm"
    label = "LightGBM"
    cat_mode = "category"
    iteration_key = "n_estimators"
    params_path = PARAMS_PATH
    model_path = MODEL_OUTPUT_PATH
    importance_path = "metadata/lightgbm_feature_importance.csv"
    predictions_path = "metadata/predictions/lightgbm_predictions.csv"

    def defaults(self):
        return {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
                "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.7,
                "reg_alpha": 0.1, "reg_lambda": 5.0}

    def suggest(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 48, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        }

    @staticmethod
    def _model_params(params):
        p = dict(params)
        # subsample only takes effect when subsample_freq >= 1
        p.update(objective="binary", subsample_freq=1, random_state=RANDOM_SEED, n_jobs=-1,
                 verbosity=-1, importance_type="gain")
        for k in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
            if k in p:
                p[k] = int(p[k])
        return p

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        p = self._model_params(params)
        p["n_estimators"] = MAX_TREES
        model = LGBMClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        best_n = int(model.best_iteration_ or MAX_TREES)
        proba = model.predict_proba(X_va, num_iteration=best_n)[:, 1]
        return proba, best_n

    def fit(self, X, y, params, cat_cols):
        model = LGBMClassifier(**self._model_params(params))
        model.fit(X, y)
        return model

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        # app.py reads artifact.get("pipeline", artifact.get("model", artifact))
        joblib.dump({"model": model, "feature_names": list(feature_cols)}, path)


SPEC = LightGBMSpec()


def train_secondary_model(
        dataset_path: str = DATASET_PATH,
        model_output_path: str = MODEL_OUTPUT_PATH,
        params_path: str = PARAMS_PATH,
        dynamic_test_window: int = HOLDOUT_DAYS,
        only_full_train: bool = False,
        importance_output_path: str = SPEC.importance_path,
        predictions_output_path: str = SPEC.predictions_path,
        **kwargs
):
    """Honest holdout evaluation, then a full-data refit saved for the live app. See engine.train_model."""
    return train_model(
        SPEC, filepath=dataset_path, dynamic_test_window=dynamic_test_window, only_full_train=only_full_train,
        params_path=params_path, model_output_path=model_output_path,
        importance_output_path=importance_output_path, predictions_output_path=predictions_output_path,
        **kwargs
    )["model"]
