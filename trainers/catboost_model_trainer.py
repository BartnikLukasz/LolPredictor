import joblib
from catboost import CatBoostClassifier

from trainers.engine import ModelSpec, train_model
from trainers.trainer_helpers import DATASET_PATH, HOLDOUT_DAYS, RANDOM_SEED

# --- PATH CONFIGURATION (unchanged; models/catboost_model.pkl is what app.py loads) ---
PARAMS_PATH = "models/catboost_best_params.json"
MODEL_OUTPUT_PATH = "models/catboost_model.pkl"

MAX_TREES = 1500            # cap during tuning only; the real count comes from early stopping inside CV folds
EARLY_STOPPING_ROUNDS = 50


class CatBoostSpec(ModelSpec):
    name = "catboost"
    label = "CatBoost"
    cat_mode = "str"
    iteration_key = "iterations"
    params_path = PARAMS_PATH
    model_path = MODEL_OUTPUT_PATH
    importance_path = "metadata/catboost_feature_importance.csv"
    predictions_path = "metadata/predictions/catboost_predictions.csv"

    def defaults(self):
        return {"iterations": 400, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 6.0,
                "random_strength": 1.0, "bagging_temperature": 1.0}

    def suggest(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        }

    @staticmethod
    def _model_params(params):
        p = dict(params)
        p.pop("n_estimators", None)
        # bagging_temperature is only valid with the Bayesian bootstrap; the default differs across versions.
        p.update(loss_function="Logloss", eval_metric="Logloss", random_seed=RANDOM_SEED, thread_count=-1,
                 bootstrap_type="Bayesian", allow_writing_files=False, verbose=False)
        p["iterations"] = int(p["iterations"])
        return p

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        p = self._model_params(params)
        p["iterations"] = MAX_TREES
        model = CatBoostClassifier(**p)
        model.fit(X_tr, y_tr, cat_features=cat_cols or None, eval_set=(X_va, y_va),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True, verbose=False)
        best_n = int(model.get_best_iteration()) + 1
        return model.predict_proba(X_va)[:, 1], best_n

    def fit(self, X, y, params, cat_cols):
        model = CatBoostClassifier(**self._model_params(params))
        model.fit(X, y, cat_features=cat_cols or None, verbose=False)
        return model

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        # app.py reads artifact.get("pipeline", artifact.get("model", artifact))
        joblib.dump({"model": model, "feature_names": list(feature_cols), "cat_features": list(cat_cols)}, path)


SPEC = CatBoostSpec()


def train_catboost(
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
