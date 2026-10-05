import numpy as np
import xgboost as xgb

from trainers.engine import ModelSpec, train_model
from trainers.trainer_helpers import DATASET_PATH, HOLDOUT_DAYS, RANDOM_SEED

# --- PATH CONFIGURATION (unchanged; models/xgboost_model.json is what app.py loads) ---
PARAMS_PATH = "models/best_params.json"
MODEL_OUTPUT_PATH = "models/xgboost_model.json"

MAX_TREES = 1500            # cap during tuning only; the real count comes from early stopping inside CV folds
EARLY_STOPPING_ROUNDS = 50


class XGBoostSpec(ModelSpec):
    name = "xgboost"
    label = "XGBoost"
    cat_mode = "category"
    iteration_key = "n_estimators"
    params_path = PARAMS_PATH
    model_path = MODEL_OUTPUT_PATH
    importance_path = "metadata/xgboost_feature_importance.csv"
    predictions_path = "metadata/predictions/xgboost_predictions.csv"

    def defaults(self):
        return {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10,
                "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 0.5, "gamma": 0.0}

    def suggest(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }

    @staticmethod
    def _model_params(params, cat_cols):
        p = dict(params)
        p.update(tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss",
                 enable_categorical=bool(cat_cols))
        return p

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        p = self._model_params(params, cat_cols)
        p.update(n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_n = int(model.best_iteration) + 1
        proba = model.predict_proba(X_va, iteration_range=(0, best_n))[:, 1]
        return proba, best_n

    def fit(self, X, y, params, cat_cols):
        model = xgb.XGBClassifier(**self._model_params(params, cat_cols))
        model.fit(X, y, verbose=False)
        return model

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        model.save_model(path)


SPEC = XGBoostSpec()


def train_lol_prediction_model(
        filepath: str = DATASET_PATH,
        test_split_ratio: float = 0.20,            # kept for signature compatibility; unused
        dynamic_test_window: int = HOLDOUT_DAYS,
        only_full_train: bool = False,
        params_filepath: str = PARAMS_PATH,
        output_model_path: str = MODEL_OUTPUT_PATH,
        importance_output_path: str = SPEC.importance_path,
        predictions_output_path: str = SPEC.predictions_path,
        **kwargs
) -> xgb.XGBClassifier:
    """Honest holdout evaluation, then a full-data refit saved for the live app. See engine.train_model."""
    return train_model(
        SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, only_full_train=only_full_train,
        params_path=params_filepath, model_output_path=output_model_path,
        importance_output_path=importance_output_path, predictions_output_path=predictions_output_path,
        **kwargs
    )["model"]
