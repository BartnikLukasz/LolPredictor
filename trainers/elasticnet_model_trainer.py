import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from trainers.engine import ModelSpec, train_model
from trainers.trainer_helpers import DATASET_PATH, HOLDOUT_DAYS, RANDOM_SEED

# --- PATH CONFIGURATION (unchanged; models/elasticnet_model.joblib is what app.py loads) ---
PARAMS_PATH = "models/elasticnet_best_params.json"
MODEL_OUTPUT_PATH = "models/elasticnet_model.joblib"

# scikit-learn 1.8 deprecated `penalty` (an l1_ratio in (0, 1) now means elastic net); older versions need it.
_SKLEARN_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


class ElasticNetSpec(ModelSpec):
    name = "elasticnet"
    label = "ElasticNet"
    cat_mode = "str"
    params_path = PARAMS_PATH
    model_path = MODEL_OUTPUT_PATH
    importance_path = "metadata/elasticnet_feature_importance.csv"
    predictions_path = "metadata/predictions/elasticnet_predictions.csv"

    def defaults(self):
        # Same solver settings in tuning and training (the old tuner used max_iter=200, tol=1e-2).
        return {"C": 0.05, "l1_ratio": 0.5, "max_iter": 5000, "tol": 1e-4, "random_state": RANDOM_SEED}

    def suggest(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def _build(self, X, params, cat_cols):
        num_cols = [c for c in X.columns if c not in cat_cols]
        transformers = [("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols)]
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                          sparse_output=True)),
            ]), cat_cols))
        pre = ColumnTransformer(transformers, verbose_feature_names_out=False)

        kwargs = {"solver": "saga", "C": params["C"], "l1_ratio": params["l1_ratio"],
                  "max_iter": params["max_iter"], "tol": params["tol"], "random_state": params["random_state"]}
        if not _SKLEARN_GE_18:
            kwargs["penalty"] = "elasticnet"
        return Pipeline([("preprocessor", pre), ("classifier", LogisticRegression(**kwargs))])

    def fit(self, X, y, params, cat_cols):
        return self._build(X, params, cat_cols).fit(X, y)

    def fit_eval(self, X_tr, y_tr, X_va, y_va, params, cat_cols):
        return self.predict(self.fit(X_tr, y_tr, params, cat_cols), X_va), None

    def save(self, model, path, feature_cols, cat_cols, X_check=None):
        joblib.dump(model, path)          # app.py: a bare sklearn Pipeline is loaded as-is


SPEC = ElasticNetSpec()


def train_elasticnet_model(
        filepath: str = DATASET_PATH,
        dynamic_test_window: int = HOLDOUT_DAYS,
        only_full_train: bool = False,
        params_json_path: str = PARAMS_PATH,
        model_output_path: str = MODEL_OUTPUT_PATH,
        importance_output_path: str = SPEC.importance_path,
        predictions_output_path: str = SPEC.predictions_path,
        **kwargs
):
    """Honest holdout evaluation, then a full-data refit saved for the live app. See engine.train_model."""
    return train_model(
        SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, only_full_train=only_full_train,
        params_path=params_json_path, model_output_path=model_output_path,
        importance_output_path=importance_output_path, predictions_output_path=predictions_output_path,
        **kwargs
    )["model"]
