"""
Hyperparameter tuning for all five models (Optuna, walk-forward cross-validation).

What changed vs. the old tuner
  * Objective = mean log-loss over several walk-forward folds taken BEFORE the final holdout window.
    The holdout (last HOLDOUT_DAYS before the dataset's latest match) is never used for tuning or early stopping.
  * Early stopping happens inside each fold; the mean best iteration becomes the tuned n_estimators/iterations,
    so the final model is trained with the same number of trees that tuning actually validated.
  * Saved params are replaced only if they beat the saved params re-scored on the SAME folds by a margin.
  * One pass, seeded sampler, search ranges that are not narrowed around earlier winners.

Run from the project root (same working directory as main.py):  python hyperparameter_tuner.py
"""
from trainers.engine import tune_model
from trainers.trainer_helpers import CV_FOLDS, DATASET_PATH, HOLDOUT_DAYS


def optimize_xgboost_hyperparameters(filepath=DATASET_PATH, dynamic_test_window=HOLDOUT_DAYS, n_trials=100,
                                     output_json_path=None, **kwargs):
    from trainers.model_trainer import SPEC
    return tune_model(SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, n_trials=n_trials,
                      params_path=output_json_path, **kwargs)


def optimize_lightgbm_hyperparameters(filepath=DATASET_PATH, dynamic_test_window=HOLDOUT_DAYS, n_trials=100,
                                      output_json_path=None, **kwargs):
    from trainers.lightgbm_model_trainer import SPEC
    return tune_model(SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, n_trials=n_trials,
                      params_path=output_json_path, **kwargs)


def optimize_catboost_hyperparameters(filepath=DATASET_PATH, dynamic_test_window=HOLDOUT_DAYS, n_trials=40,
                                      output_json_path=None, **kwargs):
    from trainers.catboost_model_trainer import SPEC
    return tune_model(SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, n_trials=n_trials,
                      params_path=output_json_path, **kwargs)


def optimize_elastictree_hyperparameters(filepath=DATASET_PATH, dynamic_test_window=HOLDOUT_DAYS, n_trials=40,
                                         output_json_path=None, **kwargs):
    from trainers.elastictree_model_trainer import SPEC
    return tune_model(SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, n_trials=n_trials,
                      params_path=output_json_path, **kwargs)


def optimize_elasticnet_hyperparameters(filepath=DATASET_PATH, dynamic_test_window=HOLDOUT_DAYS, n_trials=60,
                                        output_json_path=None, **kwargs):
    from trainers.elasticnet_model_trainer import SPEC
    return tune_model(SPEC, filepath=filepath, dynamic_test_window=dynamic_test_window, n_trials=n_trials,
                      params_path=output_json_path, **kwargs)


if __name__ == "__main__":
    # feature_profile / use_raw_champions default to the values in trainers/trainer_helpers.py.
    # IMPORTANT: tune and train with the same profile (params files record which one they were tuned for).
    i = 0
    while i < 5:
        optimize_xgboost_hyperparameters(n_trials=200)
        optimize_lightgbm_hyperparameters(n_trials=200)
        optimize_catboost_hyperparameters(n_trials=200)
        optimize_elastictree_hyperparameters(n_trials=200)
        optimize_elasticnet_hyperparameters(n_trials=200)
        i+=1
