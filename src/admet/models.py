"""Model zoo: sklearn pipelines plus hyperparameter search spaces.

Every model is wrapped in a ``Pipeline`` so that preprocessing is fitted inside
each cross-validation fold. Median imputation and scaling learned from a whole
dataset would leak test information into training; keeping them in the pipeline
makes that impossible.

Linear models get a ``StandardScaler`` because the descriptor block mixes wildly
different units (molecular weight in the hundreds, counts in single digits).
Tree ensembles are scale-invariant, so they skip it and train faster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["dummy", "logreg", "random_forest", "xgboost"]


def _imputer() -> SimpleImputer:
    # Descriptors carry NaNs (see features.py); fingerprint bits never do.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def build_model(
    name: str,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[Pipeline, dict[str, Any]]:
    """Return ``(pipeline, param_distributions)`` for a named model.

    ``param_distributions`` is in ``RandomizedSearchCV`` form, keyed with the
    ``model__`` pipeline prefix. The dummy model has an empty search space.

    ``n_jobs`` defaults to 1 because these pipelines are normally fitted inside
    a ``RandomizedSearchCV`` that already parallelises over candidates. Letting
    both layers claim every core oversubscribes the CPU and runs slower.
    """
    if name == "dummy":
        # Predicts the training class prior for every molecule. Its ROC-AUC is
        # 0.5 by construction, which is the floor every real model must clear.
        pipeline = Pipeline(
            [("impute", _imputer()), ("model", DummyClassifier(strategy="prior"))]
        )
        return pipeline, {}

    if name == "logreg":
        pipeline = Pipeline(
            [
                ("impute", _imputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        # Only C is searched: 'penalty' is deprecated in scikit-learn 1.8 and
        # L2 is already the default.
        params = {"model__C": np.logspace(-3, 2, 20)}
        return pipeline, params

    if name == "random_forest":
        pipeline = Pipeline(
            [
                ("impute", _imputer()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        params = {
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.1],
        }
        return pipeline, params

    if name == "xgboost":
        from xgboost import XGBClassifier

        pipeline = Pipeline(
            [
                ("impute", _imputer()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=600,
                        learning_rate=0.05,
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        params = {
            "model__max_depth": [3, 4, 6, 8],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.3, 0.6, 0.9],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [0.5, 1.0, 5.0],
        }
        return pipeline, params

    raise ValueError(f"Unknown model {name!r}. Available: {MODEL_NAMES}")


def set_scale_pos_weight(pipeline: Pipeline, y_train: np.ndarray) -> Pipeline:
    """Balance XGBoost on skewed labels.

    BBB penetration is ~76% positive, so an unweighted model is tempted to
    predict "crosses" for everything. Other estimators handle this through
    ``class_weight``; XGBoost needs ``scale_pos_weight`` instead.
    """
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "scale_pos_weight"):
        return pipeline

    n_positive = float(np.sum(y_train == 1))
    n_negative = float(np.sum(y_train == 0))
    if n_positive > 0:
        pipeline.set_params(model__scale_pos_weight=n_negative / n_positive)
    return pipeline
