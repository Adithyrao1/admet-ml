"""Run one experiment: a (dataset, model, split, seed) combination.

The protocol, and the reason for each step:

1. Split the dataset *before* anything else, so no test molecule influences
   featurization statistics or hyperparameter choice.
2. Featurize each split independently.
3. Tune hyperparameters with cross-validation **inside the training split**.
4. Fit the best configuration on the full training split.
5. Choose the decision threshold on the **validation** split.
6. Score the **test** split once, with that frozen threshold.

Steps 3-5 never touch test data, which is what keeps the reported numbers
honest. Threshold-free metrics (ROC-AUC, PR-AUC) are unaffected by step 5.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from admet.data import LABEL_COL, get_split, get_spec
from admet.evaluate import best_f1_threshold, classification_metrics
from admet.features import FeatureConfig, featurize_cached
from admet.models import build_model, set_scale_pos_weight


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one run."""

    dataset: str
    model: str
    split: str = "scaffold"
    seed: int = 42
    use_fingerprints: bool = True
    use_descriptors: bool = True
    n_search_iter: int = 20
    cv_folds: int = 5

    @property
    def run_id(self) -> str:
        features = (
            "fp+desc"
            if self.use_fingerprints and self.use_descriptors
            else ("fp" if self.use_fingerprints else "desc")
        )
        return f"{self.dataset}__{self.model}__{self.split}__{features}__seed{self.seed}"

    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(
            use_fingerprints=self.use_fingerprints,
            use_descriptors=self.use_descriptors,
        )


def _featurize_split(
    split_df: pd.DataFrame, config: FeatureConfig, tag: str
) -> tuple[np.ndarray, np.ndarray]:
    smiles = split_df["canonical_smiles"].tolist()
    X, _, valid_mask = featurize_cached(smiles, config, tag=tag)
    y = split_df[LABEL_COL].to_numpy()[valid_mask]
    return X, y


def run_single(
    config: RunConfig,
    cache_dir: str | Path = "data/raw",
    verbose: bool = True,
) -> dict:
    """Execute one run and return a flat result row."""
    spec = get_spec(config.dataset)
    started = time.perf_counter()

    splits = get_split(
        config.dataset, method=config.split, seed=config.seed, cache_dir=cache_dir
    )
    feature_config = config.feature_config()

    X_train, y_train = _featurize_split(
        splits["train"], feature_config, tag=f"{config.dataset}_train_s{config.seed}_{config.split}"
    )
    X_valid, y_valid = _featurize_split(
        splits["valid"], feature_config, tag=f"{config.dataset}_valid_s{config.seed}_{config.split}"
    )
    X_test, y_test = _featurize_split(
        splits["test"], feature_config, tag=f"{config.dataset}_test_s{config.seed}_{config.split}"
    )

    pipeline, param_distributions = build_model(config.model, seed=config.seed)
    pipeline = set_scale_pos_weight(pipeline, y_train)

    search_started = time.perf_counter()
    best_params: dict = {}
    cv_score = float("nan")

    if param_distributions:
        cv = StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=config.seed
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=config.n_search_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=config.seed,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        best_params = {k: _jsonable(v) for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)
    else:
        pipeline.fit(X_train, y_train)
        fitted = pipeline

    train_seconds = time.perf_counter() - search_started

    valid_prob = fitted.predict_proba(X_valid)[:, 1]
    threshold = best_f1_threshold(y_valid, valid_prob)
    valid_metrics = classification_metrics(y_valid, valid_prob, threshold=threshold)

    test_prob = fitted.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_prob, threshold=threshold)

    row = {
        "run_id": config.run_id,
        **asdict(config),
        "metric_of_record": spec.metric,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "n_test": int(len(y_test)),
        "train_pos_rate": float(np.mean(y_train)),
        "test_pos_rate": float(np.mean(y_test)),
        "n_features": int(X_train.shape[1]),
        "cv_roc_auc": cv_score,
        "best_params": json.dumps(best_params),
        "train_seconds": round(train_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
    row.update({f"test_{k}": v for k, v in test_metrics.items()})

    if verbose:
        print(
            f"  {config.run_id}: test ROC-AUC={row['test_roc_auc']:.4f} "
            f"PR-AUC={row['test_pr_auc']:.4f} MCC={row['test_mcc']:.3f} "
            f"({row['train_seconds']:.1f}s)"
        )
    return row


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def append_result(row: dict, results_path: str | Path) -> None:
    """Append one result row to the CSV, creating it if needed.

    Written after every run so an interrupted sweep keeps its finished work.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset="run_id", keep="last")
    else:
        combined = frame
    combined.to_csv(results_path, index=False)


def completed_run_ids(results_path: str | Path) -> set[str]:
    """Run IDs already present in the results CSV, for resuming a sweep."""
    results_path = Path(results_path)
    if not results_path.exists():
        return set()
    try:
        return set(pd.read_csv(results_path)["run_id"].astype(str))
    except (KeyError, pd.errors.EmptyDataError):
        return set()
