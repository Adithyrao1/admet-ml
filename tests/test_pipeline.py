"""Tests for the invariants that keep reported metrics honest.

The data tests hit the network on first run and then use the on-disk cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from admet.data import DATASETS, get_split, load_dataset, murcko_scaffold
from admet.evaluate import aggregate, best_f1_threshold, classification_metrics
from admet.features import FeatureConfig, featurize

# TDC's published dataset sizes; our loader must reproduce them exactly.
EXPECTED_SIZES = {"bbb_martins": 1975, "ames": 7255}


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_dataset_size_matches_tdc(name):
    df = load_dataset(name)
    assert len(df) == EXPECTED_SIZES[name]
    assert df["canonical_smiles"].is_unique
    assert set(df["Y"].unique()) <= {0, 1}


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("method", ["random", "scaffold"])
def test_splits_partition_the_dataset(name, method):
    splits = get_split(name, method=method, seed=42)
    total = sum(len(part) for part in splits.values())
    assert total == EXPECTED_SIZES[name]

    train = set(splits["train"]["canonical_smiles"])
    valid = set(splits["valid"]["canonical_smiles"])
    test = set(splits["test"]["canonical_smiles"])
    # No molecule may appear in more than one split.
    assert not (train & test)
    assert not (train & valid)
    assert not (valid & test)


def test_scaffold_split_shares_no_scaffold_between_train_and_test():
    """The point of a scaffold split: test holds unseen chemical series."""
    splits = get_split("bbb_martins", method="scaffold", seed=42)
    train = {murcko_scaffold(s) for s in splits["train"]["canonical_smiles"]}
    test = {murcko_scaffold(s) for s in splits["test"]["canonical_smiles"]}
    assert not ((train & test) - {None})


def test_splits_are_deterministic_for_a_seed():
    first = get_split("bbb_martins", method="scaffold", seed=7)
    second = get_split("bbb_martins", method="scaffold", seed=7)
    assert list(first["test"]["canonical_smiles"]) == list(second["test"]["canonical_smiles"])


def test_different_seeds_give_different_scaffold_splits():
    first = get_split("bbb_martins", method="scaffold", seed=1)
    second = get_split("bbb_martins", method="scaffold", seed=2)
    assert list(first["test"]["canonical_smiles"]) != list(second["test"]["canonical_smiles"])


def test_featurize_drops_unparseable_smiles():
    X, names, mask = featurize(["CCO", "not_a_molecule", "c1ccccc1"], FeatureConfig())
    assert mask.tolist() == [True, False, True]
    # X has one row per *valid* molecule, so callers must mask their labels.
    assert X.shape[0] == 2
    assert X.shape[1] == len(names)


def test_featurize_blocks_have_expected_widths():
    only_fp = FeatureConfig(use_fingerprints=True, use_descriptors=False, n_bits=512)
    X, names, _ = featurize(["CCO"], only_fp)
    assert X.shape == (1, 512)
    assert all(name.startswith("fp_") for name in names)

    only_desc = FeatureConfig(use_fingerprints=False, use_descriptors=True)
    X, names, _ = featurize(["CCO"], only_desc)
    assert all(name.startswith("desc_") for name in names)


def test_featurize_produces_no_infinities():
    """Ipc and friends overflow on large molecules; they must become NaN."""
    big = "C" * 60
    X, _, _ = featurize([big, "CCO"], FeatureConfig())
    assert not np.isinf(X).any()


def test_featurize_requires_at_least_one_block():
    with pytest.raises(ValueError):
        featurize(["CCO"], FeatureConfig(use_fingerprints=False, use_descriptors=False))


def test_constant_predictor_scores_at_the_floor():
    """A model that ignores its input must not look good on any headline metric."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.full(500, 0.76)

    threshold = best_f1_threshold(y_true, y_prob)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_best_f1_threshold_can_select_all_positive():
    """Regression test: a rounded cutoff once landed above every probability,
    silently flipping all predictions to negative."""
    y_true = np.array([1, 1, 1, 0])
    y_prob = np.full(4, 0.7380321)
    threshold = best_f1_threshold(y_true, y_prob)
    assert (y_prob >= threshold).all()


def test_perfect_predictions_score_one():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


def test_single_class_ground_truth_yields_nan_auc():
    metrics = classification_metrics(np.ones(5), np.linspace(0.1, 0.9, 5))
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_aggregate_reports_mean_and_std():
    runs = [{"roc_auc": 0.90}, {"roc_auc": 0.92}, {"roc_auc": 0.94}]
    summary = aggregate(runs)
    assert summary["n_seeds"] == 3
    assert summary["roc_auc_mean"] == pytest.approx(0.92)
    assert summary["roc_auc_std"] == pytest.approx(0.02)
