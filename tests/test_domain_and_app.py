"""Tests for the domain-knowledge rules and the demo app's artifacts."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from admet.domain import BBB_EXPECTED_DIRECTION, LIPINSKI_RULES, check_direction

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "streamlit_app.py"
MODELS_DIR = ROOT / "models"


def test_expected_directions_are_well_formed():
    assert BBB_EXPECTED_DIRECTION, "expectations must not be empty"
    for descriptor, value in BBB_EXPECTED_DIRECTION.items():
        direction, rationale = value
        assert direction in {"positive", "negative"}, descriptor
        assert rationale.strip(), f"{descriptor} needs a stated rationale"


def test_check_direction_matches_and_differs():
    # TPSA is expected to lower BBB penetration, so a negative correlation
    # between its value and its SHAP contribution is a match.
    matches, rationale = check_direction("TPSA", -0.81)
    assert matches is True
    assert rationale

    differs, _ = check_direction("TPSA", +0.5)
    assert differs is False


def test_check_direction_is_none_for_undocumented_descriptor():
    assert check_direction("PEOE_VSA7", 0.4) is None


def test_lipinski_rules_are_well_formed():
    assert len(LIPINSKI_RULES) == 4
    for key, limit, message in LIPINSKI_RULES:
        assert isinstance(key, str) and key
        assert isinstance(limit, (int, float))
        assert message.strip()


def test_app_module_compiles():
    """Catches syntax errors without executing Streamlit's runtime."""
    py_compile.compile(str(APP), doraise=True)


@pytest.mark.skipif(
    not (MODELS_DIR / "bbb_martins_full.joblib").exists(),
    reason="model artifacts not built; run scripts/build_artifacts.py",
)
@pytest.mark.parametrize("dataset", ["bbb_martins", "ames"])
def test_artifacts_present_and_consistent(dataset):
    import joblib
    import numpy as np
    import pandas as pd

    bundle = joblib.load(MODELS_DIR / f"{dataset}_full.joblib")
    assert "pipeline" in bundle and "feature_names" in bundle

    fingerprints = np.load(MODELS_DIR / f"{dataset}_trainfp.npy")
    train_smiles = (
        (MODELS_DIR / f"{dataset}_train_smiles.txt")
        .read_text(encoding="utf-8")
        .split("\n")
    )
    # One packed fingerprint row per training molecule.
    assert fingerprints.shape[0] == len(train_smiles)
    assert fingerprints.shape[1] == 2048 // 8

    holdout = pd.read_csv(MODELS_DIR / f"{dataset}_holdout.csv")
    assert {"canonical_smiles", "Y", "split"} <= set(holdout.columns)
    assert set(holdout["split"].unique()) <= {"valid", "test"}

    # The demo must never present a training molecule as held-out evidence.
    assert not (set(holdout["canonical_smiles"]) & set(train_smiles))


@pytest.mark.skipif(
    not (MODELS_DIR / "bbb_martins_full.joblib").exists(),
    reason="model artifacts not built; run scripts/build_artifacts.py",
)
def test_app_runs_without_exceptions():
    """Execute the whole app headlessly.

    Streamlit runs the script top to bottom, so every tab body executes in this
    single run -- this catches a broken tab that a smoke test of the underlying
    functions would miss.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    assert not app.exception, [str(e) for e in app.exception]
    # All five tabs should be present.
    assert len(app.tabs) >= 5, f"expected 5 tabs, found {len(app.tabs)}"
    # And the app should not have bailed out early on a missing-artifacts error.
    assert not any("No model artifacts found" in str(e.value) for e in app.error)
