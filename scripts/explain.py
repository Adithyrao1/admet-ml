"""Explain what the models learned, and check it against known pharmacology.

Produces:

* ROC curves for every tuned model on one dataset.
* A SHAP ranking of the physicochemical descriptors that move predictions most.
* A signed direction check: for each top descriptor, does a *higher* value push
  the prediction up or down?

The direction check is the point. For blood-brain barrier penetration, medicinal
chemistry says a molecule needs to be small and not very polar: high topological
polar surface area (TPSA), high molecular weight and many hydrogen-bond donors
all argue against getting into the brain. If SHAP recovers that from the data,
the model has learned chemistry rather than an artefact of the split.

SHAP runs on the descriptor-only feature block: fingerprint bits are individually
meaningless to a reader, while descriptors are quantities a chemist recognises.

Usage::

    python scripts/explain.py --dataset bbb_martins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from admet.data import LABEL_COL, get_split  # noqa: E402
from admet.features import FeatureConfig, featurize_cached  # noqa: E402
from admet.models import build_model, set_scale_pos_weight  # noqa: E402
from admet.plots import plot_descriptor_importance, plot_roc_curves  # noqa: E402

# Descriptors whose expected direction is well established for BBB penetration.
BBB_EXPECTED_DIRECTION = {
    "TPSA": "negative",
    "MolWt": "negative",
    "NumHDonors": "negative",
    "NHOHCount": "negative",
    "MolLogP": "positive",
}


def _featurize(split_df: pd.DataFrame, config: FeatureConfig, tag: str):
    smiles = split_df["canonical_smiles"].tolist()
    X, names, valid = featurize_cached(smiles, config, tag=tag)
    return X, names, split_df[LABEL_COL].to_numpy()[valid]


def build_roc_curves(dataset: str, splits, results: pd.DataFrame | None, seed: int):
    """Refit each model with its tuned parameters and collect ROC curves."""
    config = FeatureConfig()
    X_train, _, y_train = _featurize(splits["train"], config, f"{dataset}_train_s{seed}_scaffold")
    X_test, _, y_test = _featurize(splits["test"], config, f"{dataset}_test_s{seed}_scaffold")

    curves = {}
    for model_name in ("logreg", "random_forest", "xgboost"):
        pipeline, _ = build_model(model_name, seed=seed)
        pipeline = set_scale_pos_weight(pipeline, y_train)

        # Reuse the hyperparameters the sweep selected, when available.
        if results is not None:
            match = results[
                (results["dataset"] == dataset)
                & (results["model"] == model_name)
                & (results["split"] == "scaffold")
                & (results["seed"] == seed)
            ]
            if len(match):
                try:
                    params = json.loads(match.iloc[0]["best_params"])
                    if params:
                        pipeline.set_params(**params)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = float(np.trapezoid(tpr, fpr))
        curves[model_name] = (fpr, tpr, auc)
        print(f"  {model_name}: AUC {auc:.4f}")
    return curves


def descriptor_shap(dataset: str, splits, seed: int):
    """Fit a descriptor-only XGBoost and return (names, mean|SHAP|, directions)."""
    import shap

    config = FeatureConfig(use_fingerprints=False, use_descriptors=True)
    X_train, names, y_train = _featurize(
        splits["train"], config, f"{dataset}_train_s{seed}_scaffold_desc"
    )
    X_test, _, _ = _featurize(splits["test"], config, f"{dataset}_test_s{seed}_scaffold_desc")

    pipeline, _ = build_model("xgboost", seed=seed)
    pipeline = set_scale_pos_weight(pipeline, y_train)
    pipeline.fit(X_train, y_train)

    # SHAP needs the matrix the booster actually saw, i.e. after imputation.
    imputed_test = pipeline.named_steps["impute"].transform(X_test)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(imputed_test)
    if isinstance(shap_values, list):  # older SHAP returns one array per class
        shap_values = shap_values[1]

    magnitude = np.abs(shap_values).mean(axis=0)

    # Direction: correlate each descriptor's value with its SHAP contribution.
    directions = {}
    for index, name in enumerate(names):
        column = imputed_test[:, index]
        if np.std(column) < 1e-12 or np.std(shap_values[:, index]) < 1e-12:
            directions[name] = 0.0
            continue
        directions[name] = float(np.corrcoef(column, shap_values[:, index])[0, 1])

    return names, magnitude, directions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain ADMET model predictions.")
    parser.add_argument("--dataset", default="bbb_martins")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--skip-roc", action="store_true")
    args = parser.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore")

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    results_path = Path(args.results)
    results = pd.read_csv(results_path) if results_path.exists() else None

    splits = get_split(args.dataset, method="scaffold", seed=args.seed)
    written: list[Path] = []

    if not args.skip_roc:
        print("Refitting tuned models for ROC curves...")
        curves = build_roc_curves(args.dataset, splits, results, args.seed)
        written += plot_roc_curves(curves, args.dataset, figures_dir)

    print("Computing SHAP values on descriptors...")
    names, magnitude, directions = descriptor_shap(args.dataset, splits, args.seed)
    written += plot_descriptor_importance(names, magnitude, args.dataset, figures_dir)

    order = np.argsort(magnitude)[::-1][:15]
    rows = []
    for index in order:
        clean = names[index].replace("desc_", "")
        correlation = directions[names[index]]
        rows.append(
            {
                "descriptor": clean,
                "mean_abs_shap": round(float(magnitude[index]), 5),
                "direction": "higher → more likely" if correlation > 0 else "higher → less likely",
                "value_shap_corr": round(correlation, 3),
            }
        )
    importance = pd.DataFrame(rows)
    importance_path = out_dir / f"{args.dataset}_descriptor_importance.csv"
    importance.to_csv(importance_path, index=False)

    print()
    print(importance.to_string(index=False))

    if args.dataset == "bbb_martins":
        print("\nPharmacology sanity check (expected from CNS drug design):")
        lookup = {row["descriptor"]: row["value_shap_corr"] for row in rows}
        for descriptor, expected in BBB_EXPECTED_DIRECTION.items():
            if descriptor not in lookup:
                print(f"  {descriptor:<12} not in top 15")
                continue
            correlation = lookup[descriptor]
            observed = "positive" if correlation > 0 else "negative"
            verdict = "MATCHES" if observed == expected else "DIFFERS FROM"
            print(
                f"  {descriptor:<12} expected {expected:<8} observed {observed:<8} "
                f"(r={correlation:+.2f})  {verdict} expectation"
            )

    print(f"\n  {importance_path}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
