"""Pre-train the models the demo app serves, and save them to ``models/``.

The app loads these instead of training on startup. That matters because the
deployed app runs in a small shared container: training at load would cost
every cold start a minute of CPU and hold both training sets in memory, while
loading a fitted model costs a few MB and a second.

Two models per endpoint:

* ``<dataset>_full``  -- fingerprints + descriptors, used for predictions.
  Best accuracy, but individual fingerprint bits explain nothing to a human.
* ``<dataset>_desc``  -- descriptors only, used for SHAP explanations, where
  every feature is a quantity a chemist recognises.

Also saved: packed Morgan fingerprints of each training set, so the app can
report Tanimoto similarity to the nearest training molecule without loading
the datasets themselves.

Usage::

    python scripts/build_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from admet.data import DATASETS, LABEL_COL, get_split  # noqa: E402
from admet.features import FeatureConfig, featurize  # noqa: E402
from admet.models import build_model, set_scale_pos_weight  # noqa: E402

# Settled by the sweep in results/results.csv; see README for the search space.
APP_PARAMS = {"model__max_depth": 6, "model__colsample_bytree": 0.6}


def packed_training_fingerprints(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprints as packed bits (8x smaller than one byte per bit)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(generator.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.packbits(np.array(rows, dtype=np.uint8), axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build demo model artifacts.")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore")
    import joblib

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "seed": args.seed,
        "split": "scaffold",
        "built_at": time.strftime("%Y-%m-%d"),
        "datasets": {},
    }

    for name, spec in DATASETS.items():
        print(f"=== {name} ===")
        splits = get_split(name, method="scaffold", seed=args.seed)
        train_smiles = splits["train"]["canonical_smiles"].tolist()

        for tag, config in (
            ("full", FeatureConfig()),
            ("desc", FeatureConfig(use_fingerprints=False, use_descriptors=True)),
        ):
            X, feature_names, valid = featurize(train_smiles, config)
            y = splits["train"][LABEL_COL].to_numpy()[valid]

            pipeline, _ = build_model("xgboost", seed=args.seed)
            pipeline = set_scale_pos_weight(pipeline, y)
            pipeline.set_params(**APP_PARAMS)
            pipeline.fit(X, y)

            path = out_dir / f"{name}_{tag}.joblib"
            joblib.dump({"pipeline": pipeline, "feature_names": feature_names}, path, compress=3)
            size_mb = path.stat().st_size / 1e6
            print(f"  {tag:<5} {X.shape[0]} x {X.shape[1]} -> {path.name} ({size_mb:.1f} MB)")

        fingerprints = packed_training_fingerprints(train_smiles)
        fp_path = out_dir / f"{name}_trainfp.npy"
        np.save(fp_path, fingerprints)
        print(
            f"  fps   {fingerprints.shape} -> {fp_path.name} "
            f"({fp_path.stat().st_size / 1e6:.1f} MB)"
        )

        # Canonical SMILES of the training molecules, so the app can say
        # exactly whether a queried molecule was memorised rather than
        # inferring it from a fingerprint collision.
        smiles_path = out_dir / f"{name}_train_smiles.txt"
        smiles_path.write_text("\n".join(train_smiles), encoding="utf-8")
        print(
            f"  smi   {len(train_smiles)} -> {smiles_path.name} "
            f"({smiles_path.stat().st_size / 1e6:.2f} MB)"
        )

        # Held-out molecules with their true labels. These let the demo show
        # evidence of generalization instead of predictions on molecules the
        # model memorised during training.
        holdout = pd.concat(
            [
                splits["valid"].assign(split="valid"),
                splits["test"].assign(split="test"),
            ],
            ignore_index=True,
        )[["canonical_smiles", LABEL_COL, "split"]]
        holdout_path = out_dir / f"{name}_holdout.csv"
        holdout.to_csv(holdout_path, index=False)
        print(
            f"  hold  {len(holdout)} molecules -> {holdout_path.name} "
            f"({holdout_path.stat().st_size / 1e6:.2f} MB)"
        )

        metadata["datasets"][name] = {
            "n_train": int(len(y)),
            "n_holdout": int(len(holdout)),
            "train_pos_rate": round(float(np.mean(y)), 4),
            "positive_meaning": spec.positive_meaning,
            "description": spec.description,
        }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"\nArtifacts in {out_dir}/ — {total:.1f} MB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
