"""Turn SMILES strings into numeric feature matrices with RDKit.

Two complementary feature blocks are produced:

* **Morgan fingerprints** -- a 2048-bit vector where each bit answers "does this
  molecule contain substructure *k*?". Dense signal, but individual bits mean
  little to a human.
* **Physicochemical descriptors** -- ~200 interpretable numbers (molecular
  weight, LogP, topological polar surface area, hydrogen-bond donors, ...).
  These are what SHAP can explain in terms a chemist recognises.

Descriptor values that come back infinite (a few RDKit descriptors overflow on
unusual structures) are converted to NaN here and left for the modelling
pipeline to impute. Imputation deliberately happens inside the sklearn pipeline
so the fill values are learned from training folds only, never from test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


@dataclass(frozen=True)
class FeatureConfig:
    """Which feature blocks to build."""

    use_fingerprints: bool = True
    use_descriptors: bool = True
    radius: int = DEFAULT_RADIUS
    n_bits: int = DEFAULT_N_BITS

    def cache_key(self) -> str:
        return (
            f"fp{int(self.use_fingerprints)}"
            f"_desc{int(self.use_descriptors)}"
            f"_r{self.radius}_b{self.n_bits}"
        )


def _descriptor_names() -> list[str]:
    from rdkit.Chem import Descriptors

    return [name for name, _ in Descriptors.descList]


def _descriptor_functions():
    from rdkit.Chem import Descriptors

    return list(Descriptors.descList)


def featurize(
    smiles: list[str] | np.ndarray,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Featurize a list of SMILES.

    Returns ``(X, feature_names, valid_mask)``. Rows of ``X`` correspond to the
    entries of ``smiles`` where ``valid_mask`` is True; molecules RDKit cannot
    parse are excluded, so ``X`` has ``valid_mask.sum()`` rows. Callers must
    apply ``valid_mask`` to their labels.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    config = config or FeatureConfig()

    if not (config.use_fingerprints or config.use_descriptors):
        raise ValueError("At least one of use_fingerprints/use_descriptors must be set")

    smiles = list(smiles)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_mask = np.array([m is not None for m in mols], dtype=bool)
    valid_mols = [m for m in mols if m is not None]

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if config.use_fingerprints:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=config.radius, fpSize=config.n_bits
        )
        fps = np.zeros((len(valid_mols), config.n_bits), dtype=np.float32)
        for i, mol in enumerate(valid_mols):
            fps[i] = generator.GetFingerprintAsNumPy(mol)
        blocks.append(fps)
        feature_names += [f"fp_{i}" for i in range(config.n_bits)]

    if config.use_descriptors:
        functions = _descriptor_functions()
        descriptors = np.full((len(valid_mols), len(functions)), np.nan, dtype=np.float64)
        for i, mol in enumerate(valid_mols):
            for j, (_, function) in enumerate(functions):
                try:
                    descriptors[i, j] = function(mol)
                except Exception:
                    # A handful of descriptors raise on exotic valences; leave
                    # the value missing rather than discarding the molecule.
                    descriptors[i, j] = np.nan
        # Some descriptors (notably Ipc) overflow to +/-inf on large molecules.
        descriptors[~np.isfinite(descriptors)] = np.nan
        blocks.append(descriptors.astype(np.float32))
        feature_names += [f"desc_{name}" for name, _ in functions]

    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    return X, feature_names, valid_mask


def featurize_cached(
    smiles: list[str] | np.ndarray,
    config: FeatureConfig | None = None,
    cache_dir: str | Path | None = "data/features",
    tag: str = "",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """``featurize`` with an on-disk cache keyed by the SMILES and config.

    Featurizing ~7k molecules takes tens of seconds; caching keeps repeated
    experiment runs fast.
    """
    config = config or FeatureConfig()
    if cache_dir is None:
        return featurize(smiles, config)

    import hashlib

    smiles = list(smiles)
    digest = hashlib.sha256(
        ("\n".join(smiles) + "|" + config.cache_key()).encode("utf-8")
    ).hexdigest()[:16]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}_{config.cache_key()}_{digest}.npz"

    if path.exists():
        with np.load(path, allow_pickle=True) as payload:
            return (
                payload["X"],
                [str(name) for name in payload["feature_names"]],
                payload["valid_mask"],
            )

    X, feature_names, valid_mask = featurize(smiles, config)
    # Write through a handle: np.savez_compressed appends ".npz" to a plain
    # path that lacks it, which would defeat the atomic rename below.
    tmp = path.with_suffix(".npz.part")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            X=X,
            feature_names=np.array(feature_names, dtype=object),
            valid_mask=valid_mask,
        )
    tmp.replace(path)
    return X, feature_names, valid_mask
