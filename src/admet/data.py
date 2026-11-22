"""Dataset loading and splitting for ADMET endpoints.

Data is downloaded directly from the Harvard Dataverse files that back the
Therapeutics Data Commons (TDC). We deliberately do not depend on the ``PyTDC``
package at runtime: it pins ``numpy<2``, ``pandas<3`` and ``rdkit<2024.3.1`` and
pulls in a large protein/single-cell stack that is irrelevant here, which makes
the project hard to install. The file IDs and the scaffold-split algorithm below
mirror TDC so that results stay comparable with the TDC leaderboard.

Scaffold split reimplements ``tdc.utils.split.create_scaffold_split``, which in
turn follows chemprop's balanced scaffold split.
"""

from __future__ import annotations

import io
import random as _random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

SMILES_COL = "Drug"
LABEL_COL = "Y"
ID_COL = "Drug_ID"

DEFAULT_FRAC = (0.7, 0.1, 0.2)


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one ADMET endpoint."""

    name: str
    file_id: int
    task: str
    metric: str
    official_split: str
    positive_meaning: str
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "bbb_martins": DatasetSpec(
        name="bbb_martins",
        file_id=4259566,
        task="classification",
        metric="roc_auc",
        official_split="scaffold",
        positive_meaning="crosses the blood-brain barrier",
        description=(
            "Blood-brain barrier penetration (Martins et al.). Binary: does the "
            "compound reach the central nervous system?"
        ),
    ),
    "ames": DatasetSpec(
        name="ames",
        file_id=4259564,
        task="classification",
        metric="roc_auc",
        official_split="scaffold",
        positive_meaning="mutagenic in the Ames test",
        description=(
            "Ames mutagenicity. Binary: is the compound mutagenic to Salmonella "
            "typhimurium, a standard early genotoxicity screen?"
        ),
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}"
        ) from None


def _cache_path(spec: DatasetSpec, cache_dir: Path) -> Path:
    return cache_dir / f"{spec.name}.tab"


def download_raw(name: str, cache_dir: str | Path = "data/raw") -> Path:
    """Download the raw endpoint file, caching it on disk.

    Returns the path to the cached file.
    """
    spec = get_spec(name)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(spec, cache_dir)

    if path.exists() and path.stat().st_size > 0:
        return path

    url = DATAVERSE_URL.format(file_id=spec.file_id)
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    # Write via a temporary file so an interrupted download cannot leave a
    # truncated file that later looks like a valid cache hit.
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


def canonicalize(smiles: str) -> str | None:
    """Return the canonical SMILES, or None if RDKit cannot parse the string."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_dataset(
    name: str,
    cache_dir: str | Path = "data/raw",
    deduplicate: bool = True,
    on_conflict: str = "keep_first",
) -> pd.DataFrame:
    """Load one endpoint as a dataframe with columns ``Drug_ID``/``Drug``/``Y``.

    Invalid SMILES are dropped. When ``deduplicate`` is set, molecules are
    collapsed on canonical SMILES.

    A few molecules appear twice with *contradictory* labels. ``on_conflict``
    chooses what to do with them:

    * ``"keep_first"`` (default) keeps the first occurrence. This reproduces
      TDC's own counts exactly (bbb_martins 1975, ames 7255), so results stay
      comparable with the TDC leaderboard.
    * ``"drop"`` removes every molecule involved in a label conflict, which is
      stricter but yields slightly smaller datasets.

    Either way the conflict count is recorded in ``df.attrs["provenance"]``.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    path = download_raw(name, cache_dir=cache_dir)
    df = pd.read_csv(path, sep="\t")

    missing = {ID_COL, SMILES_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing expected columns {sorted(missing)}")

    n_raw = len(df)
    df = df.copy()
    df["canonical_smiles"] = df[SMILES_COL].map(canonicalize)

    n_invalid = int(df["canonical_smiles"].isna().sum())
    df = df[df["canonical_smiles"].notna()].reset_index(drop=True)

    n_conflicting = 0
    n_duplicate = 0
    if deduplicate:
        if on_conflict not in {"keep_first", "drop"}:
            raise ValueError(
                f"on_conflict must be 'keep_first' or 'drop', got {on_conflict!r}"
            )

        label_spread = df.groupby("canonical_smiles")[LABEL_COL].transform("nunique")
        conflicting = label_spread > 1
        n_conflicting = int(conflicting.sum())

        if on_conflict == "drop":
            df = df[~conflicting]

        before = len(df)
        df = df.drop_duplicates(subset="canonical_smiles", keep="first")
        n_duplicate = before - len(df)
        df = df.reset_index(drop=True)

    df.attrs["provenance"] = {
        "dataset": name,
        "n_raw": n_raw,
        "n_invalid_smiles": n_invalid,
        "n_conflicting_label_rows": n_conflicting,
        "conflict_policy": on_conflict if deduplicate else "none",
        "n_duplicates_removed": n_duplicate,
        "n_final": len(df),
    }
    return df


def split_random(
    df: pd.DataFrame,
    seed: int,
    frac: tuple[float, float, float] = DEFAULT_FRAC,
) -> dict[str, pd.DataFrame]:
    """Random split. Mirrors ``tdc.utils.split.create_fold``."""
    _, val_frac, test_frac = frac

    test = df.sample(frac=test_frac, replace=False, random_state=seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=seed
    )
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """Bemis-Murcko scaffold as SMILES, or None if RDKit fails."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return None


def split_scaffold(
    df: pd.DataFrame,
    seed: int,
    frac: tuple[float, float, float] = DEFAULT_FRAC,
    smiles_col: str = "canonical_smiles",
) -> dict[str, pd.DataFrame]:
    """Balanced scaffold split.

    Molecules sharing a Bemis-Murcko scaffold always land in the same fold, so
    the test set contains chemical series the model never saw. This is TDC's
    official protocol for these endpoints and gives markedly lower -- and more
    honest -- scores than a random split.

    Reimplements ``tdc.utils.split.create_scaffold_split``.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    if smiles_col not in df.columns:
        smiles_col = SMILES_COL

    smiles = df[smiles_col].values
    scaffolds: dict[str, set[int]] = defaultdict(set)
    n_errors = 0
    for i, smi in enumerate(smiles):
        scaffold = murcko_scaffold(smi)
        if scaffold is None:
            n_errors += 1
            continue
        scaffolds[scaffold].add(i)

    n_usable = len(df) - n_errors
    train_size = int(n_usable * frac[0])
    val_size = int(n_usable * frac[1])
    test_size = n_usable - train_size - val_size

    # Scaffold groups large enough to unbalance a fold are shuffled separately
    # and placed first, so no single chemical series dominates valid or test.
    big_index_sets: list[list[int]] = []
    small_index_sets: list[list[int]] = []
    for index_set in scaffolds.values():
        ordered = sorted(index_set)
        if len(ordered) > val_size / 2 or len(ordered) > test_size / 2:
            big_index_sets.append(ordered)
        else:
            small_index_sets.append(ordered)

    rng = _random.Random(seed)
    rng.shuffle(big_index_sets)
    rng.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
        elif len(val) + len(index_set) <= val_size:
            val += index_set
        else:
            test += index_set

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def get_split(
    name: str,
    method: str = "scaffold",
    seed: int = 42,
    frac: tuple[float, float, float] = DEFAULT_FRAC,
    cache_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    """Load an endpoint and split it into train/valid/test."""
    if abs(sum(frac) - 1.0) > 1e-9:
        raise ValueError(f"frac must sum to 1.0, got {frac} summing to {sum(frac)}")

    df = load_dataset(name, cache_dir=cache_dir)
    provenance = df.attrs.get("provenance", {})

    if method == "random":
        splits = split_random(df, seed=seed, frac=frac)
    elif method == "scaffold":
        splits = split_scaffold(df, seed=seed, frac=frac)
    else:
        raise ValueError(f"Unknown split method {method!r}; use 'random' or 'scaffold'")

    for split in splits.values():
        split.attrs["provenance"] = provenance
    return splits
