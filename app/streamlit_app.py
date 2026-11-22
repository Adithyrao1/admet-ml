"""Interactive ADMET predictor.

Paste a molecule as SMILES and get blood-brain barrier and Ames mutagenicity
predictions, the physicochemical profile behind them, and an honest read on how
far the molecule sits from the training data.

Models are trained on first load and cached, so the repository carries no binary
model files and what the app serves is always what the code produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from admet.data import LABEL_COL, DATASETS, get_split  # noqa: E402
from admet.features import FeatureConfig, featurize  # noqa: E402
from admet.models import build_model, set_scale_pos_weight  # noqa: E402

st.set_page_config(page_title="ADMET Predictor", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Caffeine (CNS active)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Diazepam (CNS active)": "CN1c2ccc(Cl)cc2C(=Nc2ccccc2)CC1=O",
    "Atenolol (poor CNS)": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Sucrose (poor CNS)": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
    "2-Nitrofluorene (mutagen)": "O=[N+]([O-])c1ccc2c(c1)Cc1ccccc1-2",
}

DESCRIPTOR_VIEW = [
    ("MolWt", "Molecular weight", "Da"),
    ("MolLogP", "LogP (lipophilicity)", ""),
    ("TPSA", "Polar surface area", "Å²"),
    ("NumHDonors", "H-bond donors", ""),
    ("NumHAcceptors", "H-bond acceptors", ""),
    ("NumRotatableBonds", "Rotatable bonds", ""),
]


@st.cache_resource(show_spinner=False)
def train_models() -> dict:
    """Train one model per endpoint on the scaffold-split training fold."""
    config = FeatureConfig()
    trained = {}
    for name in DATASETS:
        splits = get_split(name, method="scaffold", seed=42)
        smiles = splits["train"]["canonical_smiles"].tolist()
        X, _, valid = featurize(smiles, config)
        y = splits["train"][LABEL_COL].to_numpy()[valid]

        pipeline, _ = build_model("xgboost", seed=42)
        pipeline = set_scale_pos_weight(pipeline, y)
        pipeline.set_params(model__max_depth=6, model__colsample_bytree=0.6)
        pipeline.fit(X, y)
        trained[name] = {"pipeline": pipeline, "train_X": X}
    return trained


@st.cache_data(show_spinner=False)
def descriptor_profile(smiles: str) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }


def lipinski_violations(profile: dict) -> list[str]:
    violations = []
    if profile["MolWt"] > 500:
        violations.append("molecular weight > 500")
    if profile["MolLogP"] > 5:
        violations.append("LogP > 5")
    if profile["NumHDonors"] > 5:
        violations.append("H-bond donors > 5")
    if profile["NumHAcceptors"] > 10:
        violations.append("H-bond acceptors > 10")
    return violations


def nearest_training_similarity(smiles: str, dataset: str) -> float | None:
    """Tanimoto similarity to the most similar training molecule.

    A low value means the prediction is an extrapolation, which is the single
    most useful caveat to show beside a number like this.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    splits = get_split(dataset, method="scaffold", seed=42)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query = generator.GetFingerprint(mol)

    best = 0.0
    for train_smiles in splits["train"]["canonical_smiles"]:
        train_mol = Chem.MolFromSmiles(train_smiles)
        if train_mol is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(
            query, generator.GetFingerprint(train_mol)
        )
        best = max(best, similarity)
    return best


def render_prediction(label: str, probability: float, positive_meaning: str):
    st.metric(label, f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Probability that the molecule {positive_meaning}.")


st.title("🧪 ADMET Property Predictor")
st.caption(
    "Classical ML on RDKit features. Predicts blood-brain barrier penetration and "
    "Ames mutagenicity from a molecular structure."
)

with st.sidebar:
    st.header("Input")
    choice = st.selectbox("Example molecule", ["(type my own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    smiles_input = st.text_input("SMILES", value=default)
    check_similarity = st.checkbox(
        "Check similarity to training data", value=True,
        help="Slower, but tells you whether the prediction is an extrapolation.",
    )
    st.divider()
    st.caption(
        "**Not for clinical or regulatory use.** These are research baselines "
        "trained on a few thousand molecules."
    )

smiles = smiles_input.strip()
if not smiles:
    st.info("Enter a SMILES string to get started.")
    st.stop()

profile = descriptor_profile(smiles)
if profile is None:
    st.error(f"RDKit could not parse `{smiles}` as a molecule. Check the SMILES string.")
    st.stop()

with st.spinner("Training models on first run (about a minute), then cached..."):
    models = train_models()

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Structure")
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    st.image(Draw.MolToImage(mol, size=(360, 280)), use_container_width=False)

    st.subheader("Physicochemical profile")
    rows = [
        {"Property": pretty, "Value": f"{profile[key]:.2f} {unit}".strip()}
        for key, pretty, unit in DESCRIPTOR_VIEW
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    violations = lipinski_violations(profile)
    if violations:
        st.warning("Lipinski rule-of-5 violations: " + ", ".join(violations))
    else:
        st.success("Passes all four Lipinski rule-of-5 criteria.")

with right:
    st.subheader("Predictions")
    features, _, valid = featurize([smiles], FeatureConfig())
    if not valid[0]:
        st.error("Featurization failed for this molecule.")
        st.stop()

    for name, spec in DATASETS.items():
        pipeline = models[name]["pipeline"]
        probability = float(pipeline.predict_proba(features)[0, 1])
        title = "Blood-brain barrier" if name == "bbb_martins" else "Ames mutagenicity"
        with st.container(border=True):
            render_prediction(title, probability, spec.positive_meaning)

            if check_similarity:
                similarity = nearest_training_similarity(smiles, name)
                if similarity is not None:
                    st.caption(
                        f"Nearest training molecule: Tanimoto {similarity:.2f}"
                    )
                    if similarity < 0.4:
                        st.warning(
                            "This molecule is unlike anything in the training set, "
                            "so treat the number as a rough extrapolation."
                        )

    st.subheader("How to read this")
    st.markdown(
        """
Each number is a **probability**, not a yes/no verdict. Converting it to a
decision needs a cutoff, and the right cutoff depends on whether a false
positive or a false negative costs you more.

Both models were trained on a **scaffold split**, so the reported accuracy
reflects performance on chemical series the model had never seen — a harder
and more realistic test than a random split.
"""
    )
