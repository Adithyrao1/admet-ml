"""ADMET property prediction demo.

Prediction is one tab; the rest expose the evaluation behind it. Models are
pre-trained by ``scripts/build_artifacts.py`` and loaded from ``models/`` so
cold starts cost a second rather than a minute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from admet.domain import BBB_EXPECTED_DIRECTION, LIPINSKI_RULES  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"

ENDPOINTS = {
    "bbb_martins": {
        "title": "Blood-brain barrier",
        "meaning": "crosses the blood-brain barrier",
        "short": "BBB",
    },
    "ames": {
        "title": "Ames mutagenicity",
        "meaning": "is mutagenic in the Ames test",
        "short": "Ames",
    },
}

EXAMPLES = {
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Diazepam": "CN1c2ccc(Cl)cc2C(=Nc2ccccc2)CC1=O",
    "Atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Sucrose": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
    "2-Nitrofluorene": "O=[N+]([O-])c1ccc2c(c1)Cc1ccccc1-2",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}

DESCRIPTOR_VIEW = [
    ("MolWt", "MolWt", "Da"),
    ("MolLogP", "LogP", ""),
    ("TPSA", "TPSA", "A^2"),
    ("NumHDonors", "H-bond donors", ""),
    ("NumHAcceptors", "H-bond acceptors", ""),
    ("NumRotatableBonds", "Rotatable bonds", ""),
]

METRIC_LABELS = {
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_f1": "F1",
    "test_mcc": "MCC",
    "test_balanced_accuracy": "Balanced acc.",
}

MODEL_LABELS = {
    "dummy": "Baseline",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

STATUS_NOTE = {
    "train": "In training set. Prediction is recall, not evidence.",
    "holdout": "Held out from training.",
    "unseen": "Not in the dataset.",
}

st.set_page_config(page_title="ADMET Prediction", layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    import joblib

    if not MODELS_DIR.exists():
        return {}
    bundle = {}
    for name in ENDPOINTS:
        full = MODELS_DIR / f"{name}_full.joblib"
        desc = MODELS_DIR / f"{name}_desc.joblib"
        if not (full.exists() and desc.exists()):
            continue
        fps = MODELS_DIR / f"{name}_trainfp.npy"
        train_smiles = MODELS_DIR / f"{name}_train_smiles.txt"
        holdout = MODELS_DIR / f"{name}_holdout.csv"
        bundle[name] = {
            "full": joblib.load(full),
            "desc": joblib.load(desc),
            "train_fps": np.load(fps) if fps.exists() else None,
            "train_smiles": (
                set(train_smiles.read_text(encoding="utf-8").split("\n"))
                if train_smiles.exists()
                else set()
            ),
            "holdout": pd.read_csv(holdout) if holdout.exists() else None,
        }
    metadata = MODELS_DIR / "metadata.json"
    if metadata.exists():
        bundle["_metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
    return bundle


@st.cache_resource(show_spinner=False)
def load_explainer(dataset: str):
    import shap

    models = load_models()
    if dataset not in models:
        return None, []
    bundle = models[dataset]["desc"]
    names = [n.replace("desc_", "") for n in bundle["feature_names"]]
    return shap.TreeExplainer(bundle["pipeline"].named_steps["model"]), names


@st.cache_resource(show_spinner=False)
def unpacked_training_fingerprints(dataset: str) -> np.ndarray | None:
    """Unpacked once and kept; unpacking per query would redo ~10 MB of work."""
    packed = load_models().get(dataset, {}).get("train_fps")
    if packed is None:
        return None
    return np.unpackbits(packed, axis=1).astype(bool)


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame | None:
    path = RESULTS_DIR / "summary.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_descriptor_importance() -> pd.DataFrame | None:
    path = RESULTS_DIR / "bbb_martins_descriptor_importance.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------
# Chemistry
# --------------------------------------------------------------------------
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


@st.cache_data(show_spinner=False)
def featurize_one(smiles: str, with_fingerprints: bool) -> np.ndarray | None:
    from admet.features import FeatureConfig, featurize

    config = FeatureConfig(use_fingerprints=with_fingerprints, use_descriptors=True)
    X, _, valid = featurize([smiles], config)
    return X if valid[0] else None


@st.cache_data(show_spinner=False)
def nearest_similarity(smiles: str, dataset: str) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    train = unpacked_training_fingerprints(dataset)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if train is None or mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query = generator.GetFingerprintAsNumPy(mol).astype(bool)
    intersection = (train & query).sum(axis=1)
    union = (train | query).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(union > 0, intersection / union, 0.0).max())


@st.cache_data(show_spinner=False)
def membership(smiles: str, dataset: str) -> str:
    """train / holdout / unseen -- separates recall from inference."""
    from admet.data import canonicalize

    models = load_models()
    if dataset not in models:
        return "unknown"
    canonical = canonicalize(smiles)
    if canonical is None:
        return "unknown"
    if canonical in models[dataset]["train_smiles"]:
        return "train"
    holdout = models[dataset]["holdout"]
    if holdout is not None and canonical in set(holdout["canonical_smiles"]):
        return "holdout"
    return "unseen"


@st.cache_data(show_spinner=False)
def true_label(smiles: str, dataset: str) -> int | None:
    from admet.data import canonicalize

    holdout = load_models().get(dataset, {}).get("holdout")
    if holdout is None:
        return None
    match = holdout[holdout["canonical_smiles"] == canonicalize(smiles)]
    return int(match.iloc[0]["Y"]) if len(match) else None


def predict(smiles: str, dataset: str) -> float | None:
    models = load_models()
    if dataset not in models:
        return None
    features = featurize_one(smiles, with_fingerprints=True)
    if features is None:
        return None
    return float(models[dataset]["full"]["pipeline"].predict_proba(features)[0, 1])


def explain_one(smiles: str, dataset: str, top_n: int = 10) -> pd.DataFrame | None:
    explainer, names = load_explainer(dataset)
    features = featurize_one(smiles, with_fingerprints=False)
    if explainer is None or features is None:
        return None

    pipeline = load_models()[dataset]["desc"]["pipeline"]
    imputed = pipeline.named_steps["impute"].transform(features)
    values = explainer.shap_values(imputed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values).reshape(-1)

    order = np.argsort(np.abs(values))[::-1][:top_n]
    return pd.DataFrame(
        {
            "descriptor": [names[i] for i in order],
            "value": [float(imputed[0, i]) for i in order],
            "shap": [float(values[i]) for i in order],
        }
    )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
with st.sidebar:
    choice = st.selectbox("Example", ["Custom"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, EXAMPLES["Caffeine"])
    smiles = st.text_input("SMILES", value=default).strip()

    with st.expander("Method"):
        st.markdown(
            """
- XGBoost on 2048-bit Morgan fingerprints + 217 RDKit descriptors
- TDC scaffold split, 70/10/20, seeds 42/43/44
- Preprocessing fitted inside the CV fold
- Threshold set on validation; test scored once
- SHAP computed on a descriptor-only model
"""
        )
    st.caption("Research baselines. Not for clinical use.")

models = load_models()
if not models:
    st.error("No artifacts in `models/`. Run `python scripts/build_artifacts.py`.")
    st.stop()

st.title("ADMET Property Prediction")

tab_predict, tab_bench, tab_split, tab_interp, tab_limits = st.tabs(
    ["Predict", "Benchmark", "Splits", "Interpretability", "Limitations"]
)


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
with tab_predict:
    profile = descriptor_profile(smiles) if smiles else None
    if not smiles:
        st.info("Enter a SMILES string.")
    elif profile is None:
        st.error("RDKit could not parse this SMILES.")
    else:
        left, right = st.columns([1, 1.4])

        with left:
            from rdkit import Chem
            from rdkit.Chem import Draw

            st.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(320, 240)))
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Property": pretty, "Value": f"{profile[key]:.2f} {unit}".strip()}
                        for key, pretty, unit in DESCRIPTOR_VIEW
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            violations = [
                message for key, limit, message in LIPINSKI_RULES if profile.get(key, 0) > limit
            ]
            st.caption(
                "Lipinski: " + (", ".join(violations) if violations else "no violations")
            )

        with right:
            for dataset, meta in ENDPOINTS.items():
                probability = predict(smiles, dataset)
                if probability is None:
                    continue
                similarity = nearest_similarity(smiles, dataset)
                where = membership(smiles, dataset)

                with st.container(border=True):
                    head, sim = st.columns([2, 1])
                    head.metric(meta["title"], f"{probability:.1%}")
                    if similarity is not None:
                        sim.metric("Nearest train", f"{similarity:.2f}")

                    note = STATUS_NOTE.get(where, "")
                    if where == "holdout":
                        actual = true_label(smiles, dataset)
                        if actual is not None:
                            correct = (probability >= 0.5) == bool(actual)
                            note += (
                                f" True label {'positive' if actual else 'negative'};"
                                f" model is {'right' if correct else 'wrong'}."
                            )
                    elif where == "unseen" and similarity is not None and similarity < 0.4:
                        note += f" Nearest neighbour {similarity:.2f} - extrapolation."
                    st.caption(note)

            endpoint = st.radio(
                "SHAP endpoint",
                list(ENDPOINTS),
                format_func=lambda d: ENDPOINTS[d]["title"],
                horizontal=True,
                label_visibility="collapsed",
            )
            contributions = explain_one(smiles, endpoint)
            if contributions is not None and not contributions.empty:
                contributions = contributions.assign(
                    direction=np.where(contributions["shap"] >= 0, "increases", "decreases"),
                    label=[
                        f"{row.descriptor} = {row.value:.1f}"
                        for row in contributions.itertuples()
                    ],
                )
                st.altair_chart(
                    alt.Chart(contributions)
                    .mark_bar()
                    .encode(
                        x=alt.X("shap:Q", title="SHAP (log-odds)"),
                        y=alt.Y("label:N", sort="-x", title=None),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(
                                domain=["increases", "decreases"], range=[BLUE, ORANGE]
                            ),
                            legend=alt.Legend(title=None, orient="bottom"),
                        ),
                        tooltip=[
                            alt.Tooltip("descriptor:N"),
                            alt.Tooltip("value:Q", format=".2f"),
                            alt.Tooltip("shap:Q", format="+.3f"),
                        ],
                    )
                    .properties(height=280),
                    use_container_width=True,
                )


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
with tab_bench:
    summary = load_summary()
    if summary is None:
        st.info("Run `python scripts/analyze.py`.")
    else:
        controls = st.columns([1, 1, 2])
        picked_dataset = controls[0].selectbox(
            "Dataset", sorted(summary["dataset"].unique()), key="bench_dataset"
        )
        picked_split = controls[1].selectbox(
            "Split", ["scaffold", "random"], key="bench_split"
        )

        frame = summary[
            (summary["dataset"] == picked_dataset) & (summary["split"] == picked_split)
        ].copy()
        order = {name: index for index, name in enumerate(MODEL_LABELS)}
        frame = frame.sort_values("model", key=lambda s: s.map(order))

        table = pd.DataFrame({"Model": frame["model"].map(MODEL_LABELS)})
        for metric, label in METRIC_LABELS.items():
            table[label] = [
                f"{m:.3f} +/- {s:.3f}"
                for m, s in zip(frame[f"{metric}_mean"], frame[f"{metric}_std"])
            ]
        table["Train time"] = [f"{t:.0f}s" for t in frame["train_seconds_mean"]]
        st.dataframe(table, hide_index=True, use_container_width=True)

        chart_data = pd.DataFrame(
            {
                "Model": frame["model"].map(MODEL_LABELS),
                "ROC-AUC": frame["test_roc_auc_mean"],
                "sd": frame["test_roc_auc_std"],
                "Baseline": frame["model"] == "dummy",
            }
        )
        bars = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", sort=None, title=None),
                y=alt.Y("ROC-AUC:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Baseline:N",
                    scale=alt.Scale(domain=[False, True], range=[BLUE, ORANGE]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("ROC-AUC:Q", format=".3f"),
                    alt.Tooltip("sd:Q", format=".3f", title="SD"),
                ],
            )
            .properties(height=300)
        )
        rule = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(strokeDash=[4, 3], color="#52514e")
            .encode(y="y:Q")
        )
        st.altair_chart(bars + rule, use_container_width=True)
        st.caption("48 runs: 4 models x 2 datasets x 2 splits x 3 seeds. Dashed line: chance.")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
with tab_split:
    summary = load_summary()
    if summary is None:
        st.info("Run `python scripts/analyze.py`.")
    else:
        controls = st.columns([1, 1, 2])
        dataset = controls[0].selectbox(
            "Dataset", sorted(summary["dataset"].unique()), key="split_dataset"
        )
        split_choice = controls[1].selectbox(
            "Evaluated on", ["scaffold", "random"], key="split_choice"
        )
        other = "random" if split_choice == "scaffold" else "scaffold"

        real = summary[(summary["dataset"] == dataset) & (summary["model"] != "dummy")]
        current = real[real["split"] == split_choice]
        comparison = real[real["split"] == other]

        columns = st.columns(len(METRIC_LABELS))
        for column, (metric, label) in zip(columns, METRIC_LABELS.items()):
            value = float(current[f"{metric}_mean"].mean())
            column.metric(
                label, f"{value:.3f}", f"{value - float(comparison[f'{metric}_mean'].mean()):+.3f}"
            )
        st.caption(
            f"Mean over the three non-baseline models. Delta vs the {other} split."
        )

        gap_rows = []
        for ds in sorted(summary["dataset"].unique()):
            frame = summary[(summary["dataset"] == ds) & (summary["model"] != "dummy")]
            for metric, label in METRIC_LABELS.items():
                gap_rows.append(
                    {
                        "Dataset": ds,
                        "Metric": label,
                        "Inflation": frame[frame["split"] == "random"][f"{metric}_mean"].mean()
                        - frame[frame["split"] == "scaffold"][f"{metric}_mean"].mean(),
                    }
                )

        st.altair_chart(
            alt.Chart(pd.DataFrame(gap_rows))
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", sort=list(METRIC_LABELS.values()), title=None),
                y=alt.Y("Inflation:Q", title="Random minus scaffold"),
                color=alt.Color(
                    "Dataset:N",
                    scale=alt.Scale(range=[BLUE, ORANGE]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                xOffset="Dataset:N",
                tooltip=[
                    alt.Tooltip("Dataset:N"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Inflation:Q", format="+.3f"),
                ],
            )
            .properties(height=300),
            use_container_width=True,
        )
        st.caption(
            "MCC drops ~0.12 on both datasets under a scaffold split; ROC-AUC moves "
            "+0.061 on AMES and +0.003 on BBB."
        )


# --------------------------------------------------------------------------
# Interpretability
# --------------------------------------------------------------------------
with tab_interp:
    importance = load_descriptor_importance()
    if importance is None:
        st.info("Run `python scripts/explain.py --dataset bbb_martins`.")
    else:
        lookup = dict(zip(importance["descriptor"], importance["value_shap_corr"]))
        checks = []
        for descriptor, (expected, _) in BBB_EXPECTED_DIRECTION.items():
            if descriptor not in lookup:
                checks.append(
                    {
                        "Descriptor": descriptor,
                        "Expected": expected,
                        "Observed": "-",
                        "r": None,
                        "Match": "not in top 15",
                    }
                )
                continue
            correlation = float(lookup[descriptor])
            observed = "positive" if correlation > 0 else "negative"
            checks.append(
                {
                    "Descriptor": descriptor,
                    "Expected": expected,
                    "Observed": observed,
                    "r": round(correlation, 2),
                    "Match": "yes" if observed == expected else "no",
                }
            )

        st.markdown("**BBB direction check** (expectations recorded in `src/admet/domain.py`)")
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

        st.altair_chart(
            alt.Chart(importance.head(15))
            .mark_bar(color=BLUE)
            .encode(
                x=alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
                y=alt.Y("descriptor:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("descriptor:N"),
                    alt.Tooltip("mean_abs_shap:Q", format=".3f"),
                    alt.Tooltip("direction:N"),
                ],
            )
            .properties(height=400),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------
with tab_limits:
    st.markdown("**Held-out molecules**")
    controls = st.columns([1, 1, 2])
    eval_dataset = controls[0].selectbox(
        "Endpoint",
        list(ENDPOINTS),
        format_func=lambda d: ENDPOINTS[d]["title"],
        key="fail_dataset",
    )
    sample_seed = controls[1].number_input(
        "Sample", min_value=0, max_value=9999, value=0, step=1, key="fail_seed"
    )

    holdout = models.get(eval_dataset, {}).get("holdout")
    if holdout is None or holdout.empty:
        st.info("Run `python scripts/build_artifacts.py`.")
    else:
        sample = holdout.sample(n=min(8, len(holdout)), random_state=int(sample_seed))
        rows, correct_count = [], 0
        for row in sample.itertuples():
            probability = predict(row.canonical_smiles, eval_dataset)
            if probability is None:
                continue
            is_correct = int(probability >= 0.5) == int(row.Y)
            correct_count += is_correct
            rows.append(
                {
                    "SMILES": row.canonical_smiles[:40],
                    "Predicted": f"{probability:.1%}",
                    "True": "positive" if row.Y else "negative",
                    "Correct": "yes" if is_correct else "no",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"{correct_count}/{len(rows)} on this draw. Full-holdout numbers are on "
            "the Benchmark tab."
        )

    st.markdown("**Training-set molecules**")
    memorised = []
    for label, case_smiles, dataset in [
        ("Caffeine", EXAMPLES["Caffeine"], "bbb_martins"),
        ("Atenolol", EXAMPLES["Atenolol"], "bbb_martins"),
        ("Aspirin", EXAMPLES["Aspirin"], "bbb_martins"),
        ("2-Nitrofluorene", EXAMPLES["2-Nitrofluorene"], "ames"),
    ]:
        probability = predict(case_smiles, dataset)
        if probability is None:
            continue
        memorised.append(
            {
                "Molecule": label,
                "Endpoint": ENDPOINTS[dataset]["short"],
                "Predicted": f"{probability:.1%}",
                "Status": membership(case_smiles, dataset),
            }
        )
    st.dataframe(pd.DataFrame(memorised), hide_index=True, use_container_width=True)
    st.caption("Recalled, not inferred. Excluded from any claim about accuracy.")

    st.markdown("**Out of domain**")
    sucrose = EXAMPLES["Sucrose"]
    probability = predict(sucrose, "bbb_martins")
    similarity = nearest_similarity(sucrose, "bbb_martins")
    columns = st.columns([1, 1, 2])
    columns[0].metric("Sucrose, BBB", f"{probability:.1%}" if probability else "-")
    columns[1].metric("Nearest train", f"{similarity:.2f}" if similarity else "-")
    columns[2].caption(
        "A polar sugar that does not reach the brain, scored near chance. Not in "
        "the BBB dataset, so there is no label to check against; the expected "
        "answer comes from pharmacology."
    )

    st.markdown(
        """
**Known limits**

- Two binary endpoints; no ADME regression tasks.
- 2k and 7k molecules. A graph neural network would likely lead at 100k.
- Ames is one assay, not a full genotoxicity assessment.
- Tanimoto similarity is a heuristic, not calibrated confidence.
"""
    )
