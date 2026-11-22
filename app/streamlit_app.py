"""ADMET predictor — an interactive demo of the benchmark in this repository.

Deliberately not just a prediction box. Prediction is one tab; the rest show the
evaluation that decides whether the predictions deserve trust: the baseline that
a useless model achieves, what a scaffold split does to each metric, which
chemical properties drive the model, and where it breaks.

Runs server-side. Models are pre-trained by ``scripts/build_artifacts.py`` and
loaded from ``models/`` so cold starts cost a second rather than a minute.
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
    "Caffeine — CNS active": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Diazepam — CNS active": "CN1c2ccc(Cl)cc2C(=Nc2ccccc2)CC1=O",
    "Atenolol — poor CNS": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Sucrose — does not cross": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
    "2-Nitrofluorene — mutagen": "O=[N+]([O-])c1ccc2c(c1)Cc1ccccc1-2",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}

DESCRIPTOR_VIEW = [
    ("MolWt", "Molecular weight", "Da"),
    ("MolLogP", "LogP (lipophilicity)", ""),
    ("TPSA", "Polar surface area", "Å²"),
    ("NumHDonors", "H-bond donors", ""),
    ("NumHAcceptors", "H-bond acceptors", ""),
    ("NumRotatableBonds", "Rotatable bonds", ""),
]

METRIC_LABELS = {
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_f1": "F1",
    "test_mcc": "MCC",
    "test_balanced_accuracy": "Balanced accuracy",
}

MODEL_LABELS = {
    "dummy": "Dummy (baseline)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

st.set_page_config(page_title="ADMET Predictor", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> dict:
    import joblib

    if not MODELS_DIR.exists():
        return {}
    bundle = {}
    for name in ENDPOINTS:
        full = MODELS_DIR / f"{name}_full.joblib"
        desc = MODELS_DIR / f"{name}_desc.joblib"
        fps = MODELS_DIR / f"{name}_trainfp.npy"
        if not (full.exists() and desc.exists()):
            continue
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
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        bundle["_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bundle


@st.cache_resource(show_spinner=False)
def load_explainer(dataset: str):
    """TreeExplainer over the descriptor-only model."""
    import shap

    models = load_models()
    if dataset not in models:
        return None, []
    bundle = models[dataset]["desc"]
    pipeline = bundle["pipeline"]
    names = [n.replace("desc_", "") for n in bundle["feature_names"]]
    return shap.TreeExplainer(pipeline.named_steps["model"]), names


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame | None:
    path = RESULTS_DIR / "summary.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_descriptor_importance() -> pd.DataFrame | None:
    path = RESULTS_DIR / "bbb_martins_descriptor_importance.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------
# Chemistry helpers
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


@st.cache_resource(show_spinner=False)
def unpacked_training_fingerprints(dataset: str) -> np.ndarray | None:
    """Training fingerprints as a bool matrix, unpacked once and kept.

    Unpacking on every lookup would redo ~10 MB of work per new molecule; the
    unpacked matrices together are small enough to simply hold.
    """
    packed = load_models().get(dataset, {}).get("train_fps")
    if packed is None:
        return None
    return np.unpackbits(packed, axis=1).astype(bool)


@st.cache_data(show_spinner=False)
def nearest_similarity(smiles: str, dataset: str) -> float | None:
    """Max Tanimoto similarity to the training set, from packed fingerprints.

    This is the applicability-domain check: a confident number on a molecule
    unlike anything in training is an extrapolation, not a prediction.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    train = unpacked_training_fingerprints(dataset)
    if train is None:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query = generator.GetFingerprintAsNumPy(mol).astype(bool)

    intersection = (train & query).sum(axis=1)
    union = (train | query).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(union > 0, intersection / union, 0.0)
    return float(tanimoto.max())


@st.cache_data(show_spinner=False)
def membership(smiles: str, dataset: str) -> str:
    """Where this molecule sits relative to the model: train, holdout, or unseen.

    This is the difference between a prediction that proves something and one
    that is simply recalled. A demo that only shows training molecules proves
    nothing at all.
    """
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
    """Ground-truth label for a held-out molecule, when we have one."""
    from admet.data import canonicalize

    models = load_models()
    holdout = models.get(dataset, {}).get("holdout")
    if holdout is None:
        return None
    canonical = canonicalize(smiles)
    match = holdout[holdout["canonical_smiles"] == canonical]
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
    """Per-molecule SHAP contributions from the descriptor-only model."""
    explainer, names = load_explainer(dataset)
    if explainer is None:
        return None

    features = featurize_one(smiles, with_fingerprints=False)
    if features is None:
        return None

    models = load_models()
    pipeline = models[dataset]["desc"]["pipeline"]
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
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Molecule")
    choice = st.selectbox("Example", ["(type my own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, EXAMPLES["Caffeine — CNS active"])
    smiles = st.text_input("SMILES", value=default).strip()

    st.divider()
    with st.expander("Methodology"):
        st.markdown(
            """
**Leakage controls**
- Imputation and scaling live inside the sklearn `Pipeline`, so they are fitted
  per CV fold, never on the full dataset.
- Molecules are deduplicated on canonical SMILES before splitting.
- The decision threshold is chosen on **validation**; the test set is scored
  once.
- A test asserts that no molecule and no scaffold appears in both train and
  test.

**Protocol**
- TDC scaffold split (official) and a random split, 70/10/20.
- 3 seeds (42, 43, 44); every figure is mean ± SD.
- Equal hyperparameter budget for every model within a dataset.

**Served model**
- XGBoost on 2048-bit Morgan fingerprints + 217 RDKit descriptors.
- Explanations come from a descriptor-only twin, because a fingerprint bit
  means nothing to a reader.
"""
        )
    st.caption(
        "Research baselines on a few thousand molecules. "
        "**Not for clinical or regulatory use.**"
    )

models = load_models()
if not models:
    st.error(
        "No model artifacts found in `models/`. Build them with:\n\n"
        "```bash\npython scripts/build_artifacts.py\n```"
    )
    st.stop()

st.title("🧪 ADMET Property Predictor")
st.caption(
    "Classical ML on RDKit features — and the evaluation that decides whether "
    "to believe it."
)

tab_predict, tab_bench, tab_split, tab_learned, tab_fails = st.tabs(
    [
        "🔬 Predict",
        "📊 Benchmark",
        "⚠️ Why the split matters",
        "🧠 What it learned",
        "💥 Where it fails",
    ]
)


# --------------------------------------------------------------------------
# Tab 1 — Predict
# --------------------------------------------------------------------------
with tab_predict:
    profile = descriptor_profile(smiles) if smiles else None
    if not smiles:
        st.info("Enter a SMILES string in the sidebar.")
    elif profile is None:
        st.error(f"RDKit could not parse `{smiles}`. Check the SMILES string.")
    else:
        left, right = st.columns([1, 1.4])

        with left:
            from rdkit import Chem
            from rdkit.Chem import Draw

            st.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(340, 260)))

            st.markdown("**Physicochemical profile**")
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
                message
                for key, limit, message in LIPINSKI_RULES
                if profile.get(key, 0) > limit
            ]
            if violations:
                st.warning("Lipinski violations: " + ", ".join(violations))
            else:
                st.success("Passes all four Lipinski rule-of-5 criteria.")

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
                        sim.metric("Nearest training molecule", f"{similarity:.2f}")
                    st.progress(min(max(probability, 0.0), 1.0))
                    st.caption(f"Probability that the molecule {meta['meaning']}.")

                    if where == "train":
                        st.error(
                            "**This molecule is in the training set.** The model "
                            "has seen it and its answer is recall, not evidence "
                            "of generalization. Judge the model on the held-out "
                            "results instead."
                        )
                    elif where == "holdout":
                        actual = true_label(smiles, dataset)
                        if actual is not None:
                            correct = (probability >= 0.5) == bool(actual)
                            st.success(
                                f"**Held out from training.** True label: "
                                f"{'positive' if actual else 'negative'} — "
                                f"the model is {'right' if correct else 'wrong'} here."
                            )
                    elif similarity is not None and similarity < 0.4:
                        st.warning(
                            f"Tanimoto {similarity:.2f} to the nearest training "
                            "molecule, and not in the dataset at all — this is an "
                            "extrapolation, and the percentage above should not "
                            "be trusted."
                        )

            st.markdown("#### Why this prediction?")
            endpoint = st.radio(
                "Endpoint",
                list(ENDPOINTS),
                format_func=lambda d: ENDPOINTS[d]["title"],
                horizontal=True,
                label_visibility="collapsed",
            )
            contributions = explain_one(smiles, endpoint)
            if contributions is None or contributions.empty:
                st.info("No explanation available.")
            else:
                contributions = contributions.copy()
                contributions["direction"] = np.where(
                    contributions["shap"] >= 0, "increases", "decreases"
                )
                contributions["label"] = [
                    f"{row.descriptor} = {row.value:.1f}"
                    for row in contributions.itertuples()
                ]
                chart = (
                    alt.Chart(contributions)
                    .mark_bar(cornerRadius=3)
                    .encode(
                        x=alt.X("shap:Q", title="SHAP contribution (log-odds)"),
                        y=alt.Y("label:N", sort="-x", title=None),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(
                                domain=["increases", "decreases"],
                                range=[BLUE, ORANGE],
                            ),
                            legend=alt.Legend(title="Effect on probability"),
                        ),
                        tooltip=[
                            alt.Tooltip("descriptor:N"),
                            alt.Tooltip("value:Q", format=".2f"),
                            alt.Tooltip("shap:Q", format="+.3f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "Bars left of zero push the prediction down, right push it up. "
                    "Computed per molecule from the descriptor-only model."
                )


# --------------------------------------------------------------------------
# Tab 2 — Benchmark
# --------------------------------------------------------------------------
with tab_bench:
    summary = load_summary()
    if summary is None:
        st.info("Run `python scripts/analyze.py` to generate results.")
    else:
        st.markdown(
            "48 runs: 4 models × 2 datasets × 2 splits × 3 seeds. "
            "**The Dummy row is the point** — on BBB it reaches F1 0.879 while "
            "ignoring the molecule entirely."
        )

        picked_dataset = st.selectbox(
            "Dataset", sorted(summary["dataset"].unique()), key="bench_dataset"
        )
        picked_split = st.radio(
            "Split", ["scaffold", "random"], horizontal=True, key="bench_split"
        )

        frame = summary[
            (summary["dataset"] == picked_dataset) & (summary["split"] == picked_split)
        ].copy()
        order = {name: index for index, name in enumerate(MODEL_LABELS)}
        frame = frame.sort_values("model", key=lambda s: s.map(order))

        table = pd.DataFrame({"Model": frame["model"].map(MODEL_LABELS)})
        for metric, label in METRIC_LABELS.items():
            table[label] = [
                f"{m:.3f} ± {s:.3f}"
                for m, s in zip(frame[f"{metric}_mean"], frame[f"{metric}_std"])
            ]
        table["Train time"] = [f"{t:.0f}s" for t in frame["train_seconds_mean"]]

        st.dataframe(table, hide_index=True, use_container_width=True)

        chart_data = pd.DataFrame(
            {
                "Model": frame["model"].map(MODEL_LABELS),
                "ROC-AUC": frame["test_roc_auc_mean"],
                "sd": frame["test_roc_auc_std"],
                "is_baseline": frame["model"] == "dummy",
            }
        )
        bars = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Model:N", sort=None, title=None),
                y=alt.Y(
                    "ROC-AUC:Q", scale=alt.Scale(domain=[0, 1]), title="Test ROC-AUC"
                ),
                color=alt.Color(
                    "is_baseline:N",
                    scale=alt.Scale(domain=[False, True], range=[BLUE, ORANGE]),
                    legend=alt.Legend(title="Baseline?"),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("ROC-AUC:Q", format=".3f"),
                    alt.Tooltip("sd:Q", format=".3f", title="± SD"),
                ],
            )
            .properties(height=320)
        )
        rule = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(strokeDash=[4, 3], color="#52514e")
            .encode(y="y:Q")
        )
        st.altair_chart(bars + rule, use_container_width=True)
        st.caption("Dashed line = random guessing (0.5). Error bars are in the table.")


# --------------------------------------------------------------------------
# Tab 3 — Why the split matters
# --------------------------------------------------------------------------
with tab_split:
    summary = load_summary()
    if summary is None:
        st.info("Run `python scripts/analyze.py` to generate results.")
    else:
        st.markdown(
            """
A **random split** lets molecules from the same chemical series land in both
training and test. A **scaffold split** does not: every test molecule belongs to
a chemical series the model has never seen. The second is what deployment
actually looks like.

Switch the split below and watch which metrics notice.
"""
        )

        dataset = st.selectbox(
            "Dataset", sorted(summary["dataset"].unique()), key="split_dataset"
        )
        split_choice = st.radio(
            "Evaluated on",
            ["scaffold", "random"],
            horizontal=True,
            format_func=lambda s: (
                "Scaffold split (honest)" if s == "scaffold" else "Random split (flattering)"
            ),
            key="split_choice",
        )
        other = "random" if split_choice == "scaffold" else "scaffold"

        real = summary[(summary["dataset"] == dataset) & (summary["model"] != "dummy")]
        current = real[real["split"] == split_choice]
        comparison = real[real["split"] == other]

        columns = st.columns(len(METRIC_LABELS))
        for column, (metric, label) in zip(columns, METRIC_LABELS.items()):
            value = float(current[f"{metric}_mean"].mean())
            delta = value - float(comparison[f"{metric}_mean"].mean())
            column.metric(label, f"{value:.3f}", f"{delta:+.3f}")

        st.caption(
            f"Averaged over Logistic Regression, Random Forest and XGBoost. "
            f"Delta is versus the {other} split."
        )

        gap_rows = []
        for ds in sorted(summary["dataset"].unique()):
            frame = summary[(summary["dataset"] == ds) & (summary["model"] != "dummy")]
            for metric, label in METRIC_LABELS.items():
                scaffold = frame[frame["split"] == "scaffold"][f"{metric}_mean"].mean()
                random_split = frame[frame["split"] == "random"][f"{metric}_mean"].mean()
                gap_rows.append(
                    {"Dataset": ds, "Metric": label, "Inflation": random_split - scaffold}
                )
        gaps = pd.DataFrame(gap_rows)

        st.altair_chart(
            alt.Chart(gaps)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Metric:N", sort=list(METRIC_LABELS.values()), title=None),
                y=alt.Y("Inflation:Q", title="Random minus scaffold"),
                color=alt.Color(
                    "Dataset:N", scale=alt.Scale(range=[BLUE, ORANGE])
                ),
                xOffset="Dataset:N",
                tooltip=[
                    alt.Tooltip("Dataset:N"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Inflation:Q", format="+.3f"),
                ],
            )
            .properties(height=320),
            use_container_width=True,
        )

        st.info(
            "**MCC falls by roughly 0.12 on both datasets, while ROC-AUC moves "
            "+0.061 on AMES and only +0.003 on BBB.** A report using ROC-AUC "
            "alone — the metric the TDC leaderboard ranks on — would conclude "
            "BBB has no generalization problem at all. What degrades is "
            "threshold-dependent performance: the model still *ranks* unfamiliar "
            "molecules well, but a cutoff tuned on validation transfers poorly."
        )


# --------------------------------------------------------------------------
# Tab 4 — What it learned
# --------------------------------------------------------------------------
with tab_learned:
    importance = load_descriptor_importance()
    st.markdown(
        """
A good score is not proof that a model learned chemistry — it may have learned
an artefact of the split. So the descriptors driving it are checked against what
medicinal chemistry already knew, with the expected directions written down
in `src/admet/domain.py` **before** looking at the results.
"""
    )

    if importance is None:
        st.info("Run `python scripts/explain.py --dataset bbb_martins` first.")
    else:
        checks = []
        lookup = dict(zip(importance["descriptor"], importance["value_shap_corr"]))
        for descriptor, (expected, rationale) in BBB_EXPECTED_DIRECTION.items():
            if descriptor not in lookup:
                checks.append(
                    {
                        "Descriptor": descriptor,
                        "Expected": expected,
                        "Observed": "—",
                        "r": None,
                        "Verdict": "⚪ not in top 15",
                        "Why expected": rationale,
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
                    "Verdict": "✅ matches" if observed == expected else "❌ differs",
                    "Why expected": rationale,
                }
            )

        matched = sum(1 for c in checks if c["Verdict"].startswith("✅"))
        testable = sum(1 for c in checks if not c["Verdict"].startswith("⚪"))
        st.success(
            f"**{matched} of {testable} testable expectations match.** The model "
            "recovered the polarity-versus-lipophilicity trade-off behind CNS "
            "drug design from 1,382 training molecules, without being told it."
        )
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

        top = importance.head(15).copy()
        st.altair_chart(
            alt.Chart(top)
            .mark_bar(cornerRadius=3, color=BLUE)
            .encode(
                x=alt.X("mean_abs_shap:Q", title="mean |SHAP value|"),
                y=alt.Y("descriptor:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("descriptor:N"),
                    alt.Tooltip("mean_abs_shap:Q", format=".3f"),
                    alt.Tooltip("direction:N"),
                ],
            )
            .properties(height=420),
            use_container_width=True,
        )
        st.caption(
            "TPSA — topological polar surface area — leads by nearly 2×, which "
            "is exactly what CNS drug design predicts: polar molecules do not "
            "cross the blood-brain barrier."
        )


# --------------------------------------------------------------------------
# Tab 5 — Where it fails
# --------------------------------------------------------------------------
with tab_fails:
    st.markdown(
        """
Most demos pick a few famous drugs, show confident correct answers, and prove
nothing — because those drugs were in the training set. This tab separates the
three cases that actually matter.
"""
    )

    st.subheader("1. Held-out molecules — the only real evidence")
    st.caption(
        "Sampled live from the scaffold-split holdout: molecules excluded from "
        "training, with their true labels. Reshuffle to draw a different sample."
    )

    eval_dataset = st.selectbox(
        "Endpoint",
        list(ENDPOINTS),
        format_func=lambda d: ENDPOINTS[d]["title"],
        key="fail_dataset",
    )
    sample_seed = st.number_input(
        "Sample seed", min_value=0, max_value=9999, value=0, step=1, key="fail_seed"
    )

    holdout = models.get(eval_dataset, {}).get("holdout")
    if holdout is None or holdout.empty:
        st.info("No holdout artifacts found. Run `python scripts/build_artifacts.py`.")
    else:
        sample = holdout.sample(n=min(8, len(holdout)), random_state=int(sample_seed))
        rows, correct_count = [], 0
        for row in sample.itertuples():
            probability = predict(row.canonical_smiles, eval_dataset)
            if probability is None:
                continue
            predicted = int(probability >= 0.5)
            is_correct = predicted == int(row.Y)
            correct_count += is_correct
            rows.append(
                {
                    "SMILES": (
                        row.canonical_smiles[:38] + "…"
                        if len(row.canonical_smiles) > 38
                        else row.canonical_smiles
                    ),
                    "Predicted": f"{probability:.1%}",
                    "True label": "positive" if row.Y else "negative",
                    "Result": "✅" if is_correct else "❌",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"{correct_count}/{len(rows)} correct on this sample — a small draw, "
            "so treat it as illustration; the benchmark tab has the real numbers "
            "over the full holdout and three seeds."
        )

    st.divider()
    st.subheader("2. Memorised molecules — impressive-looking and meaningless")

    memorised = []
    for label, case_smiles, dataset in [
        ("Caffeine", EXAMPLES["Caffeine — CNS active"], "bbb_martins"),
        ("Atenolol", EXAMPLES["Atenolol — poor CNS"], "bbb_martins"),
        ("Aspirin", EXAMPLES["Aspirin"], "bbb_martins"),
        ("2-Nitrofluorene", EXAMPLES["2-Nitrofluorene — mutagen"], "ames"),
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
    st.warning(
        "Every molecule above is **in the training set**. The model recalls them "
        "almost perfectly, which looks convincing and demonstrates nothing. "
        "Checking this is why the app ships the training SMILES alongside the "
        "model — a demo that cannot tell recall from inference is not a "
        "demonstration of anything."
    )

    st.divider()
    st.subheader("3. Out of domain — Sucrose")

    sucrose = EXAMPLES["Sucrose — does not cross"]
    probability = predict(sucrose, "bbb_martins")
    similarity = nearest_similarity(sucrose, "bbb_martins")
    left, right = st.columns([1, 2])
    left.metric("BBB prediction", f"{probability:.1%}" if probability else "—")
    left.metric("Nearest training molecule", f"{similarity:.2f}" if similarity else "—")
    right.error(
        "Sucrose is a highly polar sugar that certainly does not reach the "
        "brain — yet the model returns roughly a **coin flip**.\n\n"
        "It is not in the BBB dataset at all, and its nearest training "
        "neighbour sits at Tanimoto ~0.41. Sugars are barely represented among "
        "the 1,382 training molecules, so this is extrapolation, not judgement. "
        "The expected answer here comes from pharmacology, not from the "
        "benchmark — there is no ground-truth label to check against."
    )

    st.markdown(
        """
#### Why this matters more than the accuracy number

A model is only trustworthy **inside its applicability domain** — the region of
chemical space its training data covers. Outside it, the model still returns a
confident-looking percentage, because nothing in a probability score knows what
it has never seen.

That is why every prediction on the first tab ships with its training-set
status and the Tanimoto similarity to the nearest training molecule. A number
without that context invites exactly the mistake Sucrose demonstrates.

#### Other known limits

- **Two endpoints**, both classification. No solubility or clearance regression.
- **Small data** — 2k and 7k molecules. At 100k a graph neural network would
  likely overtake these models.
- **Ames is one assay**, not a full genotoxicity assessment.
- **Tanimoto similarity is a heuristic**, not a calibrated confidence estimate.
"""
    )

    st.markdown(
        """
#### Why this matters more than the accuracy number

A model is only trustworthy **inside its applicability domain** — the region of
chemical space its training data covers. Outside it, the model still returns a
confident-looking percentage, because nothing in a softmax knows what it has
never seen.

That is why every prediction on the first tab ships with the Tanimoto similarity
to the nearest training molecule, and warns below 0.4. A number without that
context invites exactly the mistake Sucrose demonstrates.

#### Other known limits

- **Two endpoints**, both classification. No solubility or clearance regression.
- **Small data** — 2k and 7k molecules. At 100k a graph neural network would
  likely overtake these models.
- **Ames is one assay**, not a full genotoxicity assessment.
- **Tanimoto similarity is a heuristic**, not a calibrated confidence estimate.
"""
    )
