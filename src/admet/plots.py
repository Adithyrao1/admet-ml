"""Figure generation.

Every figure is rendered twice -- once for a light surface and once for a dark
one -- so the README looks right under either GitHub theme. The two variants are
stepped from the same hues rather than being an automatic inversion.

Colours come from a validated categorical palette: the slot order is the
colour-blind-safety mechanism, so series are assigned slots in order and never
cycled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#d9d8d4",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#f0efec",
    text_secondary="#c3c2b7",
    grid="#3a3a38",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)

THEMES = (LIGHT, DARK)

MODEL_LABELS = {
    "dummy": "Dummy\n(baseline)",
    "logreg": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
}

SPLIT_LABELS = {
    "scaffold": "Scaffold split (TDC official)",
    "random": "Random split",
}


def _style_axes(ax, theme: Theme, ylabel: str = "", title: str = "", subtitle: str = ""):
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)

    # Recessive grid and axes: the data should carry the attention.
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(colors=theme.text_secondary, labelsize=10, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.text_secondary, fontsize=10)
    if title:
        ax.set_title(
            title,
            color=theme.text_primary,
            fontsize=13,
            fontweight="bold",
            loc="left",
            pad=24 if subtitle else 12,
        )
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=theme.text_secondary,
            fontsize=9.5,
            va="bottom",
        )


def _save(fig, out_dir: Path, stem: str, theme: Theme) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{theme.name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_model_comparison(
    summary,
    dataset: str,
    out_dir: Path,
    metric: str = "test_roc_auc",
) -> list[Path]:
    """Grouped bars: metric per model, one series per split.

    The two splits are the story -- the scaffold bars show what the model does
    on chemistry it has never seen, the random bars show the optimistic number
    most write-ups report.
    """
    frame = summary[summary["dataset"] == dataset]
    models = [m for m in MODEL_LABELS if m in set(frame["model"])]
    splits = [s for s in ("scaffold", "random") if s in set(frame["split"])]

    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        width = 0.36
        positions = np.arange(len(models))

        for series_index, split in enumerate(splits):
            means, errors = [], []
            for model in models:
                row = frame[(frame["model"] == model) & (frame["split"] == split)]
                means.append(float(row[f"{metric}_mean"].iloc[0]) if len(row) else np.nan)
                errors.append(float(row[f"{metric}_std"].iloc[0]) if len(row) else 0.0)

            # 2px-equivalent gap between adjacent bars keeps the fills separate.
            offset = (series_index - (len(splits) - 1) / 2) * (width + 0.02)
            bars = ax.bar(
                positions + offset,
                means,
                width,
                label=SPLIT_LABELS.get(split, split),
                color=theme.series[series_index],
                yerr=errors,
                capsize=3,
                error_kw={"ecolor": theme.text_secondary, "elinewidth": 1.2},
                zorder=3,
            )
            # Direct labels: identity and value are never colour-alone.
            for bar, mean in zip(bars, means):
                if np.isnan(mean):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + 0.012,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=theme.text_primary,
                )

        # The 0.5 line is what a model that ignores the molecule scores. It is
        # named in the subtitle rather than inline, which would collide with the
        # right-hand bars.
        ax.axhline(0.5, color=theme.text_secondary, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], color=theme.text_secondary)
        # Bars start at zero: a truncated axis would exaggerate differences that
        # are in fact inside the seed-to-seed noise.
        ax.set_ylim(0, 1.06)
        _style_axes(
            ax,
            theme,
            ylabel="Test ROC-AUC",
            title=f"{dataset}: model performance by split",
            subtitle="Mean ± SD over 3 seeds. Dashed line marks random guessing (0.5).",
        )
        legend = ax.legend(
            frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=2
        )
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)

        paths.append(_save(fig, out_dir, f"{dataset}_model_comparison", theme))
    return paths


GAP_METRICS = [
    ("test_roc_auc", "ROC-AUC"),
    ("test_pr_auc", "PR-AUC"),
    ("test_f1", "F1"),
    ("test_balanced_accuracy", "Balanced\naccuracy"),
    ("test_mcc", "MCC"),
]


def plot_generalization_gap(summary, out_dir: Path) -> list[Path]:
    """How much each metric is inflated by scoring on a random split.

    The point of the figure: MCC falls by a similar, large amount on both
    datasets when test molecules no longer share scaffolds with training, while
    ROC-AUC's sensitivity is dataset-dependent -- substantial on AMES, almost
    nil on BBB. A write-up reporting ROC-AUC alone can therefore show no
    generalization gap where one clearly exists.

    Values are averaged over the real models (the dummy is excluded, since its
    scores are fixed by the class prior and would dilute the effect).
    """
    real_models = summary[summary["model"] != "dummy"]
    datasets = sorted(real_models["dataset"].unique())

    rows: dict[str, list[float]] = {}
    for dataset in datasets:
        frame = real_models[real_models["dataset"] == dataset]
        values = []
        for metric, _ in GAP_METRICS:
            scaffold = frame[frame["split"] == "scaffold"][f"{metric}_mean"]
            random_split = frame[frame["split"] == "random"][f"{metric}_mean"]
            if scaffold.empty or random_split.empty:
                values.append(np.nan)
            else:
                values.append(float(random_split.mean()) - float(scaffold.mean()))
        rows[dataset] = values

    if not rows:
        return []

    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        width = 0.36 if len(datasets) > 1 else 0.5
        positions = np.arange(len(GAP_METRICS))

        for series_index, dataset in enumerate(datasets):
            offset = (series_index - (len(datasets) - 1) / 2) * (width + 0.02)
            values = rows[dataset]
            bars = ax.bar(
                positions + offset,
                values,
                width,
                label=dataset,
                color=theme.series[series_index],
                zorder=3,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.004 if value >= 0 else -0.012),
                    f"{value:+.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                    color=theme.text_primary,
                )

        ax.axhline(0.0, color=theme.text_secondary, linewidth=1.2, zorder=2)
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label in GAP_METRICS], color=theme.text_secondary)

        # Headroom below zero so a negative bar's label cannot land on the tick
        # labels underneath the axis.
        finite = [v for values in rows.values() for v in values if not np.isnan(v)]
        if finite:
            low, high = min(finite), max(finite)
            ax.set_ylim(min(0.0, low) - 0.028, max(0.0, high) * 1.14 + 0.004)
        _style_axes(
            ax,
            theme,
            ylabel="Random minus scaffold",
            title="MCC exposes a generalization gap that ROC-AUC can miss",
            subtitle="Metric improvement when test molecules are allowed to share scaffolds with training.",
        )
        if len(datasets) > 1:
            legend = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
            for text in legend.get_texts():
                text.set_color(theme.text_secondary)
        paths.append(_save(fig, out_dir, "generalization_gap", theme))
    return paths


def plot_roc_curves(curves: dict[str, tuple], dataset: str, out_dir: Path) -> list[Path]:
    """ROC curves for several models on one dataset.

    ``curves`` maps a model name to ``(fpr, tpr, auc)``. AUC values ride in the
    legend text, which doubles as the relief required for lower-contrast slots.
    """
    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        for index, (model, (fpr, tpr, auc)) in enumerate(curves.items()):
            ax.plot(
                fpr,
                tpr,
                color=theme.series[index % len(theme.series)],
                linewidth=2.0,
                label=f"{model.replace('_', ' ')} — AUC {auc:.3f}",
                zorder=3,
            )
        ax.plot([0, 1], [0, 1], color=theme.text_secondary, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("False positive rate", color=theme.text_secondary, fontsize=10)
        _style_axes(
            ax,
            theme,
            ylabel="True positive rate",
            title=f"{dataset}: ROC curves",
            subtitle="Scaffold split, seed 42.",
        )
        ax.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.7, zorder=0)
        legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)
        paths.append(_save(fig, out_dir, f"{dataset}_roc_curves", theme))
    return paths


def plot_descriptor_importance(
    names: list[str],
    values: np.ndarray,
    dataset: str,
    out_dir: Path,
    top_n: int = 15,
) -> list[Path]:
    """Horizontal bars of mean |SHAP| for the most influential descriptors.

    Magnitude ranking with long category names, so horizontal bars and a single
    sequential hue -- these are one quantity, not separate identities.
    """
    order = np.argsort(values)[::-1][:top_n][::-1]
    labels = [names[i].replace("desc_", "") for i in order]
    magnitudes = values[order]

    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(labels) + 1.8))
        positions = np.arange(len(labels))
        ax.barh(positions, magnitudes, height=0.68, color=theme.series[0], zorder=3)
        for position, magnitude in zip(positions, magnitudes):
            ax.text(
                magnitude + magnitudes.max() * 0.015,
                position,
                f"{magnitude:.3f}",
                va="center",
                fontsize=8.5,
                color=theme.text_secondary,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, color=theme.text_secondary, fontsize=9)
        ax.set_xlim(0, magnitudes.max() * 1.18)
        _style_axes(
            ax,
            theme,
            title=f"{dataset}: most influential descriptors",
            subtitle="Mean |SHAP value| — how much each property moves the prediction.",
        )
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_xlabel("mean |SHAP value|", color=theme.text_secondary, fontsize=10)
        paths.append(_save(fig, out_dir, f"{dataset}_descriptor_importance", theme))
    return paths
