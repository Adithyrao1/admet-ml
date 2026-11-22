"""Turn the raw results CSV into summary tables and figures.

Usage::

    python scripts/analyze.py --results results/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from admet.plots import (  # noqa: E402
    MODEL_LABELS,
    plot_generalization_gap,
    plot_model_comparison,
)

METRICS = ["test_roc_auc", "test_pr_auc", "test_f1", "test_mcc", "test_balanced_accuracy"]
METRIC_HEADERS = {
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_f1": "F1",
    "test_mcc": "MCC",
    "test_balanced_accuracy": "Balanced acc.",
}
MODEL_ORDER = list(MODEL_LABELS)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds into mean/std per (dataset, model, split)."""
    group_keys = ["dataset", "model", "split"]
    aggregations = {metric: ["mean", "std", "count"] for metric in METRICS}
    aggregations["train_seconds"] = ["mean"]

    summary = results.groupby(group_keys).agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    # A single seed has no spread; report 0 rather than NaN.
    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)

    summary["n_seeds"] = summary[f"{METRICS[0]}_count"].astype(int)
    summary = summary.drop(columns=[f"{m}_count" for m in METRICS])

    summary["__order"] = summary["model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    summary = summary.sort_values(["dataset", "split", "__order"]).drop(columns="__order")
    return summary.reset_index(drop=True)


def markdown_table(summary: pd.DataFrame, dataset: str, split: str) -> str:
    frame = summary[(summary["dataset"] == dataset) & (summary["split"] == split)]
    if frame.empty:
        return ""

    header = "| Model | " + " | ".join(METRIC_HEADERS[m] for m in METRICS) + " | Train time |"
    divider = "|---" * (len(METRICS) + 2) + "|"
    lines = [header, divider]

    best = frame[frame["model"] != "dummy"]["test_roc_auc_mean"].max()
    for _, row in frame.iterrows():
        label = MODEL_LABELS[row["model"]].replace("\n", " ")
        cells = []
        for metric in METRICS:
            cell = f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            if metric == "test_roc_auc" and row["test_roc_auc_mean"] == best:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {row['train_seconds_mean']:.0f}s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ADMET results.")
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"No results at {results_path}. Run scripts/run_experiment.py first.")
        return 1

    results = pd.read_csv(results_path)
    summary = summarize(results)

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    sections = []
    for dataset in sorted(summary["dataset"].unique()):
        for split in ("scaffold", "random"):
            table = markdown_table(summary, dataset, split)
            if not table:
                continue
            label = "scaffold split (official TDC protocol)" if split == "scaffold" else "random split"
            sections.append(f"### {dataset} — {label}\n\n{table}\n")

    tables_md = "\n".join(sections)
    tables_path = out_dir / "tables.md"
    tables_path.write_text(tables_md, encoding="utf-8")

    figures = []
    for dataset in sorted(summary["dataset"].unique()):
        figures += plot_model_comparison(summary, dataset, figures_dir)
    figures += plot_generalization_gap(summary, figures_dir)

    print(f"Runs summarized: {len(results)} -> {len(summary)} configurations")
    print(f"  {summary_path}")
    print(f"  {tables_path}")
    for path in figures:
        print(f"  {path}")
    print()
    print(tables_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
