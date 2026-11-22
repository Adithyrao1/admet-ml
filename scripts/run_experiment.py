"""Run a sweep of ADMET experiments and append results to a CSV.

Examples
--------
Reproduce the headline table (both datasets, all models, both splits, 3 seeds)::

    python scripts/run_experiment.py --all

A single quick run::

    python scripts/run_experiment.py --datasets bbb_martins --models xgboost \
        --splits scaffold --seeds 42

The sweep is resumable: every finished run is appended to the results CSV
immediately, and re-running skips any run_id already present. Use --force to
recompute them.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# Allow running as `python scripts/run_experiment.py` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from admet.data import DATASETS  # noqa: E402
from admet.experiment import (  # noqa: E402
    RunConfig,
    append_result,
    completed_run_ids,
    run_single,
)
from admet.models import MODEL_NAMES  # noqa: E402

DEFAULT_RESULTS = "results/results.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADMET classical-ML experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["bbb_martins"], choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--splits", nargs="+", default=["scaffold"], choices=["scaffold", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--features",
        default="fp+desc",
        choices=["fp+desc", "fp", "desc"],
        help="Which feature blocks to use.",
    )
    parser.add_argument("--n-search-iter", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shorthand for the full headline sweep: both datasets, all models, "
        "both splits, seeds 42/43/44.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute finished runs.")
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing.")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    use_fingerprints = args.features in {"fp+desc", "fp"}
    use_descriptors = args.features in {"fp+desc", "desc"}

    configs = []
    for dataset in args.datasets:
        for split in args.splits:
            for model in args.models:
                for seed in args.seeds:
                    configs.append(
                        RunConfig(
                            dataset=dataset,
                            model=model,
                            split=split,
                            seed=seed,
                            use_fingerprints=use_fingerprints,
                            use_descriptors=use_descriptors,
                            n_search_iter=args.n_search_iter,
                            cv_folds=args.cv_folds,
                        )
                    )
    return configs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.all:
        args.datasets = sorted(DATASETS)
        args.models = MODEL_NAMES
        args.splits = ["scaffold", "random"]
        args.seeds = [42, 43, 44]

    # sklearn/xgboost emit a lot of convergence and deprecation chatter that
    # would bury the per-run result lines.
    warnings.filterwarnings("ignore")

    configs = build_configs(args)
    done = set() if args.force else completed_run_ids(args.results)
    pending = [c for c in configs if c.run_id not in done]

    print(f"Planned runs: {len(configs)}  already done: {len(configs) - len(pending)}  to run: {len(pending)}")
    if args.dry_run:
        for config in pending:
            print("  would run:", config.run_id)
        return 0

    started = time.perf_counter()
    failures: list[tuple[str, str]] = []

    for index, config in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {config.run_id}")
        try:
            row = run_single(config)
        except Exception as error:  # keep the sweep alive; report at the end
            print(f"  FAILED: {type(error).__name__}: {error}")
            failures.append((config.run_id, f"{type(error).__name__}: {error}"))
            continue
        append_result(row, args.results)

    elapsed = time.perf_counter() - started
    print(f"\nFinished {len(pending) - len(failures)}/{len(pending)} runs in {elapsed / 60:.1f} min")
    print(f"Results: {args.results}")

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for run_id, message in failures:
            print(f"  {run_id}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
