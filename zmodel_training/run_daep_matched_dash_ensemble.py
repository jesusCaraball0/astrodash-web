#!/usr/bin/env python3
"""
Train Dash 1D CNN ensemble runs on DAEP-matched splits (+z and/or -z).

Prerequisite:
  python zmodel_training/create_daep_matched_dash_split.py

Usage:
  python zmodel_training/run_daep_matched_dash_ensemble.py
  python zmodel_training/run_daep_matched_dash_ensemble.py --seeds 0 1 2
  python zmodel_training/run_daep_matched_dash_ensemble.py --no-redshift-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DASH_RETRAIN = SCRIPT_DIR / "dash_retrain.py"
CREATE_SPLIT = SCRIPT_DIR / "create_daep_matched_dash_split.py"


def run_one(seed: int, has_redshift: bool) -> None:
    cmd = [
        sys.executable,
        str(DASH_RETRAIN),
        "--daep-matched",
        "--seed",
        str(seed),
    ]
    if not has_redshift:
        cmd.append("--no-redshift")
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAEP-matched Dash 1D CNN ensemble.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Training seeds (default: 0..9).",
    )
    parser.add_argument(
        "--redshift-only",
        action="store_true",
        help="Train only +redshift models.",
    )
    parser.add_argument(
        "--no-redshift-only",
        action="store_true",
        help="Train only -redshift models.",
    )
    parser.add_argument(
        "--skip-split-create",
        action="store_true",
        help="Skip running create_daep_matched_dash_split.py first.",
    )
    args = parser.parse_args()

    if args.redshift_only and args.no_redshift_only:
        raise SystemExit("Use at most one of --redshift-only / --no-redshift-only.")

    if not args.skip_split_create:
        subprocess.run(
            [sys.executable, str(CREATE_SPLIT), "--both"],
            check=True,
            cwd=PROJECT_ROOT,
        )

    variants = []
    if not args.no_redshift_only:
        variants.append(True)
    if not args.redshift_only:
        variants.append(False)

    for has_redshift in variants:
        tag = "+z" if has_redshift else "-z"
        print(f"\n### Ensemble variant: {tag} ###")
        for seed in args.seeds:
            run_one(seed, has_redshift=has_redshift)

    print("\nDone. Evaluate with:")
    print(
        "  python zmodel_training/roc_ensemble_daep_comparison.py "
        "data/pre_trained_models/daep_comparison_z"
    )
    print(
        "  python zmodel_training/dash_ensemble_plots.py "
        "data/pre_trained_models/daep_comparison_z"
    )
    print(
        "  python zmodel_training/dash_ensemble_plots.py "
        "data/pre_trained_models/daep_comparison_noz"
    )


if __name__ == "__main__":
    main()
