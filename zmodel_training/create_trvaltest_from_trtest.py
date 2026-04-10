#!/usr/bin/env python3
"""
Writes wiserep_splits_train_val_test.json: same keys as 80/10/10 (train, val, test) but WISEREP_* ids.
Val = IAU-stratified holdout from train; test = colleague test_spectrum_ids.

  python zmodel_training/create_trvaltest_from_trtest.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

import constants as const
import helpers as helpers
import parquet_dataset as rpd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=const.SEED)
    p.add_argument("--output", type=Path, default=rpd.RUIYAO_TRAIN_VAL_TEST_JSON)
    p.add_argument("-f", "--force", action="store_true")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists (use --force)")

    colleague = helpers.load_json(rpd.RUIYAO_SPLIT_JSON)
    train_ids = list(colleague["train_spectrum_ids"])
    test_ids = list(colleague["test_spectrum_ids"])
    df = pd.read_parquet(rpd.RUIYAO_PARQUET)
    if "IAU name" not in df.columns:
        raise SystemExit("Parquet needs 'IAU name' column")

    tr, va, stats = rpd.build_object_stratified_train_val(df, train_ids, args.val_frac, args.seed)
    out = {
        "train": tr,
        "val": va,
        "test": test_ids,
        "seed": args.seed,
        "val_object_frac": args.val_frac,
        **stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}  train={len(tr)} val={len(va)} test={len(test_ids)}")


if __name__ == "__main__":
    main()
