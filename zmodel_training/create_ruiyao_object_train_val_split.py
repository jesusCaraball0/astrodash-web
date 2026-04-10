#!/usr/bin/env python3
"""
Writes wiserep_splits_train_val_test.json: same keys as 80/10/10 (train, val, test) but WISEREP_* ids.
Val = IAU-stratified holdout from train; test = colleague test_spectrum_ids.

  python zmodel_training/create_ruiyao_object_train_val_split.py --force
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

import constants as const
import helpers as helpers
import parquet_dataset as rpd


def build_object_stratified_train_val(
    df: pd.DataFrame,
    colleague_train_ids: List[str],
    val_object_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Group spectra by IAU (else treat each sid as its own object); stratify by first-seen class."""
    n = len(df)
    iau_to_sids: Dict[str, List[str]] = defaultdict(list)
    iau_stratum: Dict[str, int] = {}

    # group spectra by IAU for objects with multiple observations
    for sid in colleague_train_ids:
        i = rpd.wiserep_spectrum_id_to_iloc(sid)
        row = df.iloc[i]
        raw = row["IAU name"]
        iau = str(raw).strip() if pd.notna(raw) and str(raw).strip() else sid
        iau_to_sids[iau].append(sid)
        if iau not in iau_stratum:
            typ = "" if pd.isna(row["parent_class"]) else str(row["parent_class"]).strip()
            lab = helpers.normalize_label(typ)
            iau_stratum[iau] = const.CLASS_TO_IDX[lab] if lab is not None else -1

    by_stratum: Dict[int, List[str]] = defaultdict(list)
    for iau in iau_to_sids:
        by_stratum[iau_stratum[iau]].append(iau)

    # randomly assign based on splits
    rng = random.Random(seed)
    train_iau: List[str] = []
    val_iau: List[str] = []
    for k in sorted(by_stratum.keys()):
        objs = by_stratum[k]
        rng.shuffle(objs)
        n_val = min(max(1, int(round(len(objs) * val_object_frac))), len(objs) - 1)
        val_iau.extend(objs[:n_val])
        train_iau.extend(objs[n_val:])

    train_sids = sorted(s for iau in train_iau for s in iau_to_sids[iau])
    val_sids = sorted(s for iau in val_iau for s in iau_to_sids[iau])
    exp = set(colleague_train_ids)
    if set(train_sids) | set(val_sids) != exp or set(train_sids) & set(val_sids):
        raise ValueError("Object split failed to partition colleague train_spectrum_ids")
    return train_sids, val_sids, {"n_train_objects": len(train_iau), "n_val_objects": len(val_iau)}


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

    tr, va, stats = build_object_stratified_train_val(df, train_ids, args.val_frac, args.seed)
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
