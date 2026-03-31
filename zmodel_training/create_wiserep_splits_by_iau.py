#!/usr/bin/env python
"""
Create train/val/test splits by IAU name so no object (IAU name) appears in
more than one split. Stratified by class, grouped by object (IAU name).

Two modes (--mode):
- train-val-test: 80% train, 10% val, 10% test. Output: wiserep_splits_by_iau_80_10_10.json (default path).
- train-test:      90% train, 10% test (no val). Output: wiserep_splits_by_iau_90_10.json (default path).

Reads: data/wiserep/wiserep_metadata.csv. Filters to spectra in data/wiserep/wiserep_data_noSEDM (hardcoded).

Usage:
    python zmodel_training/create_wiserep_splits_by_iau.py --mode train-val-test
    python zmodel_training/create_wiserep_splits_by_iau.py --mode train-test
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

from constants import (
    PROJECT_ROOT,
    METADATA_CSV,
    CLASS_NAMES,
    DEFAULT_OUTPUT_80_10_10,
    DEFAULT_OUTPUT_90_10,
    SEED,
    SPECTRA_DIR,
)
from helpers import normalize_label


def load_iau_to_files_and_class(metadata_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Build: (1) IAU name -> list of Ascii file names, (2) IAU name -> canonical class.
    Uses first Obj. Type seen per IAU for class. Only includes rows with IAU + Ascii file.
    """
    iau_to_files: dict[str, list[str]] = defaultdict(list)
    iau_to_class: dict[str, str] = {}
    seen_files: set[str] = set()

    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("Ascii file") or "").strip()
            iau = (row.get("IAU name") or "").strip()
            obj_type = (row.get("Obj. Type") or "").strip()
            if not fname or not iau:
                continue
            if fname in seen_files:
                continue
            seen_files.add(fname)
            iau_to_files[iau].append(fname)
            if iau not in iau_to_class:
                cls = normalize_label(obj_type)
                if cls is not None:
                    iau_to_class[iau] = cls

    return dict(iau_to_files), iau_to_class


def _class_counts(
    iau_set: set[str],
    iau_to_files: dict[str, list[str]],
    iau_to_class: dict[str, str],
) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for iau in iau_set:
        cls = iau_to_class.get(iau, "_other")
        for _ in iau_to_files[iau]:
            c[cls] += 1
    return dict(c)


def create_splits_80_10_10(
    iau_to_files: dict[str, list[str]],
    iau_to_class: dict[str, str],
    rng: random.Random,
) -> tuple[list[str], list[str], list[str], set[str], set[str], set[str]]:
    """
    Stratified 80% train, 10% val, 10% test by IAU name (grouped by object, stratified by class).
    Returns (train_files, val_files, test_files, train_iau, val_iau, test_iau).
    """
    class_to_iau: dict[str, list[str]] = defaultdict(list)
    for iau in iau_to_files:
        cls = iau_to_class.get(iau)
        if cls in CLASS_NAMES:
            class_to_iau[cls].append(iau)
        else:
            class_to_iau["_other"].append(iau)

    train_iau: set[str] = set()
    val_iau: set[str] = set()
    test_iau: set[str] = set()

    for cls, iau_list in class_to_iau.items():
        rng.shuffle(iau_list)
        n = len(iau_list)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        n_test = n - n_train - n_val
        train_iau.update(iau_list[:n_train])
        val_iau.update(iau_list[n_train : n_train + n_val])
        test_iau.update(iau_list[n_train + n_val :])

    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []
    for iau, files in iau_to_files.items():
        if iau in train_iau:
            train_files.extend(files)
        elif iau in val_iau:
            val_files.extend(files)
        else:
            test_files.extend(files)

    return train_files, val_files, test_files, train_iau, val_iau, test_iau


def create_splits_90_10(
    iau_to_files: dict[str, list[str]],
    iau_to_class: dict[str, str],
    rng: random.Random,
) -> tuple[list[str], list[str], list[str], set[str], set[str], set[str]]:
    """
    Stratified 90% train, 10% test by IAU name (no val). Val is returned empty.
    Returns (train_files, val_files, test_files, train_iau, val_iau, test_iau).
    """
    class_to_iau: dict[str, list[str]] = defaultdict(list)
    for iau in iau_to_files:
        cls = iau_to_class.get(iau)
        if cls in CLASS_NAMES:
            class_to_iau[cls].append(iau)
        else:
            class_to_iau["_other"].append(iau)

    train_iau: set[str] = set()
    test_iau: set[str] = set()

    for cls, iau_list in class_to_iau.items():
        rng.shuffle(iau_list)
        n = len(iau_list)
        n_train = int(n * 0.90)
        train_iau.update(iau_list[:n_train])
        test_iau.update(iau_list[n_train:])

    train_files = []
    test_files = []
    for iau, files in iau_to_files.items():
        if iau in train_iau:
            train_files.extend(files)
        else:
            test_files.extend(files)

    return train_files, [], test_files, train_iau, [], test_iau


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits by IAU name (stratified, no object leakage)."
    )
    parser.add_argument(
        "--mode",
        choices=["train-val-test", "train-test"],
        required=True,
        help="train-val-test: 80/10/10 train/val/test. train-test: 90/10 train/test (no val).",
    )
    args = parser.parse_args()

    metadata_path = METADATA_CSV if METADATA_CSV.is_absolute() else PROJECT_ROOT / METADATA_CSV

    print(f"Loading metadata from {metadata_path}")
    iau_to_files, iau_to_class = load_iau_to_files_and_class(metadata_path)
    n_spectra = sum(len(files) for files in iau_to_files.values())
    print(f"  Unique IAU names: {len(iau_to_files)}")
    print(f"  Total spectra (with IAU name): {n_spectra}")

    base = SPECTRA_DIR if SPECTRA_DIR.is_absolute() else PROJECT_ROOT / SPECTRA_DIR
    existing = {f.name for f in base.iterdir() if f.is_file()}
    iau_to_files = {
        iau: [f for f in files if f in existing]
        for iau, files in iau_to_files.items()
    }
    iau_to_files = {iau: files for iau, files in iau_to_files.items() if files}
    iau_to_class = {iau: iau_to_class[iau] for iau in iau_to_files if iau in iau_to_class}
    n_spectra = sum(len(files) for files in iau_to_files.values())
    print(f"  After filtering to existing files in {base}:")
    print(f"  Unique IAU names: {len(iau_to_files)}, Spectra: {n_spectra}")

    rng = random.Random(SEED)

    if args.mode == "train-val-test":
        train_files, val_files, test_files, train_iau, val_iau, test_iau = create_splits_80_10_10(
            iau_to_files, iau_to_class, rng
        )
        default_output = DEFAULT_OUTPUT_80_10_10
        split_label = "80/10/10"
    else:
        train_files, val_files, test_files, train_iau, val_iau, test_iau = create_splits_90_10(
            iau_to_files, iau_to_class, rng
        )
        default_output = DEFAULT_OUTPUT_90_10
        split_label = "90/10"

    total_spectra = len(train_files) + len(val_files) + len(test_files)
    train_dist = _class_counts(train_iau, iau_to_files, iau_to_class)
    val_dist = _class_counts(val_iau, iau_to_files, iau_to_class)
    test_dist = _class_counts(test_iau, iau_to_files, iau_to_class)

    print(f"\nClass distribution — stratified {split_label}:")
    for cls in CLASS_NAMES + ["_other"]:
        t = train_dist.get(cls, 0)
        v = val_dist.get(cls, 0)
        te = test_dist.get(cls, 0)
        tot = t + v + te
        pct = (tot / total_spectra * 100) if total_spectra else 0
        print(f"  {cls}: train={t} val={v} test={te}  (total {tot}, {pct:.1f}%)")

    splits = {
        "total": total_spectra,
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }

    out_path = default_output if default_output.is_absolute() else PROJECT_ROOT / default_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplit (stratified {split_label}, seed={SEED}):")
    print(f"  Objects: train={len(train_iau)}  val={len(val_iau)}  test={len(test_iau)}")
    print(f"  Spectra: train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")
    print(f"\nWrote {out_path}")
    print("Use this file in dash_retrain (SPLITS_JSON) or dash_retrain_kfold.")


if __name__ == "__main__":
    main()
