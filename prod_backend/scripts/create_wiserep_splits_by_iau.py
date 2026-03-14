#!/usr/bin/env python
"""
Create train/val/test splits by IAU name so no object (IAU name) appears in
more than one split. Uses stratified splitting so class distributions are
similar across train, val, and test.

Reads:  data/wiserep/wiserep_metadata.csv  (uses "IAU name", "Ascii file", "Obj. Type")
Writes: data/wiserep/wiserep_splits_by_iau.json  (same format as wiserep_splits.json)

Split: 90% train, 10% test at the object (IAU name) level, stratified by
classification (Obj. Type mapped to 5 classes). No separate val split; val
is output as an empty list for compatibility. Spectra are assigned to the
split of their object.

Usage (from repo root):
    python prod_backend/scripts/create_wiserep_splits_by_iau.py
    python prod_backend/scripts/create_wiserep_splits_by_iau.py --seed 0 --spectra-dir data/wiserep/wiserep_data_noSEDM
    python prod_backend/scripts/create_wiserep_splits_by_iau.py --output data/wiserep/wiserep_splits_by_iau_noSEDM.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
METADATA_CSV = WISEREP_DIR / "wiserep_metadata.csv"
DEFAULT_OUTPUT = WISEREP_DIR / "wiserep_splits_by_iau.json"

TRAIN_FRAC = 0.90
TEST_FRAC = 0.10

# 5-class label mapping (must match dash_retrain / eval)
LABEL_MAP: dict[str, str] = {
    "SN Ia": "SN Ia", "SN Ia-CSM": "SN Ia", "SN Ia-91T-like": "SN Ia", "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia", "SN Ia-pec": "SN Ia", "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia", "Computed-Ia": "SN Ia",
    "SN Ib": "SN Ib/c", "SN Ic": "SN Ib/c", "SN Ib/c": "SN Ib/c", "SN Ib-Ca-rich": "SN Ib/c",
    "SN Ib-pec": "SN Ib/c", "SN Ibn": "SN Ib/c", "SN Ic-BL": "SN Ib/c", "SN Ic-Ca-rich": "SN Ib/c",
    "SN Ic-pec": "SN Ib/c", "SN Icn": "SN Ib/c", "SN Ib/c-Ca-rich": "SN Ib/c", "SN Ibn/Icn": "SN Ib/c",
    "SN II": "SN II", "SN IIP": "SN II", "SN IIL": "SN II", "SN II-pec": "SN II", "SN IIb": "SN II",
    "Computed-IIP": "SN II", "Computed-IIb": "SN II",
    "SN IIn": "SN IIn", "SN IIn-pec": "SN IIn",
    "SLSN-I": "SLSN-I", "SLSN-II": "SLSN-I", "SLSN-R": "SLSN-I",
}
CLASS_NAMES = ["SN Ia", "SN Ib/c", "SN II", "SN IIn", "SLSN-I"]


def normalize_class(raw_type: str) -> str | None:
    """Map Obj. Type to one of CLASS_NAMES, or None if unmapped."""
    raw = (raw_type or "").strip()
    if not raw:
        return None
    canonical = LABEL_MAP.get(raw) or LABEL_MAP.get(raw.replace(" ", ""))
    return canonical if canonical in CLASS_NAMES else None


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
                cls = normalize_class(obj_type)
                if cls is not None:
                    iau_to_class[iau] = cls

    return dict(iau_to_files), iau_to_class


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits by IAU name (no object leakage)."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_CSV,
        help=f"Path to metadata CSV (default: {METADATA_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--spectra-dir",
        type=Path,
        default=None,
        help="If set, only include filenames that exist under this dir (e.g. wiserep_data_noSEDM). Ensures split matches on-disk data.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata if args.metadata.is_absolute() else PROJECT_ROOT / args.metadata
    if not metadata_path.exists():
        print(f"Error: metadata not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metadata from {metadata_path}")
    iau_to_files, iau_to_class = load_iau_to_files_and_class(metadata_path)
    n_spectra = sum(len(files) for files in iau_to_files.values())
    print(f"  Unique IAU names: {len(iau_to_files)}")
    print(f"  Total spectra (with IAU name): {n_spectra}")

    # Optionally restrict to files that exist under spectra_dir (e.g. wiserep_data_noSEDM)
    spectra_dir = getattr(args, "spectra_dir", None)
    if spectra_dir is not None:
        base = spectra_dir if spectra_dir.is_absolute() else PROJECT_ROOT / spectra_dir
        if not base.is_dir():
            print(f"Error: --spectra-dir not found: {base}", file=sys.stderr)
            sys.exit(1)
        existing = {f.name for f in base.iterdir() if f.is_file()}
        iau_to_files = {
            iau: [f for f in files if f in existing]
            for iau, files in iau_to_files.items()
        }
        iau_to_files = {iau: files for iau, files in iau_to_files.items() if files}
        # Keep only iau_to_class for IAUs still present
        iau_to_class = {iau: iau_to_class[iau] for iau in iau_to_files if iau in iau_to_class}
        n_spectra = sum(len(files) for files in iau_to_files.values())
        print(f"  After filtering to existing files in {base}:")
        print(f"  Unique IAU names: {len(iau_to_files)}, Spectra: {n_spectra}")

    # Group IAUs by class (stratification); IAUs without a valid class go to "_other"
    class_to_iau: dict[str, list[str]] = defaultdict(list)
    for iau in iau_to_files:
        cls = iau_to_class.get(iau)
        if cls in CLASS_NAMES:
            class_to_iau[cls].append(iau)
        else:
            class_to_iau["_other"].append(iau)

    rng = __import__("random").Random(args.seed)
    train_iau: set[str] = set()
    val_iau: set[str] = set()
    test_iau: set[str] = set()

    for cls, iau_list in class_to_iau.items():
        rng.shuffle(iau_list)
        n = len(iau_list)
        n_train = int(n * TRAIN_FRAC)
        n_test = n - n_train
        train_iau.update(iau_list[:n_train])
        test_iau.update(iau_list[n_train:])

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

    # Class distribution per split (spectra)
    def _class_counts(iau_set: set[str]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for iau in iau_set:
            cls = iau_to_class.get(iau, "_other")
            for _ in iau_to_files[iau]:
                c[cls] += 1
        return dict(c)

    train_dist = _class_counts(train_iau)
    test_dist = _class_counts(test_iau)
    total_spectra = len(train_files) + len(test_files)
    print("\nClass distribution (spectra) — stratified 90/10:")
    for cls in CLASS_NAMES + ["_other"]:
        t = train_dist.get(cls, 0)
        te = test_dist.get(cls, 0)
        tot = t + te
        pct = (tot / total_spectra * 100) if total_spectra else 0
        print(f"  {cls}: train={t} test={te}  (total {tot}, {pct:.1f}%)")

    val_files = []  # no val split; empty for compatibility
    total = len(train_files) + len(test_files)
    splits = {
        "total": total,
        "counts": {
            "train": len(train_files),
            "val": 0,
            "test": len(test_files),
        },
        "train": sorted(train_files),
        "val": [],
        "test": sorted(test_files),
    }

    out_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplit (stratified 90 train / 10 test, seed={args.seed}):")
    print(f"  Objects: train={len(train_iau)}  test={len(test_iau)}")
    print(f"  Spectra: train={len(train_files)}  test={len(test_files)}")
    print(f"\nWrote {out_path}")
    print("\nUse this file in dash_retrain.py by setting SPLITS_JSON to the output path.")


if __name__ == "__main__":
    main()
