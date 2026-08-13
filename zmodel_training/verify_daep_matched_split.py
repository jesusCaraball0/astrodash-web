#!/usr/bin/env python3
"""
Verify that DAEP-aligned Dash split JSON matches DAEP classifier row counts.

Usage:
  python zmodel_training/verify_daep_matched_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WISEREP_DATA_DIR = PROJECT_ROOT / "WiserepData"

for path in (PROJECT_ROOT, SCRIPT_DIR, str(WISEREP_DATA_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from TwinsClassifier_Wiserep import (
    IAU_COLUMN,
    IAU_TEST_FRAC,
    IAU_TRAIN_FRAC,
    IAU_VAL_FRAC,
    LABEL_COLUMN,
    filter_indices_mapped,
)


def daep_split_counts(meta_csv: Path) -> dict[str, int]:
    meta = pd.read_csv(meta_csv, low_memory=False).reset_index(drop=True)
    tr, va, te = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        IAU_TRAIN_FRAC,
        IAU_VAL_FRAC,
        IAU_TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    tr = filter_indices_mapped(meta, tr, LABEL_COLUMN)
    va = filter_indices_mapped(meta, va, LABEL_COLUMN)
    te = filter_indices_mapped(meta, te, LABEL_COLUMN)
    return {
        "train": int(tr.size),
        "val": int(va.size),
        "test": int(te.size),
        "total": int(tr.size + va.size + te.size),
    }


def compare_variant(label: str, meta_csv: Path, split_json: Path) -> None:
    if not split_json.is_file():
        print(f"\n{label}: MISSING split JSON — run create_daep_matched_dash_split.py")
        return

    daep = daep_split_counts(meta_csv)
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    counts = payload["counts"]

    print(f"\n{label}")
    print(f"  meta: {meta_csv}")
    print(f"  split: {split_json}")
    print("  mappable row counts (DAEP reference vs split JSON):")
    for key in ("train", "val", "test", "total"):
        daep_n = daep[key]
        json_n = counts[f"mappable_{key}"] if key != "total" else counts["mappable_total"]
        ok = "OK" if daep_n == json_n else "MISMATCH"
        print(f"    {key:5s}: daep={daep_n:6d}  json={json_n:6d}  [{ok}]")

    print("  dash loadable ascii counts:")
    print(
        f"    train={counts['train']}  val={counts['val']}  "
        f"test={counts['test']}  total={counts['total']}"
    )
    skipped = payload.get("skipped", {})
    print(
        "  skipped missing ascii on disk: "
        f"train={skipped.get('train_missing_file', 0)}  "
        f"val={skipped.get('val_missing_file', 0)}  "
        f"test={skipped.get('test_missing_file', 0)}"
    )


def main() -> None:
    compare_variant(
        "henna deredshifted (+redshift)",
        const.PROCESSED_META_HENNA_Z,
        const.SPLITS_JSON_HENNA_MATCHED_Z,
    )
    compare_variant(
        "henna Noderedshift (observed frame)",
        const.PROCESSED_META_HENNA_NOZ,
        const.SPLITS_JSON_HENNA_MATCHED_NOZ,
    )


if __name__ == "__main__":
    main()
