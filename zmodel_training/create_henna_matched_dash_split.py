#!/usr/bin/env python3
"""
Build Dash 1D CNN train/val/test JSON from Henna's deduplicated WISeREP meta.

Pipeline:
  1. Load data/wiserep_henna/{deredshifted|Noderedshift}/meta.csv (already deduped)
  2. Join spec_id -> Ascii file via data/wiserep/wiserep_metadata.csv
  3. Write wiserep_metadata_processed.csv (DASH/DAEP column names) next to meta.csv
  4. IAU-level 80/10/10 split (seed 0), keep mappable 5-class labels
  5. Emit ascii filename lists for train/val/test

Output (default):
  data/wiserep/henna_matched_split_z.json
  data/wiserep/henna_matched_split_noz.json
  data/wiserep_henna/*/wiserep_metadata_processed.csv

Usage:
  python zmodel_training/create_henna_matched_dash_split.py
  python zmodel_training/create_henna_matched_dash_split.py --variant noz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WISEREP_DATA_DIR = PROJECT_ROOT / "WiserepData"

for path in (PROJECT_ROOT, SCRIPT_DIR, str(WISEREP_DATA_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test


IAU_COLUMN = "IAU name"
LABEL_COLUMN = "Obj. Type"
ASCII_COLUMN = "Ascii file"
SPEC_COLUMN = "Spec. ID"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def load_spec_id_to_ascii(metadata_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(metadata_csv, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = (row.get("Spec. ID") or "").strip()
            ascii_name = (row.get("Ascii file") or "").strip()
            if sid and ascii_name and sid not in mapping:
                mapping[sid] = ascii_name
    return mapping


def enrich_henna_meta(meta: pd.DataFrame, spec_to_ascii: dict[str, str]) -> pd.DataFrame:
    """Add DASH/DAEP column aliases; join Ascii file from Spec. ID."""
    out = meta.copy()
    if "sn_name_used" in out.columns and IAU_COLUMN not in out.columns:
        out[IAU_COLUMN] = out["sn_name_used"]
    if "raw_type" in out.columns and LABEL_COLUMN not in out.columns:
        out[LABEL_COLUMN] = out["raw_type"]
    if "redshift_used" in out.columns and "Redshift" not in out.columns:
        out["Redshift"] = out["redshift_used"]
    if "spec_id" in out.columns and SPEC_COLUMN not in out.columns:
        out[SPEC_COLUMN] = out["spec_id"].astype(str).str.strip()
    else:
        out[SPEC_COLUMN] = out[SPEC_COLUMN].astype(str).str.strip()

    out[ASCII_COLUMN] = out[SPEC_COLUMN].map(lambda s: spec_to_ascii.get(str(s).strip(), ""))
    return out.reset_index(drop=True)


def row_class_idx(meta: pd.DataFrame, row: int) -> int:
    raw = str(meta.iloc[row][LABEL_COLUMN]).strip()
    if not raw or raw.lower() == "nan":
        return -1
    canonical = const.LABEL_MAP.get(raw)
    if canonical is None:
        canonical = const.LABEL_MAP.get(raw.replace(" ", ""))
    if canonical is None:
        return -1
    return int(const.CLASS_TO_IDX.get(canonical, -1))


def filter_indices_mapped(meta: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    keep = [int(i) for i in indices if row_class_idx(meta, int(i)) >= 0]
    return np.asarray(keep, dtype=np.int64)


def _norm_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def split_payload_for_indices(
    meta: pd.DataFrame,
    indices: np.ndarray,
    spectra_dir: Path,
    require_file: bool,
) -> dict[str, Any]:
    ascii_files: list[str] = []
    spec_ids: list[str] = []
    row_indices: list[int] = []
    skipped_no_ascii = 0
    skipped_missing_file = 0

    for i in indices:
        row = int(i)
        ascii_name = _norm_cell(meta.iloc[row][ASCII_COLUMN])
        spec_id = _norm_cell(meta.iloc[row][SPEC_COLUMN])
        if not ascii_name:
            skipped_no_ascii += 1
            continue
        if require_file and not (spectra_dir / ascii_name).is_file():
            skipped_missing_file += 1
            continue
        ascii_files.append(ascii_name)
        spec_ids.append(spec_id)
        row_indices.append(row)

    return {
        "ascii_files": ascii_files,
        "spec_ids": spec_ids,
        "row_indices": row_indices,
        "skipped_no_ascii": skipped_no_ascii,
        "skipped_missing_file": skipped_missing_file,
    }


def build_split(
    henna_meta_csv: Path,
    spectra_dir: Path,
    wiserep_metadata_csv: Path,
    out_path: Path,
    processed_meta_out: Path,
    *,
    require_file: bool,
) -> dict[str, Any]:
    raw = pd.read_csv(henna_meta_csv, low_memory=False).reset_index(drop=True)
    spec_to_ascii = load_spec_id_to_ascii(wiserep_metadata_csv)
    meta = enrich_henna_meta(raw, spec_to_ascii)

    processed_meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(processed_meta_out, index=False)

    tr_idx, va_idx, te_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
        IAU_SPLIT_SEED,
    )

    tr_idx = filter_indices_mapped(meta, tr_idx)
    va_idx = filter_indices_mapped(meta, va_idx)
    te_idx = filter_indices_mapped(meta, te_idx)

    train_part = split_payload_for_indices(meta, tr_idx, spectra_dir, require_file)
    val_part = split_payload_for_indices(meta, va_idx, spectra_dir, require_file)
    test_part = split_payload_for_indices(meta, te_idx, spectra_dir, require_file)

    iau = meta[IAU_COLUMN].astype(str).str.strip()
    n_joined = int((meta[ASCII_COLUMN].astype(str).str.strip() != "").sum())
    payload: dict[str, Any] = {
        "split_method": "henna_deduped_meta_iau_80_10_10_label_filtered",
        "seed": IAU_SPLIT_SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "iau_col": IAU_COLUMN,
        "label_col": LABEL_COLUMN,
        "henna_meta_csv": str(henna_meta_csv.resolve()),
        "processed_meta_csv": str(processed_meta_out.resolve()),
        "wiserep_metadata_csv": str(wiserep_metadata_csv.resolve()),
        "spectra_dir": str(spectra_dir.resolve()),
        "require_ascii_on_disk": require_file,
        "counts": {
            "meta_rows": int(len(meta)),
            "spec_id_joined_to_ascii": n_joined,
            "mappable_train": int(tr_idx.size),
            "mappable_val": int(va_idx.size),
            "mappable_test": int(te_idx.size),
            "mappable_total": int(tr_idx.size + va_idx.size + te_idx.size),
            "train": len(train_part["ascii_files"]),
            "val": len(val_part["ascii_files"]),
            "test": len(test_part["ascii_files"]),
            "total": len(train_part["ascii_files"])
            + len(val_part["ascii_files"])
            + len(test_part["ascii_files"]),
        },
        "skipped": {
            "train_no_ascii": train_part["skipped_no_ascii"],
            "train_missing_file": train_part["skipped_missing_file"],
            "val_no_ascii": val_part["skipped_no_ascii"],
            "val_missing_file": val_part["skipped_missing_file"],
            "test_no_ascii": test_part["skipped_no_ascii"],
            "test_missing_file": test_part["skipped_missing_file"],
        },
        "unique_iau": {
            "train": int(pd.unique(iau.iloc[tr_idx]).size) if tr_idx.size else 0,
            "val": int(pd.unique(iau.iloc[va_idx]).size) if va_idx.size else 0,
            "test": int(pd.unique(iau.iloc[te_idx]).size) if te_idx.size else 0,
        },
        "train": train_part["ascii_files"],
        "val": val_part["ascii_files"],
        "test": test_part["ascii_files"],
        "train_ascii_files": train_part["ascii_files"],
        "val_ascii_files": val_part["ascii_files"],
        "test_ascii_files": test_part["ascii_files"],
        "train_spec_ids": train_part["spec_ids"],
        "val_spec_ids": val_part["spec_ids"],
        "test_spec_ids": test_part["spec_ids"],
        "train_row_indices": train_part["row_indices"],
        "val_row_indices": val_part["row_indices"],
        "test_row_indices": test_part["row_indices"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _print_summary(label: str, payload: dict[str, Any], out_path: Path) -> None:
    counts = payload["counts"]
    skipped = payload["skipped"]
    print(f"\n{label}")
    print(f"  wrote: {out_path}")
    print(f"  processed meta: {payload['processed_meta_csv']}")
    print(
        f"  meta_rows={counts['meta_rows']}  "
        f"joined_ascii={counts['spec_id_joined_to_ascii']}"
    )
    print(
        f"  mappable (5-class): train={counts['mappable_train']}  "
        f"val={counts['mappable_val']}  test={counts['mappable_test']}  "
        f"total={counts['mappable_total']}"
    )
    print(
        f"  dash loadable: train={counts['train']}  val={counts['val']}  "
        f"test={counts['test']}  total={counts['total']}"
    )
    print(
        f"  skipped missing ascii on disk: train={skipped['train_missing_file']}  "
        f"val={skipped['val_missing_file']}  test={skipped['test_missing_file']}"
    )
    print(
        f"  unique IAU: train={payload['unique_iau']['train']}  "
        f"val={payload['unique_iau']['val']}  test={payload['unique_iau']['test']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Henna-dedup-matched Dash split JSON from wiserep_henna meta."
    )
    parser.add_argument(
        "--variant",
        choices=("z", "noz", "both"),
        default="both",
        help="Which henna bundle to use (default: both).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Alias for --variant both.",
    )
    parser.add_argument(
        "--require-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Ascii files to exist under spectra_dir (default: true).",
    )
    args = parser.parse_args()
    if args.both:
        args.variant = "both"

    jobs: list[tuple[str, Path, Path, Path]] = []
    if args.variant in ("z", "both"):
        jobs.append(
            (
                "henna deredshifted (+redshift)",
                const.WISEREP_HENNA_Z / "meta.csv",
                const.SPLITS_JSON_HENNA_MATCHED_Z,
                const.PROCESSED_META_HENNA_Z,
            )
        )
    if args.variant in ("noz", "both"):
        jobs.append(
            (
                "henna Noderedshift (observed frame)",
                const.WISEREP_HENNA_NOZ / "meta.csv",
                const.SPLITS_JSON_HENNA_MATCHED_NOZ,
                const.PROCESSED_META_HENNA_NOZ,
            )
        )

    for label, meta_csv, out_json, processed_meta in jobs:
        if not meta_csv.is_file():
            raise SystemExit(f"Missing henna metadata: {meta_csv}")
        payload = build_split(
            meta_csv,
            const.SPECTRA_DIR,
            const.METADATA_CSV,
            out_json,
            processed_meta,
            require_file=args.require_file,
        )
        _print_summary(label, payload, out_json)


if __name__ == "__main__":
    main()
