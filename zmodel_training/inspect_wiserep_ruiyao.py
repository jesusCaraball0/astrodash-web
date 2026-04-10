#!/usr/bin/env python3
"""
CLI: summarize wiserep_ruiyao parquet + colleague split JSON.

Run from repo root (or anywhere); paths are resolved from this file's project root.

  python zmodel_training/inspect_wiserep_ruiyao.py
  python zmodel_training/inspect_wiserep_ruiyao.py --stats
  python zmodel_training/inspect_wiserep_ruiyao.py --stats --train-val-test-json path/to.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

import constants as const
import helpers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUIYAO_DIR = PROJECT_ROOT / "data" / "wiserep_ruiyao"
PARQUET = RUIYAO_DIR / "wiserep_spectra_5class_optical.parquet"
SPLIT_JSON = RUIYAO_DIR / "wiserep_split_ids.json"
TRAIN_VAL_TEST_JSON = RUIYAO_DIR / "wiserep_splits_train_val_test.json"


def spectrum_id_to_iloc(sid: str) -> int:
    """WISEREP_0000000 … = row order in the parquet (pandas iloc), not the 'index' column."""
    return int(sid.split("_", 1)[1], 10)


def _canonical_class_label(row: pd.Series) -> str:
    pc = row["parent_class"]
    typ = "" if pd.isna(pc) else str(pc).strip()
    lab = helpers.normalize_label(typ)
    return lab if lab is not None else "_unmapped"


def _iau_object_key(row: pd.Series, sid: str) -> str:
    """Same object grouping as create_ruiyao_object_train_val_split.build_object_stratified_train_val."""
    raw = row["IAU name"]
    if pd.notna(raw) and str(raw).strip():
        return str(raw).strip()
    return sid


def print_train_val_test_json_stats(
    json_path: Path | None = None,
    parquet_path: Path | None = None,
) -> None:
    """
    Print observation and object counts per canonical class for train / val / test
    from wiserep_splits_train_val_test.json (or a path you pass), joined to the parquet.
    """
    jp = json_path if json_path is not None else TRAIN_VAL_TEST_JSON
    pq = parquet_path if parquet_path is not None else PARQUET

    if not jp.exists():
        print(f"Missing train/val/test JSON: {jp}", file=sys.stderr)
        sys.exit(1)
    if not pq.exists():
        print(f"Missing parquet: {pq}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(jp.read_text(encoding="utf-8"))
    need_keys = ("train", "val", "test")
    for k in need_keys:
        if k not in raw:
            print(f"JSON missing key {k!r}: {jp}", file=sys.stderr)
            sys.exit(1)

    df = pd.read_parquet(pq)
    n = len(df)
    if "parent_class" not in df.columns:
        print("Parquet needs column 'parent_class'", file=sys.stderr)
        sys.exit(1)
    if "IAU name" not in df.columns:
        print("Parquet needs column 'IAU name' (for object counts)", file=sys.stderr)
        sys.exit(1)

    split_ids: Dict[str, list[str]] = {
        "train": list(raw["train"]),
        "val": list(raw["val"]),
        "test": list(raw["test"]),
    }

    print("=== wiserep_splits_train_val_test (stats) ===")
    print(f"JSON: {jp}")
    print(f"Parquet: {pq}  rows={n}")
    for meta_key in ("seed", "val_object_frac", "n_train_objects", "n_val_objects"):
        if meta_key in raw:
            print(f"  {meta_key}: {raw[meta_key]!r}")

    spectra_per_class: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in split_ids}
    objects_per_class: Dict[str, Dict[str, Set[str]]] = {
        s: defaultdict(set) for s in split_ids
    }
    invalid: Dict[str, list[str]] = {s: [] for s in split_ids}
    all_sets: Dict[str, Set[str]] = {s: set() for s in split_ids}

    for split_name, ids in split_ids.items():
        for sid in ids:
            all_sets[split_name].add(sid)
            try:
                i = spectrum_id_to_iloc(sid)
            except ValueError:
                invalid[split_name].append(sid)
                continue
            if not (0 <= i < n):
                invalid[split_name].append(sid)
                continue
            row = df.iloc[i]
            lab = _canonical_class_label(row)
            obj = _iau_object_key(row, sid)
            spectra_per_class[split_name][lab] += 1
            objects_per_class[split_name][lab].add(obj)

    print("\n--- Spectra per split (total) ---")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(split_ids[s])}")

    print("\n--- Spectra per class (canonical label) per split ---")
    known = list(const.CLASS_NAMES)
    for cls in known:
        parts = [str(spectra_per_class[s].get(cls, 0)) for s in ("train", "val", "test")]
        print(f"  {cls!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")
    if sum(spectra_per_class[s].get("_unmapped", 0) for s in ("train", "val", "test")) > 0:
        parts = [str(spectra_per_class[s].get("_unmapped", 0)) for s in ("train", "val", "test")]
        print(f"  '_unmapped': train={parts[0]}  val={parts[1]}  test={parts[2]}")

    extras = set()
    for s in spectra_per_class.values():
        extras.update(s.keys())
    for e in sorted(extras - set(known) - {"_unmapped"}):
        parts = [str(spectra_per_class[s].get(e, 0)) for s in ("train", "val", "test")]
        print(f"  {e!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")

    print("\n--- Unique objects (IAU or lone spectrum id) per class per split ---")
    for cls in known:
        parts = [str(len(objects_per_class[s].get(cls, set()))) for s in ("train", "val", "test")]
        print(f"  {cls!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")
    if sum(len(objects_per_class[s].get("_unmapped", set())) for s in ("train", "val", "test")) > 0:
        parts = [str(len(objects_per_class[s].get("_unmapped", set()))) for s in ("train", "val", "test")]
        print(f"  '_unmapped': train={parts[0]}  val={parts[1]}  test={parts[2]}")
    for e in sorted(extras - set(known) - {"_unmapped"}):
        parts = [str(len(objects_per_class[s].get(e, set()))) for s in ("train", "val", "test")]
        print(f"  {e!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")

    print("\n--- Unique objects per split (all classes) ---")
    for s in ("train", "val", "test"):
        u: Set[str] = set()
        for cls in objects_per_class[s]:
            u |= objects_per_class[s][cls]
        print(f"  {s}: {len(u)}")

    print(
        "\n--- Spectra per class: share within each split (sums to 100% per split) ---"
    )
    totals = {s: sum(spectra_per_class[s].values()) for s in ("train", "val", "test")}

    def _pct_parts(cls: str) -> list[str]:
        out: list[str] = []
        for s in ("train", "val", "test"):
            t = totals[s]
            v = (spectra_per_class[s].get(cls, 0) / t * 100.0) if t else 0.0
            out.append(f"{v:.2f}%")
        return out

    for cls in known:
        parts = _pct_parts(cls)
        print(f"  {cls!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")
    if sum(spectra_per_class[s].get("_unmapped", 0) for s in ("train", "val", "test")) > 0:
        parts = _pct_parts("_unmapped")
        print(f"  '_unmapped': train={parts[0]}  val={parts[1]}  test={parts[2]}")
    for e in sorted(extras - set(known) - {"_unmapped"}):
        parts = _pct_parts(e)
        print(f"  {e!r}: train={parts[0]}  val={parts[1]}  test={parts[2]}")

    # Spectra-per-object distribution
    print("\n--- Spectra per object (min / median / max) within each split ---")
    for s in ("train", "val", "test"):
        counts: Dict[str, int] = defaultdict(int)
        for sid in split_ids[s]:
            try:
                i = spectrum_id_to_iloc(sid)
            except ValueError:
                continue
            if not (0 <= i < n):
                continue
            row = df.iloc[i]
            obj = _iau_object_key(row, sid)
            counts[obj] += 1
        if not counts:
            print(f"  {s}: (no valid rows)")
            continue
        arr = np.array(list(counts.values()), dtype=np.float64)
        print(
            f"  {s}: n_objects={len(counts)}  "
            f"spectra/obj min={int(arr.min())} median={float(np.median(arr)):.3g} max={int(arr.max())}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect wiserep_ruiyao parquet and split JSONs.")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics for wiserep_splits_train_val_test.json (train/val/test).",
    )
    ap.add_argument(
        "--train-val-test-json",
        type=Path,
        default=None,
        help=f"Path to train/val/test JSON (default: {TRAIN_VAL_TEST_JSON}).",
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        dest="parquet_path",
        help=f"Parquet path (default: {PARQUET}).",
    )
    args = ap.parse_args()
    if args.stats:
        print_train_val_test_json_stats(
            json_path=args.train_val_test_json,
            parquet_path=args.parquet_path,
        )
        return

    if not PARQUET.exists():
        print(f"Missing parquet: {PARQUET}", file=sys.stderr)
        sys.exit(1)
    if not SPLIT_JSON.exists():
        print(f"Missing split JSON: {SPLIT_JSON}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET)
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    tr_ids = split["train_spectrum_ids"]
    te_ids = split["test_spectrum_ids"]

    print("=== Parquet ===")
    print(f"path: {PARQUET}")
    print(f"rows: {len(df)}  cols: {len(df.columns)}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    need = ["wavelength", "flux", "parent_class", "Redshift", "split", "index", "Ascii file", "Obj. Type"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"WARNING: expected columns missing: {missing}")

    r0 = df.iloc[0]
    for c in ("wavelength", "flux"):
        v = np.asarray(r0[c], dtype=np.float64)
        print(f"  sample {c}: shape={v.shape} dtype={v.dtype} range=[{v.min():.4g}, {v.max():.4g}]")
    print(f"  sample parent_class: {r0['parent_class']!r}")
    print(f"  sample Obj. Type: {r0['Obj. Type']!r}")
    print(f"  sample Redshift: {r0['Redshift']}")
    print(f"  parquet 'split' value_counts:\n{df['split'].value_counts(dropna=False)}")

    print("\n=== Split JSON ===")
    for k in ("seed", "test_size", "method"):
        if k in split:
            print(f"  {k}: {split[k]!r}")
    print(f"  train_spectrum_ids: {len(tr_ids)}  test_spectrum_ids: {len(te_ids)}")

    n = len(df)
    bad_tr = [s for s in tr_ids if not (0 <= spectrum_id_to_iloc(s) < n)]
    bad_te = [s for s in te_ids if not (0 <= spectrum_id_to_iloc(s) < n)]
    print(f"\n=== ID ↔ row mapping (iloc) ===")
    print(f"  train ids with invalid iloc: {len(bad_tr)}")
    print(f"  test ids with invalid iloc: {len(bad_te)}")
    iloc_tr = {spectrum_id_to_iloc(s) for s in tr_ids}
    iloc_te = {spectrum_id_to_iloc(s) for s in te_ids}
    print(f"  train∩test (iloc overlap): {len(iloc_tr & iloc_te)}")

    if "index" in df.columns:
        from_index = set(df["index"].map(lambda x: f"WISEREP_{int(x):07d}"))
        print(f"\n  If IDs were built from parquet column 'index' (wrong):")
        print(f"    train in table: {len(set(tr_ids) & from_index)}/{len(tr_ids)}")

    json_train = set(tr_ids)
    json_test = set(te_ids)

    def sid_for_row(i: int) -> str:
        return f"WISEREP_{i:07d}"

    agree_train = sum(
        1
        for i in range(n)
        if df.iloc[i]["split"] == "train" and sid_for_row(i) in json_train
    )
    agree_test = sum(
        1
        for i in range(n)
        if df.iloc[i]["split"] == "test" and sid_for_row(i) in json_test
    )
    print(f"\n=== Parquet 'split' vs colleague JSON (by iloc) ===")
    print(f"  rows: parquet split=='train' AND WISEREP_i in JSON train: {agree_train}")
    print(f"  rows: parquet split=='test' AND WISEREP_i in JSON test: {agree_test}")
    in_json = json_train | json_test
    orphans = sum(1 for i in range(n) if sid_for_row(i) not in in_json)
    print(f"  rows whose WISEREP_i is in neither JSON train nor test: {orphans}")
    print("  Use train_spectrum_ids / test_spectrum_ids for training; parquet 'split' is a different partition.")

    sub = df.iloc[[spectrum_id_to_iloc(s) for s in tr_ids[: min(5000, len(tr_ids))]]]
    print(f"\n=== parent_class on first {len(sub)} JSON-train rows ===")
    print(sub["parent_class"].value_counts())


if __name__ == "__main__":
    main()
