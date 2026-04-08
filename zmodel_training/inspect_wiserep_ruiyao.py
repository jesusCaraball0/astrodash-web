#!/usr/bin/env python3
"""
CLI: summarize wiserep_ruiyao parquet + colleague split JSON.

Run from repo root (or anywhere); paths are resolved from this file's project root.

  python zmodel_training/inspect_wiserep_ruiyao.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUIYAO_DIR = PROJECT_ROOT / "data" / "wiserep_ruiyao"
PARQUET = RUIYAO_DIR / "wiserep_spectra_5class_optical.parquet"
SPLIT_JSON = RUIYAO_DIR / "wiserep_split_ids.json"


def spectrum_id_to_iloc(sid: str) -> int:
    """WISEREP_0000000 … = row order in the parquet (pandas iloc), not the 'index' column."""
    return int(sid.split("_", 1)[1], 10)


def main() -> None:
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
