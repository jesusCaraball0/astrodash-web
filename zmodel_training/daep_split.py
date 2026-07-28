#!/usr/bin/env python3
"""
Build train/val/test split JSON from IAU-level row-index splitting (numpy seed 0, 80/10/10).

The function ``split_row_indices_by_iau_train_val_test`` below is pasted exactly so collaborators get
the same row indices given the same ``wiserep_metadata.csv`` row order (``pd.read_csv(..., low_memory=False)``).

Output (default): data/wiserep/wiserep_splits_by_iau_row_indices_seed0_80_10_10.json

  python zmodel_training/daep_split.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# IAU-level train / validation / test row indices for preprocessed WISeREP data.


def split_row_indices_by_iau_train_val_test(
    meta: pd.DataFrame,
    iau_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split WISeREP spectra by IAU into train, validation, and test sets.
    """
    s = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1, got {s}")

    if iau_col not in meta.columns:
        raise KeyError(f"Missing column {iau_col!r}; have: {list(meta.columns)[:40]} ...")

    iau = meta[iau_col].astype(str).str.strip()
    valid = iau.notna() & (iau != "") & (iau.str.lower() != "nan")
    unique = np.array(pd.unique(iau[valid]), dtype=object)
    n_u = int(unique.size)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    # Integer IAU counts: largest-remainder so sizes sum to n_u and match fractions closely
    raw = np.array([train_frac, val_frac, test_frac], dtype=np.float64) * n_u
    sizes = np.floor(raw).astype(int)
    rem = int(n_u - int(sizes.sum()))
    if rem > 0:
        frac_order = np.argsort(-(raw - sizes), kind="stable")
        for j in range(rem):
            sizes[frac_order[j % 3]] += 1
    n_tr, n_va, n_te = int(sizes[0]), int(sizes[1]), int(sizes[2])

    train_iaus = set(unique[:n_tr].tolist())
    val_iaus = set(unique[n_tr : n_tr + n_va].tolist())
    test_iaus = set(unique[n_tr + n_va :].tolist())

    row_iaus = iau.to_numpy()
    v = valid.to_numpy()
    tr_idx = np.flatnonzero(v & np.isin(row_iaus, list(train_iaus)))
    va_idx = np.flatnonzero(v & np.isin(row_iaus, list(val_iaus)))
    te_idx = np.flatnonzero(v & np.isin(row_iaus, list(test_iaus)))

    if tr_idx.size == 0 or va_idx.size == 0 or te_idx.size == 0:
        raise RuntimeError(
            f"Empty split after IAU partition: train={tr_idx.size} val={va_idx.size} test={te_idx.size}"
        )

    return tr_idx.astype(np.int64), va_idx.astype(np.int64), te_idx.astype(np.int64)


def main() -> None:

    meta_path = Path("/Users/jesuscaraball0/code/personal_code/astrodash-web/data/wiserep/wiserep_metadata.csv")
    out_path = Path("/Users/jesuscaraball0/code/personal_code/astrodash-web/data/wiserep/daep_compatible_split.json")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    for p in (project_root, script_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    meta = pd.read_csv(meta_path, low_memory=False)
    iau_col = "IAU name"
    seed = 0
    tr, va, te = split_row_indices_by_iau_train_val_test(meta, iau_col, 0.8, 0.1, 0.1, seed)

    ascii_col = "Ascii file"
    spec_col = "Spec. ID"
    idx_to_ascii = [str(meta.iloc[int(i)][ascii_col] or "").strip() for i in tr]
    idx_to_ascii_va = [str(meta.iloc[int(i)][ascii_col] or "").strip() for i in va]
    idx_to_ascii_te = [str(meta.iloc[int(i)][ascii_col] or "").strip() for i in te]

    def spec_at(idx_arr: np.ndarray) -> list[str]:
        if spec_col not in meta.columns:
            return []
        out: list[str] = []
        for i in idx_arr:
            v = meta.iloc[int(i)][spec_col]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out

    try:
        meta_rel = str(meta_path.relative_to(project_root))
    except ValueError:
        meta_rel = str(meta_path)

    total_rows = int(tr.size + va.size + te.size)
    payload = {
        "split_method": "split_row_indices_by_iau_train_val_test",
        "seed": seed,
        "train_frac": 0.8,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "iau_col": iau_col,
        "metadata_csv": meta_rel,
        "pandas_read_csv_kw": {"low_memory": False},
        "total": total_rows,
        "counts": {
            "train": int(tr.size),
            "val": int(va.size),
            "test": int(te.size),
        },
        "train": idx_to_ascii,
        "val": idx_to_ascii_va,
        "test": idx_to_ascii_te,
        "train_row_indices": [int(x) for x in tr],
        "val_row_indices": [int(x) for x in va],
        "test_row_indices": [int(x) for x in te],
        "train_ascii_files": idx_to_ascii,
        "val_ascii_files": idx_to_ascii_va,
        "test_ascii_files": idx_to_ascii_te,
        "train_spec_ids": spec_at(tr),
        "val_spec_ids": spec_at(va),
        "test_spec_ids": spec_at(te),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        f"  rows — train={tr.size} val={va.size} test={te.size} (IAU-global 80/10/10, seed={seed})"
    )


if __name__ == "__main__":
    main()
