"""
IAU-level train / validation / test row indices for preprocessed WISeREP data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Fixed project-wide: train/val/test IAU partition does not vary with per-run training RNG.
IAU_SPLIT_SEED = 0


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
