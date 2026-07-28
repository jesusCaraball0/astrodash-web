#!/usr/bin/env python
"""
Print overlap between the test set used by WiserepData/train_latent.py and the
parquet-ruiyao test set used by zmodel_training/dash_retrain.py --parquet-ruiyao.

The primary comparison key is WISeREP "IAU name", because both pipelines split
or group spectra at the object level. When matching spectrum-level columns are
present in both tables, the script also prints exact value overlaps for those
columns.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
WISEREP_DIR = ROOT / "WiserepData"
ZMODEL_DIR = ROOT / "zmodel_training"

for path in (str(WISEREP_DIR), str(ZMODEL_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import parquet_dataset as ruiyao_parquet  # noqa: E402
import train_latent  # noqa: E402
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test  # noqa: E402
from TwinsClassifier_Wiserep import filter_indices_mapped  # noqa: E402


SPECTRUM_KEY_CANDIDATES = (
    "Spec. ID",
    "Ascii file",
    "Fits file",
    "sn_name_used",
    "obs_date_used",
    "jd_obs_used",
)


def _norm(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _values(series: pd.Series) -> set[str]:
    return {v for v in (_norm(x) for x in series.tolist()) if v is not None}


def _pct(numer: int, denom: int) -> str:
    return "n/a" if denom == 0 else f"{100.0 * numer / denom:.2f}%"


def _print_examples(title: str, values: Iterable[str], sample_size: int) -> None:
    sample = sorted(values)[:sample_size]
    print(f"\n{title} (showing {len(sample)} of up to {sample_size}):")
    if sample:
        for value in sample:
            print(f"  {value}")
    else:
        print("  <none>")


def latent_test_rows() -> pd.DataFrame:
    meta = pd.read_csv(train_latent.META_CSV, low_memory=False).reset_index(drop=True)
    _, _, test_idx = split_row_indices_by_iau_train_val_test(
        meta,
        train_latent.IAU_COLUMN,
        train_latent.TRAIN_FRAC,
        train_latent.VAL_FRAC,
        train_latent.TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    test_idx = filter_indices_mapped(meta, test_idx, train_latent.LABEL_COLUMN)
    return meta.iloc[np.asarray(test_idx, dtype=np.int64)].copy()


def parquet_test_rows() -> tuple[pd.DataFrame, list[str], list[str]]:
    if not ruiyao_parquet.RUIYAO_TRAIN_VAL_TEST_JSON.is_file():
        raise FileNotFoundError(
            f"Missing split JSON: {ruiyao_parquet.RUIYAO_TRAIN_VAL_TEST_JSON}"
        )
    if not ruiyao_parquet.RUIYAO_PARQUET.is_file():
        raise FileNotFoundError(f"Missing parquet: {ruiyao_parquet.RUIYAO_PARQUET}")

    splits = json.loads(ruiyao_parquet.RUIYAO_TRAIN_VAL_TEST_JSON.read_text(encoding="utf-8"))
    test_ids = list(splits.get("test", []))
    df = pd.read_parquet(ruiyao_parquet.RUIYAO_PARQUET).reset_index(drop=True)

    valid_ids: list[str] = []
    invalid_ids: list[str] = []
    ilocs: list[int] = []
    for sid in test_ids:
        try:
            iloc = ruiyao_parquet.wiserep_spectrum_id_to_iloc(str(sid))
        except (IndexError, TypeError, ValueError):
            invalid_ids.append(str(sid))
            continue
        if 0 <= iloc < len(df):
            valid_ids.append(str(sid))
            ilocs.append(iloc)
        else:
            invalid_ids.append(str(sid))

    rows = df.iloc[np.asarray(ilocs, dtype=np.int64)].copy()
    rows.insert(0, "WISEREP spectrum id", valid_ids)
    return rows, valid_ids, invalid_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare train_latent.py and dash_retrain.py --parquet-ruiyao test sets."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of overlapping values to print as examples.",
    )
    args = parser.parse_args()

    latent_rows = latent_test_rows()
    parquet_rows, parquet_valid_ids, parquet_invalid_ids = parquet_test_rows()

    if train_latent.IAU_COLUMN not in latent_rows.columns:
        raise KeyError(f"Latent metadata missing {train_latent.IAU_COLUMN!r}")
    if train_latent.IAU_COLUMN not in parquet_rows.columns:
        raise KeyError(f"Parquet missing {train_latent.IAU_COLUMN!r}")

    latent_iaus = _values(latent_rows[train_latent.IAU_COLUMN])
    parquet_iaus = _values(parquet_rows[train_latent.IAU_COLUMN])
    iau_overlap = latent_iaus & parquet_iaus

    print("Sources")
    print(f"  train_latent metadata: {train_latent.META_CSV}")
    print(f"  train_latent USE_REDSHIFT: {train_latent.USE_REDSHIFT}")
    print(f"  train_latent IAU split seed: {IAU_SPLIT_SEED}")
    print(f"  parquet: {ruiyao_parquet.RUIYAO_PARQUET}")
    print(f"  parquet split JSON: {ruiyao_parquet.RUIYAO_TRAIN_VAL_TEST_JSON}")

    print("\nTest Set Sizes")
    print(f"  latent test spectra after label filtering: {len(latent_rows)}")
    print(f"  latent test unique IAU names: {len(latent_iaus)}")
    print(f"  parquet-ruiyao test IDs in split JSON: {len(parquet_valid_ids) + len(parquet_invalid_ids)}")
    print(f"  parquet-ruiyao valid test rows: {len(parquet_rows)}")
    print(f"  parquet-ruiyao invalid/out-of-range test IDs: {len(parquet_invalid_ids)}")
    print(f"  parquet-ruiyao test unique IAU names: {len(parquet_iaus)}")

    print("\nIAU/Object Overlap")
    print(f"  overlapping IAU names: {len(iau_overlap)}")
    print(f"  percent of latent test IAU names: {_pct(len(iau_overlap), len(latent_iaus))}")
    print(f"  percent of parquet test IAU names: {_pct(len(iau_overlap), len(parquet_iaus))}")

    latent_overlap_spectra = latent_rows[
        latent_rows[train_latent.IAU_COLUMN].map(_norm).isin(iau_overlap)
    ]
    parquet_overlap_spectra = parquet_rows[
        parquet_rows[train_latent.IAU_COLUMN].map(_norm).isin(iau_overlap)
    ]
    print(f"  latent test spectra whose IAU overlaps: {len(latent_overlap_spectra)}")
    print(f"  parquet test spectra whose IAU overlaps: {len(parquet_overlap_spectra)}")

    _print_examples("Overlapping IAU names", iau_overlap, args.sample_size)

    common_key_cols = [
        col for col in SPECTRUM_KEY_CANDIDATES if col in latent_rows.columns and col in parquet_rows.columns
    ]
    if common_key_cols:
        print("\nExact Spectrum-Key Overlap")
        for col in common_key_cols:
            latent_vals = _values(latent_rows[col])
            parquet_vals = _values(parquet_rows[col])
            overlap = latent_vals & parquet_vals
            print(
                f"  {col}: {len(overlap)} overlap "
                f"({len(latent_vals)} latent values, {len(parquet_vals)} parquet values)"
            )
            if overlap:
                _print_examples(f"  Examples for {col}", overlap, args.sample_size)
    else:
        print("\nExact Spectrum-Key Overlap")
        print("  No shared spectrum-level key columns found between the two tables.")


if __name__ == "__main__":
    main()
