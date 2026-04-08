"""
Colleague WISeREP parquet + JSON split. WISEREP_0000123 = row iloc 123 (not parquet column 'index').
See inspect_wiserep_ruiyao.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import constants as const
import helpers as helpers

RUIYAO_DIR = const.PROJECT_ROOT / "data" / "wiserep_ruiyao"
RUIYAO_PARQUET = RUIYAO_DIR / "wiserep_spectra_5class_optical.parquet"
RUIYAO_SPLIT_JSON = RUIYAO_DIR / "wiserep_split_ids.json"
RUIYAO_METADATA_CACHE = RUIYAO_DIR / "parquet_spectrum_metadata_cache.json"
# train/val/test (WISEREP_* ids), same keys as 80/10/10 JSON — written by create_ruiyao_object_train_val_split.py
RUIYAO_TRAIN_VAL_TEST_JSON = RUIYAO_DIR / "wiserep_splits_train_val_test.json"
RUIYAO_VAL_FRAC = 0.1


def wiserep_spectrum_id_to_iloc(spectrum_id: str) -> int:
    return int(spectrum_id.split("_", 1)[1], 10)


def load_or_build_metadata(df: pd.DataFrame, cache_path: Path) -> Dict[str, Dict[str, str]]:
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = {str(k): v for k, v in raw.items() if not str(k).startswith("_")}
        if len(meta) == len(df):
            return meta

    meta = {}
    for i in range(len(df)):
        sid = f"WISEREP_{i:07d}"
        row = df.iloc[i]
        pc = row["parent_class"]
        typ = "" if pd.isna(pc) else str(pc).strip()
        zv = row["Redshift"]
        zs = "0" if pd.isna(zv) else str(float(zv))
        meta[sid] = {"type": typ, "redshift": zs}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def stratified_train_val_from_json_train(
    train_spectrum_ids: List[str],
    metadata: Dict[str, Dict[str, str]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """JSON has train/test only; carve stratified val from train ids."""
    by_label: Dict[int, List[str]] = {i: [] for i in range(const.NUM_CLASSES)}
    unlabeled: List[str] = []
    for sid in train_spectrum_ids:
        m = metadata.get(sid)
        if not m:
            unlabeled.append(sid)
            continue
        lab = helpers.normalize_label(m.get("type", ""))
        if lab is None:
            unlabeled.append(sid)
            continue
        by_label[const.CLASS_TO_IDX[lab]].append(sid)

    rng = random.Random(seed)
    train_out: List[str] = []
    val_out: List[str] = []
    for _li, ids in by_label.items():
        if not ids:
            continue
        ids = list(ids)
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_fraction))
        if len(ids) >= 2:
            n_val = min(max(n_val, 1), len(ids) - 1)
        else:
            n_val = 0
        if n_val == 0:
            train_out.extend(ids)
        else:
            val_out.extend(ids[:n_val])
            train_out.extend(ids[n_val:])
    rng.shuffle(unlabeled)
    n_uv = min(len(unlabeled), int(round(len(unlabeled) * val_fraction)))
    val_out.extend(unlabeled[:n_uv])
    train_out.extend(unlabeled[n_uv:])
    return train_out, val_out


class ParquetSpectrumDataset(Dataset):
    """Spectrum ids WISEREP_*; columns wavelength, flux, parent_class, Redshift."""

    def __init__(
        self,
        spectrum_ids: List[str],
        df: pd.DataFrame,
        target_length: int = const.TARGET_LENGTH,
        has_redshift: bool = True,
    ):
        self.df = df
        self.target_length = target_length
        self.has_redshift = has_redshift
        n = len(df)
        self.samples: List[Tuple[int, int, float]] = []
        for sid in spectrum_ids:
            i = wiserep_spectrum_id_to_iloc(sid)
            if not (0 <= i < n):
                continue
            row = df.iloc[i]
            pc = row["parent_class"]
            typ = "" if pd.isna(pc) else str(pc).strip()
            lab = helpers.normalize_label(typ)
            if lab is None:
                continue
            zv = row["Redshift"]
            if pd.isna(zv):
                z_val = 0.0
            else:
                try:
                    z_val = float(zv)
                except (TypeError, ValueError):
                    z_val = 0.0
            self.samples.append((i, const.CLASS_TO_IDX[lab], z_val))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        iloc, label_idx, z_val = self.samples[idx]
        row = self.df.iloc[iloc]
        try:
            wave = np.asarray(row["wavelength"], dtype=np.float64)
            flux = np.asarray(row["flux"], dtype=np.float64)
        except Exception:
            return None
        if wave.size == 0 or flux.size == 0:
            return None
        processed = helpers.preprocess_spectrum(
            wave, flux, z_val if self.has_redshift else None, self.target_length
        )
        if processed is None:
            return None
        processed = np.concatenate([processed, [z_val if self.has_redshift else 0.0]])
        return torch.from_numpy(processed.astype(np.float32)), label_idx


def load_df_metadata_train_val_ids(
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], List[str], List[str], int]:
    """Read parquet; train/val from wiserep_splits_train_val_test.json if present, else random spectrum val."""
    colleague = helpers.load_json(RUIYAO_SPLIT_JSON)
    train_json = list(colleague.get("train_spectrum_ids", []))
    test_n = len(colleague.get("test_spectrum_ids", []))
    df = pd.read_parquet(RUIYAO_PARQUET)
    for col in ("wavelength", "flux", "parent_class", "Redshift"):
        if col not in df.columns:
            raise SystemExit(f"Parquet missing column {col!r}")
    helpers.set_seed(seed)
    metadata = load_or_build_metadata(df, RUIYAO_METADATA_CACHE)

    if RUIYAO_TRAIN_VAL_TEST_JSON.exists():
        u = json.loads(RUIYAO_TRAIN_VAL_TEST_JSON.read_text(encoding="utf-8"))
        return df, metadata, list(u["train"]), list(u["val"]), len(u.get("test", []))

    train_ids, val_ids = stratified_train_val_from_json_train(
        train_json, metadata, RUIYAO_VAL_FRAC, seed,
    )
    print(f"Note: write {RUIYAO_TRAIN_VAL_TEST_JSON.name} (create_ruiyao_object_train_val_split.py) for fixed val.")
    return df, metadata, train_ids, val_ids, test_n
