"""
Colleague WISeREP parquet + JSON split. WISEREP_0000123 = row iloc 123 (not parquet column 'index').
See inspect_wiserep_ruiyao.py.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
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
# train/val/test (WISEREP_* ids), same keys as 80/10/10 JSON — written by create_trvaltest_from_trtest.py
RUIYAO_TRAIN_VAL_TEST_JSON = RUIYAO_DIR / "wiserep_splits_train_val_test.json"


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


def build_object_stratified_train_val(
    df: pd.DataFrame,
    colleague_train_ids: List[str],
    val_object_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Group spectra by IAU (else treat each sid as its own object); stratify holdout by class stratum."""
    iau_to_sids: Dict[str, List[str]] = defaultdict(list)
    iau_stratum: Dict[str, int] = {}

    for sid in colleague_train_ids:
        i = wiserep_spectrum_id_to_iloc(sid)
        row = df.iloc[i]
        raw = row["IAU name"]
        iau = str(raw).strip() if pd.notna(raw) and str(raw).strip() else sid
        iau_to_sids[iau].append(sid)
        if iau not in iau_stratum:
            typ = "" if pd.isna(row["parent_class"]) else str(row["parent_class"]).strip()
            lab = helpers.normalize_label(typ)
            iau_stratum[iau] = const.CLASS_TO_IDX[lab] if lab is not None else -1

    by_stratum: Dict[int, List[str]] = defaultdict(list)
    for iau in iau_to_sids:
        by_stratum[iau_stratum[iau]].append(iau)

    rng = random.Random(seed)
    train_iau: List[str] = []
    val_iau: List[str] = []
    for k in sorted(by_stratum.keys()):
        objs = by_stratum[k]
        rng.shuffle(objs)
        n_val = min(max(1, int(round(len(objs) * val_object_frac))), len(objs) - 1)
        val_iau.extend(objs[:n_val])
        train_iau.extend(objs[n_val:])

    train_sids = sorted(s for iau in train_iau for s in iau_to_sids[iau])
    val_sids = sorted(s for iau in val_iau for s in iau_to_sids[iau])
    exp = set(colleague_train_ids)
    if set(train_sids) | set(val_sids) != exp or set(train_sids) & set(val_sids):
        raise ValueError("Object split failed to partition colleague train_spectrum_ids")
    return train_sids, val_sids, {"n_train_objects": len(train_iau), "n_val_objects": len(val_iau)}


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
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], List[str], List[str], List[str]]:
    """Read parquet; train/val/test from wiserep_splits_train_val_test.json (see create_trvaltest_from_trtest.py)."""
    if not RUIYAO_TRAIN_VAL_TEST_JSON.is_file():
        raise SystemExit(
            f"Missing {RUIYAO_TRAIN_VAL_TEST_JSON}. Run: python zmodel_training/create_trvaltest_from_trtest.py --force"
        )
    u = json.loads(RUIYAO_TRAIN_VAL_TEST_JSON.read_text(encoding="utf-8"))
    df = pd.read_parquet(RUIYAO_PARQUET)
    for col in ("wavelength", "flux", "parent_class", "Redshift"):
        if col not in df.columns:
            raise SystemExit(f"Parquet missing column {col!r}")
    helpers.set_seed(seed)
    metadata = load_or_build_metadata(df, RUIYAO_METADATA_CACHE)
    return (
        df,
        metadata,
        list(u["train"]),
        list(u["val"]),
        list(u.get("test", [])),
    )
