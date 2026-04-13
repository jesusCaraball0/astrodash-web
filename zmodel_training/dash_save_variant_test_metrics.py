#!/usr/bin/env python
"""
For each WISeREP run, evaluate the saved model on the **test** split using preprocessing
that **matches that run's ablation** (via duplicated pipeline flags in dash_variant_preprocess.py),
then merge results into that run's model_performance.json under `test_metrics`.

This fixes eval drift when prod data_processor.py no longer matches the code state used during training.

Usage:
  python zmodel_training/dash_save_variant_test_metrics.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import dash_retrain
import helpers as helpers
from dash_eval_confusion_matrices import (
    MODELS_BASE,
    confusion_matrix_counts,
    load_class_names,
    load_model,
)
from dash_variant_preprocess import PreprocessFlags, preprocess_spectrum_variant
from parquet_dataset import ParquetSpectrumDataset

for _name in (
    "app.infrastructure.storage.file_spectrum_repository",
    "prod_backend.app.infrastructure.storage.file_spectrum_repository",
    "app.infrastructure.ml.data_processor",
    "prod_backend.app.infrastructure.ml.data_processor",
):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


# Baseline + ablations (1D CNN, Ruiyao split). Flags must match what each run was trained with.
_FULL = PreprocessFlags()
RUN_VARIANT_FLAGS: Dict[str, PreprocessFlags] = {
    "04_11_26_redshift": _FULL,
    "04_13_26_redshift": replace(_FULL, use_continuum_removal=False),
    "04_14_26_redshift": replace(_FULL, use_medfilt=False),
    "04_15_26_redshift": replace(_FULL, use_apodize=False),
    "04_16_26_redshift": replace(_FULL, initial_norm=False),
    "04_17_26_redshift": replace(_FULL, deredshift=False),
    "04_18_26_redshift": replace(
        _FULL,
        initial_norm=False,
        norm_after_slice=False,
        continuum_tail_norm=False,
        mean_zero=False,
        final_norm=False,
    ),
}


class ParquetSpectrumDatasetForVariant(ParquetSpectrumDataset):
    """Parquet rows + preprocess_spectrum_variant(..., flags) instead of helpers.preprocess_spectrum."""

    def __init__(
        self,
        spectrum_ids: List[str],
        df,
        flags: PreprocessFlags,
        target_length: int = const.TARGET_LENGTH,
        has_redshift: bool = True,
    ):
        self._flags = flags
        super().__init__(spectrum_ids, df, target_length=target_length, has_redshift=has_redshift)

    def __getitem__(self, idx: int):
        iloc, label_idx, z_val = self.samples[idx]
        row = self.df.iloc[iloc]
        try:
            wave = np.asarray(row["wavelength"], dtype=np.float64)
            flux = np.asarray(row["flux"], dtype=np.float64)
        except Exception:
            return None
        if wave.size == 0 or flux.size == 0:
            return None
        processed = preprocess_spectrum_variant(
            wave,
            flux,
            z_val if self.has_redshift else None,
            self.target_length,
            self._flags,
        )
        if processed is None:
            return None
        processed = np.concatenate([processed, [z_val if self.has_redshift else 0.0]])
        return torch.from_numpy(processed.astype(np.float32)), label_idx


def _test_cm(run_id: str, flags: PreprocessFlags):
    out_dir = MODELS_BASE / run_id
    training_config_path = out_dir / "training_config.json"
    config = helpers.load_json(training_config_path)
    splits_path = Path(config.get("splits_file", const.SPLITS_JSON_80_10_10))
    splits = helpers.load_json(splits_path)
    test_ids = list(splits.get("test", []))
    if not test_ids:
        raise ValueError(f"No test ids in {splits_path}")

    import pandas as pd

    parquet_path = config.get("parquet")
    if not parquet_path:
        raise SystemExit(f"{run_id}: training_config has no parquet path (this script supports Ruiyao parquet only).")

    df = pd.read_parquet(Path(parquet_path))
    has_redshift = config.get("has_redshift", True)
    device = helpers.get_device()
    class_names = load_class_names(out_dir / "class_mapping.json")
    n_classes = len(class_names)
    model = load_model(out_dir / "model.pth", n_classes, device)
    batch_size = int(config.get("batch_size", 64))

    loader = DataLoader(
        ParquetSpectrumDatasetForVariant(test_ids, df, flags, has_redshift=has_redshift),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dash_retrain.collate_skip_none,
        pin_memory=(device.type == "cuda"),
    )
    cm = confusion_matrix_counts(model, loader, device, n_classes)
    return class_names, cm


def _acc_from_cm(cm: List[List[int]]) -> float:
    n = len(cm)
    tot = sum(sum(row) for row in cm)
    if tot == 0:
        return 0.0
    return sum(cm[i][i] for i in range(n)) / tot


def main() -> None:
    for run_id, flags in sorted(RUN_VARIANT_FLAGS.items()):
        print(f"=== {run_id} ===")
        class_names, cm = _test_cm(run_id, flags)
        perf_path = MODELS_BASE / run_id / "model_performance.json"
        if not perf_path.is_file():
            raise FileNotFoundError(f"Missing {perf_path}")

        val_acc = _acc_from_cm(cm)
        test_block = dash_retrain.build_performance_json(
            best_epoch=0,
            val_loss=0.0,
            val_acc=val_acc,
            cm=cm,
        )
        test_block["split"] = "test"
        test_block["preprocess_flags"] = {
            "initial_norm": flags.initial_norm,
            "use_medfilt": flags.use_medfilt,
            "deredshift": flags.deredshift,
            "norm_after_slice": flags.norm_after_slice,
            "use_continuum_removal": flags.use_continuum_removal,
            "continuum_tail_norm": flags.continuum_tail_norm,
            "mean_zero": flags.mean_zero,
            "use_apodize": flags.use_apodize,
            "final_norm": flags.final_norm,
        }
        test_block["note"] = (
            "Test split; preprocessing matches this run via dash_variant_preprocess.PreprocessFlags. "
            "cumulative.loss is placeholder 0 (not computed on test)."
        )

        with open(perf_path, encoding="utf-8") as f:
            root = json.load(f)
        root["test_metrics"] = test_block
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(root, f, indent=2)
        print(f"  wrote test_metrics -> {perf_path}")


if __name__ == "__main__":
    main()
