#!/usr/bin/env python
"""
Evaluate a trained DAEP transceiver classifier (daep_classifier.py) by run_id and save
row-normalized confusion matrices under models/<run_id>/confusion_matrices/.

Training writes checkpoints to constants.DAEP_DIR (default: models/daep_classifier_small/) and saves
daep_architecture.json next to model.pth so eval matches the architecture even if constants.py changes.

To use the same --run-id as a DASH run, copy or symlink model.pth into that folder, e.g.:
    cp data/pre_trained_models/dash_wiserep/models/daep_classifier_small/model.pth \\
       data/pre_trained_models/dash_wiserep/models/04_07_26_redshift/model.pth

Or pass --run-id daep_classifier_small to point at the default DAEP output directory.

Usage:
    python zmodel_training/daep_eval_confusion_matrices.py --run-id 04_07_26_redshift
    python zmodel_training/daep_eval_confusion_matrices.py --run-id daep_classifier_small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import daep_architecture as daep_arch
import dash_retrain
import helpers as helpers
import parquet_dataset as rpd
from daep_classifier import DAEPClassifier, ParquetRawSpectrumDataset, _collate
from dash_eval_confusion_matrices import plot_confusion_matrix, row_normalized_percent

MODELS_BASE = PROJECT_ROOT / "data" / "pre_trained_models" / "dash_wiserep" / "models"


def eval_daep(run_id: str) -> None:
    out_dir = MODELS_BASE / run_id
    model_path = out_dir / "model.pth"
    if not model_path.is_file():
        raise SystemExit(f"No model at {model_path}")

    try:
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ck = torch.load(model_path, map_location="cpu")
    if not isinstance(ck, dict):
        raise SystemExit(f"Expected a state_dict at {model_path}")

    arch_path = out_dir / daep_arch.DAEP_ARCH_JSON
    if arch_path.is_file():
        daep_arch.apply_architecture(daep_arch.load_architecture_json(arch_path))
    else:
        daep_arch.apply_architecture(daep_arch.infer_architecture_from_state_dict(ck))

    device = helpers.get_device()
    nw = const.TARGET_LENGTH - 1

    df, _metadata, _train_ids, val_ids, test_ids = rpd.load_df_metadata_train_val_ids(const.SEED)

    model = DAEPClassifier().to(device)
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError as e:
        bad = model.load_state_dict(ck, strict=False)
        if bad.missing_keys:
            raise RuntimeError(
                "Partial load failed: checkpoint does not match the current daep.SpectraLayers "
                "module (use the same Perceiver submodule commit as training, or retrain). "
                f"Missing keys: {len(bad.missing_keys)}. Original error: {e}"
            ) from e
        if bad.unexpected_keys:
            print(
                "Note: ignored unexpected keys in checkpoint (often harmless if daep code drifted): "
                f"{len(bad.unexpected_keys)} keys"
            )
    crit_eval = nn.CrossEntropyLoss()

    confusion_dir = out_dir / "confusion_matrices"
    class_names = list(const.CLASS_NAMES)

    for split_name, ids in (("val", val_ids), ("test", test_ids)):
        if not ids:
            continue
        ds = ParquetRawSpectrumDataset(ids, df, nw, const.WAVE_MIN, const.WAVE_MAX)
        loader = DataLoader(
            ds,
            batch_size=const.BATCH_SIZE,
            shuffle=False,
            num_workers=const.NUM_WORKERS,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
        )
        _loss, _acc, cm = dash_retrain.evaluate(model, loader, crit_eval, device)
        plot_confusion_matrix(
            row_normalized_percent(cm),
            class_names,
            split_name,
            confusion_dir / f"confusion_{split_name}_{run_id}.png",
            [sum(row) for row in cm],
        )
        print(f"Saved {confusion_dir / f'confusion_{split_name}_{run_id}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DAEP classifier by run_id and save confusion matrices."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Subdirectory under dash_wiserep/models/ containing model.pth (e.g. daep_classifier_small or 04_07_26_redshift)",
    )
    args = parser.parse_args()
    run_id = args.run_id.strip()
    if not run_id:
        raise SystemExit("--run-id must be non-empty")
    eval_daep(run_id)


if __name__ == "__main__":
    main()
