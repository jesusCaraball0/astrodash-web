#!/usr/bin/env python
"""
K-fold cross-validation training for DASH 1D CNN on WISeREP spectra.

Uses nested validation inside each outer fold:
- outer fold val set: used only for final fold evaluation
- inner val split from outer-train: used for early stopping / LR scheduling

Usage:
    python dash_retrain_kfold.py --k-fold 8
    python dash_retrain_kfold.py --k-fold 8 --seed 42
    python dash_retrain_kfold.py --k-fold 8 --inner-val-frac 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import constants as const
import helpers

from dash_retrain import (
    DashCNN1D,
    WISeREPDataset,
    collate_skip_none,
    evaluate,
    train,
)

# Silence spectrum load errors so tqdm stays readable (skipped files are expected in WISeREP)
for _name in (
    "app.infrastructure.storage.file_spectrum_repository",
    "prod_backend.app.infrastructure.storage.file_spectrum_repository",
    "app.infrastructure.ml.data_processor",
    "prod_backend.app.infrastructure.ml.data_processor",
):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


def get_label_idx(
    fname: str,
    metadata: Dict[str, Dict[str, str]],
) -> Optional[int]:
    """Return normalized class index for filename, or None if invalid."""
    meta = metadata.get(fname)
    if meta is None:
        return None
    label = helpers.normalize_label(meta["type"])
    if label is None:
        return None
    return const.CLASS_TO_IDX[label]


def grouped_valid_filenames(
    filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
) -> Dict[int, List[str]]:
    """
    Group valid filenames by class index.
    Invalid / unlabeled files are excluded.
    """
    by_label: Dict[int, List[str]] = {i: [] for i in range(const.NUM_CLASSES)}
    for fname in filenames:
        idx = get_label_idx(fname, metadata)
        if idx is not None:
            by_label[idx].append(fname)
    return by_label


def stratified_train_val_split(
    filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Stratified split into train/val by class.
    Only valid labeled filenames are included.
    """
    rng = random.Random(seed)
    by_label = grouped_valid_filenames(filenames, metadata)

    train_out: List[str] = []
    val_out: List[str] = []

    for idx in range(const.NUM_CLASSES):
        files = by_label[idx][:]
        rng.shuffle(files)

        if not files:
            continue

        n_val = max(1, int(len(files) * val_frac))
        n_val = min(n_val, len(files) - 1) if len(files) > 1 else 1

        val_out.extend(files[:n_val])
        train_out.extend(files[n_val:])

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


def stratified_k_folds(
    filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified k-fold split by class.
    Only valid labeled filenames are included.
    Returns a list of (outer_train, outer_val) pairs.
    """
    rng = random.Random(seed)
    by_label = grouped_valid_filenames(filenames, metadata)

    folds: List[List[str]] = [[] for _ in range(k)]
    for idx in range(const.NUM_CLASSES):
        files = by_label[idx][:]
        rng.shuffle(files)
        for i, fname in enumerate(files):
            folds[i % k].append(fname)

    result: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(k):
        outer_val = list(folds[fold_idx])
        outer_train: List[str] = []
        for j in range(k):
            if j != fold_idx:
                outer_train.extend(folds[j])

        rng.shuffle(outer_train)
        rng.shuffle(outer_val)
        result.append((outer_train, outer_val))

    return result


def run_single_fold(
    outer_train_filenames: List[str],
    outer_val_filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    device: torch.device,
    fold_seed: int,
    inner_val_frac: float,
    out_dir: Path,
) -> Dict[str, float | str | int]:
    """
    Run one outer fold.

    Steps:
    - split outer_train into inner_train / inner_val
    - train with early stopping on inner_val
    - evaluate final chosen model on outer_val only
    """
    inner_train_filenames, inner_val_filenames = stratified_train_val_split(
        outer_train_filenames,
        metadata,
        val_frac=inner_val_frac,
        seed=fold_seed,
    )


    inner_train_loader = helpers.make_loader(
        inner_train_filenames, metadata, const.HAS_REDSHIFT, device,
        shuffle=True, batch_size=const.BATCH_SIZE,
    )
    inner_val_loader = helpers.make_loader(
        inner_val_filenames, metadata, const.HAS_REDSHIFT, device,
        batch_size=const.BATCH_SIZE,
    )
    outer_val_loader = helpers.make_loader(
        outer_val_filenames, metadata, const.HAS_REDSHIFT, device,
        batch_size=const.BATCH_SIZE,
    )

    class_weights = helpers.compute_class_weights_from_filenames(
        inner_train_filenames, metadata
    )

    model = DashCNN1D(
        input_length=const.TARGET_LENGTH,
        num_classes=const.NUM_CLASSES,
    ).to(device)

    best_model_path = train(
        model,
        inner_train_loader,
        inner_val_loader,
        device,
        class_weights,
        epochs=const.EPOCHS,
        lr=const.LEARNING_RATE,
        patience=const.EARLY_STOP_PATIENCE,
        val_every=const.VAL_EVERY,
        out_dir=out_dir,
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Match training-time loss definition for consistency.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    outer_val_loss, outer_val_acc, _ = evaluate(
        model,
        outer_val_loader,
        criterion,
        device,
    )

    return {
        "best_model_path": str(best_model_path),
        "outer_val_loss": float(outer_val_loss),
        "outer_val_acc": float(outer_val_acc),
        "outer_train_count": len(outer_train_filenames),
        "outer_val_count": len(outer_val_filenames),
        "inner_train_count": len(inner_train_filenames),
        "inner_val_count": len(inner_val_filenames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DASH 1D CNN on WISeREP with k-fold cross-validation."
    )
    parser.add_argument(
        "--k-fold",
        type=int,
        required=True,
        metavar="K",
        help="Number of outer folds for cross-validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold construction",
    )
    parser.add_argument(
        "--inner-val-frac",
        type=float,
        default=0.1,
        help="Fraction of each outer-train fold used for inner validation / early stopping",
    )
    args = parser.parse_args()

    k_fold = args.k_fold
    seed = args.seed
    inner_val_frac = args.inner_val_frac

    if k_fold < 2:
        raise SystemExit("--k-fold must be >= 2")
    if not (0.0 < inner_val_frac < 1.0):
        raise SystemExit("--inner-val-frac must be in (0, 1)")

    helpers.set_seed(seed)

    print("Starting nested k-fold training for DASH 1D CNN Model on Wiserep dataset")
    device = helpers.get_device()
    print(f"Device: {device}")
    print(f"K-fold: {k_fold}  Seed: {seed}  Inner val frac: {inner_val_frac}")

    splits = helpers.load_json(const.SPLITS_JSON_90_10)
    print(
        f"Splits file: train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}"
    )

    print(f"Loading metadata from {const.METADATA_CSV}")
    metadata = helpers.load_metadata(const.METADATA_CSV)
    print(f"{len(metadata)} filename -> (type, redshift) entries")

    outer_folds = stratified_k_folds(
        splits["train"],
        metadata,
        k=k_fold,
        seed=seed,
    )

    print(f"\nStratified {k_fold}-fold CV over train split")
    fold_results: List[Dict[str, float | str | int]] = []

    for fold_idx, (outer_train_filenames, outer_val_filenames) in enumerate(outer_folds):
        print(f"\n{'=' * 60}")
        print(
            f"Fold {fold_idx + 1}/{k_fold}  "
            f"outer_train={len(outer_train_filenames)}  outer_val={len(outer_val_filenames)}"
        )
        print("=" * 60)

        fold_dir = const.OUT_DIR / f"fold_{fold_idx}"
        result = run_single_fold(
            outer_train_filenames=outer_train_filenames,
            outer_val_filenames=outer_val_filenames,
            metadata=metadata,
            device=device,
            fold_seed=seed + fold_idx,
            inner_val_frac=inner_val_frac,
            out_dir=fold_dir,
        )
        fold_results.append(result)

    val_losses = [float(r["outer_val_loss"]) for r in fold_results]
    val_accs = [float(r["outer_val_acc"]) for r in fold_results]

    mean_loss = float(np.mean(val_losses))
    std_loss = float(np.std(val_losses))
    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs))

    print(f"\n{'=' * 60}")
    print(f"K-fold summary (k={k_fold})")
    print(f"  Outer val loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Outer val acc:  {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 60)

    training_config = {
        "run_id": const.RUN_ID,
        "has_redshift": const.HAS_REDSHIFT,
        "target_length": const.TARGET_LENGTH,
        "wave_min": const.WAVE_MIN,
        "wave_max": const.WAVE_MAX,
        "num_classes": const.NUM_CLASSES,
        "class_names": const.CLASS_NAMES,
        "epochs": const.EPOCHS,
        "batch_size": const.BATCH_SIZE,
        "lr": const.LEARNING_RATE,
        "patience": const.EARLY_STOP_PATIENCE,
        "splits_file": str(const.SPLITS_JSON_90_10),
        "k_fold": k_fold,
        "seed": seed,
        "inner_val_frac": inner_val_frac,
        "cv_outer_val_loss_mean": mean_loss,
        "cv_outer_val_loss_std": std_loss,
        "cv_outer_val_acc_mean": mean_acc,
        "cv_outer_val_acc_std": std_acc,
        "fold_results": fold_results,
    }

    const.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(const.OUT_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"\nModels saved under {const.OUT_DIR}/fold_0 .. fold_{k_fold - 1}")
    print(f"Config: {const.OUT_DIR / 'training_config.json'}")
    print("Done.")


if __name__ == "__main__":
    main()