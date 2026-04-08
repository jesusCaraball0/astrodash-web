#!/usr/bin/env python
"""
Evaluate a trained DASH WISeREP model (by run_id) on val and test splits and generate
row-normalized confusion matrices. Saves figures under models/<run_id>/confusion_matrices/.

Two entry points:
- eval_single_model(run_id): one model, val and test from splits
- eval_kfold_models(run_id): k models; val = mean ± std across folds,
  test = mean ± std across folds.

Usage:
    python zmodel_training/dash_eval_confusion_matrices.py --run-id 03_14_26_redshift
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import dash_retrain
import helpers as helpers
from dash_retrain_kfold import stratified_k_folds

MODELS_BASE = PROJECT_ROOT / "data" / "pre_trained_models" / "dash_wiserep" / "models"

logging.getLogger("app.infrastructure.storage.file_spectrum_repository").setLevel(logging.CRITICAL)
logging.getLogger("app.infrastructure.ml.data_processor").setLevel(logging.CRITICAL)


def load_class_names(class_mapping_path: Path) -> List[str]:
    class_mapping = helpers.load_json(class_mapping_path)
    idx_to_name = {int(v): str(k) for k, v in class_mapping.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]





def load_model(model_path: Path, n_classes: int, device: torch.device) -> nn.Module:
    model = dash_retrain.DashCNN1D(
        input_length=const.TARGET_LENGTH,
        num_classes=n_classes,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def confusion_matrix_counts(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            preds = model(x.to(device)).argmax(dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                ti, pi = int(t.item()), int(p.item())
                if 0 <= ti < n_classes and 0 <= pi < n_classes:
                    cm[ti][pi] += 1
    return cm


def row_normalized_percent(cm: List[List[int]]) -> np.ndarray:
    return np.asarray([
        [100.0 * v / sum(row) if sum(row) > 0 else 0.0 for v in row]
        for row in cm
    ], dtype=float)


def mean_std_percent(cms: List[List[List[int]]], n_classes: int):
    arr = np.asarray([row_normalized_percent(cm) for cm in cms], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    row_counts = [int(np.mean([sum(cm[i]) for cm in cms])) for i in range(n_classes)]
    return mean, std, row_counts


def plot_confusion_matrix(
    cm_or_mean: np.ndarray,
    class_names: List[str],
    set_name: str,
    out_path: Path,
    row_counts: List[int],
    cm_std: np.ndarray | None = None,
) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * n), max(6.0, 1.1 * n)))
    ax.imshow(cm_or_mean, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_title(set_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            m = float(cm_or_mean[i, j])
            if cm_std is None:
                text, fs = f"{m:.1f}", 9
            else:
                s = float(cm_std[i, j])
                text, fs = (f"{m:.1f}\n±{s:.1f}" if s > 0 else f"{m:.1f}"), 8
            ax.text(j, i, text, ha="center", va="center", fontsize=fs,
                    color="white" if m >= 50 else "black")

        ax.text(
            n - 0.5, i, f"  n={row_counts[i]}",
            ha="left", va="center", fontsize=9, color="black",
            transform=ax.transData,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def eval_single_model(run_id: str) -> None:
    out_dir = MODELS_BASE / run_id
    model_path = out_dir / "model.pth"
    class_mapping_path = out_dir / "class_mapping.json"
    training_config_path = out_dir / "training_config.json"
    confusion_dir = out_dir / "confusion_matrices"

    config = helpers.load_json(training_config_path)
    splits_path = Path(config.get("splits_file", const.SPLITS_JSON_80_10_10))
    splits = helpers.load_json(splits_path)

    has_redshift = config.get("has_redshift", True)
    device = helpers.get_device()
    class_names = load_class_names(class_mapping_path)
    n_classes = len(class_names)
    model = load_model(model_path, n_classes, device)
    batch_size = int(config.get("batch_size", 64))

    parquet_path = config.get("parquet")
    if parquet_path:
        import pandas as pd

        import ruiyao_parquet_dataset as rpd

        df = pd.read_parquet(Path(parquet_path))
        for split_name in ("val", "test"):
            ids = list(splits.get(split_name, []))
            if not ids:
                continue
            loader = DataLoader(
                rpd.ParquetSpectrumDataset(ids, df, has_redshift=has_redshift),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=dash_retrain.collate_skip_none,
                pin_memory=(device.type == "cuda"),
            )
            cm = confusion_matrix_counts(model, loader, device, n_classes)
            plot_confusion_matrix(
                row_normalized_percent(cm),
                class_names,
                split_name,
                confusion_dir / f"confusion_{split_name}_{run_id}.png",
                [sum(row) for row in cm],
            )
        return

    metadata = helpers.load_metadata(const.METADATA_CSV)
    for split_name in ("val", "test"):
        filenames = list(splits.get(split_name, []))
        if not filenames:
            continue
        loader = helpers.make_loader(filenames, metadata, has_redshift, device)
        cm = confusion_matrix_counts(model, loader, device, n_classes)
        plot_confusion_matrix(
            row_normalized_percent(cm),
            class_names,
            split_name,
            confusion_dir / f"confusion_{split_name}_{run_id}.png",
            [sum(row) for row in cm],
        )


def eval_kfold_models(run_id: str) -> None:
    out_dir = MODELS_BASE / run_id
    class_mapping_path = out_dir / "class_mapping.json"
    training_config_path = out_dir / "training_config.json"
    confusion_dir = out_dir / "confusion_matrices"

    config = helpers.load_json(training_config_path)
    has_redshift = config.get("has_redshift", True)
    seed = int(config.get("seed", 42))
    device = helpers.get_device()
    metadata = helpers.load_metadata(const.METADATA_CSV)
    class_names = load_class_names(class_mapping_path)
    n_classes = len(class_names)
    splits = helpers.load_json(const.SPLITS_JSON_90_10)

    fold_model_paths = sorted(out_dir.glob("fold_*/model.pth"))
    if not fold_model_paths:
        raise SystemExit(f"No fold models under {out_dir}/fold_*/model.pth")

    train_filenames = list(splits.get("train", []))
    if not train_filenames:
        raise SystemExit(f"Splits file has no 'train' key: {const.SPLITS_JSON_90_10}")

    folds = stratified_k_folds(train_filenames, metadata, len(fold_model_paths), seed=seed)

    val_cms = []
    for fold, model_path in enumerate(fold_model_paths):
        val_filenames = folds[fold][1]
        if not val_filenames:
            continue
        loader = helpers.make_loader(val_filenames, metadata, has_redshift, device)
        model = load_model(model_path, n_classes, device)
        val_cms.append(confusion_matrix_counts(model, loader, device, n_classes))

    if not val_cms:
        raise SystemExit("No val confusion matrices produced (check folds and models)")

    val_mean, val_std, val_row_counts = mean_std_percent(val_cms, n_classes)
    val_out = confusion_dir / f"confusion_val_{run_id}.png"
    plot_confusion_matrix(
        val_mean,
        class_names,
        "val (mean +/- std over folds)",
        val_out,
        val_row_counts,
        val_std,
    )
    print(f"Saved {val_out}")

    test_filenames = list(splits.get("test", []))
    if not test_filenames:
        return

    test_cms = []
    loader = helpers.make_loader(test_filenames, metadata, has_redshift, device)
    for model_path in fold_model_paths:
        model = load_model(model_path, n_classes, device)
        test_cms.append(confusion_matrix_counts(model, loader, device, n_classes))

    if test_cms:
        test_mean, test_std, test_row_counts = mean_std_percent(test_cms, n_classes)
        test_out = confusion_dir / f"confusion_test_{run_id}.png"
        plot_confusion_matrix(
            test_mean,
            class_names,
            "test (mean +/- std over folds)",
            test_out,
            test_row_counts,
            test_std,
        )
        print(f"Saved {test_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DASH WISeREP model by run_id and save confusion matrices."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID (subdirectory under dash_wiserep/models/, e.g. 03_14_26_redshift)",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip()
    if not run_id:
        raise SystemExit("--run-id must be non-empty")

    out_dir = MODELS_BASE / run_id

    if (out_dir / "fold_0" / "model.pth").exists():
        eval_kfold_models(run_id)
    elif (out_dir / "model.pth").exists():
        eval_single_model(run_id)
    else:
        raise SystemExit(
            f"No model found under {out_dir}. "
            f"Expected either model.pth (single run) or fold_0/model.pth, fold_1/model.pth, ... (k-fold)."
        )


if __name__ == "__main__":
    main()