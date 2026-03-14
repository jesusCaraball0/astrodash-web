#!/usr/bin/env python
"""
Evaluate a trained DASH WISeREP model (by run_id) on val and test splits and generate
row-normalized confusion matrices. Saves figures under models/<run_id>/confusion_matrices/.

- Single run: confusion_val_<run_id>.png, confusion_test_<run_id>.png (from splits val/test).
- K-fold run: the best fold is chosen by validation accuracy (from each fold's model_performance.json).
  Only that fold's model is evaluated; outputs are confusion_val_<run_id>.png and confusion_test_<run_id>.png.

Usage (from repo root):
    python prod_backend/scripts/dash_eval_confusion_matrices.py --run-id 03_14_26_redshift
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Allow importing app (prod_backend) and dash_retrain (scripts)
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(1, str(SCRIPT_DIR))

import dash_retrain

# Base paths (run-specific OUT_DIR is set from --run-id in main)
MODELS_BASE = PROJECT_ROOT / "data" / "pre_trained_models" / "dash_wiserep" / "models"
WISEREP_DIR = dash_retrain.WISEREP_DIR
SPECTRA_DIR = dash_retrain.SPECTRA_DIR
METADATA_CSV = dash_retrain.METADATA_CSV
SPLITS_JSON = dash_retrain.SPLITS_JSON
load_metadata = dash_retrain.load_metadata
collate_skip_none = dash_retrain.collate_skip_none
DashCNN1D = dash_retrain.DashCNN1D
WISeREPDataset = dash_retrain.WISeREPDataset
TARGET_LENGTH = dash_retrain.TARGET_LENGTH
stratified_k_folds = dash_retrain.stratified_k_folds

# Keep output clean
logging.getLogger("app.infrastructure.storage.file_spectrum_repository").setLevel(logging.CRITICAL)
logging.getLogger("app.infrastructure.ml.data_processor").setLevel(logging.CRITICAL)


def confusion_matrix_counts(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                ti = int(t.item())
                pi = int(p.item())
                if 0 <= ti < n_classes and 0 <= pi < n_classes:
                    cm[ti][pi] += 1
    return cm


def confusion_matrix_row_normalized_percent(cm: List[List[int]]) -> List[List[float]]:
    out: List[List[float]] = []
    for row in cm:
        s = sum(row)
        if s == 0:
            out.append([0.0 for _ in row])
        else:
            out.append([100.0 * v / s for v in row])
    return out


def plot_confusion_matrix(
    cm_counts: List[List[int]],
    class_names: List[str],
    set_name: str,
    out_path: Path,
) -> None:
    cm_pct = confusion_matrix_row_normalized_percent(cm_counts)
    n = len(class_names)
    cm_arr = np.asarray(cm_pct, dtype=float)

    fig_w = max(7.0, 1.2 * n)
    fig_h = max(6.0, 1.1 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_arr, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_title(f"DASH retrained on WISeREP {set_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells with percentages and row counts
    for i in range(n):
        row_n = sum(cm_counts[i])
        for j in range(n):
            val = cm_arr[i, j]
            text = f"{val:.1f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val >= 50 else "black",
            )
        # Row count label at end of row (outside heatmap)
        ax.text(
            n - 0.5,
            i,
            f"  n={row_n}",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            transform=ax.transData,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
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
    class_mapping_path = out_dir / "class_mapping.json"
    training_config_path = out_dir / "training_config.json"

    if not class_mapping_path.exists():
        raise SystemExit(f"Class mapping not found: {class_mapping_path}")
    if not SPLITS_JSON.exists():
        raise SystemExit(f"Splits file not found: {SPLITS_JSON}")

    # Detect k-fold run: config has k_fold > 1 and fold_0 exists
    config = {}
    if training_config_path.exists():
        try:
            config = json.loads(training_config_path.read_text())
        except Exception:
            pass
    has_redshift = config.get("has_redshift", True)
    k_fold = int(config.get("k_fold", 1))
    seed = int(config.get("seed", 42))
    is_kfold_run = k_fold > 1 and (out_dir / "fold_0" / "model.pth").exists()

    if not is_kfold_run and not (out_dir / "model.pth").exists():
        raise SystemExit(f"Model not found: {out_dir / 'model.pth'}")

    # Device (prefer MPS on macOS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    class_mapping = json.loads(class_mapping_path.read_text())
    idx_to_name = {int(v): str(k) for k, v in class_mapping.items()}
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    n_classes = len(class_names)

    metadata = load_metadata(METADATA_CSV)
    splits = json.loads(SPLITS_JSON.read_text())
    confusion_dir = out_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    def make_loader(filenames: List[str]):
        ds = WISeREPDataset(
            filenames,
            SPECTRA_DIR,
            metadata,
            target_length=TARGET_LENGTH,
            has_redshift=has_redshift,
        )
        return DataLoader(
            ds,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_skip_none,
            pin_memory=(device.type == "cuda"),
        )

    if not is_kfold_run:
        # Single run: one model, val and test from splits
        model_path = out_dir / "model.pth"
        model = DashCNN1D(input_length=TARGET_LENGTH, num_classes=n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        for split_name in ("val", "test"):
            filenames = list(splits.get(split_name, []))
            if not filenames:
                continue
            loader = make_loader(filenames)
            cm = confusion_matrix_counts(model, loader, device, n_classes)
            out_path = confusion_dir / f"confusion_{split_name}_{run_id}.png"
            plot_confusion_matrix(cm, class_names, set_name=split_name, out_path=out_path)
        return

    # K-fold run: pick best fold by val accuracy (from model_performance.json), then only that fold's confusion matrices
    folds = stratified_k_folds(splits["train"], metadata, k_fold, seed=seed)
    best_fold = 0
    best_val_acc = -1.0
    for fold in range(k_fold):
        perf_path = out_dir / f"fold_{fold}" / "model_performance.json"
        if perf_path.exists():
            try:
                perf = json.loads(perf_path.read_text())
                acc = float(perf.get("cumulative", {}).get("accuracy_pct", -1.0))
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_fold = fold
            except Exception:
                pass

    fold = best_fold
    model_path = out_dir / f"fold_{fold}" / "model.pth"
    if not model_path.exists():
        raise SystemExit(f"Best fold {fold} model not found: {model_path}")
    model = DashCNN1D(input_length=TARGET_LENGTH, num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Val: this fold's holdout set
    val_filenames = folds[fold][1]
    if val_filenames:
        loader = make_loader(val_filenames)
        cm = confusion_matrix_counts(model, loader, device, n_classes)
        out_path = confusion_dir / f"confusion_val_{run_id}.png"
        plot_confusion_matrix(cm, class_names, set_name=f"val (best fold {fold})", out_path=out_path)

    # Test: same test set (unchanged)
    test_filenames = list(splits.get("test", []))
    if test_filenames:
        loader = make_loader(test_filenames)
        cm = confusion_matrix_counts(model, loader, device, n_classes)
        out_path = confusion_dir / f"confusion_test_{run_id}.png"
        plot_confusion_matrix(cm, class_names, set_name=f"test (best fold {fold})", out_path=out_path)


if __name__ == "__main__":
    main()
