#!/usr/bin/env python
"""
One-vs-rest ROC curves for each class on the test split of a DashCNN1D checkpoint.

For WISeREP ASCII models: the test split comes from ``training_config.json`` → ``splits_file``
(DAEP-matched runs use ``daep_matched_split_*.json``; legacy runs use ``daep_compatible_split.json``).

Parquet (Ruiyao) models use ``training_config.json`` ``splits_file`` + ``parquet`` paths.

``has_redshift`` defaults to ``constants.HAS_REDSHIFT`` when missing from config (must
match training; the old default True was wrong for HAS_REDSHIFT=False runs).

Usage:
    python zmodel_training/roc_curves.py
    python zmodel_training/roc_curves.py --model-path path/to/model.pth
    python zmodel_training/roc_curves.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import dash_retrain
import helpers as helpers

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


def collect_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns y_true (N,) int and y_score (N, n_classes) softmax probabilities."""
    ys: List[int] = []
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device)
            logits = model(x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            ys.extend(y.view(-1).numpy().tolist())
            chunks.append(proba)

    if not chunks:
        raise SystemExit("No batches collected from test loader (empty split or all samples failed loading).")

    y_true = np.asarray(ys, dtype=int)
    y_score = np.vstack(chunks)
    return y_true, y_score


def _safe_filename_class(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def print_loader_coverage(info: Dict[str, Any], n_predicted: int) -> None:
    """Explain gaps between split JSON size, Dataset length, and rows used for metrics."""
    listed = int(info.get("test_ids_listed", 0))
    unique = int(info.get("test_ids_unique", 0))
    d_len = int(info.get("dataset_len", 0))
    mode = str(info.get("mode", "?"))

    print("\n--- Split vs evaluated coverage ---")
    print(
        f"  Split JSON 'test' list length: {listed}  (unique filenames/ids: {unique})"
    )
    print(
        f"  Dataset __len__ (after metadata/label filter{'; parquet row range' if mode == 'parquet' else ''}): {d_len}"
    )
    if listed > unique:
        print(
            f"  Note: {listed - unique} duplicate id(s) in the test list — "
            "dataset may repeat spectra (see identical preview rows)."
        )
    if listed > d_len:
        print(
            f"  Dropped before iteration: {listed - d_len} (missing metadata, unknown label, "
            "or out-of-range parquet index — see WISeREPDataset / ParquetSpectrumDataset)."
        )
    if d_len > n_predicted:
        print(
            f"  Dropped at load/preprocess (collate_skip_none): {d_len - n_predicted} "
            "(file missing, parse error, or preprocess returned None)."
        )
    print(
        f"  Rows used for ROC / metrics: {n_predicted}  "
        f"(fraction of listed test list: {n_predicted / max(listed, 1):.4f})"
    )
    print("---\n")


def print_test_debug_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    *,
    preview_rows: int = 10,
) -> None:
    """Console-only diagnostics: counts, accuracy, per-class metrics, confusion counts, sample softmax rows."""
    n_classes = len(class_names)
    y_pred = np.argmax(y_score, axis=1)
    n = len(y_true)

    print("\n=== Test-set debug stats ===")
    print(f"n_samples (loaded batches): {n}")
    print(
        "roc_auc / ap_ovr / confusion / accuracy are computed only on these rows. "
        "Split-vs-coverage logging does not alter predictions — if numbers match an earlier "
        "run, that is expected (same model + same successfully loaded spectra)."
    )
    print(f"overall accuracy (argmax): {(y_pred == y_true).mean():.6f}")

    print("\n" + classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    # OVR AUC uses only the softmax column for class c (not argmax). It measures whether
    # true positives tend to have *higher* P(c) than true negatives — not whether P(c) wins
    # over the other four classes. Rare classes can show high AUC but low precision/F1.
    # Average precision (AP) is often more informative for heavy class imbalance.
    print(
        f"{'class':<14} {'support':>8} {'recall':>8} {'roc_auc':>9} {'ap_ovr':>9} "
        f"{'P(c|true)':>10} {'prev':>8}"
    )
    for c in range(n_classes):
        mask = y_true == c
        support = int(mask.sum())
        prev = support / max(n, 1)
        if support == 0:
            print(f"{class_names[c]:<14} {support:>8}       —         —         —          —         —")
            continue
        recall = float(((y_pred == c) & mask).sum() / support)
        y_bin = mask.astype(int)
        try:
            auc = float(roc_auc_score(y_bin, y_score[:, c]))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_bin, y_score[:, c]))
        except ValueError:
            ap = float("nan")
        mean_pt = float(y_score[mask, c].mean())
        print(
            f"{class_names[c]:<14} {support:>8} {recall:>8.4f} {auc:>9.4f} {ap:>9.4f} "
            f"{mean_pt:>10.4f} {prev:>8.4f}"
        )

    print(
        "\nNote: roc_auc (one-vs-rest, softmax P(class c)) scores how well P(c) *ranks* true "
        "class-c spectra above non-c spectra. It does *not* require P(c) to be the largest "
        "of the five logits, so it can stay high while precision (argmax = c) stays low — "
        "typical for rare classes. ap_ovr (average precision) is usually more aligned with "
        "rare-class detection quality; compare ap_ovr to prevalence (prev) as a random baseline. "
        "`prev` is class frequency among these n_samples only (not among all IDs listed in the split JSON)."
    )

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1

    short = [name[:10] + ("…" if len(name) > 10 else "") for name in class_names]
    print("\nConfusion counts (row=true, col=pred):")
    hdr = f"{'true \\ pred':<14}" + "".join(f"{s:>12}" for s in short)
    print(hdr)
    for i in range(n_classes):
        row = "".join(f"{cm[i, j]:>12d}" for j in range(n_classes))
        print(f"{class_names[i][:12]:<14}{row}")

    show = min(preview_rows, n)
    print(f"\nFirst {show} samples (dataset order): idx  true → pred   probs[class_true]  probs[class_pred]   top-2 (idx, p)")
    for idx in range(show):
        t_i, p_i = int(y_true[idx]), int(y_pred[idx])
        probs = y_score[idx]
        top2_idx = np.argsort(-probs)[:2]
        top2_str = ", ".join(f"({int(k)}, {probs[k]:.3f})" for k in top2_idx)
        print(
            f"  {idx:4d}  {class_names[t_i]:>10} → {class_names[p_i]:<10}  "
            f"P(y={t_i})={probs[t_i]:.4f}  P(y={p_i})={probs[p_i]:.4f}   {top2_str}"
        )

    print("=== end debug stats ===\n")


def plot_class_rocs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    prefix: str = "roc_test",
) -> None:
    n_classes = len(class_names)
    if y_score.shape[1] != n_classes:
        raise SystemExit(
            f"y_score has {y_score.shape[1]} columns but class_mapping lists {n_classes} classes."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0:
            print(f"Skipping ROC for '{class_names[c]}': no positive samples in split.")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(
            y_bin,
            y_score[:, c],
            ax=ax,
            name=f"{class_names[c]} vs rest",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="chance")
        ax.set_title(f"ROC — {class_names[c]} vs rest")
        ax.legend(loc="lower right")
        ax.set_aspect("equal", adjustable="box")

        fname = f"{prefix}_{_safe_filename_class(class_names[c])}.png"
        out_path = out_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def make_test_loader_from_config(
    model_dir: Path, training_config_path: Path
) -> Tuple[DataLoader | None, Dict[str, Any]]:
    config = helpers.load_json(training_config_path)
    has_redshift = config.get("has_redshift", const.HAS_REDSHIFT)
    batch_size = int(config.get("batch_size", const.BATCH_SIZE))
    device = helpers.get_device()

    parquet_path = config.get("parquet")
    if parquet_path:
        import pandas as pd

        import parquet_dataset as rpd

        splits_path = Path(config.get("splits_file", rpd.RUIYAO_TRAIN_VAL_TEST_JSON))
        splits = helpers.load_json(splits_path)
        test_ids = list(splits.get("test", []))
        if not test_ids:
            return None, {}
        print(f"ROC eval (parquet): splits={splits_path.resolve()}  test_n={len(test_ids)}")
        df = pd.read_parquet(Path(parquet_path))
        ds = rpd.ParquetSpectrumDataset(test_ids, df, has_redshift=has_redshift)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dash_retrain.collate_skip_none,
            pin_memory=(device.type == "cuda"),
        )
        info: Dict[str, Any] = {
            "mode": "parquet",
            "test_ids_listed": len(test_ids),
            "test_ids_unique": len(set(test_ids)),
            "dataset_len": len(ds),
        }
        print(f"  has_redshift={has_redshift}  dataset_len={info['dataset_len']}")
        return loader, info

    splits_path = Path(config.get("splits_file", const.SPLITS_JSON_80_10_10)).resolve()
    print(f"ROC eval: test split from {splits_path}")

    splits = helpers.load_json(splits_path)
    test_ids = list(splits.get("test", []))
    if not test_ids:
        return None, {}

    data_mode = config.get("data_mode", "ascii")
    if data_mode == "daep_matched_ascii":
        meta_csv = Path(config.get("processed_meta_csv", const.PROCESSED_META_Z))
        print(f"ROC eval: metadata from processed CSV {meta_csv}")
        metadata = helpers.load_metadata_from_processed_csv(meta_csv)
    else:
        metadata = helpers.load_metadata(const.METADATA_CSV)
    loader = helpers.make_loader(test_ids, metadata, has_redshift, device, batch_size=batch_size)
    info = {
        "mode": "ascii",
        "test_ids_listed": len(test_ids),
        "test_ids_unique": len(set(test_ids)),
        "dataset_len": len(loader.dataset),
    }
    print(
        f"  test list length={info['test_ids_listed']}  unique={info['test_ids_unique']}  "
        f"dataset_len={info['dataset_len']}  has_redshift={has_redshift}"
    )
    return loader, info


def run_for_model(model_path: Path) -> None:
    model_dir = model_path.resolve().parent
    class_mapping_path = model_dir / "class_mapping.json"
    training_config_path = model_dir / "training_config.json"

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not class_mapping_path.exists():
        raise SystemExit(f"Missing class_mapping.json next to model: {class_mapping_path}")
    if not training_config_path.exists():
        raise SystemExit(f"Missing training_config.json next to model: {training_config_path}")

    class_names = load_class_names(class_mapping_path)
    n_classes = len(class_names)
    loader, split_info = make_test_loader_from_config(model_dir, training_config_path)
    if loader is None:
        raise SystemExit("Splits file has no 'test' entries (or could not build loader).")

    device = helpers.get_device()
    model = load_model(model_path, n_classes, device)
    y_true, y_score = collect_test_predictions(model, loader, device)
    print_loader_coverage(split_info, len(y_true))
    print_test_debug_stats(y_true, y_score, class_names)
    plot_class_rocs(y_true, y_score, class_names, out_dir=model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class ROC on test split for DashCNN1D.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=f"Path to model.pth (default: {const.OUT_DIR / 'model.pth'})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help=f"If set, use {MODELS_BASE}/<run-id>/model.pth",
    )
    args = parser.parse_args()

    if args.run_id.strip():
        model_path = MODELS_BASE / args.run_id.strip() / "model.pth"
    elif args.model_path is not None:
        model_path = args.model_path.expanduser().resolve()
    else:
        model_path = (const.OUT_DIR / "model.pth").resolve()

    run_for_model(model_path)


if __name__ == "__main__":
    main()
