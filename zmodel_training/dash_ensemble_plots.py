#!/usr/bin/env python3
"""
Aggregate Dash 1D CNN ensemble runs (iter_*) and save latent_plots-style PNGs:
  - test confusion matrix (precision + recall %, mean ± std across runs)
  - train/validation loss curves (mean ± std across runs)

Uses the same test loader path as roc_curves.py / roc_ensemble_daep_comparison.py
(reads each run's training_config.json → splits_file + processed_meta_csv).

Usage:
  python zmodel_training/dash_ensemble_plots.py data/pre_trained_models/henna_matched_comparison_z
  python zmodel_training/dash_ensemble_plots.py data/pre_trained_models/henna_matched_comparison_noz
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import helpers as helpers
import roc_curves as rc
from dash_eval_confusion_matrices import confusion_matrix_counts

for _name in (
    "app.infrastructure.storage.file_spectrum_repository",
    "prod_backend.app.infrastructure.storage.file_spectrum_repository",
    "app.infrastructure.ml.data_processor",
    "prod_backend.app.infrastructure.ml.data_processor",
):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

CKPT_NAME = "model.pth"
ITER_DIR_RE = re.compile(r"^iter[_]?(\d+)$")


def discover_run_dirs(root: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = ITER_DIR_RE.match(p.name)
        if m and (p / CKPT_NAME).is_file() and (p / "model_performance.json").is_file():
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def _row_normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm)
    mask = row_sums[:, 0] > 0
    out[mask] = cm[mask] / row_sums[mask]
    return out


def _col_normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    col_sums = cm.sum(axis=0, keepdims=True)
    out = np.zeros_like(cm)
    mask = col_sums[0, :] > 0
    out[:, mask] = cm[:, mask] / col_sums[:, mask]
    return out


def _plot_normalized_cm(
    ax,
    mat_pct: np.ndarray,
    title: str,
    cbar_label: str,
    class_names: list[str],
    mat_std_pct: np.ndarray | None = None,
) -> None:
    n = len(class_names)
    im = ax.imshow(
        mat_pct,
        aspect="equal",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        origin="upper",
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            label = f"{mat_pct[i, j]:.1f}"
            if mat_std_pct is not None:
                label = f"{label}\n±{mat_std_pct[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _loss_curves_from_json(perf_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    rows = perf.get("loss_by_epoch")
    if not rows:
        raise KeyError(f"{perf_path} has no 'loss_by_epoch'")
    ep, tr, va = [], [], []
    for r in rows:
        ep.append(int(r[0]))
        tr.append(float(r[1]) if r[1] is not None else float("nan"))
        va.append(float(r[2]))
    return np.asarray(ep), np.asarray(tr), np.asarray(va)


def _stack_loss_curves(
    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_epoch = max(int(epochs[-1]) for epochs, _, _ in curves if len(epochs))
    epoch_grid = np.arange(1, max_epoch + 1)
    train = np.full((len(curves), max_epoch), np.nan, dtype=np.float64)
    val = np.full((len(curves), max_epoch), np.nan, dtype=np.float64)

    for row, (epochs, train_loss, val_loss) in enumerate(curves):
        cols = epochs.astype(int) - 1
        train[row, cols] = train_loss
        val[row, cols] = val_loss

    return (
        epoch_grid,
        np.nanmean(train, axis=0),
        np.nanstd(train, axis=0),
        np.nanmean(val, axis=0),
        np.nanstd(val, axis=0),
    )


def _title_prefix(has_redshift: bool) -> str:
    if has_redshift:
        return "Dash 1D CNN with Redshift"
    return "Dash 1D CNN without Redshift"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ensemble test CM + loss curves for Dash 1D CNN iter_* runs."
    )
    parser.add_argument(
        "comparison_root",
        type=Path,
        nargs="?",
        default=const.OUT_DIR_HENNA_MATCHED_Z,
        help="Folder containing iter_0, iter_1, … (default: henna_matched_comparison_z)",
    )
    parser.add_argument(
        "--cm-out",
        type=Path,
        default=None,
        help="Confusion matrix PNG (default: <comparison_root>/dash_cm.png)",
    )
    parser.add_argument(
        "--loss-out",
        type=Path,
        default=None,
        help="Loss curves PNG (default: <comparison_root>/dash_loss_curves.png)",
    )
    args = parser.parse_args()

    root = args.comparison_root.expanduser().resolve()
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        raise SystemExit(
            f"No runs with {CKPT_NAME} + model_performance.json under {root}"
        )

    cm_out = (args.cm_out or root / "dash_cm.png").expanduser().resolve()
    loss_out = (args.loss_out or root / "dash_loss_curves.png").expanduser().resolve()

    device = helpers.get_device()
    first_cfg = helpers.load_json(run_dirs[0] / "training_config.json")
    has_redshift = bool(first_cfg.get("has_redshift", True))
    class_names = list(first_cfg.get("class_names") or const.CLASS_NAMES)
    n_classes = len(class_names)

    cm_raw_runs: list[np.ndarray] = []
    cm_recall_runs: list[np.ndarray] = []
    cm_precision_runs: list[np.ndarray] = []
    loss_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    print(f"Aggregating {len(run_dirs)} run(s) from {root}")
    for run_dir in run_dirs:
        training_config_path = run_dir / "training_config.json"
        class_mapping_path = run_dir / "class_mapping.json"
        if not training_config_path.is_file() or not class_mapping_path.is_file():
            raise FileNotFoundError(
                f"{run_dir}: need training_config.json and class_mapping.json"
            )

        names = rc.load_class_names(class_mapping_path)
        if names != class_names:
            raise RuntimeError(f"Class order/names differ at {run_dir} vs first run")

        loader, _info = rc.make_test_loader_from_config(run_dir, training_config_path)
        if loader is None:
            raise RuntimeError(f"No test split / loader for {run_dir}")

        model = rc.load_model(run_dir / CKPT_NAME, n_classes, device)
        cm_list = confusion_matrix_counts(model, loader, device, n_classes)
        cm = np.asarray(cm_list, dtype=np.float64)
        cm_raw_runs.append(cm)
        cm_recall_runs.append(_row_normalize_cm(cm) * 100.0)
        cm_precision_runs.append(_col_normalize_cm(cm) * 100.0)
        loss_curves.append(_loss_curves_from_json(run_dir / "model_performance.json"))
        print(f"Loaded {run_dir.name}")

    cm_recall_stack = np.stack(cm_recall_runs, axis=0)
    cm_precision_stack = np.stack(cm_precision_runs, axis=0)
    cm_recall = np.mean(cm_recall_stack, axis=0)
    cm_recall_std = np.std(cm_recall_stack, axis=0)
    cm_precision = np.mean(cm_precision_stack, axis=0)
    cm_precision_std = np.std(cm_precision_stack, axis=0)
    acc_runs = [
        float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0 for cm in cm_raw_runs
    ]
    acc = float(np.mean(acc_runs))
    acc_std = float(np.std(acc_runs))

    epochs, train_loss, train_loss_std, val_loss, val_loss_std = _stack_loss_curves(
        loss_curves
    )

    loss_out.parent.mkdir(parents=True, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, train_loss, label="Train", color="C0")
    ax_loss.fill_between(
        epochs,
        train_loss - train_loss_std,
        train_loss + train_loss_std,
        color="C0",
        alpha=0.2,
        label="Train ±1 std",
    )
    ax_loss.plot(epochs, val_loss, label="Validation", color="C1")
    ax_loss.fill_between(
        epochs,
        val_loss - val_loss_std,
        val_loss + val_loss_std,
        color="C1",
        alpha=0.2,
        label="Validation ±1 std",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Train and validation loss ({len(run_dirs)} runs)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(loss_out, dpi=160, bbox_inches="tight")
    plt.close(fig_loss)

    fig_cm, (ax_prec, ax_rec) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_normalized_cm(
        ax_prec,
        cm_precision,
        title="Precision",
        cbar_label="Predicted Class %",
        class_names=class_names,
        mat_std_pct=cm_precision_std,
    )
    _plot_normalized_cm(
        ax_rec,
        cm_recall,
        title="Recall",
        cbar_label="True Class %",
        class_names=class_names,
        mat_std_pct=cm_recall_std,
    )
    fig_cm.suptitle(
        f"{_title_prefix(has_redshift)} Confusion Matrix | "
        f"accuracy {100.0 * acc:.1f} ± {100.0 * acc_std:.1f}%",
        y=1.02,
    )
    fig_cm.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig_cm.savefig(cm_out, dpi=160, bbox_inches="tight")
    plt.close(fig_cm)

    print(f"Wrote {loss_out}")
    print(f"Wrote {cm_out}")


if __name__ == "__main__":
    main()
