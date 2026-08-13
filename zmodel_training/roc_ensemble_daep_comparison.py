#!/usr/bin/env python
"""
Average multiclass OvR ROC curves across DashCNN1D runs under ``COMPARISON_ROOT``
(e.g. ``iter_0`` … ``iter_9``). Reports mean ± std macro/micro F1 and OvR AUC
across runs; plots mean TPR ± std band per class on a shared FPR grid.

Uses the same test loader and evaluation path as ``roc_curves.py`` (via shared helpers).

Usage:
  python zmodel_training/roc_ensemble_daep_comparison.py
  python zmodel_training/roc_ensemble_daep_comparison.py /path/to/daep_comparison
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import helpers as helpers
import roc_curves as rc

COMPARISON_ROOT = PROJECT_ROOT / "data" / "pre_trained_models" / "daep_comparison_z"
CKPT_NAME = "model.pth"
FPR_GRID = np.linspace(0.0, 1.0, 101)
IS_MACRO = False


def discover_run_dirs(root: Path) -> List[Path]:
    """Prefer ``iter_*`` training-seed runs; else ``split_*`` data-split runs."""
    iter_pat = re.compile(r"^iter[_]?(\d+)$")
    split_pat = re.compile(r"^split_(\d+)$")
    if not root.is_dir():
        raise FileNotFoundError(root)

    def _collect(pat: re.Pattern[str]) -> List[tuple[int, Path]]:
        found: List[tuple[int, Path]] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            m = pat.fullmatch(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda x: x[0])
        return found

    found = _collect(iter_pat)
    if not found:
        found = _collect(split_pat)
    return [p for _, p in found]


def std_ddof1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def collect_run_predictions(
    run_dirs: List[Path],
    device: torch.device,
) -> tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Returns per-run y_true, per-run softmax scores, and class names.

    Each run is evaluated on its own test split from training_config.json
    (required for data-split ensembles where test membership differs).
    """
    all_y_true: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    class_names: List[str] | None = None

    for run_dir in run_dirs:
        model_path = run_dir / CKPT_NAME
        training_config_path = run_dir / "training_config.json"
        class_mapping_path = run_dir / "class_mapping.json"

        if not model_path.is_file():
            continue
        if not training_config_path.is_file() or not class_mapping_path.is_file():
            raise FileNotFoundError(
                f"{run_dir}: need {CKPT_NAME}, training_config.json, and class_mapping.json"
            )

        names = rc.load_class_names(class_mapping_path)
        n_classes = len(names)
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise RuntimeError(f"Class order/names differ at {run_dir} vs first run")

        loader, _split_info = rc.make_test_loader_from_config(run_dir, training_config_path)
        if loader is None:
            raise RuntimeError(f"No test split / loader for {run_dir}")

        model = rc.load_model(model_path, n_classes, device)
        y_true, y_score = rc.collect_test_predictions(model, loader, device)
        all_y_true.append(y_true)
        all_scores.append(y_score)
        print(f"  {run_dir.name}: test_n={len(y_true)}")

    if not all_y_true or not all_scores or class_names is None:
        raise RuntimeError(
            f"No valid runs with {CKPT_NAME} under given directories "
            f"(need training_config.json + class_mapping.json)."
        )

    return all_y_true, all_scores, class_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble OvR ROC + macro/micro F1 / OvR AUC over daep_comparison iter_* runs."
    )
    parser.add_argument(
        "comparison_root",
        type=Path,
        nargs="?",
        default=COMPARISON_ROOT,
        help=f"Folder containing iter_0, iter_1, … (default: {COMPARISON_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <comparison_root>/ensemble_roc_ovr_<average>.png)",
    )
    args = parser.parse_args()

    root = args.comparison_root.expanduser().resolve()
    run_dirs = [d for d in discover_run_dirs(root) if (d / CKPT_NAME).is_file()]
    if not run_dirs:
        raise FileNotFoundError(f"No subdirs with {CKPT_NAME} under {root}")

    device = helpers.get_device()
    all_y_true, all_scores, class_names = collect_run_predictions(run_dirs, device)
    n_runs = len(all_scores)
    n_classes = len(class_names)
    average_kind = "macro" if IS_MACRO else "micro"
    data_split_ensemble = all(re.fullmatch(r"split_\d+", d.name) for d in run_dirs)

    avg_f1_runs = [
        f1_score(y_true, np.argmax(s, axis=1), average=average_kind, zero_division=0)
        for y_true, s in zip(all_y_true, all_scores)
    ]
    avg_auc_runs = [
        roc_auc_score(
            y_true,
            s,
            multi_class="ovr",
            average=average_kind,
            labels=np.arange(n_classes),
        )
        for y_true, s in zip(all_y_true, all_scores)
    ]

    f1_m, f1_s = float(np.mean(avg_f1_runs)), std_ddof1(np.array(avg_f1_runs))
    auc_m, auc_s = float(np.mean(avg_auc_runs)), std_ddof1(np.array(avg_auc_runs))

    print(f"Runs used (n={n_runs}):")
    for d in run_dirs:
        print(f"  {d.name}")
    print(f"{average_kind} F1      = {f1_m:.4f} ± {f1_s:.4f}")
    print(f"{average_kind} OvR AUC = {auc_m:.4f} ± {auc_s:.4f}")

    fig, ax = plt.subplots(figsize=(9, 8))

    for i, name in enumerate(class_names):
        tpr_rows: List[np.ndarray] = []
        auc_i: List[float] = []
        for y_true, y_score in zip(all_y_true, all_scores):
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            if y_bin.ndim == 1:
                y_bin = np.column_stack([1 - y_bin, y_bin])
            pos = int(y_bin[:, i].sum())
            if pos == 0 or pos == len(y_true):
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            tpr_rows.append(np.interp(FPR_GRID, fpr, tpr))
            auc_i.append(auc(fpr, tpr))

        if not tpr_rows:
            print(f"Skipping ROC curve for '{name}': no run had both classes")
            continue

        tpr_mean = np.mean(tpr_rows, axis=0)
        tpr_std = np.std(tpr_rows, axis=0, ddof=1) if len(tpr_rows) > 1 else np.zeros_like(tpr_mean)
        auc_mean = float(np.mean(auc_i))
        auc_std = std_ddof1(np.array(auc_i))

        (line,) = ax.plot(
            FPR_GRID,
            tpr_mean,
            lw=2,
            label=f"{name} (AUC={auc_mean:.3f}±{auc_std:.3f})",
        )
        ax.fill_between(
            FPR_GRID,
            np.clip(tpr_mean - tpr_std, 0, 1),
            np.clip(tpr_mean + tpr_std, 0, 1),
            color=line.get_color(),
            alpha=0.2,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ensemble_tag = "data-split" if data_split_ensemble else "training-seed"
    ax.set_title(
        f"ROC  Dash1D CNN with redshift ({ensemble_tag} ensemble)\n"
        f"{average_kind} F1 = {f1_m:.3f} ± {f1_s:.3f}  |  "
        f"{average_kind} avg AUC = {auc_m:.3f} ± {auc_s:.3f}"
    )
    fig.tight_layout()

    out = (
        args.output
        if args.output is not None
        else root / f"ensemble_roc_ovr_{average_kind}.png"
    )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
