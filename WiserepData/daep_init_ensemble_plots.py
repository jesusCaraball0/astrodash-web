#!/usr/bin/env python3
"""
Aggregate DAEP no-diffusion (init) ensemble runs (iter_*) and save
dash_ensemble_plots-style PNGs:
  - test confusion matrix (precision + recall %, mean ± std across runs)
  - train/validation loss curves (mean ± std across runs)

Usage:
  python WiserepData/daep_init_ensemble_plots.py WiserepData/Test/daep_comparison_init
  python WiserepData/daep_init_ensemble_plots.py WiserepData/Test/daep_comparison_init_noz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WISEREP_DIR = pathlib.Path(__file__).resolve().parent
if str(WISEREP_DIR) not in sys.path:
    sys.path.insert(0, str(WISEREP_DIR))

from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from TwinsClassifier_Wiserep import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    IAU_COLUMN,
    IAU_TEST_FRAC,
    IAU_TRAIN_FRAC,
    IAU_VAL_FRAC,
    LABEL_COLUMN,
    NUM_CLASSES,
    REPO,
    LabeledSpectra,
    MeanPoolClassifier,
    collate_labeled,
    data_dir_for,
    evaluate,
    filter_indices_mapped,
    load_preprocessed_for_classifier,
)
from TwinsModel_Wiserep import build_daep, default_cfg, device_from_str, repo_import_setup
from TwinsTrain_Wiserep import set_seeds

ITER_DIR_RE = re.compile(r"^iter_(\d+)$")
CKPT_NAME = "classifier_best.pt"


def _iter_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = ITER_DIR_RE.fullmatch(path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def discover_run_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / CKPT_NAME).is_file()
        and (p / "model_performance.json").is_file()
        and ITER_DIR_RE.fullmatch(p.name)
    ]
    if not runs:
        raise FileNotFoundError(
            f"No runs with {CKPT_NAME} and model_performance.json under {root.resolve()}"
        )
    return sorted(runs, key=_iter_sort_key)


def _load_cfg(run_dir: pathlib.Path, ckpt: dict) -> dict:
    cfg = default_cfg()
    cfg_json = run_dir / "cfg_used.json"
    if cfg_json.is_file():
        cfg.update(json.loads(cfg_json.read_text(encoding="utf-8")))
    if isinstance(ckpt.get("cfg"), dict):
        cfg.update(ckpt["cfg"])
    return cfg


def _has_redshift_from_root(root: pathlib.Path) -> bool:
    name = root.name.lower()
    return not ("noz" in name or "no_z" in name or "nodered" in name)


def _title_prefix(has_redshift: bool) -> str:
    if has_redshift:
        return "DAEP (no diffusion) with Redshift"
    return "DAEP (no diffusion) without Redshift"


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
    mat_std_pct: np.ndarray | None = None,
) -> None:
    im = ax.imshow(
        mat_pct,
        aspect="equal",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        origin="upper",
    )
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    n = mat_pct.shape[0]
    for i in range(n):
        for j in range(n):
            label = f"{mat_pct[i, j]:.1f}"
            if mat_std_pct is not None:
                label = f"{label}\n±{mat_std_pct[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _loss_curves_from_json(perf_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _build_model(cfg: dict, device: torch.device) -> MeanPoolClassifier:
    repo_import_setup(str(REPO.resolve()))
    return MeanPoolClassifier(
        build_daep(cfg).to(device).encoder,
        int(cfg["bottleneck_dim"]) * int(cfg["bottleneck_length"]),
        NUM_CLASSES,
        head_hidden=int(cfg["ff_dim"]),
        head_dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)


def main() -> None:
    default_root = WISEREP_DIR / "Test" / "daep_comparison_init"
    parser = argparse.ArgumentParser(
        description="Plot ensemble test CM + loss curves for DAEP no-diffusion iter_* runs."
    )
    parser.add_argument(
        "comparison_root",
        type=pathlib.Path,
        nargs="?",
        default=default_root,
        help="Folder containing iter_0, iter_1, …",
    )
    parser.add_argument(
        "--cm-out",
        type=pathlib.Path,
        default=None,
        help="Confusion matrix PNG (default: <root>/daep_cm.png)",
    )
    parser.add_argument(
        "--loss-out",
        type=pathlib.Path,
        default=None,
        help="Loss curves PNG (default: <root>/daep_loss_curves.png)",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_root.expanduser().resolve()
    run_dirs = discover_run_dirs(comparison_dir)
    cm_out = (args.cm_out or comparison_dir / "daep_cm.png").expanduser().resolve()
    loss_out = (args.loss_out or comparison_dir / "daep_loss_curves.png").expanduser().resolve()

    set_seeds(0)
    device = device_from_str(DEVICE)
    has_redshift = _has_redshift_from_root(comparison_dir)
    data_dir = data_dir_for(has_redshift).resolve()

    meta, wave, flux, mask = load_preprocessed_for_classifier(data_dir)
    _, _, te_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        IAU_TRAIN_FRAC,
        IAU_VAL_FRAC,
        IAU_TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)
    te_load = DataLoader(
        LabeledSpectra(meta, wave, flux, mask, te_idx, LABEL_COLUMN),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_labeled,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()
    cm_raw_runs: list[np.ndarray] = []
    cm_recall_runs: list[np.ndarray] = []
    cm_precision_runs: list[np.ndarray] = []
    loss_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    print(f"Aggregating {len(run_dirs)} run(s) from {comparison_dir}")
    print(f"Data dir: {data_dir}")
    for run_dir in run_dirs:
        ckpt_path = run_dir / CKPT_NAME
        perf_path = run_dir / "model_performance.json"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = _load_cfg(run_dir, ckpt)

        model = _build_model(cfg, device)
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.head.load_state_dict(ckpt["head_state_dict"])

        _, _, cm_list = evaluate(
            model,
            te_load,
            loss_fn,
            device,
            metrics_title=f"Test metrics ({run_dir.name})",
            print_summary=False,
        )
        cm = np.asarray(cm_list, dtype=np.float64)
        cm_raw_runs.append(cm)
        cm_recall_runs.append(_row_normalize_cm(cm) * 100.0)
        cm_precision_runs.append(_col_normalize_cm(cm) * 100.0)
        loss_curves.append(_loss_curves_from_json(perf_path))
        print(f"Loaded {run_dir.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cm_recall_stack = np.stack(cm_recall_runs, axis=0)
    cm_precision_stack = np.stack(cm_precision_runs, axis=0)
    cm_recall = np.mean(cm_recall_stack, axis=0)
    cm_recall_std = np.std(cm_recall_stack, axis=0)
    cm_precision = np.mean(cm_precision_stack, axis=0)
    cm_precision_std = np.std(cm_precision_stack, axis=0)
    acc_runs = [float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0 for cm in cm_raw_runs]
    acc = float(np.mean(acc_runs))
    acc_std = float(np.std(acc_runs))

    epochs, train_loss, train_loss_std, val_loss, val_loss_std = _stack_loss_curves(loss_curves)

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
        mat_std_pct=cm_precision_std,
    )
    _plot_normalized_cm(
        ax_rec,
        cm_recall,
        title="Recall",
        cbar_label="True Class %",
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
