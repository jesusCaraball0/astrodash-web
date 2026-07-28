"""
Load a trained DAEP classifier checkpoint, evaluate on the IAU test split,
and save two PNGs: test confusion matrix (precision + recall heatmaps, %) and
train/validation loss vs epoch (from model_performance.json).

``DATA_DIR`` / ``OUT_DIR`` for artifacts come from ``TwinsClassifier_Wiserep`` constants.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iau_train_val_test_split import IAU_SPLIT_SEED
from TwinsClassifier_Wiserep import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    DEVICE,
    IAU_COLUMN,
    IAU_TEST_FRAC,
    IAU_TRAIN_FRAC,
    IAU_VAL_FRAC,
    LABEL_COLUMN,
    NUM_CLASSES,
    REPO,
    SEED,
    LabeledSpectra,
    MeanPoolClassifier,
    collate_labeled,
    evaluate,
    filter_indices_mapped,
    load_preprocessed_for_classifier,
    split_row_indices_by_iau_train_val_test,
)
from TwinsModel_Wiserep import build_daep, default_cfg, device_from_str, repo_import_setup
from TwinsTrain_Wiserep import set_seeds

OUT_DIR = DATA_DIR / "ClassifierOutput"
CKPT_PATH = OUT_DIR / "classifier_best.pt"
PERF_PATH = OUT_DIR / "model_performance.json"
FIGURE_LOSS = OUT_DIR / "daep_loss_curves.png"
FIGURE_CONFUSION = OUT_DIR / "daep_cm.png"
DAEP_CFG_JSON = OUT_DIR / "cfg_used.json"


def _load_daep_cfg() -> dict:
    cfg = default_cfg()
    cfg.update(json.loads(DAEP_CFG_JSON.read_text(encoding="utf-8")))
    return cfg


def _row_normalize_cm(cm: np.ndarray) -> np.ndarray:
    """Recall view: each row (true class) sums to 1; empty rows stay zero."""
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm)
    mask = row_sums[:, 0] > 0
    out[mask] = cm[mask] / row_sums[mask]
    return out


def _col_normalize_cm(cm: np.ndarray) -> np.ndarray:
    """Precision view: each column (predicted class) sums to 1; empty cols stay zero."""
    cm = np.asarray(cm, dtype=np.float64)
    col_sums = cm.sum(axis=0, keepdims=True)
    out = np.zeros_like(cm)
    mask = col_sums[0, :] > 0
    out[:, mask] = cm[:, mask] / col_sums[:, mask]
    return out


def _plot_normalized_cm(ax, mat_pct: np.ndarray, title: str, cbar_label: str) -> None:
    """Single heatmap: percentages."""
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
            ax.text(j, i, f"{mat_pct[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9)
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


def main() -> None:
    data_dir = DATA_DIR.resolve()
    ckpt_path = CKPT_PATH.resolve()
    perf_path = PERF_PATH.resolve()
    loss_out = FIGURE_LOSS.resolve()
    cm_out = FIGURE_CONFUSION.resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not perf_path.is_file():
        raise FileNotFoundError(f"Missing metrics JSON: {perf_path}")
    if not DAEP_CFG_JSON.is_file():
        raise FileNotFoundError(f"Missing DAEP cfg: {DAEP_CFG_JSON.resolve()}")

    repo_import_setup(str(REPO.resolve()))
    set_seeds(SEED)
    device = device_from_str(DEVICE)
    cfg = _load_daep_cfg()

    meta, wave, flux, mask = load_preprocessed_for_classifier(data_dir)
    tr_idx, va_idx, te_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        IAU_TRAIN_FRAC,
        IAU_VAL_FRAC,
        IAU_TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)

    test_ds = LabeledSpectra(meta, wave, flux, mask, te_idx, LABEL_COLUMN)
    te_load = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_labeled,
        num_workers=0,
    )

    model = MeanPoolClassifier(
        build_daep(cfg).to(device).encoder,
        int(cfg["bottleneck_dim"]),
        NUM_CLASSES,
        head_hidden=int(cfg["ff_dim"]),
        head_dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    _, _, cm_list = evaluate(
        model,
        te_load,
        loss_fn,
        device,
        metrics_title="Test metrics (for confusion matrix)",
        print_summary=False,
    )
    cm = np.asarray(cm_list, dtype=np.float64)
    cm_recall = _row_normalize_cm(cm) * 100.0
    cm_precision = _col_normalize_cm(cm) * 100.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0

    epochs, train_loss, val_loss = _loss_curves_from_json(perf_path)

    loss_out.parent.mkdir(parents=True, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, train_loss, label="Train", color="C0")
    ax_loss.plot(epochs, val_loss, label="Validation", color="C1")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train and validation loss")
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
    )
    _plot_normalized_cm(
        ax_rec,
        cm_recall,
        title="Recall",
        cbar_label="True Class %",
    )
    fig_cm.suptitle(
        f"DAEP Classifier Confusion Matrix — accuracy {100.0 * acc:.1f}%",
        y=1.02,
    )
    fig_cm.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig_cm.savefig(cm_out, dpi=160, bbox_inches="tight")
    plt.close(fig_cm)

    print(f"Wrote {loss_out}")
    print(f"Wrote {cm_out}")


if __name__ == "__main__":
    main()
