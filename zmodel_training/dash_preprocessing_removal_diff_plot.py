#!/usr/bin/env python
"""
Single figure: one panel per variant run vs full-pipeline baseline on the **test** split.

**Preferred:** per-run `model_performance.json` → `test_metrics` (written by
`dash_save_variant_test_metrics.py`, which applies the matching preprocessing variant for each run).

**Fallback:** live model eval with the **current** prod preprocessor (only if `test_metrics` is missing).

Fixed mapping (baseline + variants, 1D CNN, Ruiyao split):
  04_11_26_redshift — full preprocessing (baseline)
  04_13_26_redshift — no continuum removal
  04_14_26_redshift — no med filtering
  04_15_26_redshift — no apodize
  04_16_26_redshift — no initial normalization
  04_17_26_redshift — no deredshifting wave
  04_18_26_redshift — no norm anywhere

Δ = variant - baseline (per-class recall on test).

Output:
  data/pre_trained_models/dash_wiserep/models/preprocessing_removal_difference_vs_full.png

Usage:
  python zmodel_training/dash_save_variant_test_metrics.py   # refresh test_metrics first
  python zmodel_training/dash_preprocessing_removal_diff_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import constants as const
import dash_retrain
import helpers as helpers
from dash_eval_confusion_matrices import (
    MODELS_BASE,
    confusion_matrix_counts,
    load_class_names,
    load_model,
)

for _name in (
    "app.infrastructure.storage.file_spectrum_repository",
    "prod_backend.app.infrastructure.storage.file_spectrum_repository",
    "app.infrastructure.ml.data_processor",
    "prod_backend.app.infrastructure.ml.data_processor",
):
    __import__("logging").getLogger(_name).setLevel(__import__("logging").CRITICAL)

OUT_PNG = MODELS_BASE / "preprocessing_removal_difference_vs_full.png"

_BASELINE_ID = "04_11_26_redshift"
_VARIANTS = [
    ("04_13_26_redshift", "no continuum removal"),
    ("04_14_26_redshift", "no med filtering"),
    ("04_15_26_redshift", "no apodize"),
    ("04_16_26_redshift", "no initial normalization"),
    ("04_17_26_redshift", "no deredshifting wave"),
    ("04_18_26_redshift", "no norm anywhere"),
]


def _load_test_metrics_from_json(run_id: str) -> Optional[Tuple[List[str], Dict[str, float], float]]:
    path = MODELS_BASE / run_id / "model_performance.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tm = data.get("test_metrics")
    if not tm or tm.get("split") != "test":
        return None
    labels = list(tm["confusion_matrix_labels"])
    pc = {str(c): float(tm["per_class"][c]["accuracy_pct"]) for c in labels}
    overall = float(tm["cumulative"]["accuracy_pct"])
    return labels, pc, overall


def _recall_pct_from_cm(cm: List[List[int]], class_names: List[str]) -> Tuple[Dict[str, float], float]:
    n = len(class_names)
    per_class: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        row = cm[i]
        s = sum(row)
        per_class[name] = 100.0 * row[i] / s if s > 0 else 0.0
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    overall = 100.0 * correct / total if total > 0 else 0.0
    return per_class, overall


def _test_cm_for_run(run_id: str) -> Tuple[List[str], List[List[int]]]:
    out_dir = MODELS_BASE / run_id
    model_path = out_dir / "model.pth"
    class_mapping_path = out_dir / "class_mapping.json"
    training_config_path = out_dir / "training_config.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path}")

    config = helpers.load_json(training_config_path)
    splits_path = Path(config.get("splits_file", const.SPLITS_JSON_80_10_10))
    splits = helpers.load_json(splits_path)
    test_ids = list(splits.get("test", []))
    if not test_ids:
        raise ValueError(f"No 'test' split in {splits_path} for run {run_id}")

    has_redshift = config.get("has_redshift", True)
    device = helpers.get_device()
    class_names = load_class_names(class_mapping_path)
    n_classes = len(class_names)
    model = load_model(model_path, n_classes, device)
    batch_size = int(config.get("batch_size", 64))

    parquet_path = config.get("parquet")
    if parquet_path:
        import pandas as pd

        import parquet_dataset as rpd

        df = pd.read_parquet(Path(parquet_path))
        loader = DataLoader(
            rpd.ParquetSpectrumDataset(test_ids, df, has_redshift=has_redshift),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dash_retrain.collate_skip_none,
            pin_memory=(device.type == "cuda"),
        )
    else:
        metadata = helpers.load_metadata(const.METADATA_CSV)
        loader = helpers.make_loader(
            test_ids, metadata, has_redshift, device, batch_size=batch_size
        )

    cm = confusion_matrix_counts(model, loader, device, n_classes)
    return class_names, cm


def main() -> None:
    all_ids = [_BASELINE_ID] + [r for r, _ in _VARIANTS]
    use_saved = all(_load_test_metrics_from_json(r) is not None for r in all_ids)

    if use_saved:
        print(
            "Using model_performance.json['test_metrics'] (from dash_save_variant_test_metrics.py)."
        )
        base_t = _load_test_metrics_from_json(_BASELINE_ID)
        assert base_t is not None
        base_names, base_pc, base_overall = base_t
    else:
        print(
            "Warning: missing test_metrics in some runs; using live eval with prod preprocessor. "
            "Run: python zmodel_training/dash_save_variant_test_metrics.py"
        )
        base_names, base_cm = _test_cm_for_run(_BASELINE_ID)
        base_pc, base_overall = _recall_pct_from_cm(base_cm, base_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Performance Change vs. Full Preprocessing", fontsize=14, fontweight="bold")

    for ax, (run_id, panel_title) in zip(np.ravel(axes), _VARIANTS):
        if use_saved:
            loaded = _load_test_metrics_from_json(run_id)
            assert loaded is not None
            names, var_pc, var_overall = loaded
        else:
            names, cm = _test_cm_for_run(run_id)
            var_pc, var_overall = _recall_pct_from_cm(cm, names)

        if names != base_names:
            raise ValueError(f"Class order/names differ: baseline vs {run_id}")

        d_acc = var_overall - base_overall
        d_class = [float(var_pc[c]) - float(base_pc[c]) for c in base_names]

        x = np.arange(len(base_names))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in d_class]
        ax.bar(x, d_class, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(base_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy Change (pp)")
        ax.set_title(panel_title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        src = "saved test_metrics" if use_saved else "live eval (prod preprocessor)"
        ax.text(
            0.02,
            0.98,
            f"Δ overall acc (test): {d_acc:+.2f} pp",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )

    fig.text(
        0.5,
        0.02,
        f"Baseline test set: overall accuracy = {base_overall:.2f}%",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.subplots_adjust(bottom=0.08, top=0.92)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e
