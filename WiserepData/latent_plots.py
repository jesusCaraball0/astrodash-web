"""
Aggregate DAEP diffusion (latent MLP) ensemble runs (iter_*) and save
dash_ensemble_plots-style PNGs:
  - test confusion matrix (precision + recall %, mean ± std across runs)
  - train/validation loss curves (mean ± std across runs)

Does not overwrite existing plots unless --cm-out / --loss-out point at them.

Usage:
  python WiserepData/latent_plots.py WiserepData/Test/daep_comparison
  python WiserepData/latent_plots.py WiserepData/Test/daep_comparison_noz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WISEREP_DIR = pathlib.Path(__file__).resolve().parent
if str(WISEREP_DIR) not in sys.path:
    sys.path.insert(0, str(WISEREP_DIR))

from train_latent import (
    BATCH_SIZE,
    DEVICE,
    IAU_COLUMN,
    LABEL_COLUMN,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    LatentClassifier,
    LatentDataset,
    collate_latent,
    load_assignment_indices_from_dir,
    normalize_latent_meta,
)
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from TwinsClassifier_Wiserep import (
    CLASS_NAMES,
    NUM_CLASSES,
    evaluate,
    filter_indices_mapped,
)
from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import set_seeds

# Prefer underscore form (iter_0) from train_latent; also support split_{N} data-split runs.
ITER_DIR_RE = re.compile(r"^iter_(\d+)$")
ITER_DIR_RE_LEGACY = re.compile(r"^iter(\d+)$")
SPLIT_DIR_RE = re.compile(r"^split_(\d+)$")


def _run_sort_key(path: pathlib.Path) -> tuple[int, int, str]:
    for pattern, group in (
        (ITER_DIR_RE, 0),
        (SPLIT_DIR_RE, 1),
        (ITER_DIR_RE_LEGACY, 2),
    ):
        match = pattern.fullmatch(path.name)
        if match:
            return (group, int(match.group(1)), path.name)
    return (3, sys.maxsize, path.name)


def discover_run_dirs(
    root: pathlib.Path,
    *,
    run_style: str = "auto",
) -> list[pathlib.Path]:
    """Discover ensemble run dirs.

    ``run_style``:
      - ``auto``: prefer ``iter_*``, else ``split_*``, else legacy ``iterN``
      - ``underscore``: only ``iter_*``
      - ``legacy``: only ``iterN`` (no underscore), excluding ``iter_N``
      - ``split``: only ``split_*``
    """

    def _collect(pattern: re.Pattern[str] | None = None) -> list[pathlib.Path]:
        out: list[pathlib.Path] = []
        for p in root.iterdir():
            if not (
                p.is_dir()
                and (p / "classifier_best.pt").is_file()
                and (p / "model_performance.json").is_file()
            ):
                continue
            if pattern is None or pattern.fullmatch(p.name):
                out.append(p)
        return out

    if run_style == "underscore":
        runs = _collect(ITER_DIR_RE)
    elif run_style == "legacy":
        runs = _collect(ITER_DIR_RE_LEGACY)
    elif run_style == "split":
        runs = _collect(SPLIT_DIR_RE)
    elif run_style == "auto":
        runs = _collect(ITER_DIR_RE)
        if not runs:
            runs = _collect(SPLIT_DIR_RE)
        if not runs:
            runs = _collect(ITER_DIR_RE_LEGACY)
    else:
        raise ValueError(f"Unknown run_style={run_style!r}")

    if not runs:
        raise FileNotFoundError(
            f"No runs with classifier_best.pt and model_performance.json under {root.resolve()} "
            f"(run_style={run_style})"
        )
    return sorted(runs, key=_run_sort_key)


def _should_use_assignment(cfg: dict) -> bool:
    """Data-split runs record split_source / run_id=split_*; training-seed iters use IAU seed 0."""
    split_source = cfg.get("split_source")
    if isinstance(split_source, str) and split_source.startswith("assignment:"):
        return True
    run_id = str(cfg.get("run_id", ""))
    return bool(re.fullmatch(r"split_\d+", run_id))


def _latent_dir_for_assignment(latent_npz: str | pathlib.Path) -> pathlib.Path:
    """Parent of the configured latent path, without following symlinks.

    Staging dirs often symlink ``latent_raw_z_best.npz`` at a real dump (e.g. Output6)
    while keeping ``split_assignment*.json`` next to the symlink. ``Path.resolve()``
    would jump to the dump and miss the assignment.
    """
    return pathlib.Path(latent_npz).expanduser().parent


def _test_indices_for_cfg(cfg: dict, meta: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Use latent-dir assignment for data-split runs; else IAU seed 0 (training-seed ensembles)."""
    if _should_use_assignment(cfg):
        latent_npz = cfg.get("latent_npz")
        if not latent_npz:
            raise KeyError("data-split cfg missing latent_npz")
        latent_dir = _latent_dir_for_assignment(latent_npz)
        _, _, te_idx, src = load_assignment_indices_from_dir(latent_dir, meta)
        te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)
        return te_idx, src

    _, _, te_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)
    return te_idx, f"iau_split_seed={IAU_SPLIT_SEED}"


def _latent_cfg(ckpt: dict, cfg_json: pathlib.Path) -> dict:
    # Prefer on-disk cfg_used.json (always rewritten by the latest train run).
    if cfg_json.is_file():
        return json.loads(cfg_json.read_text(encoding="utf-8"))
    if isinstance(ckpt.get("cfg"), dict):
        return dict(ckpt["cfg"])
    raise FileNotFoundError(f"Checkpoint has no 'cfg' and missing: {cfg_json}")


def _resolve_path(raw: str | pathlib.Path, fallback: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(raw)
    name = p.name
    candidates = [
        p,
        WISEREP_DIR / name,
        WISEREP_DIR / "Test" / "data_z" / name,
        WISEREP_DIR / "Test" / "data_no_z" / name,
        fallback,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Missing file {raw!r} (also tried {[str(c) for c in candidates[1:]]})")


def _load_z_and_meta(cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    if not cfg.get("latent_npz") or not cfg.get("meta_csv"):
        raise KeyError("cfg must include 'latent_npz' and 'meta_csv'")
    latent_npz = _resolve_path(cfg["latent_npz"], pathlib.Path(cfg["latent_npz"]))
    meta_csv = _resolve_path(cfg["meta_csv"], pathlib.Path(cfg["meta_csv"]))
    z = np.load(latent_npz)["z"].astype(np.float32, copy=False)
    meta = normalize_latent_meta(pd.read_csv(meta_csv, low_memory=False))
    if len(meta) != z.shape[0]:
        raise ValueError(f"meta rows ({len(meta)}) != z rows ({z.shape[0]})")
    return z, meta


def _has_redshift_from_cfg(cfg: dict) -> bool:
    latent = str(cfg.get("latent_npz", "")).lower()
    meta = str(cfg.get("meta_csv", "")).lower()
    joined = f"{latent} {meta}"
    if "nodered" in joined or "no_z" in joined or "noz" in joined:
        return False
    return True


def _title_prefix(has_redshift: bool, *, data_split_ensemble: bool) -> str:
    z_tag = "with Redshift" if has_redshift else "without Redshift"
    split_tag = "data-split ensemble" if data_split_ensemble else "training-seed ensemble"
    return f"DAEP (diffusion) {z_tag} ({split_tag})"


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


def _plot_normalized_cm(
    ax,
    mat_pct: np.ndarray,
    title: str,
    cbar_label: str,
    mat_std_pct: np.ndarray | None = None,
) -> None:
    """Single heatmap: percentages 0–100, TL→BR diagonal = correct when same class on both axes."""
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


def _build_model(cfg: dict, embed_dim: int, device: torch.device) -> LatentClassifier:
    return LatentClassifier(
        embed_dim=embed_dim,
        n_cls=NUM_CLASSES,
        head_hidden=int(cfg.get("ff_dim", 512)),
        head_dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)


def _refuse_overwrite(path: pathlib.Path) -> None:
    if path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing plot: {path}\n"
            f"Pass a different --cm-out / --loss-out path, or move/rename the old file first."
        )


def main() -> None:
    default_root = WISEREP_DIR / "Test" / "daep_comparison"
    parser = argparse.ArgumentParser(
        description="Plot ensemble test CM + loss curves for DAEP latent iter_* runs."
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
        help="Confusion matrix PNG (default: <root>/plots/<tag>_test_cm.png)",
    )
    parser.add_argument(
        "--loss-out",
        type=pathlib.Path,
        default=None,
        help="Loss curves PNG (default: <root>/plots/<tag>_loss_curves.png)",
    )
    parser.add_argument(
        "--run-style",
        choices=("auto", "underscore", "legacy", "split"),
        default="auto",
        help="Which run directories to aggregate (default: auto).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap after discovery/sort (e.g. 10 for legacy iter0..iter9).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Permit writing over an existing PNG (off by default).",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_root.expanduser().resolve()
    run_dirs = discover_run_dirs(comparison_dir, run_style=args.run_style)
    if args.max_runs is not None:
        run_dirs = run_dirs[: max(0, args.max_runs)]
        if not run_dirs:
            raise FileNotFoundError(f"No runs left after --max-runs={args.max_runs}")
    data_split_ensemble = all(SPLIT_DIR_RE.fullmatch(p.name) for p in run_dirs)
    if data_split_ensemble:
        tag = "data_split_ensemble"
    elif args.run_style == "legacy" or all(ITER_DIR_RE_LEGACY.fullmatch(p.name) for p in run_dirs):
        tag = "legacy_iter_training_seed_ensemble"
    else:
        tag = "training_seed_ensemble"

    plots_dir = comparison_dir / "plots"
    cm_out = (args.cm_out or plots_dir / f"{tag}_test_cm.png").expanduser().resolve()
    loss_out = (args.loss_out or plots_dir / f"{tag}_loss_curves.png").expanduser().resolve()
    if not args.allow_overwrite:
        _refuse_overwrite(cm_out)
        _refuse_overwrite(loss_out)

    set_seeds(0)
    device = device_from_str(DEVICE)
    first_ckpt = torch.load(
        run_dirs[0] / "classifier_best.pt", map_location=device, weights_only=False
    )
    first_cfg = _latent_cfg(first_ckpt, run_dirs[0] / "cfg_used.json")
    has_redshift = _has_redshift_from_cfg(first_cfg)

    loss_fn = nn.CrossEntropyLoss()
    cm_raw_runs = []
    cm_recall_runs = []
    cm_precision_runs = []
    loss_curves = []

    print(f"Aggregating {len(run_dirs)} run(s) from {comparison_dir}")
    for run_dir in run_dirs:
        ckpt_path = run_dir / "classifier_best.pt"
        perf_path = run_dir / "model_performance.json"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = _latent_cfg(ckpt, run_dir / "cfg_used.json")
        z, meta = _load_z_and_meta(cfg)
        te_idx, split_src = _test_indices_for_cfg(cfg, meta)
        embed_dim = int(np.prod(z.shape[1:]))
        te_load = DataLoader(
            LatentDataset(meta, z, te_idx, LABEL_COLUMN),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_latent,
            num_workers=0,
        )

        model = _build_model(cfg, embed_dim, device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.head.load_state_dict(ckpt["head_state_dict"], strict=True)

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
        print(f"Loaded {run_dir.name}  test_n={len(te_idx)}  split={split_src}")

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
        f"{_title_prefix(has_redshift, data_split_ensemble=data_split_ensemble)} "
        f"Confusion Matrix | accuracy {100.0 * acc:.1f} ± {100.0 * acc_std:.1f}%",
        y=1.02,
    )
    fig_cm.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig_cm.savefig(cm_out, dpi=160, bbox_inches="tight")
    plt.close(fig_cm)

    print(f"Wrote {loss_out}")
    print(f"Wrote {cm_out}")


if __name__ == "__main__":
    main()
