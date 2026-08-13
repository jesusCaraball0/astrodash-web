"""
Average multiclass avg ROC curves

Usage:
  python roc_ensemble_wiserep.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

WISEREP_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(WISEREP_DIR))

from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from train_latent import (
    LatentClassifier,
    LatentDataset,
    collate_latent,
    load_assignment_indices_from_dir,
    normalize_latent_meta,
)
from TwinsClassifier_Wiserep import (
    BATCH_SIZE,
    CLASS_NAMES,
    data_dir_for,
    DEVICE,
    IAU_COLUMN,
    IAU_TEST_FRAC,
    IAU_TRAIN_FRAC,
    IAU_VAL_FRAC,
    LABEL_COLUMN,
    LabeledSpectra,
    MeanPoolClassifier,
    NUM_CLASSES,
    REPO,
    TEST_ROOT,
    collate_labeled,
    filter_indices_mapped,
    load_preprocessed_for_classifier,
)
from TwinsModel_Wiserep import build_daep, default_cfg, device_from_str, repo_import_setup
from TwinsTrain_Wiserep import to_device

HAS_REDSHIFT = True
IS_LATENT = False
IS_MACRO = True

DATA_DIR = data_dir_for(HAS_REDSHIFT)
COMPARISON_ROOT = TEST_ROOT / (
    ("daep_comparison" if HAS_REDSHIFT else "daep_comparison_noz")
    if IS_LATENT
    else ("daep_comparison_init" if HAS_REDSHIFT else "daep_comparison_init_noz")
)
CKPT_NAME = "classifier_best.pt"
META_CSV = DATA_DIR / "wiserep_metadata_processed.csv"
LATENT_NPZ_FALLBACK = DATA_DIR / "latent_raw_z.npz"

FPR_GRID = np.linspace(0.0, 1.0, 101)
model_type = (("Daep (diffusion) with redshift" if IS_LATENT else "DAEP (no diffusion) with redshift") 
    if HAS_REDSHIFT 
    else ("DAEP (diffusion) without redshift" if IS_LATENT else "DAEP (no diffusion) without redshift")
)


def md5_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_all_ckpt_dirs(
    root: pathlib.Path,
    *,
    run_style: str = "auto",
) -> list[pathlib.Path]:
    """Immediate child dirs that contain ``CKPT_NAME``.

    ``run_style``: ``auto`` (prefer ``iter_*`` over legacy ``iterN``), ``underscore``,
    ``legacy``, or ``split``.
    """
    underscore: list[pathlib.Path] = []
    legacy: list[pathlib.Path] = []
    split_dirs: list[pathlib.Path] = []
    other: list[pathlib.Path] = []
    for p in root.iterdir():
        if not (p.is_dir() and (p / CKPT_NAME).is_file()):
            continue
        if re.fullmatch(r"iter_\d+", p.name, re.I):
            underscore.append(p)
        elif re.fullmatch(r"iter\d+", p.name, re.I):
            legacy.append(p)
        elif re.fullmatch(r"split_\d+", p.name, re.I):
            split_dirs.append(p)
        else:
            other.append(p)

    def sort_key(p: pathlib.Path):
        m = re.match(r"^iter_?(\d+)$", p.name, re.I)
        if m:
            return (0, int(m.group(1)))
        m = re.match(r"^split_(\d+)$", p.name, re.I)
        if m:
            return (0, int(m.group(1)))
        return (1, p.name.lower())

    if run_style == "underscore":
        chosen = underscore
    elif run_style == "legacy":
        chosen = legacy
    elif run_style == "split":
        chosen = split_dirs
    elif run_style == "auto":
        # Prefer underscore form (henna / current); ignore legacy iter0.. when both exist.
        chosen = underscore if underscore else (split_dirs if split_dirs else legacy)
    else:
        raise ValueError(f"Unknown run_style={run_style!r}")

    chosen = sorted(chosen, key=sort_key) + (
        [] if run_style != "auto" else sorted(other, key=lambda p: p.name.lower())
    )
    return chosen


def has_redshift_from_root(root: pathlib.Path) -> bool:
    name = root.name.lower()
    return not ("noz" in name or "no_z" in name or "nodered" in name)


def _load_ckpt_meta(ckpt_path: pathlib.Path, device: torch.device) -> dict:
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def resolve_meta_csv_for_ensemble(
    run_dirs: list[pathlib.Path],
    device: torch.device,
) -> pathlib.Path:
    """
    Latent checkpoints store ``meta_csv`` (often under ``data_no_z``) while ``TwinsClassifier_Wiserep``
    may point at ``data_z`` if ``USE_REDSHIFT`` differs — load meta from checkpoint cfg for latent-only
    ensembles. DAEP runs use ``META_CSV`` from the classifier module.
    """
    latent_meta: pathlib.Path | None = None
    has_daep = False

    for run_dir in run_dirs:
        ckpt_path = run_dir / CKPT_NAME
        if not ckpt_path.is_file():
            continue
        ckpt = _load_ckpt_meta(ckpt_path, device)
        if "encoder_state_dict" in ckpt:
            has_daep = True
            break
        zcfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
        m = zcfg.get("meta_csv")
        if m:
            mp = pathlib.Path(m)
            if mp.is_file():
                r = mp.resolve()
                if latent_meta is None:
                    latent_meta = r
                elif latent_meta != r:
                    raise RuntimeError(
                        f"Inconsistent meta_csv across latent runs: {latent_meta} vs {r}"
                    )

    if has_daep:
        return META_CSV.resolve()
    if latent_meta is not None:
        return latent_meta
    return META_CSV.resolve()


def dedupe_dirs_by_checkpoint(run_dirs: list[pathlib.Path]) -> tuple[list[pathlib.Path], int]:
    """
    Keep one folder per distinct ``classifier_best.pt`` content (MD5).
    Returns (unique_dirs_in_order, num_input_dirs).
    """
    seen: set[str] = set()
    out: list[pathlib.Path] = []
    for d in run_dirs:
        digest = md5_file(d / CKPT_NAME)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(d)
    return out, len(run_dirs)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        y = batch.pop("y").to(device)
        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def build_loader_daep(
    run_dir: pathlib.Path,
    ckpt: dict,
    meta: pd.DataFrame,
    te_idx: np.ndarray,
    wave: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.nn.Module, DataLoader]:
    cfg_p = run_dir / "cfg_used.json"
    if not cfg_p.is_file():
        cfg_p = DATA_DIR / "Output" / "cfg_used.json"
    cfg = default_cfg()
    if cfg_p.is_file():
        cfg.update(json.loads(cfg_p.read_text(encoding="utf-8")))
    if isinstance(ckpt.get("cfg"), dict):
        cfg.update(ckpt["cfg"])
    repo_import_setup(str(REPO.resolve()))
    model = MeanPoolClassifier(
        build_daep(cfg).to(device).encoder,
        int(cfg["bottleneck_dim"]) * int(cfg["bottleneck_length"]),
        NUM_CLASSES,
        int(cfg["ff_dim"]),
        float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    ds = LabeledSpectra(meta, wave, flux, mask, te_idx, LABEL_COLUMN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_labeled, num_workers=0)
    return model, loader


def _cfg_from_ckpt(run_dir: pathlib.Path, ckpt: dict) -> dict:
    cfg: dict = {}
    cfg_p = run_dir / "cfg_used.json"
    if cfg_p.is_file():
        cfg.update(json.loads(cfg_p.read_text(encoding="utf-8")))
    if isinstance(ckpt.get("cfg"), dict):
        cfg.update(ckpt["cfg"])
    return cfg


def _resolve_existing_path(raw: str | pathlib.Path, *fallbacks: pathlib.Path) -> pathlib.Path | None:
    p = pathlib.Path(raw)
    for cand in (p, *fallbacks):
        if cand.is_file():
            return cand.resolve()
    return None


def build_loader_latent(
    ckpt: dict,
    meta: pd.DataFrame,
    te_idx: np.ndarray,
    device: torch.device,
    *,
    z: np.ndarray | None = None,
) -> tuple[torch.nn.Module, DataLoader]:
    zcfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
    if z is None:
        if zcfg.get("latent_npz"):
            for cand in (
                pathlib.Path(zcfg["latent_npz"]),
                WISEREP_DIR / pathlib.Path(zcfg["latent_npz"]).name,
                LATENT_NPZ_FALLBACK,
            ):
                if cand.is_file():
                    z = np.load(cand)["z"].astype(np.float32, copy=False)
                    break
        if z is None:
            z = np.load(LATENT_NPZ_FALLBACK)["z"].astype(np.float32, copy=False)
    if len(meta) != z.shape[0]:
        raise ValueError(
            f"meta rows ({len(meta)}) != latent z rows ({z.shape[0]}). "
            "Latent checkpoints must include cfg['meta_csv'] matching the NPZ in cfg['latent_npz'] "
            "(re-save from train_latent.py, or set TwinsClassifier_Wiserep.USE_REDSHIFT to match train_latent)."
        )
    cfg = zcfg
    model = LatentClassifier(
        int(np.prod(z.shape[1:])),
        NUM_CLASSES,
        int(cfg.get("ff_dim", 512)),
        float(cfg.get("dropout", 0.1)),
    ).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.head.load_state_dict(ckpt["head_state_dict"], strict=True)
    ds = LatentDataset(meta, z, te_idx, LABEL_COLUMN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_latent, num_workers=0)
    return model, loader


def _should_use_assignment(cfg: dict) -> bool:
    split_source = cfg.get("split_source")
    if isinstance(split_source, str) and split_source.startswith("assignment:"):
        return True
    run_id = str(cfg.get("run_id", ""))
    return bool(re.fullmatch(r"split_\d+", run_id))


def _latent_dir_for_assignment(latent_npz: str | pathlib.Path) -> pathlib.Path:
    """Parent of the configured latent path, without following symlinks."""
    return pathlib.Path(latent_npz).expanduser().parent


def _test_idx_for_latent(cfg: dict, meta: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Assignment for data-split runs; IAU seed 0 for training-seed iter_* ensembles."""
    if _should_use_assignment(cfg):
        latent_npz = cfg.get("latent_npz")
        if not latent_npz:
            raise KeyError("data-split cfg missing latent_npz")
        latent_dir = _latent_dir_for_assignment(latent_npz)
        _, _, te_idx, src = load_assignment_indices_from_dir(latent_dir, meta)
        return filter_indices_mapped(meta, te_idx, LABEL_COLUMN), src
    _, _, te_idx = split_row_indices_by_iau_train_val_test(
        meta, IAU_COLUMN, IAU_TRAIN_FRAC, IAU_VAL_FRAC, IAU_TEST_FRAC, IAU_SPLIT_SEED
    )
    return filter_indices_mapped(meta, te_idx, LABEL_COLUMN), f"iau_split_seed={IAU_SPLIT_SEED}"


def run_models(
    run_dirs: list[pathlib.Path],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Evaluate each run on its own test split.

    Latent runs load z/meta from that checkpoint's cfg and use assignment indices when present.
    DAEP-init runs share the IAU seed-0 test split on the classifier data_dir bundle.
    """
    wave = flux = mask = None
    shared_daep_meta: pd.DataFrame | None = None
    shared_daep_te: np.ndarray | None = None
    all_y_true: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for run_dir in run_dirs:
        ckpt_path = run_dir / CKPT_NAME
        if not ckpt_path.is_file():
            continue
        ckpt = _load_ckpt_meta(ckpt_path, device)

        if "encoder_state_dict" in ckpt:
            if shared_daep_meta is None:
                shared_daep_meta, wave, flux, mask = load_preprocessed_for_classifier(DATA_DIR)
                _, _, shared_daep_te = split_row_indices_by_iau_train_val_test(
                    shared_daep_meta,
                    IAU_COLUMN,
                    IAU_TRAIN_FRAC,
                    IAU_VAL_FRAC,
                    IAU_TEST_FRAC,
                    IAU_SPLIT_SEED,
                )
                shared_daep_te = filter_indices_mapped(
                    shared_daep_meta, shared_daep_te, LABEL_COLUMN
                )
            model, loader = build_loader_daep(
                run_dir, ckpt, shared_daep_meta, shared_daep_te, wave, flux, mask, device
            )
            split_src = f"iau_split_seed={IAU_SPLIT_SEED}"
        else:
            cfg = _cfg_from_ckpt(run_dir, ckpt)
            ckpt = {**ckpt, "cfg": {**(ckpt.get("cfg") or {}), **cfg}}
            meta_csv = _resolve_existing_path(
                cfg.get("meta_csv", META_CSV),
                META_CSV,
            )
            if meta_csv is None:
                raise FileNotFoundError(f"{run_dir}: missing meta_csv in cfg")
            meta = normalize_latent_meta(pd.read_csv(meta_csv, low_memory=False))
            latent_npz = _resolve_existing_path(
                cfg.get("latent_npz", LATENT_NPZ_FALLBACK),
                LATENT_NPZ_FALLBACK,
            )
            if latent_npz is None:
                raise FileNotFoundError(f"{run_dir}: missing latent_npz in cfg")
            z = np.load(latent_npz)["z"].astype(np.float32, copy=False)
            te_idx, split_src = _test_idx_for_latent(cfg, meta)
            model, loader = build_loader_latent(ckpt, meta, te_idx, device, z=z)

        y_true, y_score = predict_probs(model, loader, device)
        all_y_true.append(y_true)
        all_scores.append(y_score)
        print(f"  {run_dir.name}: test_n={len(y_true)}  split={split_src}")

    if not all_y_true or not all_scores:
        raise RuntimeError("No valid checkpoints found under given directories.")
    return all_y_true, all_scores


def std_ddof1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def main(
    comparison_root: pathlib.Path | None = None,
    *,
    average: str | None = None,
    output: pathlib.Path | None = None,
    run_style: str = "auto",
    max_runs: int | None = None,
) -> None:
    global DATA_DIR, HAS_REDSHIFT, LATENT_NPZ_FALLBACK

    root = pathlib.Path(comparison_root or COMPARISON_ROOT).resolve()
    HAS_REDSHIFT = has_redshift_from_root(root)
    DATA_DIR = data_dir_for(HAS_REDSHIFT)
    LATENT_NPZ_FALLBACK = DATA_DIR / "latent_raw_z.npz"

    all_dirs = discover_all_ckpt_dirs(root, run_style=run_style)
    if max_runs is not None:
        all_dirs = all_dirs[: max(0, max_runs)]
    if not all_dirs:
        raise FileNotFoundError(f"No subdirs with {CKPT_NAME} under {root} (run_style={run_style})")

    run_dirs, n_folders_total = dedupe_dirs_by_checkpoint(all_dirs)
    n_dup_skipped = n_folders_total - len(run_dirs)
    if n_dup_skipped:
        print(
            f"[roc_ensemble_wiserep] Skipped {n_dup_skipped} folders whose {CKPT_NAME} "
            f"is byte-identical (MD5) to another folder — std only reflects {len(run_dirs)} unique weight(s).",
            file=sys.stderr,
        )

    device = device_from_str(DEVICE)
    all_y_true, all_scores = run_models(run_dirs, device)
    n_runs = len(all_scores)
    average_kind = average if average is not None else ("macro" if IS_MACRO else "micro")
    data_split_ensemble = all(re.fullmatch(r"split_\d+", d.name) for d in run_dirs)
    # Detect latent vs daep-init from first run for the title.
    first_ckpt = _load_ckpt_meta(run_dirs[0] / CKPT_NAME, device)
    is_latent = "encoder_state_dict" not in first_ckpt
    title_model = (
        ("DAEP (diffusion) with redshift" if HAS_REDSHIFT else "DAEP (diffusion) without redshift")
        if is_latent
        else (
            "DAEP (no diffusion) with redshift"
            if HAS_REDSHIFT
            else "DAEP (no diffusion) without redshift"
        )
    )
    if data_split_ensemble:
        ensemble_tag = "data-split"
        file_tag = "data_split_ensemble"
    elif run_style == "legacy" or all(re.fullmatch(r"iter\d+", d.name, re.I) for d in run_dirs):
        ensemble_tag = "legacy training-seed"
        file_tag = "legacy_iter_training_seed_ensemble"
    else:
        ensemble_tag = "training-seed"
        file_tag = "training_seed_ensemble"

    f1_runs = [
        f1_score(y_true, np.argmax(s, axis=1), average=average_kind)
        for y_true, s in zip(all_y_true, all_scores)
    ]
    auc_runs = [
        roc_auc_score(y_true, s, multi_class="ovr", average=average_kind)
        for y_true, s in zip(all_y_true, all_scores)
    ]
    f1_m, f1_s = float(np.mean(f1_runs)), std_ddof1(np.array(f1_runs))
    auc_m, auc_s = float(np.mean(auc_runs)), std_ddof1(np.array(auc_runs))
    if n_dup_skipped:
        print(
            f"[roc_ensemble_wiserep] Skipped {n_dup_skipped} duplicate ckpt file(s).",
            file=sys.stderr,
        )

    fig, ax = plt.subplots(figsize=(9, 8))

    for i, name in enumerate(CLASS_NAMES):
        tpr_rows: list[np.ndarray] = []
        auc_i: list[float] = []
        for y_true, y_score in zip(all_y_true, all_scores):
            y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
            if y_bin.ndim == 1:
                y_bin = np.column_stack([1 - y_bin, y_bin])
            pos = int(y_bin[:, i].sum())
            if pos == 0 or pos == len(y_true):
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            tpr_rows.append(np.interp(FPR_GRID, fpr, tpr))
            auc_i.append(auc(fpr, tpr))
        if not tpr_rows:
            continue
        tpr_mean = np.mean(tpr_rows, axis=0)
        tpr_std = np.std(tpr_rows, axis=0, ddof=1) if len(tpr_rows) > 1 else np.zeros_like(tpr_mean)
        auc_mean = float(np.mean(auc_i))
        auc_std = std_ddof1(np.array(auc_i))
        auc_sd_str = f"{auc_std:.4f}" if auc_std >= 1e-4 else f"{auc_std:.2e}"
        (line,) = ax.plot(
            FPR_GRID,
            tpr_mean,
            lw=2,
            label=f"{name} (AUC={auc_mean:.3f}±{auc_sd_str})",
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
    f1_s_str = f"{f1_s:.4f}" if f1_s >= 1e-4 else f"{f1_s:.2e}"
    auc_s_str = f"{auc_s:.4f}" if auc_s >= 1e-4 else f"{auc_s:.2e}"
    ax.set_title(
        f"ROC — {title_model} ({ensemble_tag} ensemble, n={n_runs})\n"
        f"{average_kind} F1 = {f1_m:.3f} ± {f1_s_str}  |  "
        f"{average_kind} avg AUC = {auc_m:.3f} ± {auc_s_str}"
    )
    fig.tight_layout()
    out = (
        output.expanduser().resolve()
        if output is not None
        else (root / "plots" / f"{file_tag}_roc_ovr_{average_kind}.png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(
        f"  {average_kind} F1 = {f1_m:.4f} ± {f1_s_str} | "
        f"{average_kind} AUC = {auc_m:.4f} ± {auc_s_str} | n={n_runs}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble OvR ROC for DAEP WISeREP runs.")
    parser.add_argument(
        "comparison_root",
        type=pathlib.Path,
        nargs="?",
        default=None,
        help="Folder containing iter_* / iterN / split_* runs",
    )
    parser.add_argument(
        "--average",
        choices=("micro", "macro"),
        default=None,
        help="Averaging for F1/AUC title metrics (default: module IS_MACRO)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output PNG (default: <root>/plots/<tag>_roc_ovr_<average>.png)",
    )
    parser.add_argument(
        "--run-style",
        choices=("auto", "underscore", "legacy", "split"),
        default="auto",
        help="Which run directories to include",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap after discovery/sort (e.g. 10 for legacy iter0..iter9)",
    )
    ns = parser.parse_args()
    main(
        ns.comparison_root,
        average=ns.average,
        output=ns.output,
        run_style=ns.run_style,
        max_runs=ns.max_runs,
    )
