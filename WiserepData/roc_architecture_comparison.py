"""
Compare micro-average ensemble ROC curves across model architectures.

Six henna-matched ensembles:
  - DAEP diffusion ±z     → WiserepData/Test/daep_comparison(_noz)
  - DAEP no-diffusion ±z  → WiserepData/Test/daep_comparison_init(_noz)
  - Dash 1D CNN ±z        → data/pre_trained_models/henna_matched_comparison_{z,noz}

Usage:
  # write a NEW file (keeps existing architecture_comparison_micro.png)
  python WiserepData/roc_architecture_comparison.py \\
      WiserepData/Test/architecture_comparison_micro_henna.png

  # default path (overwrites Test/architecture_comparison_micro.png)
  python WiserepData/roc_architecture_comparison.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZMODEL_DIR = PROJECT_ROOT / "zmodel_training"

for path in (SCRIPT_DIR, PROJECT_ROOT, ZMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import roc_ensemble_daep_comparison as dash_roc
import roc_ensemble_wiserep as wiserep_roc
from train_latent import latent_dir_for, normalize_latent_meta, resolve_latent_npz

FPR_GRID = np.linspace(0.0, 1.0, 101)
DEFAULT_OUTPUT = SCRIPT_DIR / "Test" / "architecture_comparison_micro.png"


@dataclass(frozen=True)
class EnsembleSpec:
    label: str
    root: pathlib.Path
    kind: str
    has_redshift: bool
    is_latent: bool | None = None


@dataclass
class EnsembleResult:
    label: str
    root: pathlib.Path
    n_total: int
    n_unique: int
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float
    tpr_mean: np.ndarray
    tpr_std: np.ndarray


def wiserep_title(has_redshift: bool, is_latent: bool) -> str:
    return (
        ("Daep (diffusion) with redshift" if is_latent else "DAEP (no diffusion) with redshift")
        if has_redshift
        else ("DAEP (diffusion) without redshift" if is_latent else "DAEP (no diffusion) without redshift")
    )


def dash_title(has_redshift: bool) -> str:
    base_title = "Dash1D CNN"
    return f"{base_title} with redshift" if has_redshift else f"{base_title} without redshift"


SPECS = [
    EnsembleSpec(
        label=wiserep_title(True, True),
        root=wiserep_roc.TEST_ROOT / "daep_comparison",
        kind="wiserep",
        has_redshift=True,
        is_latent=True,
    ),
    EnsembleSpec(
        label=wiserep_title(True, False),
        root=wiserep_roc.TEST_ROOT / "daep_comparison_init",
        kind="wiserep",
        has_redshift=True,
        is_latent=False,
    ),
    EnsembleSpec(
        label=wiserep_title(False, False),
        root=wiserep_roc.TEST_ROOT / "daep_comparison_init_noz",
        kind="wiserep",
        has_redshift=False,
        is_latent=False,
    ),
    EnsembleSpec(
        label=wiserep_title(False, True),
        root=wiserep_roc.TEST_ROOT / "daep_comparison_noz",
        kind="wiserep",
        has_redshift=False,
        is_latent=True,
    ),
    EnsembleSpec(
        label=dash_title(True),
        root=PROJECT_ROOT / "data" / "pre_trained_models" / "henna_matched_comparison_z",
        kind="dash",
        has_redshift=True,
    ),
    EnsembleSpec(
        label=dash_title(False),
        root=PROJECT_ROOT / "data" / "pre_trained_models" / "henna_matched_comparison_noz",
        kind="dash",
        has_redshift=False,
    ),
]


def md5_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def std_ddof1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def dedupe_dirs_by_checkpoint(
    run_dirs: list[pathlib.Path],
    ckpt_name: str,
) -> tuple[list[pathlib.Path], int]:
    seen: set[str] = set()
    out: list[pathlib.Path] = []
    for run_dir in run_dirs:
        digest = md5_file(run_dir / ckpt_name)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(run_dir)
    return out, len(run_dirs)


def resolve_wiserep_meta_csv(
    run_dirs: list[pathlib.Path],
    device: torch.device,
    default_meta_csv: pathlib.Path,
) -> pathlib.Path:
    latent_meta: pathlib.Path | None = None
    has_daep = False

    for run_dir in run_dirs:
        ckpt_path = run_dir / wiserep_roc.CKPT_NAME
        if not ckpt_path.is_file():
            continue
        ckpt = wiserep_roc._load_ckpt_meta(ckpt_path, device)
        if "encoder_state_dict" in ckpt:
            has_daep = True
            break
        zcfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
        meta_csv = zcfg.get("meta_csv")
        if not meta_csv:
            continue
        candidate = pathlib.Path(meta_csv)
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if latent_meta is None:
            latent_meta = resolved
        elif latent_meta != resolved:
            raise RuntimeError(
                f"Inconsistent meta_csv across latent runs: {latent_meta} vs {resolved}"
            )

    if has_daep:
        return default_meta_csv.resolve()
    if latent_meta is not None:
        return latent_meta
    return default_meta_csv.resolve()


def build_wiserep_loader_daep(
    run_dir: pathlib.Path,
    ckpt: dict,
    meta: pd.DataFrame,
    te_idx: np.ndarray,
    wave: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    data_dir: pathlib.Path,
) -> tuple[torch.nn.Module, DataLoader]:
    cfg_p = run_dir / "cfg_used.json"
    if not cfg_p.is_file():
        cfg_p = data_dir / "Output" / "cfg_used.json"
    cfg = wiserep_roc.default_cfg()
    if cfg_p.is_file():
        cfg.update(json.loads(cfg_p.read_text(encoding="utf-8")))
    if isinstance(ckpt.get("cfg"), dict):
        cfg.update(ckpt["cfg"])
    wiserep_roc.repo_import_setup(str(wiserep_roc.REPO.resolve()))
    model = wiserep_roc.MeanPoolClassifier(
        wiserep_roc.build_daep(cfg).to(device).encoder,
        int(cfg["bottleneck_dim"]) * int(cfg["bottleneck_length"]),
        wiserep_roc.NUM_CLASSES,
        int(cfg["ff_dim"]),
        float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    dataset = wiserep_roc.LabeledSpectra(meta, wave, flux, mask, te_idx, wiserep_roc.LABEL_COLUMN)
    loader = DataLoader(
        dataset,
        batch_size=wiserep_roc.BATCH_SIZE,
        shuffle=False,
        collate_fn=wiserep_roc.collate_labeled,
        num_workers=0,
    )
    return model, loader


def build_wiserep_loader_latent(
    ckpt: dict,
    meta: pd.DataFrame,
    te_idx: np.ndarray,
    device: torch.device,
    latent_npz_fallback: pathlib.Path,
) -> tuple[torch.nn.Module, DataLoader]:
    zcfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
    z = None
    if zcfg.get("latent_npz"):
        for candidate in (
            pathlib.Path(zcfg["latent_npz"]),
            SCRIPT_DIR / pathlib.Path(zcfg["latent_npz"]).name,
            latent_npz_fallback,
        ):
            if candidate.is_file():
                z = np.load(candidate)["z"].astype(np.float32, copy=False)
                break
    if z is None:
        z = np.load(latent_npz_fallback)["z"].astype(np.float32, copy=False)
    if len(meta) != z.shape[0]:
        raise ValueError(
            f"meta rows ({len(meta)}) != latent z rows ({z.shape[0]}). "
            "Latent checkpoints must include cfg['meta_csv'] matching the NPZ in cfg['latent_npz']."
        )
    model = wiserep_roc.LatentClassifier(
        int(np.prod(z.shape[1:])),
        wiserep_roc.NUM_CLASSES,
        int(zcfg.get("ff_dim", 512)),
        float(zcfg.get("dropout", 0.1)),
    ).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.head.load_state_dict(ckpt["head_state_dict"], strict=True)
    dataset = wiserep_roc.LatentDataset(meta, z, te_idx, wiserep_roc.LABEL_COLUMN)
    loader = DataLoader(
        dataset,
        batch_size=wiserep_roc.BATCH_SIZE,
        shuffle=False,
        collate_fn=wiserep_roc.collate_latent,
        num_workers=0,
    )
    return model, loader


def collect_wiserep_predictions(
    run_dirs: list[pathlib.Path],
    device: torch.device,
    has_redshift: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    data_dir = wiserep_roc.data_dir_for(has_redshift)
    default_meta_csv = data_dir / "wiserep_metadata_processed.csv"
    # Latents live under 1024d2 / nodered1024d2, not the spectra dirs.
    try:
        latent_npz_fallback = resolve_latent_npz(latent_dir_for(has_redshift))
    except FileNotFoundError:
        latent_npz_fallback = data_dir / "latent_raw_z.npz"

    meta_csv = resolve_wiserep_meta_csv(run_dirs, device, default_meta_csv)
    # Henna meta_universal.csv uses sn_name_used / raw_type; map to IAU name / Obj. Type.
    meta = normalize_latent_meta(pd.read_csv(meta_csv, low_memory=False))
    _, _, te_idx = wiserep_roc.split_row_indices_by_iau_train_val_test(
        meta,
        wiserep_roc.IAU_COLUMN,
        wiserep_roc.IAU_TRAIN_FRAC,
        wiserep_roc.IAU_VAL_FRAC,
        wiserep_roc.IAU_TEST_FRAC,
        wiserep_roc.IAU_SPLIT_SEED,
    )
    te_idx = wiserep_roc.filter_indices_mapped(meta, te_idx, wiserep_roc.LABEL_COLUMN)

    wave = flux = mask = None
    y_ref: np.ndarray | None = None
    all_scores: list[np.ndarray] = []

    for run_dir in run_dirs:
        ckpt_path = run_dir / wiserep_roc.CKPT_NAME
        if not ckpt_path.is_file():
            continue
        ckpt = wiserep_roc._load_ckpt_meta(ckpt_path, device)
        if "encoder_state_dict" in ckpt:
            if wave is None:
                _, wave, flux, mask = wiserep_roc.load_preprocessed_for_classifier(data_dir)
            model, loader = build_wiserep_loader_daep(
                run_dir,
                ckpt,
                meta,
                te_idx,
                wave,
                flux,
                mask,
                device,
                data_dir,
            )
        else:
            model, loader = build_wiserep_loader_latent(
                ckpt,
                meta,
                te_idx,
                device,
                latent_npz_fallback,
            )

        y_true, y_score = wiserep_roc.predict_probs(model, loader, device)
        if y_ref is None:
            y_ref = y_true
        elif not np.array_equal(y_ref, y_true):
            raise RuntimeError(f"Label mismatch vs first run at {run_dir}")
        all_scores.append(y_score)

    if y_ref is None or not all_scores:
        raise RuntimeError("No valid WISeREP checkpoints found under given directories.")
    return y_ref, all_scores


def collect_dash_predictions(
    run_dirs: list[pathlib.Path],
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    y_true, all_scores, class_names = dash_roc.collect_run_predictions(run_dirs, device)
    if class_names != wiserep_roc.CLASS_NAMES:
        raise RuntimeError(
            f"Dash class names differ from WISeREP class names: {class_names} vs {wiserep_roc.CLASS_NAMES}"
        )
    return y_true, all_scores


def summarize_micro_ensemble(
    y_true: np.ndarray,
    all_scores: list[np.ndarray],
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    n_classes = all_scores[0].shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    f1_runs: list[float] = []
    auc_runs: list[float] = []
    tpr_rows: list[np.ndarray] = []

    for y_score in all_scores:
        f1_runs.append(
            f1_score(y_true, np.argmax(y_score, axis=1), average="micro", zero_division=0)
        )
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        tpr_interp = np.interp(FPR_GRID, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_rows.append(tpr_interp)
        auc_runs.append(auc(fpr, tpr))

    tpr_mean = np.mean(tpr_rows, axis=0)
    tpr_std = np.std(tpr_rows, axis=0, ddof=1) if len(tpr_rows) > 1 else np.zeros_like(tpr_mean)
    f1_mean = float(np.mean(f1_runs))
    f1_std = std_ddof1(np.array(f1_runs))
    auc_mean = float(np.mean(auc_runs))
    auc_std = std_ddof1(np.array(auc_runs))
    return f1_mean, f1_std, auc_mean, auc_std, tpr_mean, tpr_std


def load_ensemble_result(spec: EnsembleSpec, device: torch.device) -> EnsembleResult:
    if spec.kind == "wiserep":
        all_dirs = wiserep_roc.discover_all_ckpt_dirs(spec.root)
        if not all_dirs:
            raise FileNotFoundError(f"No subdirs with {wiserep_roc.CKPT_NAME} under {spec.root}")
        run_dirs, n_total = dedupe_dirs_by_checkpoint(all_dirs, wiserep_roc.CKPT_NAME)
        y_true, all_scores = collect_wiserep_predictions(run_dirs, device, spec.has_redshift)
    elif spec.kind == "dash":
        all_dirs = [
            run_dir
            for run_dir in dash_roc.discover_run_dirs(spec.root)
            if (run_dir / dash_roc.CKPT_NAME).is_file()
        ]
        if not all_dirs:
            raise FileNotFoundError(f"No subdirs with {dash_roc.CKPT_NAME} under {spec.root}")
        run_dirs, n_total = dedupe_dirs_by_checkpoint(all_dirs, dash_roc.CKPT_NAME)
        y_true, all_scores = collect_dash_predictions(run_dirs, device)
    else:
        raise ValueError(f"Unknown ensemble kind: {spec.kind}")

    f1_mean, f1_std, auc_mean, auc_std, tpr_mean, tpr_std = summarize_micro_ensemble(
        y_true,
        all_scores,
    )
    return EnsembleResult(
        label=spec.label,
        root=spec.root,
        n_total=n_total,
        n_unique=len(run_dirs),
        f1_mean=f1_mean,
        f1_std=f1_std,
        auc_mean=auc_mean,
        auc_std=auc_std,
        tpr_mean=tpr_mean,
        tpr_std=tpr_std,
    )


def fmt_sd(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.2e}"


def main(output_path: pathlib.Path | None = None) -> None:
    output = (output_path or DEFAULT_OUTPUT).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = dash_roc.helpers.get_device()
    results = [load_ensemble_result(spec, device) for spec in SPECS]

    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results:
        auc_sd_str = fmt_sd(result.auc_std)
        f1_sd_str = fmt_sd(result.f1_std)
        (line,) = ax.plot(
            FPR_GRID,
            result.tpr_mean,
            lw=2,
            label=(
                f"{result.label} (AUC={result.auc_mean:.3f}±{auc_sd_str}, "
                f"micro F1={result.f1_mean:.3f}±{f1_sd_str})"
            ),
        )
        ax.fill_between(
            FPR_GRID,
            np.clip(result.tpr_mean - result.tpr_std, 0, 1),
            np.clip(result.tpr_mean + result.tpr_std, 0, 1),
            color=line.get_color(),
            alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Architecture Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")
    print("Micro F1 summary:")
    for result in results:
        skipped = result.n_total - result.n_unique
        print(
            f"  {result.label}: micro F1 = {result.f1_mean:.4f} ± {fmt_sd(result.f1_std)}"
            f" | micro AUC = {result.auc_mean:.4f} ± {fmt_sd(result.auc_std)}"
            f" | unique runs = {result.n_unique}/{result.n_total}"
            f" | duplicates skipped = {skipped}"
        )


if __name__ == "__main__":
    arg_output = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg_output)
