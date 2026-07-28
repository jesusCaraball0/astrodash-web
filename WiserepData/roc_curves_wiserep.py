"""Multiclass OvR ROC for a Wiserep classifier checkpoint.

``MODEL_DIR`` defaults to ``TwinsClassifier_Wiserep.OUT_DIR``; spectra paths from ``DATA_DIR`` there.
"""

from __future__ import annotations

import json
import pathlib
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
from train_latent import LatentClassifier, LatentDataset, collate_latent
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
    LabeledSpectra,
    MeanPoolClassifier,
    NUM_CLASSES,
    OUT_DIR,
    REPO,
    collate_labeled,
    filter_indices_mapped,
    load_preprocessed_for_classifier,
)
from TwinsModel_Wiserep import build_daep, default_cfg, device_from_str, repo_import_setup
from TwinsTrain_Wiserep import to_device

MODEL_DIR = OUT_DIR
CKPT_NAME = "classifier_best.pt"
META_CSV = DATA_DIR / "wiserep_metadata_processed.csv"
LATENT_NPZ_FALLBACK = DATA_DIR / "latent_raw_z.npz"

model_type = "DAEP (no diffusion)" if "init" in MODEL_DIR.stem else "Latent classifier"

def main() -> None:
    device = device_from_str(DEVICE)
    meta = pd.read_csv(META_CSV, low_memory=False).reset_index(drop=True)
    _, _, te_idx = split_row_indices_by_iau_train_val_test(
        meta, IAU_COLUMN, IAU_TRAIN_FRAC, IAU_VAL_FRAC, IAU_TEST_FRAC, IAU_SPLIT_SEED
    )
    te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)

    ckpt_path = MODEL_DIR / CKPT_NAME
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "encoder_state_dict" in ckpt:
        cfg_p = MODEL_DIR / "cfg_used.json"
        if not cfg_p.is_file():
            cfg_p = DATA_DIR / "Output" / "cfg_used.json"
        cfg = default_cfg()
        if cfg_p.is_file():
            cfg.update(json.loads(cfg_p.read_text(encoding="utf-8")))
        if isinstance(ckpt.get("cfg"), dict):
            cfg.update(ckpt["cfg"])
        repo_import_setup(str(REPO.resolve()))
        _, wave, flux, mask = load_preprocessed_for_classifier(DATA_DIR)
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
    else:
        zcfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
        z = None
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
        assert len(meta) == z.shape[0]
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

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            ys.append(y.cpu().numpy())
            ps.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    y_pred = np.argmax(y_score, axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

    y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, name in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"ROC {model_type} \nmacro F1 = {macro_f1:.3f}  |  macro avg AUC = {macro_auc:.3f}"
    )
    fig.tight_layout()
    out = MODEL_DIR / f"{ckpt_path.stem}_roc_ovr.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
