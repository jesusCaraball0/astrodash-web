"""
DAEP-style classifier: SpectraTransceiverEncoder (wave, flux) + raw redshift concat + MLP head.
Training uses Ruiyao parquet + parquet_dataset.load_df_metadata_train_val_ids.
Outputs go to constants.DAEP_DIR; resume by re-running (loads training_checkpoint.pth if present).

Encoder blocks adapted from
https://github.com/YunyiShen/Perceiver-diffusion-autoencoder (SpectraLayers, Perceiver, util_layers).

https://github.com/hnemeh/Twinsanity/ (Training and helpers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_ZM = Path(__file__).resolve().parent
if str(_ZM / "Perceiver-diffusion-autoencoder" / "package") not in sys.path:
    sys.path.insert(0, str(_ZM / "Perceiver-diffusion-autoencoder" / "package"))

from daep.SpectraLayers import spectraTransceiverEncoder  # noqa: E402

import constants as const
import dash_retrain
import helpers as helpers
import parquet_dataset as rpd


class DAEPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = const.DAEP_BOTTLENECK_DIM
        h = const.DAEP_HEAD_HIDDEN
        self.encoder = spectraTransceiverEncoder(
            const.DAEP_BOTTLENECK_LENGTH,
            d,
            model_dim=const.DAEP_MODEL_DIM,
            num_heads=const.DAEP_NUM_HEADS,
            num_layers=const.DAEP_NUM_LAYERS,
            ff_dim=const.DAEP_FF_DIM,
            dropout=const.DAEP_DROPOUT,
            selfattn=const.DAEP_SELFATTN,
            concat=const.DAEP_CONCAT,
        )
        self.head = nn.Sequential(
            nn.Linear(d + 1, h),
            nn.ReLU(inplace=True),
            nn.Dropout(const.DAEP_DROPOUT),
            nn.Linear(h, const.NUM_CLASSES),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        rz = x["redshift"]
        enc_in = {k: x[k] for k in ("flux", "wavelength", "phase", "mask")}
        z = self.encoder(enc_in)
        pooled = z.mean(dim=1)
        return self.head(torch.cat([pooled, rz], dim=-1))


def _resample(wave: np.ndarray, flux: np.ndarray, n_bins: int, w0: float, w1: float):
    wave = np.asarray(wave, dtype=np.float64).ravel()
    flux = np.asarray(flux, dtype=np.float64).ravel()
    o = np.argsort(wave)
    wave, flux = wave[o], flux[o]
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.linspace(float(w0), float(w1), n_bins, dtype=np.float64)
    fi = np.interp(grid, wave, flux, left=np.nan, right=np.nan)
    bad = np.isnan(fi)
    fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
    fi = np.clip(fi, -1e6, 1e6)
    g = np.clip(grid, float(w0), float(w1)).astype(np.float32)
    return g, fi.astype(np.float32), bad


def _redshift_from_row(row: pd.Series) -> float:
    """Parquet Redshift column; missing -> 0.0 (same spirit as dash_retrain WISeREP metadata)."""
    if "Redshift" not in row.index:
        return 0.0
    zv = row["Redshift"]
    if pd.isna(zv):
        return 0.0
    try:
        z = float(zv)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(z) or z < 0.0:
        return 0.0
    return float(min(z, 5.0))


class ParquetRawSpectrumDataset(Dataset):
    def __init__(self, spectrum_ids: List[str], df: pd.DataFrame, n_bins: int, w0: float, w1: float):
        self.df, self.n_bins, self.w0, self.w1 = df, n_bins, w0, w1
        n = len(df)
        self.samples: List[Tuple[int, int]] = []
        for sid in spectrum_ids:
            i = rpd.wiserep_spectrum_id_to_iloc(sid)
            if not (0 <= i < n):
                continue
            t = "" if pd.isna(df.iloc[i]["parent_class"]) else str(df.iloc[i]["parent_class"]).strip()
            lab = helpers.normalize_label(t)
            if lab is not None:
                self.samples.append((i, const.CLASS_TO_IDX[lab]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[Dict[str, torch.Tensor], int]]:
        iloc, y = self.samples[idx]
        row = self.df.iloc[iloc]
        try:
            g, f, bad = _resample(row["wavelength"], row["flux"], self.n_bins, self.w0, self.w1)
        except Exception:
            return None
        f = np.asarray(f, dtype=np.float64)
        s = float(np.nanmax(np.abs(f)))
        if not np.isfinite(s) or s < 1e-12:
            s = 1.0
        f = np.clip(f / s, -20.0, 20.0).astype(np.float32)
        z = _redshift_from_row(row)
        return (
            {
                "flux": torch.from_numpy(f),
                "wavelength": torch.from_numpy(g),
                "phase": torch.tensor(0.0, dtype=torch.float32),
                "mask": torch.from_numpy(bad.astype(bool)),
                "redshift": torch.tensor([z], dtype=torch.float32),
            },
            y,
        )


def _class_weights_from_dataset(ds: ParquetRawSpectrumDataset) -> torch.Tensor:
    """sklearn-style balanced: n / (K * count_c)."""
    counts = [0] * const.NUM_CLASSES
    for _iloc, y in ds.samples:
        counts[y] += 1
    n, k = sum(counts), const.NUM_CLASSES
    return torch.tensor([(n / (k * c)) if c > 0 else 0.0 for c in counts], dtype=torch.float32)


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return (
        {
            "flux": torch.stack([x["flux"] for x in xs]),
            "wavelength": torch.stack([x["wavelength"] for x in xs]),
            "phase": torch.stack([x["phase"] for x in xs]),
            "mask": torch.stack([x["mask"] for x in xs]).bool(),
            "redshift": torch.stack([x["redshift"] for x in xs]),
        },
        torch.tensor(ys, dtype=torch.long),
    )


CHECKPOINT_FNAME = "training_checkpoint.pth"
MODEL_FNAME = "model.pth"


def train() -> None:
    helpers.set_seed(const.SEED)
    device = helpers.get_device()
    nw = const.TARGET_LENGTH - 1
    out = const.DAEP_DIR
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CHECKPOINT_FNAME
    best_path = out / MODEL_FNAME

    df, _metadata, train_ids, val_ids, _ = rpd.load_df_metadata_train_val_ids(const.SEED)
    train_ds = ParquetRawSpectrumDataset(train_ids, df, nw, const.WAVE_MIN, const.WAVE_MAX)
    val_ds = ParquetRawSpectrumDataset(val_ids, df, nw, const.WAVE_MIN, const.WAVE_MAX)

    train_counts = [0] * const.NUM_CLASSES
    for _i, y in train_ds.samples:
        train_counts[y] += 1
    print("train samples per class:", dict(zip(const.CLASS_NAMES, train_counts)))
    weights = _class_weights_from_dataset(train_ds).to(device)
    print("CE train class weights (balanced):", dict(zip(const.CLASS_NAMES, weights.tolist())))

    train_loader = DataLoader(
        train_ds,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=const.NUM_WORKERS,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=const.NUM_WORKERS,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    model = DAEPClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.DAEP_WEIGHT_DECAY)
    crit_train = nn.CrossEntropyLoss(weight=weights)
    crit_eval = nn.CrossEntropyLoss()

    start_ep = 1
    best_val, best_ep, stall = float("inf"), 0, 0
    if ckpt_path.is_file():
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_ep = int(ck["next_epoch"])
        best_val = float(ck["best_val"])
        best_ep = int(ck["best_epoch"])
        stall = 0 if const.DAEP_RESET_PATIENCE_ON_RESUME else int(ck["stall"])
        print(
            f"Resuming from {ckpt_path.name}: next epoch {start_ep}/{const.EPOCHS}, "
            f"best val loss {best_val:.4f} @ epoch {best_ep}"
            + (" (stall reset)" if const.DAEP_RESET_PATIENCE_ON_RESUME else "")
        )

    if start_ep > const.EPOCHS:
        print(
            f"Nothing to train: checkpoint next_epoch={start_ep} > EPOCHS={const.EPOCHS}. "
            f"Increase EPOCHS in constants or delete {ckpt_path} to start over."
        )
        return

    def _save_checkpoint(next_epoch: int) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "next_epoch": next_epoch,
                "best_val": best_val,
                "best_epoch": best_ep,
                "stall": stall,
            },
            ckpt_path,
        )

    for ep in range(start_ep, const.EPOCHS + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{ep}/{const.EPOCHS}", leave=False):
            if batch is None:
                continue
            x, y = batch
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            opt.zero_grad()
            loss = crit_train(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * y.size(0)
            n += y.size(0)

        print(f"epoch {ep:3d}  train_loss {tr_loss / max(n, 1):.4f}")
        # Unweighted val loss + metrics (weighted CE on val drives misleading loss + collapse)
        va_loss, acc, cm = dash_retrain.evaluate(model, val_loader, crit_eval, device)

        if va_loss < best_val:
            best_val, best_ep, stall = va_loss, ep, 0
            torch.save(model.state_dict(), best_path)
            perf = dash_retrain.build_performance_json(best_ep, va_loss, acc, cm)
            perf["parquet_path"] = str(rpd.RUIYAO_PARQUET)
            perf["n_bins"] = nw
            perf["seed"] = const.SEED
            perf["uses_redshift"] = True
            with open(out / "model_performance.json", "w") as f:
                json.dump(perf, f, indent=2)
        else:
            stall += 1

        _save_checkpoint(ep + 1)

        if stall >= const.EARLY_STOP_PATIENCE:
            print(f"early stop (best epoch {best_ep}, val_loss {best_val:.4f})")
            break

    print(f"done: {best_path} (checkpoint state: {ckpt_path})")


if __name__ == "__main__":
    train()
