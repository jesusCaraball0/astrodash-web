#!/usr/bin/env python
"""
Retrain a DASH-style 1D CNN on WISeREP spectra.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


RUN_ID = "03_14_26_redshift"
HAS_REDSHIFT = True



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Ensure prod_backend is importable
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from app.config.settings import get_settings
from app.infrastructure.ml.data_processor import DashSpectrumProcessor
from app.shared.utils.validators import ValidationError
from app.infrastructure.storage.file_spectrum_repository import FileSpectrumRepository
import logging

# silence logs so tqdm stays readable
logging.getLogger("app.infrastructure.storage.file_spectrum_repository").setLevel(logging.CRITICAL)
logging.getLogger("app.infrastructure.ml.data_processor").setLevel(logging.CRITICAL)

WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
SPECTRA_DIR = WISEREP_DIR / "wiserep_data_noSEDM"
METADATA_CSV = WISEREP_DIR / "wiserep_metadata.csv"
SPLITS_JSON = WISEREP_DIR / "wiserep_splits_by_iau_noSEDM.json"
OUT_DIR = PROJECT_ROOT / "data" / "pre_trained_models" / "dash_wiserep" / "models" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5
VAL_EVERY = 1
NUM_WORKERS = 0

# 5-class label mapping (from Lanqing)
LABEL_MAP: Dict[str, str] = {
    # Type Ia
    "SN Ia": "SN Ia",
    "SN Ia-CSM": "SN Ia",
    "SN Ia-91T-like": "SN Ia",
    "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia",
    "SN Ia-pec": "SN Ia",
    "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia",
    "Computed-Ia": "SN Ia",
    # Type Ib/c (all Ib and Ic variants)
    "SN Ib": "SN Ib/c",
    "SN Ic": "SN Ib/c",
    "SN Ib/c": "SN Ib/c",
    "SN Ib-Ca-rich": "SN Ib/c",
    "SN Ib-pec": "SN Ib/c",
    "SN Ibn": "SN Ib/c",
    "SN Ic-BL": "SN Ib/c",
    "SN Ic-Ca-rich": "SN Ib/c",
    "SN Ic-pec": "SN Ib/c",
    "SN Icn": "SN Ib/c",
    "SN Ib/c-Ca-rich": "SN Ib/c",
    "SN Ibn/Icn": "SN Ib/c",
    # Type II (NOT including IIn)
    "SN II": "SN II",
    "SN IIP": "SN II",
    "SN IIL": "SN II",
    "SN II-pec": "SN II",
    "SN IIb": "SN II",
    "Computed-IIP": "SN II",
    "Computed-IIb": "SN II",
    # Type IIn (separate)
    "SN IIn": "SN IIn",
    "SN IIn-pec": "SN IIn",
    # SLSN
    "SLSN-I": "SLSN-I",
    "SLSN-II": "SLSN-I",
    "SLSN-R": "SLSN-I",
}

CLASS_NAMES = ["SN Ia", "SN Ib/c", "SN II", "SN IIn", "SLSN-I"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# match backend DASH preprocessing params
_SETTINGS = get_settings()
# Model input = spectrum bins (nw) + 1 redshift feature
TARGET_LENGTH = _SETTINGS.nw + 1
WAVE_MIN, WAVE_MAX = _SETTINGS.w0, _SETTINGS.w1

# global DashSpectrumProcessor (outputs nw bins; we append z to get TARGET_LENGTH)
_PROCESSOR = DashSpectrumProcessor(WAVE_MIN, WAVE_MAX, _SETTINGS.nw)
_FILE_REPO = FileSpectrumRepository(_SETTINGS)

def load_spectrum(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a spectrum from disk using the same parsing logic as the prod backend
    (FileSpectrumRepository). Returns (wave, flux) arrays filtered to the
    repository's wavelength range, or None on failure.
    """
    try:
        # wrap the path in a simple object that mimics fastapi's UploadFile
        class _LocalFile:
            def __init__(self, path: Path):
                self.filename = path.name
                self.file = open(path, "rb")

        wrapper = _LocalFile(filepath)
        try:
            spectrum = _FILE_REPO.get_from_file(wrapper)
        finally:
            try:
                wrapper.file.close()
            except Exception:
                pass

        if spectrum is None:
            return None
        wave = np.asarray(spectrum.x, dtype=float)
        flux = np.asarray(spectrum.y, dtype=float)
        if wave.size == 0 or flux.size == 0:
            return None
        return wave, flux
    except Exception as e:
        return None


def load_metadata(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    """Build map: ascii_filename -> {'type': ..., 'redshift': ...}."""
    info: Dict[str, Dict[str, str]] = {}
    if not metadata_path.exists():
        return info
    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("Ascii file") or "").strip()
            if not fname or fname in info:
                continue
            info[fname] = {
                "type": (row.get("Obj. Type") or "").strip(),
                "redshift": (row.get("Redshift") or "0").strip(),
            }
    return info

def normalize_label(raw_type: str) -> Optional[str]:
    raw = (raw_type or "").strip()
    if not raw:
        return None
    canonical = LABEL_MAP.get(raw)
    if canonical is None:
        # handles small formatting differences
        canonical = LABEL_MAP.get(raw.replace(" ", ""))
    if canonical is None:
        return None
    if canonical in CLASS_TO_IDX:
        return canonical
    return None


def stratified_train_val_split(
    train_filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split train_filenames into train/val by class, so val has ~val_frac of each class.
    Only includes files with a valid label. Returns (train_list, val_list).
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = {i: [] for i in range(NUM_CLASSES)}
    for fname in train_filenames:
        meta = metadata.get(fname)
        if meta is None:
            continue
        label = normalize_label(meta["type"])
        if label is None:
            continue
        idx = CLASS_TO_IDX[label]
        by_label[idx].append(fname)
    train_out: List[str] = []
    val_out: List[str] = []
    for idx in range(NUM_CLASSES):
        files = by_label[idx][:]
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac)) if files else 0
        n_train = len(files) - n_val
        train_out.extend(files[:n_train])
        val_out.extend(files[n_train:])
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


def stratified_k_folds(
    train_filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    k: int,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified k-fold split. Returns k pairs (train_filenames, val_filenames)
    so that each fold uses 1/k of the data as val (by class). Only files with valid labels.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = {i: [] for i in range(NUM_CLASSES)}
    for fname in train_filenames:
        meta = metadata.get(fname)
        if meta is None:
            continue
        label = normalize_label(meta["type"])
        if label is None:
            continue
        idx = CLASS_TO_IDX[label]
        by_label[idx].append(fname)
    # Assign fold index 0..k-1 to each file per class
    fold_assignments: Dict[str, int] = {}
    for idx in range(NUM_CLASSES):
        files = by_label[idx][:]
        rng.shuffle(files)
        for i, fname in enumerate(files):
            fold_assignments[fname] = i % k
    # Unlabeled files (not in any by_label) get a random fold
    for fname in train_filenames:
        if fname not in fold_assignments:
            fold_assignments[fname] = rng.randint(0, k - 1) if k > 1 else 0
    # Build k (train, val) pairs
    result: List[Tuple[List[str], List[str]]] = []
    for fold in range(k):
        train_f = [f for f in train_filenames if fold_assignments.get(f, 0) != fold]
        val_f = [f for f in train_filenames if fold_assignments.get(f, 0) == fold]
        result.append((train_f, val_f))
    return result


def preprocess_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    redshift: Optional[float] = None,
    target_length: int = TARGET_LENGTH 
) -> Optional[np.ndarray]:
    """
    Preprocess spectrum using backend original DASH preprocessing.

    This uses DashSpectrumProcessor(w0=3500, w1=10000, nw=1024)
    with the given redshift. 
    returns: processed flux array | None if processing/validation fails
    """

    try:
        if redshift is not None:
            processed_flux, _, _, _ = _PROCESSOR.process(
                wave, flux, float(redshift)
            )
        else:
            processed_flux, _, _ = _PROCESSOR.process_no_redshift(
                wave, flux
            )




    except Exception:
        return None

    return processed_flux.astype(np.float32)


# PyTorch Dataset
class WISeREPDataset(Dataset):
    """
    Loads spectra from given a list of filenames,
    applies preprocessing, and returns (flux_tensor, label_index).
    """

    def __init__(
        self,
        filenames: List[str],
        spectra_dir: Path,
        metadata: Dict[str, Dict[str, str]],
        target_length: int = TARGET_LENGTH,
        has_redshift: bool = True,
    ):
        self.spectra_dir = spectra_dir
        self.target_length = target_length
        self.has_redshift = has_redshift

        # only keep files that have a valid label and redshift
        # samples: (filename, label_idx, redshift)
        self.samples: List[Tuple[str, int, float]] = []
        skipped = 0
        for fname in filenames:
            meta = metadata.get(fname)
            if meta is None:
                skipped += 1
                continue
            label = normalize_label(meta["type"])
            if label is None:
                skipped += 1
                continue
            # Redshift, default 0.0 on parse failure
            z_raw = meta.get("redshift", "0").strip()
            try:
                z_val = float(z_raw) if z_raw else 0.0
            except ValueError:
                z_val = 0.0
            self.samples.append((fname, CLASS_TO_IDX[label], z_val))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        fname, label_idx, z_val = self.samples[idx]
        filepath = self.spectra_dir / fname
        result = load_spectrum(filepath)
        if result is None:
            return None
        wave, flux = result
        processed = preprocess_spectrum(wave, flux, z_val if self.has_redshift else None, self.target_length)
        if processed is None:
            return None
        processed = np.concatenate([processed, [z_val if self.has_redshift else 0.0]])
        return torch.from_numpy(processed.astype(np.float32)), label_idx


def collate_skip_none(batch):
    """Custom collate that drops None samples (failed to load/process)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


# 1D CNN model: 3 conv layers -> 2 linear layers
class DashCNN1D(nn.Module):
    """
    1D CNN for supernova spectrum classification.
    3 convolutional blocks (Conv1d + BatchNorm + ReLU + MaxPool)
    followed by 2 fully-connected layers.
    """

    def __init__(self, input_length: int = TARGET_LENGTH, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # Compute flattened size after convolutions
        conv_out_length = input_length // (4 * 4 * 4)
        self.flat_size = 128 * conv_out_length

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_length)
        x = x.unsqueeze(1)  # (batch, 1, input_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)  # (batch, flat_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float], Dict[str, float], List[List[int]]]:
    """
    Evaluate model on a DataLoader.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    # confusion matrix: rows = true, cols = pred
    cm = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            for t, p in zip(y.view(-1), preds.view(-1)):
                t_i = int(t.item())
                p_i = int(p.item())
                if 0 <= t_i < NUM_CLASSES and 0 <= p_i < NUM_CLASSES:
                    cm[t_i][p_i] += 1
                    per_class_total[t_i] += 1
                    if t_i == p_i:
                        per_class_correct[t_i] += 1

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    per_class_acc: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}

    # compute per-class precision/recall/F1 from confusion matrix
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(NUM_CLASSES)) - tp
        fp = sum(cm[i][c] for i in range(NUM_CLASSES)) - tp
        tn = total - tp - fp - fn

        acc_c = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_acc[name] = acc_c
        per_class_f1[name] = f1

    # printing summary
    print("\nValidation metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # confusion matrix
    print("\n  Confusion matrix (rows = true, cols = pred):")
    header = " " * 10 + "".join(f"{name:>10}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:>10d}" for val in row)
        print(f"{CLASS_NAMES[i]:>10}{row_str}")

    # per-class metrics
    print("\n  Per-class metrics:")
    print(f"{'Class':>10}  {'Acc':>6}  {'F1':>6}")
    for name in CLASS_NAMES:
        acc_c = per_class_acc.get(name, 0.0)
        f1_c = per_class_f1.get(name, 0.0)
        print(f"{name:>10}  {acc_c:6.3f}  {f1_c:6.3f}")

    return avg_loss, accuracy, per_class_acc, per_class_f1, cm


def build_performance_json(
    best_epoch: int,
    val_loss: float,
    val_acc: float,
    per_class_acc: Dict[str, float],
    per_class_f1: Dict[str, float],
    cm: List[List[int]],
) -> Dict:
    """Build model_performance.json structure: cumulative and per-class, counts and percentages."""
    total_count = sum(cm[i][j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES))
    correct_count = sum(cm[c][c] for c in range(NUM_CLASSES))

    per_class = {}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        support = sum(cm[c][j] for j in range(NUM_CLASSES))
        correct_c = cm[c][c]
        tp = correct_c
        fn = support - tp
        fp = sum(cm[i][c] for i in range(NUM_CLASSES)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc_frac = per_class_acc.get(name, 0.0)
        f1_frac = per_class_f1.get(name, 0.0)
        per_class[name] = {
            "count": support,
            "correct_count": correct_c,
            "accuracy_pct": round(100.0 * acc_frac, 2),
            "precision_pct": round(100.0 * precision, 2),
            "recall_pct": round(100.0 * recall, 2),
            "f1_pct": round(100.0 * f1_frac, 2),
        }

    return {
        "best_epoch": best_epoch,
        "cumulative": {
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy_pct": round(100.0 * val_acc, 2),
            "loss": round(val_loss, 6),
        },
        "per_class": per_class,
        "confusion_matrix_raw": cm,
        "confusion_matrix_labels": CLASS_NAMES,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int,
    val_every: int,
    out_dir: Optional[Path] = None,
) -> Path:
    """Training loop with early stopping. Returns path to best saved model."""
    from tqdm import tqdm

    out = (out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    best_model_path = out / "model.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            ncols=100,
        )
        for batch in pbar:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_total += bs
            pbar.set_postfix(
                loss=f"{epoch_loss / max(epoch_total, 1):.4f}",
                acc=f"{epoch_correct / max(epoch_total, 1):.3f}",
            )

        # val
        if epoch % val_every == 0:
            val_loss, val_acc, per_class, per_class_f1, cm = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                perf = build_performance_json(
                    best_epoch, val_loss, val_acc, per_class, per_class_f1, cm
                )
                with open(out / "model_performance.json", "w") as f:
                    json.dump(perf, f, indent=2)
                print(f"New best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement}/{patience} epochs "
                    f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})"
                )

            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
                )
                break

    print(f"\nTraining complete. Best model at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    return best_model_path


# Main

def _run_single_fold(
    train_filenames: List[str],
    val_filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
    device: torch.device,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, float, float]:
    """Build datasets/loaders, train one fold. Returns (best_model_path, best_val_loss, best_val_acc)."""
    train_ds = WISeREPDataset(train_filenames, SPECTRA_DIR, metadata, has_redshift=HAS_REDSHIFT)
    val_ds = WISeREPDataset(val_filenames, SPECTRA_DIR, metadata, has_redshift=HAS_REDSHIFT)

    train_counts = [0] * NUM_CLASSES
    for _, label_idx, _ in train_ds.samples:
        train_counts[label_idx] += 1
    total_train = sum(train_counts)
    class_weights_list: List[float] = []
    for i, cnt in enumerate(train_counts):
        w = (total_train / cnt) if cnt > 0 else 0.0
        class_weights_list.append(w)
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_skip_none,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_skip_none,
        pin_memory=(device.type == "cuda"),
    )

    model = DashCNN1D(input_length=TARGET_LENGTH, num_classes=NUM_CLASSES).to(device)
    best_path = train(
        model, train_loader, val_loader, device, class_weights,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=EARLY_STOP_PATIENCE,
        val_every=VAL_EVERY,
        out_dir=out_dir,
    )
    # Reload best and get its val metrics
    model.load_state_dict(torch.load(best_path, map_location=device))
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
    return best_path, val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train DASH 1D CNN on WISeREP.")
    parser.add_argument(
        "--k-fold",
        type=int,
        default=1,
        metavar="K",
        help="Number of folds for cross-validation (1 = single train/val split, no CV)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified splits and k-fold",
    )
    args = parser.parse_args()
    k_fold = args.k_fold
    seed = args.seed
    if k_fold < 1:
        raise SystemExit("--k-fold must be >= 1")

    print("Starting training for DASH 1D CNN Model on Wiserep dataset")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"K-fold: {k_fold}  Seed: {seed}")

    if not SPLITS_JSON.exists():
        raise SystemExit(f"Splits file not found: {SPLITS_JSON}")
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    print(
        f"Splits: train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}"
    )

    print(f"Loading metadata from {METADATA_CSV}")
    metadata = load_metadata(METADATA_CSV)
    print(f"{len(metadata)} filename -> (type, redshift) entries")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_mapping = {name: idx for name, idx in CLASS_TO_IDX.items()}
    with open(OUT_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    if k_fold == 1:
        # Single run: use val if present, else stratified 90/10 of train
        if splits["val"]:
            train_filenames, val_filenames = splits["train"], splits["val"]
            print("\nUsing train/val from splits file")
        else:
            train_filenames, val_filenames = stratified_train_val_split(
                splits["train"], metadata, val_frac=0.1, seed=seed
            )
            print(f"\nStratified 90/10 train/val from train split: {len(train_filenames)} train, {len(val_filenames)} val")

        best_path, best_val_loss, best_val_acc = _run_single_fold(
            train_filenames, val_filenames, metadata, device, out_dir=None
        )
        # Class weights from train_filenames for config
        train_counts = [0] * NUM_CLASSES
        for fname in train_filenames:
            meta = metadata.get(fname)
            if meta is None:
                continue
            label = normalize_label(meta["type"])
            if label is None:
                continue
            train_counts[CLASS_TO_IDX[label]] += 1
        total_train = sum(train_counts)
        class_weights_list = [(total_train / c) if c > 0 else 0.0 for c in train_counts]

        training_config_path = OUT_DIR / "training_config.json"
        if training_config_path.exists():
            with open(training_config_path) as f:
                training_config = json.load(f)
        else:
            training_config = {}
        training_config["run_id"] = RUN_ID
        training_config["has_redshift"] = HAS_REDSHIFT
        training_config["target_length"] = TARGET_LENGTH
        training_config["wave_min"] = WAVE_MIN
        training_config["wave_max"] = WAVE_MAX
        training_config["num_classes"] = NUM_CLASSES
        training_config["class_names"] = CLASS_NAMES
        training_config["epochs"] = EPOCHS
        training_config["batch_size"] = BATCH_SIZE
        training_config["lr"] = LEARNING_RATE
        training_config["patience"] = EARLY_STOP_PATIENCE
        training_config["class_weights"] = class_weights_list
        training_config["splits_file"] = str(SPLITS_JSON)
        training_config["k_fold"] = 1
        training_config["seed"] = seed
        with open(training_config_path, "w") as f:
            json.dump(training_config, f, indent=2)

        print(f"\nSaved: {best_path}")
        print(f"  Config: {OUT_DIR / 'training_config.json'}")
        print(f"  Performance: {OUT_DIR / 'model_performance.json'}")
        return

    # K-fold cross-validation
    folds = stratified_k_folds(splits["train"], metadata, k_fold, seed=seed)
    print(f"\nStratified {k_fold}-fold CV over train set")
    val_losses: List[float] = []
    val_accs: List[float] = []

    for fold in range(k_fold):
        train_f, val_f = folds[fold]
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k_fold}  train={len(train_f)}  val={len(val_f)}")
        print("="*60)
        fold_dir = OUT_DIR / f"fold_{fold}"
        best_path, val_loss, val_acc = _run_single_fold(
            train_f, val_f, metadata, device, out_dir=fold_dir
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    mean_loss = float(np.mean(val_losses))
    std_loss = float(np.std(val_losses))
    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs))
    print(f"\n{'='*60}")
    print(f"K-fold summary (k={k_fold})")
    print(f"  Val loss:  {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Val acc:   {mean_acc:.4f} ± {std_acc:.4f}")
    print("="*60)

    training_config = {
        "run_id": RUN_ID,
        "has_redshift": HAS_REDSHIFT,
        "target_length": TARGET_LENGTH,
        "wave_min": WAVE_MIN,
        "wave_max": WAVE_MAX,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "patience": EARLY_STOP_PATIENCE,
        "splits_file": str(SPLITS_JSON),
        "k_fold": k_fold,
        "seed": seed,
        "cv_val_loss_mean": mean_loss,
        "cv_val_loss_std": std_loss,
        "cv_val_acc_mean": mean_acc,
        "cv_val_acc_std": std_acc,
    }
    with open(OUT_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"\nModels saved under {OUT_DIR}/fold_0 .. fold_{k_fold - 1}")
    print(f"Config: {OUT_DIR / 'training_config.json'}")
    print(f"Eval a fold: python prod_backend/scripts/dash_eval_confusion_matrices.py --run-id {RUN_ID}  # (eval script may need fold support)")
    print("Done.")


if __name__ == "__main__":
    main()
