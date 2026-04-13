#!/usr/bin/env python
"""
Retrain a DASH-style 1D CNN on WISeREP spectra.

Default: ASCII spectra under data/wiserep + wiserep_splits_by_iau_80_10_10.json.

New dataset: python dash_retrain.py --parquet-ruiyao
(requires wiserep_splits_train_val_test.json from create_trvaltest_from_trtest.py; same train/val/test keys as 80/10/10.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import constants as const
import logging

# Silence spectrum load errors so tqdm stays readable (skipped files are expected in WISeREP)
for _name in (
    "app.infrastructure.storage.file_spectrum_repository",
    "prod_backend.app.infrastructure.storage.file_spectrum_repository",
    "app.infrastructure.ml.data_processor",
    "prod_backend.app.infrastructure.ml.data_processor",
):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

import helpers as helpers

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
        target_length: int = const.TARGET_LENGTH,
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
            label = helpers.normalize_label(meta["type"])
            if label is None:
                skipped += 1
                continue
            # Redshift, default 0.0 on parse failure
            z_raw = meta.get("redshift", "0").strip()
            try:
                z_val = float(z_raw) if z_raw else 0.0
            except ValueError:
                z_val = 0.0
            self.samples.append((fname, const.CLASS_TO_IDX[label], z_val))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        fname, label_idx, z_val = self.samples[idx]
        filepath = self.spectra_dir / fname
        result = helpers.load_spectrum(filepath)
        if result is None:
            return None
        wave, flux = result
        processed = helpers.preprocess_spectrum(wave, flux, z_val if self.has_redshift else None, self.target_length)
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

    def __init__(self, input_length: int = const.TARGET_LENGTH, num_classes: int = const.NUM_CLASSES):
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
) -> Tuple[float, float, List[List[int]]]:
    """
    Evaluate model on a DataLoader.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = [0] * const.NUM_CLASSES
    per_class_total = [0] * const.NUM_CLASSES
    # confusion matrix: rows = true, cols = pred
    cm = [[0 for _ in range(const.NUM_CLASSES)] for _ in range(const.NUM_CLASSES)]

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
            else:
                x = x.to(device)
            y = y.to(device)
            bs = y.size(0)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += bs
            for t, p in zip(y.view(-1), preds.view(-1)):
                t_i = int(t.item())
                p_i = int(p.item())
                if 0 <= t_i < const.NUM_CLASSES and 0 <= p_i < const.NUM_CLASSES:
                    cm[t_i][p_i] += 1
                    per_class_total[t_i] += 1
                    if t_i == p_i:
                        per_class_correct[t_i] += 1

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    per_class_acc = {
        const.CLASS_NAMES[c]: (per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0)
        for c in range(const.NUM_CLASSES)
    }

    # printing summary
    print("\nValidation metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # confusion matrix
    print("\n  Confusion matrix (rows = true, cols = pred):")
    header = " " * 10 + "".join(f"{name:>10}" for name in const.CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:>10d}" for val in row)
        print(f"{const.CLASS_NAMES[i]:>10}{row_str}")

    # per-class metrics
    print("\n  Per-class metrics:")
    print(f"{'Class':>10}  {'Acc':>6}")
    for name in const.CLASS_NAMES:
        acc_c = per_class_acc.get(name, 0.0)
        print(f"{name:>10}  {acc_c:6.3f}")

    return avg_loss, accuracy, cm


def build_performance_json(
    best_epoch: int,
    val_loss: float,
    val_acc: float,
    cm: List[List[int]],
    *,
    loss_by_epoch: Optional[List[List[float]]] = None,
) -> Dict:
    """Build model_performance.json structure: cumulative and per-class, counts and percentages.

    loss_by_epoch: optional list of [epoch, train_loss, val_loss] triples (one row per validation step).
    """
    total_count = sum(cm[i][j] for i in range(const.NUM_CLASSES) for j in range(const.NUM_CLASSES))
    correct_count = sum(cm[c][c] for c in range(const.NUM_CLASSES))

    per_class = {}
    for c in range(const.NUM_CLASSES):
        name = const.CLASS_NAMES[c]
        support = sum(cm[c][j] for j in range(const.NUM_CLASSES))
        correct_c = cm[c][c]
        acc_pct = round(100.0 * (correct_c / support), 2) if support > 0 else 0.0
        per_class[name] = {
            "count": support,
            "correct_count": correct_c,
            "accuracy_pct": acc_pct,
        }

    out: Dict = {
        "best_epoch": best_epoch,
        "cumulative": {
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy_pct": round(100.0 * val_acc, 2),
            "loss": round(val_loss, 6),
        },
        "per_class": per_class,
        "confusion_matrix_raw": cm,
        "confusion_matrix_labels": const.CLASS_NAMES,
    }
    if loss_by_epoch is not None:
        out["loss_by_epoch"] = loss_by_epoch
    return out


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
    *,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
    plateau_mode: str = "min",
    plateau_min_lr: float = 0.0,
    plateau_threshold: float = 1e-4,
    plateau_threshold_mode: str = "rel",
    plateau_cooldown: int = 0,
) -> Path:
    """Training loop with early stopping. Resumes from checkpoint if present."""
    out = out_dir if out_dir is not None else const.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    best_model_path = out / "model.pth"
    checkpoint_path = out / "checkpoint.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=plateau_mode,
        factor=plateau_factor,
        patience=plateau_patience,
        min_lr=plateau_min_lr,
        threshold=plateau_threshold,
        threshold_mode=plateau_threshold_mode,
        cooldown=plateau_cooldown,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    loss_history: List[List[float]] = []
    last_perf: Optional[Dict] = None

    if checkpoint_path.exists():
        try:
            ck = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_val_loss = ck["best_val_loss"]
            best_epoch = ck["best_epoch"]
            epochs_without_improvement = ck.get("epochs_without_improvement", 0)
            loss_history = ck.get("loss_history", [])
            print(f"Resuming from epoch {start_epoch} (best so far: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
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

        train_loss = epoch_loss / max(epoch_total, 1)

        # val
        if epoch % val_every == 0:
            val_loss, val_acc, cm = evaluate(
                model, val_loader, criterion, device
            )
            loss_history.append(
                [float(epoch), round(train_loss, 6), round(val_loss, 6)]
            )
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                perf = build_performance_json(
                    best_epoch, val_loss, val_acc, cm, loss_by_epoch=loss_history
                )
                last_perf = perf
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

        # Save checkpoint so we can resume later
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "loss_history": loss_history,
            },
            checkpoint_path,
        )

    if last_perf is not None:
        last_perf["loss_by_epoch"] = loss_history
        with open(out / "model_performance.json", "w") as f:
            json.dump(last_perf, f, indent=2)

    print(f"\nTraining complete. Best model at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    return best_model_path


def main() -> None:
    """Normal training: one train/val split, one model."""

    parser = argparse.ArgumentParser(description="Retrain DASH 1D CNN on WISeREP spectra.")
    parser.add_argument(
        "--parquet-ruiyao",
        action="store_true",
        help="Use parquet_dataset (parquet + wiserep_split_ids.json) instead of ASCII + 80/10/10 JSON.",
    )
    args = parser.parse_args()

    print("Starting training for DASH 1D CNN Model on Wiserep dataset")

    device = helpers.get_device()
    print(f"Device: {device}")

    const.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(const.OUT_DIR / "class_mapping.json", "w") as f:
        json.dump({n: const.CLASS_TO_IDX[n] for n in const.CLASS_NAMES}, f, indent=2)

    parquet_training_config_extra: Optional[Dict] = None
    if args.parquet_ruiyao:
        import parquet_dataset as rpd

        df, metadata, train_ids, val_ids, test_ids = rpd.load_df_metadata_train_val_ids(const.SEED)
        test_n = len(test_ids)
        splits_file = str(rpd.RUIYAO_TRAIN_VAL_TEST_JSON)
        class_weights = helpers.compute_class_weights_from_filenames(train_ids, metadata)
        train_loader = DataLoader(
            rpd.ParquetSpectrumDataset(train_ids, df, has_redshift=const.HAS_REDSHIFT),
            batch_size=const.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_skip_none,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            rpd.ParquetSpectrumDataset(val_ids, df, has_redshift=const.HAS_REDSHIFT),
            batch_size=const.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_skip_none,
            pin_memory=(device.type == "cuda"),
        )
        parquet_training_config_extra = {
            "data_mode": "ruiyao_parquet",
            "parquet": str(rpd.RUIYAO_PARQUET),
            "metadata_cache": str(rpd.RUIYAO_METADATA_CACHE),
            "test_spectrum_ids_count": test_n,
        }
    else:
        splits = helpers.load_json(const.SPLITS_JSON_80_10_10)
        print(
            f"Splits: train={len(splits.get('train', []))}  val={len(splits.get('val', []))}  "
            f"test={len(splits.get('test', []))}"
        )
        print(f"Loading metadata from {const.METADATA_CSV}")
        metadata = helpers.load_metadata(const.METADATA_CSV)
        print(f"{len(metadata)} filename -> (type, redshift) entries")
        train_filenames = list(splits["train"])
        val_filenames = list(splits["val"])
        splits_file = str(const.SPLITS_JSON_80_10_10)
        class_weights = helpers.compute_class_weights_from_filenames(train_filenames, metadata)
        train_loader = helpers.make_loader(
            train_filenames, metadata, const.HAS_REDSHIFT, device,
            shuffle=True, batch_size=const.BATCH_SIZE,
        )
        val_loader = helpers.make_loader(
            val_filenames, metadata, const.HAS_REDSHIFT, device,
            batch_size=const.BATCH_SIZE,
        )

    model = DashCNN1D(input_length=const.TARGET_LENGTH, num_classes=const.NUM_CLASSES).to(device)
    best_path = train(
        model, train_loader, val_loader, device, class_weights,
        epochs=const.EPOCHS, lr=const.LEARNING_RATE, patience=const.EARLY_STOP_PATIENCE,
        val_every=const.VAL_EVERY, out_dir=None,
    )

    training_config = {
        "run_id": const.RUN_ID,
        "has_redshift": const.HAS_REDSHIFT,
        "target_length": const.TARGET_LENGTH,
        "wave_min": const.WAVE_MIN,
        "wave_max": const.WAVE_MAX,
        "num_classes": const.NUM_CLASSES,
        "class_names": const.CLASS_NAMES,
        "epochs": const.EPOCHS,
        "batch_size": const.BATCH_SIZE,
        "lr": const.LEARNING_RATE,
        "patience": const.EARLY_STOP_PATIENCE,
        "class_weights": class_weights.tolist(),
        "splits_file": splits_file,
        "k_fold": 1,
        "seed": const.SEED,
    }
    if parquet_training_config_extra is not None:
        training_config.update(parquet_training_config_extra)

    with open(const.OUT_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"\nSaved: {best_path}")
    print(f"  Config: {const.OUT_DIR / 'training_config.json'}")
    print(f"  Performance: {const.OUT_DIR / 'model_performance.json'}")


if __name__ == "__main__":
    main()
