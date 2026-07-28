from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from TwinsClassifier_Wiserep import (
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    EPOCHS as DEFAULT_EPOCHS,
    LABEL_COLUMN,
    LABEL_MAP,
    LR as DEFAULT_LR,
    NUM_CLASSES,
    WEIGHT_DECAY as DEFAULT_WEIGHT_DECAY,
    build_performance_json,
    evaluate,
    filter_indices_mapped,
    inverse_frequency_class_weights,
    report_split_label_counts,
    row_class_idx,
)
from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import set_seeds, to_device


WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = WISEREP_DIR.parent

# Train both redshift variants × seeds 0..9 (DASH-style iter_* folders).
USE_REDSHIFT_VALUES = [True, False]
SEED_VALUES = range(10)

HENNA_ROOT = _PROJECT_ROOT / "data" / "wiserep_henna"
TEST_ROOT = WISEREP_DIR / "Test"


def latent_dir_for(use_redshift: bool) -> pathlib.Path:
    # 1024d2 = deredshifted; nodered1024d2 = observed-frame
    return HENNA_ROOT / ("1024d2" if use_redshift else "nodered1024d2")


def out_dir_for(use_redshift: bool, seed: int) -> pathlib.Path:
    return TEST_ROOT / ("daep_comparison" if use_redshift else "daep_comparison_noz") / f"iter_{seed}"


# Training / split (latent MLP)
IAU_COLUMN = "IAU name"
DEVICE = "auto"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

BATCH_SIZE = DEFAULT_BATCH_SIZE
LR = DEFAULT_LR
WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY
EPOCHS = DEFAULT_EPOCHS
EARLY_STOPPING_PATIENCE = 10
DROPOUT = .25


class LatentDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, z: np.ndarray, indices: np.ndarray, label_col: str):
        self.meta = meta.reset_index(drop=True)
        self.z = z.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = int(self.indices[i])
        return {
            "idx": torch.tensor(row, dtype=torch.long),
            "z": torch.from_numpy(self.z[row]),
            "y": torch.tensor(row_class_idx(self.meta, row, self.label_col), dtype=torch.long),
        }


def collate_latent(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "idx": torch.stack([b["idx"] for b in batch], dim=0),
        "z": torch.stack([b["z"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
    }


class LatentClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_cls: int,
        head_hidden: int,
        head_dropout: float,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            # To match `TwinsClassifier_Wiserep.py`, uncomment the second hidden layer below.
            # nn.Linear(head_hidden, head_hidden),
            # nn.GELU(),
            # nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_cls),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_batch = batch["z"]
        return self.head(z_batch.reshape(z_batch.shape[0], -1))


def latent_checkpoint_payload(model: LatentClassifier, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "cfg": cfg,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "label_map": LABEL_MAP,
        "label_column": LABEL_COLUMN,
    }


def train_latent_classifier(
    model: LatentClassifier,
    tr_load: DataLoader,
    va_load: DataLoader,
    te_load: DataLoader,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: Dict[str, Any],
    *,
    class_weights: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
) -> pathlib.Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_ckpt_path = out_dir / "classifier_best.pt"
    final_ckpt_path = out_dir / "classifier.pt"
    perf_path = out_dir / "model_performance.json"

    print(
        f"[train] epochs={epochs}  early_stop_patience={early_stopping_patience}  "
        f"lr={lr}  wd={weight_decay}  out_dir={out_dir}",
        flush=True,
    )

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    loss_by_epoch: List[List[Any]] = []
    best_metric = (float("inf"), -1.0)
    best_epoch = 0
    best_val_loss = float("nan")
    best_val_acc = float("nan")
    best_val_cm: List[List[int]] | None = None
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tl, ta = [], []
        pbar = tqdm(tr_load, desc=f"Epoch {ep}/{epochs} train", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            logits = model(batch)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append(loss.item())
            ta.append((logits.argmax(-1) == y).float().mean().item())
            pbar.set_postfix(loss=f"{np.mean(tl):.4f}", acc=f"{np.mean(ta):.4f}")

        val_loss, val_acc, val_cm = evaluate(
            model,
            va_load,
            loss_fn,
            device,
            metrics_title="Validation metrics",
            print_summary=True,
        )
        test_loss, test_acc, _ = evaluate(
            model,
            te_load,
            loss_fn,
            device,
            metrics_title="Test metrics",
            print_summary=False,
        )

        train_loss_ep = _mean(tl)
        loss_by_epoch.append(
            [
                ep,
                float(train_loss_ep) if np.isfinite(train_loss_ep) else None,
                float(val_loss),
            ]
        )

        dt = time.perf_counter() - t_ep
        print(
            f"epoch {ep}/{epochs} done in {dt:.1f}s  "
            f"train loss {_mean(tl):.4f} acc {_mean(ta):.4f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  |  "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}",
            flush=True,
        )

        metric = (val_loss, -val_acc)
        if metric < best_metric:
            epochs_no_improve = 0
            best_metric = metric
            best_epoch = ep
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_val_cm = val_cm
            perf = build_performance_json(
                best_epoch=best_epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                cm=val_cm,
                loss_by_epoch=[list(r) for r in loss_by_epoch],
            )
            perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")

            ckpt = latent_checkpoint_payload(model, cfg)
            ckpt["best_epoch"] = best_epoch
            ckpt["val_loss"] = val_loss
            ckpt["val_acc"] = val_acc
            torch.save(ckpt, best_ckpt_path)
            print(
                f"[best] epoch={best_epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f} -> "
                f"{perf_path.name} + {best_ckpt_path.name}",
                flush=True,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[early stop] no val improvement for {early_stopping_patience} epochs "
                    f"(best epoch {best_epoch})",
                    flush=True,
                )
                break

    if best_val_cm is not None:
        perf = build_performance_json(
            best_epoch=best_epoch,
            val_loss=best_val_loss,
            val_acc=best_val_acc,
            cm=best_val_cm,
            loss_by_epoch=[list(r) for r in loss_by_epoch],
        )
        perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")
        print(
            f"[done] refreshed {perf_path.name} with {len(loss_by_epoch)} epochs of loss history",
            flush=True,
        )

    print("\n[saving] last-epoch checkpoint ...", flush=True)
    torch.save(latent_checkpoint_payload(model, cfg), final_ckpt_path)
    cfg_path = out_dir / "cfg_used.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("[done] wrote", final_ckpt_path.resolve(), flush=True)
    print("[done] wrote", cfg_path.resolve(), flush=True)
    return best_ckpt_path


def normalize_latent_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Map Henna meta_universal columns onto IAU name / Obj. Type aliases."""
    out = meta.copy()
    if IAU_COLUMN not in out.columns and "sn_name_used" in out.columns:
        out[IAU_COLUMN] = out["sn_name_used"]
    if LABEL_COLUMN not in out.columns and "raw_type" in out.columns:
        out[LABEL_COLUMN] = out["raw_type"]
    return out.reset_index(drop=True)


def resolve_latent_npz(latent_dir: pathlib.Path) -> pathlib.Path:
    best = latent_dir / "latent_raw_z_best.npz"
    if best.is_file():
        return best
    fallback = latent_dir / "latent_raw_z.npz"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No latent_raw_z*.npz under {latent_dir}")


def resolve_latent_meta_csv(latent_dir: pathlib.Path) -> pathlib.Path:
    meta_path = latent_dir / "meta_universal.csv"
    if meta_path.is_file():
        return meta_path
    raise FileNotFoundError(f"Missing meta_universal.csv under {latent_dir}")


def load_latent_and_meta(
    latent_dir: pathlib.Path,
) -> tuple[pd.DataFrame, np.ndarray, pathlib.Path, pathlib.Path]:
    latent_npz = resolve_latent_npz(latent_dir)
    meta_csv = resolve_latent_meta_csv(latent_dir)

    latent = np.load(latent_npz)
    for key, array in latent.items():
        print(f"key: {key} has shape: {array.shape}", flush=True)

    z = latent["z"].astype(np.float32)
    meta = normalize_latent_meta(pd.read_csv(meta_csv, low_memory=False))

    if z.ndim != 3:
        raise ValueError(f"Expected latent z with shape (N, L, D), got {z.shape}")
    if len(meta) != z.shape[0]:
        raise ValueError(f"metadata rows ({len(meta)}) != latent rows ({z.shape[0]})")
    if IAU_COLUMN not in meta.columns or LABEL_COLUMN not in meta.columns:
        raise KeyError(
            f"Need {IAU_COLUMN!r} and {LABEL_COLUMN!r} after normalize; have {list(meta.columns)[:20]}"
        )

    print(f"[load] latent_dir={latent_dir}", flush=True)
    print(f"[load] latent_npz={latent_npz.name}  meta={meta_csv.name}", flush=True)
    print(
        f"meta rows={len(meta)}  z shape={z.shape}  flattened dim={int(np.prod(z.shape[1:]))}",
        flush=True,
    )
    print(meta[[IAU_COLUMN, LABEL_COLUMN]].head(), flush=True)
    print(f"classes: {CLASS_TO_IDX}", flush=True)
    print(f"label map covers {len(LABEL_MAP)} raw labels", flush=True)
    return meta, z, latent_npz, meta_csv


def split_and_filter(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
        IAU_SPLIT_SEED,
    )

    iau = meta[IAU_COLUMN].astype(str).str.strip()
    print(
        "[split] IAU (unique): "
        f"train={pd.unique(iau.iloc[train_idx]).size}  "
        f"val={pd.unique(iau.iloc[val_idx]).size}  "
        f"test={pd.unique(iau.iloc[test_idx]).size}  |  "
        f"spectra: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  "
        f"(iau_split_seed={IAU_SPLIT_SEED})",
        flush=True,
    )

    train_idx = filter_indices_mapped(meta, train_idx, LABEL_COLUMN)
    val_idx = filter_indices_mapped(meta, val_idx, LABEL_COLUMN)
    test_idx = filter_indices_mapped(meta, test_idx, LABEL_COLUMN)

    report_split_label_counts(meta, train_idx, LABEL_COLUMN, "train")
    report_split_label_counts(meta, val_idx, LABEL_COLUMN, "val")
    report_split_label_counts(meta, test_idx, LABEL_COLUMN, "test")
    return train_idx, val_idx, test_idx


def build_cfg(
    z: np.ndarray,
    *,
    latent_npz: pathlib.Path,
    meta_csv: pathlib.Path,
    cfg_json: pathlib.Path,
    seed: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if cfg_json.is_file():
        cfg.update(json.loads(cfg_json.read_text(encoding="utf-8")))
        print(f"[cfg] loaded {cfg_json}", flush=True)

    cfg.update(
        {
            "latent_npz": str(latent_npz),
            "meta_csv": str(meta_csv),
            "latent_shape": list(z.shape),
            "classifier_kind": "latent_flatten_mlp",
            "seed": int(seed),
            "run_id": f"iter_{seed}",
            "iau_split_seed": IAU_SPLIT_SEED,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        }
    )
    return cfg


def run_one(
    use_redshift: bool,
    seed: int,
    *,
    meta: pd.DataFrame,
    z: np.ndarray,
    latent_npz: pathlib.Path,
    meta_csv: pathlib.Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg_json: pathlib.Path,
) -> None:
    print(f"=== train_latent start | USE_REDSHIFT={use_redshift} SEED={seed} ===", flush=True)
    set_seeds(seed)

    out_dir = out_dir_for(use_redshift, seed).resolve()

    train_ds = LatentDataset(meta, z, train_idx, LABEL_COLUMN)
    val_ds = LatentDataset(meta, z, val_idx, LABEL_COLUMN)
    test_ds = LatentDataset(meta, z, test_idx, LABEL_COLUMN)

    train_load = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_latent,
        num_workers=0,
    )
    val_load = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_latent,
        num_workers=0,
    )
    test_load = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_latent,
        num_workers=0,
    )

    print(f"dataset rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    print(f"[out] {out_dir}", flush=True)

    cfg = build_cfg(
        z,
        latent_npz=latent_npz,
        meta_csv=meta_csv,
        cfg_json=cfg_json,
        seed=seed,
    )
    embed_dim = int(np.prod(z.shape[1:]))
    head_hidden = int(cfg.get("ff_dim", 512))
    head_dropout = float(cfg.get("dropout", DROPOUT))

    device = device_from_str(DEVICE)
    model = LatentClassifier(
        embed_dim=embed_dim,
        n_cls=NUM_CLASSES,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] device={device} embed_dim={embed_dim} hidden={head_hidden} dropout={head_dropout}",
        flush=True,
    )
    print(f"[model] parameters total={n_params:,}", flush=True)

    ce_weights = inverse_frequency_class_weights(meta, train_idx, LABEL_COLUMN).to(device)
    print(
        "[train] class weights:",
        {name: float(ce_weights[i].detach().cpu()) for i, name in enumerate(CLASS_NAMES)},
        flush=True,
    )

    best_path = train_latent_classifier(
        model,
        train_load,
        val_load,
        test_load,
        device,
        out_dir,
        cfg,
        class_weights=ce_weights,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    print("[done] best checkpoint:", best_path.resolve(), flush=True)
    print(f"=== train_latent finished | USE_REDSHIFT={use_redshift} SEED={seed} ===", flush=True)


def main() -> None:
    for use_redshift in USE_REDSHIFT_VALUES:
        latent_dir = latent_dir_for(use_redshift).resolve()
        cfg_json = latent_dir / "cfg_used.json"
        meta, z, latent_npz, meta_csv = load_latent_and_meta(latent_dir)
        train_idx, val_idx, test_idx = split_and_filter(meta)

        for seed in SEED_VALUES:
            run_one(
                use_redshift,
                seed,
                meta=meta,
                z=z,
                latent_npz=latent_npz,
                meta_csv=meta_csv,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                cfg_json=cfg_json,
            )


if __name__ == "__main__":
    main()
