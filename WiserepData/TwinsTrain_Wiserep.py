# TwinsTrain_Wiserep.py

import argparse
import pathlib
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from TwinsModel_Wiserep import repo_import_setup, build_daep, default_cfg, device_from_str

# Paths — CLI defaults (independent of TwinsClassifier)
_WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _WISEREP_DIR.parent
TEST_ROOT = _WISEREP_DIR / "Test"
HENNA_ROOT = _PROJECT_ROOT / "data" / "wiserep_henna"
USE_REDSHIFT_CORRECTED_DATA = True
DATA_DIR = HENNA_ROOT / ("deredshifted" if USE_REDSHIFT_CORRECTED_DATA else "Noderedshift")


class WiserepDataset(Dataset):
    def __init__(self, meta_df, wave, flux, mask_valid):
        self.meta       = meta_df.reset_index(drop=True)
        self.wave       = wave.astype(np.float32)
        self.flux       = flux.astype(np.float32)
        self.mask_valid = mask_valid.astype(bool)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        phase = 0.0
        for col in ("Phase (days)", "phase_used", "phase_rest", "phase_obs"):
            if col in self.meta.columns:
                phase = self.meta.loc[i, col]
                break
        try:
            phase = float(phase)
            if not np.isfinite(phase):
                phase = 0.0
        except Exception:
            phase = 0.0

        valid_mask       = self.mask_valid[i]
        daep_padding_mask = ~valid_mask          # daep convention: True = ignore

        return {
            "idx":        i,
            "flux":       torch.from_numpy(self.flux[i]),
            "wavelength": torch.from_numpy(self.wave),
            "phase":      torch.tensor(phase, dtype=torch.float32),
            "mask":       torch.from_numpy(daep_padding_mask).bool(),
        }


def collate_fixed(batch):
    return {
        "idx":        torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "flux":       torch.stack([b["flux"]       for b in batch], dim=0),
        "mask":       torch.stack([b["mask"]       for b in batch], dim=0).bool(),
        "phase":      torch.stack([b["phase"]      for b in batch], dim=0).view(-1),
        "wavelength": torch.stack([b["wavelength"] for b in batch], dim=0),
    }


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "manual_seed"):
            mps.manual_seed(seed)


def load_data(data_dir: pathlib.Path):
    data_dir = pathlib.Path(data_dir)
    # Henna layout: meta.csv / flux.npy / mask.npy / wavelength.npy
    # Legacy layout: wiserep_metadata_processed.csv / wiserep_*.npy
    meta_path = data_dir / "wiserep_metadata_processed.csv"
    if not meta_path.is_file():
        meta_path = data_dir / "meta.csv"
    flux_path = data_dir / "flux.npy"
    if not flux_path.is_file():
        flux_path = data_dir / "wiserep_flux.npy"
    mask_path = data_dir / "mask.npy"
    if not mask_path.is_file():
        mask_path = data_dir / "wiserep_mask.npy"
    wave_path = data_dir / "wavelength.npy"
    if not wave_path.is_file():
        wave_path = data_dir / "wiserep_wavelength.npy"

    meta = pd.read_csv(meta_path, low_memory=False)
    flux = np.load(flux_path).astype(np.float32)
    mask = np.load(mask_path).astype(bool)
    wavelength = np.load(wave_path).astype(np.float32)

    assert len(meta) == flux.shape[0], (
        f"metadata rows ({len(meta)}) != flux rows ({flux.shape[0]})"
    )
    assert flux.shape == mask.shape, (
        f"flux shape {flux.shape} != mask shape {mask.shape}"
    )
    assert flux.shape[1] == len(wavelength), (
        f"flux bins ({flux.shape[1]}) != wavelength bins ({len(wavelength)})"
    )
    return meta, wavelength, flux, mask


def save_history(outdir, epochs, train_hist, val_hist):
    out_npz = outdir / "loss_history.npz"
    np.savez_compressed(
        out_npz,
        epoch=np.array(epochs, dtype=np.int32),
        train=np.array(train_hist, dtype=np.float32),
        val=np.array(val_hist, dtype=np.float32),
    )

    out_csv = outdir / "loss_history.csv"
    pd.DataFrame({"epoch": epochs, "train": train_hist, "val": val_hist}).to_csv(
        out_csv, index=False
    )

    out_png = outdir / "loss_curves.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(epochs, train_hist, label="train")
        plt.plot(epochs, val_hist,   label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("DAEP training curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
    except Exception as e:
        print(f"[warn] could not write {out_png.name}: {e}")

    return str(out_npz), str(out_csv), str(out_png)


def rebuild_best_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_daep(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return ckpt, model


def print_dataset_diagnostics(meta, flux, mask_valid):
    valid_frac   = mask_valid.mean(axis=1)
    valid_counts = mask_valid.sum(axis=1)

    print("dataset diagnostics")
    print("  N spectra          =", flux.shape[0])
    print("  n_wave bins        =", flux.shape[1])
    print("  flux shape         =", tuple(flux.shape))
    print("  mask_valid shape   =", tuple(mask_valid.shape))
    print("  valid fraction min =", float(np.min(valid_frac)))
    print("  valid fraction med =", float(np.median(valid_frac)))
    print("  valid fraction max =", float(np.max(valid_frac)))
    print("  valid bins min     =", int(np.min(valid_counts)))
    print("  valid bins med     =", int(np.median(valid_counts)))
    print("  valid bins max     =", int(np.max(valid_counts)))

    if "Phase (days)" in meta.columns:
        phase = pd.to_numeric(meta["Phase (days)"], errors="coerce")
        print("  phase finite count =", int(np.isfinite(phase).sum()))
        if np.isfinite(phase).any():
            print("  phase min/med/max  =",
                  float(np.nanmin(phase)),
                  float(np.nanmedian(phase)),
                  float(np.nanmax(phase)))

    print("  finite flux frac   =", float(np.isfinite(flux).mean()))
    print("  any NaN in flux    =", bool(np.isnan(flux).any()))
    print("  any inf in flux    =", bool(np.isinf(flux).any()))
    print("  note: stored mask  True=valid bin")
    print("  note: DAEP batch mask True=padding/ignore bin")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo",     type=str, required=True)
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--outdir",   type=str, default=str(DATA_DIR / "Output"))
    ap.add_argument("--device",   type=str, default="auto")

    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--val_frac",    type=float, default=0.1)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--latent_batch",type=int,   default=128)

    ap.add_argument("--max_epochs", type=int,   default=500)
    ap.add_argument("--min_epochs", type=int,   default=1)
    ap.add_argument("--min_delta",  type=float, default=1e-5)

    ap.add_argument("--use_patience", action="store_true")
    ap.add_argument("--patience",     type=int, default=20)

    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--export_only", action="store_true")

    ap.add_argument("--subset_n",         type=int,   default=None)
    ap.add_argument("--model_dim",        type=int,   default=None)
    ap.add_argument("--num_layers",       type=int,   default=None)
    ap.add_argument("--num_heads",        type=int,   default=None)
    ap.add_argument("--ff_dim",           type=int,   default=None)
    ap.add_argument("--dropout",          type=float, default=None)
    ap.add_argument("--bottleneck_length",type=int,   default=None)
    ap.add_argument("--bottleneck_dim",   type=int,   default=None)

    args = ap.parse_args()

    repo_import_setup(args.repo)
    set_seeds(args.seed)

    data_dir = pathlib.Path(args.data_dir).resolve()
    outdir   = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path      = outdir / "best_ckpt.pt"
    out_latent_raw = outdir / "latent_raw_z.npz"

    meta, wave, flux, mask_valid = load_data(data_dir)
    print_dataset_diagnostics(meta, flux, mask_valid)

    cfg = default_cfg()

    def maybe_set(key, val):
        if val is None:
            return
        if key not in cfg:
            raise KeyError(f"cfg has no key '{key}'. Available: {list(cfg.keys())}")
        cfg[key] = val

    maybe_set("model_dim",         args.model_dim)
    maybe_set("num_layers",        args.num_layers)
    maybe_set("num_heads",         args.num_heads)
    maybe_set("ff_dim",            args.ff_dim)
    maybe_set("dropout",           args.dropout)
    maybe_set("bottleneck_length", args.bottleneck_length)
    maybe_set("bottleneck_dim",    args.bottleneck_dim)

    (outdir / "cfg_used.json").write_text(json.dumps(cfg, indent=2))

    device = device_from_str(args.device)
    print("device:", device)
    print("cuda_available:", torch.cuda.is_available())
    if device.type == "cuda" and torch.cuda.is_available():
        print("cuda_device_name:", torch.cuda.get_device_name(0))

    ds_full = WiserepDataset(meta, wave, flux, mask_valid)

    if args.subset_n is not None:
        subset_n = min(int(args.subset_n), len(ds_full))
        ds_base  = Subset(ds_full, np.arange(subset_n))
        print(f"using subset_n={subset_n} for debugging")
    else:
        ds_base = ds_full
        print("using full dataset")

    if not args.export_only:
        n_val = max(1, int(len(ds_base) * args.val_frac))
        n_tr  = len(ds_base) - n_val

        tr_ds, va_ds = random_split(
            ds_base, [n_tr, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                               collate_fn=collate_fixed, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                               collate_fn=collate_fixed, num_workers=0)

        model = build_daep(cfg).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        use_amp = bool(args.amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val   = float("inf")
        best_epoch = 0
        bad_epochs = 0
        epochs_hist, train_hist, val_hist = [], [], []

        print(f"max_epochs={args.max_epochs}  use_patience={args.use_patience}  amp={use_amp}")

        for ep in range(1, args.max_epochs + 1):
            model.train(True)
            tr_losses = []
            for batch in tr_loader:
                batch = to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss = model(batch)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
#                else:
#                    loss = model(batch)
#                    loss.backward()
#                    opt.step()
                else:
                    loss = model(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                tr_losses.append(float(loss.detach().cpu()))

            model.train(False)
            va_losses = []
            with torch.no_grad():
                for batch in va_loader:
                    batch = to_device(batch, device)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            loss = model(batch)
                    else:
                        loss = model(batch)
                    va_losses.append(float(loss.detach().cpu()))

            tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
            va = float(np.mean(va_losses)) if va_losses else float("nan")

            epochs_hist.append(ep)
            train_hist.append(tr)
            val_hist.append(va)

            improved = np.isfinite(va) and (va < (best_val - float(args.min_delta)))

            print(f"epoch {ep:03d}  train {tr:.6f}  val {va:.6f}  "
                  f"best_val {best_val:.6f} (ep {best_epoch})  "
                  f"bad_epochs={bad_epochs}  improved={improved}")

            npz_path, csv_path, png_path = save_history(outdir, epochs_hist, train_hist, val_hist)

            if improved:
                best_val   = va
                best_epoch = ep
                bad_epochs = 0
                torch.save({
                    "model_kind": "daep",
                    "cfg":        cfg,
                    "state_dict": model.state_dict(),
                    "best_val":   float(best_val),
                    "best_epoch": int(best_epoch),
                    "data_dir":   str(data_dir),
                    "subset_n":   args.subset_n,
                    "history": {
                        "epoch":    epochs_hist,
                        "train":    train_hist,
                        "val":      val_hist,
                        "loss_npz": str(npz_path),
                        "loss_csv": str(csv_path),
                        "loss_png": str(png_path),
                    },
                }, ckpt_path)
                print(f"[saved best] {ckpt_path}  (best_val={best_val:.6f} @ epoch {best_epoch})")
            else:
                bad_epochs += 1

            if args.use_patience and ep >= args.min_epochs and bad_epochs >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs. "
                      f"Best epoch {best_epoch} (val={best_val:.6f}).")
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt, model = rebuild_best_model(ckpt_path, device)
    print(f"[loaded best] epoch={ckpt.get('best_epoch','NA')}  val={ckpt.get('best_val','NA')}")

    full_loader = DataLoader(ds_full, batch_size=args.latent_batch, shuffle=False,
                             collate_fn=collate_fixed, num_workers=0)

    z_raw_out = None
    with torch.no_grad():
        for batch in full_loader:
            idx   = batch["idx"].cpu().numpy()
            batch = to_device(batch, device)
            z_raw = model.encoder.encode_raw(batch)
            z_raw = z_raw.detach().float().cpu().numpy().astype(np.float32)
            if z_raw_out is None:
                z_raw_out = np.zeros((len(ds_full),) + z_raw.shape[1:], dtype=np.float32)
            z_raw_out[idx] = z_raw

    np.savez_compressed(out_latent_raw, z=z_raw_out)
    print(str(out_latent_raw))
    print("raw encoder z.shape =", tuple(z_raw_out.shape))
    print("raw latent_dim =", int(np.prod(z_raw_out.shape[1:])))


if __name__ == "__main__":
    main()
