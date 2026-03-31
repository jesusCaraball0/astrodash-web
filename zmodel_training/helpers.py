from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import csv
import json
import torch
import random
import constants as const
import dash_retrain
from torch.utils.data import DataLoader

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
            spectrum = const._FILE_REPO.get_from_file(wrapper)
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
    canonical = const.LABEL_MAP.get(raw)
    if canonical is None:
        # handles small formatting differences
        canonical = const.LABEL_MAP.get(raw.replace(" ", ""))
    if canonical is None:
        return None
    if canonical in const.CLASS_TO_IDX:
        return canonical
    return None

def preprocess_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    redshift: Optional[float] = None,
    target_length: int = const.TARGET_LENGTH 
) -> Optional[np.ndarray]:
    """
    Preprocess spectrum using backend original DASH preprocessing.

    This uses DashSpectrumProcessor(w0=3500, w1=10000, nw=1024)
    with the given redshift. 
    returns: processed flux array | None if processing/validation fails
    """

    try:
        if redshift is not None:
            processed_flux, _, _, _ = const._PROCESSOR.process(
                wave, flux, float(redshift)
            )
        else:
            processed_flux, _, _ = const._PROCESSOR.process_no_redshift(
                wave, flux
            )
            
    except Exception:
        return None

    return processed_flux.astype(np.float32)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except Exception:
        return {} if default is None else default


def require(path: Path, name: str) -> None:
    if not path.exists():
        raise SystemExit(f"{name} not found: {path}")

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader(
    filenames: List[str],
    metadata,
    has_redshift: bool,
    device: torch.device,
    shuffle: bool = False,
    batch_size: int = 64,
) -> DataLoader:
    ds = dash_retrain.WISeREPDataset(
        filenames,
        const.SPECTRA_DIR,
        metadata,
        target_length=const.TARGET_LENGTH,
        has_redshift=has_redshift,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dash_retrain.collate_skip_none,
        pin_memory=(device.type == "cuda"),
    )

def compute_class_weights_from_filenames(
    filenames: List[str],
    metadata: Dict[str, Dict[str, str]],
) -> torch.Tensor:
    """Compute inverse-frequency class weights from filenames + metadata (no spectrum loading)."""
    counts = [0] * const.NUM_CLASSES
    for fname in filenames:
        label = normalize_label((metadata.get(fname) or {}).get("type", ""))
        if label is not None and label in const.CLASS_TO_IDX:
            counts[const.CLASS_TO_IDX[label]] += 1
    total = sum(counts)
    weights = [(total / c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Compute inverse-frequency class weights from training DataLoader (slow: loads all spectra)."""
    counts = [0] * const.NUM_CLASSES
    for batch in train_loader:
        if batch is None:
            continue
        _, y = batch
        for label_idx in y.tolist():
            if 0 <= label_idx < const.NUM_CLASSES:
                counts[label_idx] += 1
    total = sum(counts)
    weights = [(total / c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)