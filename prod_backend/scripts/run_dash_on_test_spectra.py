#!/usr/bin/env python

import os
import sys
import re
from pathlib import Path

import numpy as np

# Paths setup (follow pattern from other scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Ensure prod_backend is on sys.path so imports work when run from project root
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from app.infrastructure.ml.classifiers.dash_classifier import DashClassifier


class SimpleSpectrum:
    """Minimal spectrum object compatible with DashClassifier.classify_sync."""

    def __init__(self, x, y, redshift: float = 0.0, file_name: str | None = None):
        self.x = list(x)
        self.y = list(y)
        self.redshift = float(redshift)
        self.file_name = file_name
        self.meta = {}
        self.calculate_rlap = False


def read_lnw_file(path: Path):
    """
    Minimal .lnw reader, mirroring the old backend logic:
    - Treat first two columns as wavelength and flux
    - Keep only 4000–9000 Å
    """
    spectrum = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                wavelength = float(parts[0])
                flux = float(parts[1])
            except ValueError:
                continue

            if 4000.0 <= wavelength <= 9000.0:
                spectrum.append((wavelength, flux))

    if not spectrum:
        raise ValueError(f"No valid spectrum data found in {path}")

    spectrum.sort(key=lambda x: x[0])
    wave = np.array([w for w, _ in spectrum], dtype=float)
    flux = np.array([f for _, f in spectrum], dtype=float)
    return wave, flux


def _is_numeric_line(line: str) -> bool:
    """True if line looks like two or more numbers (wave flux ...)."""
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def read_ascii_file(path: Path):
    """
    .ascii reader (WISeREP-style): skip # lines and leading non-numeric header,
    then load two-column wave/flux; filter to 4000–9000 Å.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        no_comment = [l for l in f if not l.strip().startswith("#")]
    start = None
    for i, line in enumerate(no_comment):
        if _is_numeric_line(line):
            start = i
            break
    if start is None:
        raise ValueError(f"No numeric data found in {path}")
    data = np.loadtxt(no_comment[start:], delimiter=None)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data shape in {path}")
    wave = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    mask = (wave >= 4000) & (wave <= 9000)
    if not np.any(mask):
        raise ValueError(f"No data in wavelength range 4000-9000 in {path}")
    return wave[mask], flux[mask]


def read_flm_file(path: Path):
    """
    .flm reader (WISeREP-style): three columns wavelength, flux, flux_err;
    use first two; filter to 4000–9000 Å.
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data shape in {path}")
    wave = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    mask = (wave >= 4000) & (wave <= 9000)
    if not np.any(mask):
        raise ValueError(f"No data in wavelength range 4000-9000 in {path}")
    return wave[mask], flux[mask]


def read_csv_file(path: Path):
    """
    .csv reader: first row is header (e.g. WAVE,FLUX,...); use first two columns;
    filter to 4000–9000 Å.
    """
    import pandas as pd

    df = pd.read_csv(path, header=0, comment="#")
    if len(df.columns) < 2:
        raise ValueError(f"CSV has fewer than 2 columns in {path}")
    # Prefer named columns if present
    if "WAVE" in df.columns and "FLUX" in df.columns:
        wave = np.asarray(df["WAVE"], dtype=float)
        flux = np.asarray(df["FLUX"], dtype=float)
    else:
        wave = np.asarray(df.iloc[:, 0], dtype=float)
        flux = np.asarray(df.iloc[:, 1], dtype=float)
    mask = (wave >= 4000) & (wave <= 9000)
    if not np.any(mask):
        raise ValueError(f"No data in wavelength range 4000-9000 in {path}")
    return wave[mask], flux[mask]


def read_dat_file(path: Path):
    """
    .dat reader matching FileSpectrumRepository._read_text_file:
    - Uses pandas first (tries tab, comma, space separators)
    - Falls back to manual parsing if pandas fails
    - Filters to 4000–9000 Å
    - Preserves original order (no sorting, matching prod_backend)
    """
    import io
    
    # Read file content
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # Try pandas first (matching prod_backend)
    try:
        import pandas as pd
        for sep in ['\t', ',', ' ']:
            try:
                df = pd.read_csv(io.StringIO(content), sep=sep, header=None, comment='#')
                if len(df.columns) >= 2:
                    wavelength = df[0].tolist()
                    flux = df[1].tolist()
                    break
            except:
                continue
        else:
            # Pandas failed, fall back to manual parsing
            raise ValueError("Pandas parsing failed")
    except:
        # Manual parsing fallback
        lines = content.splitlines()
        data = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        data.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
        if not data:
            raise ValueError(f"No valid data found in {path}")
        wavelength = [d[0] for d in data]
        flux = [d[1] for d in data]
    
    # Filter to 4000-9000 Å (matching prod_backend)
    filtered_data = [(w, f) for w, f in zip(wavelength, flux) if 4000 <= w <= 9000]
    
    if not filtered_data:
        raise ValueError(f"No data in wavelength range 4000-9000 in {path}")
    
    # Extract arrays (preserve order, no sorting - matching prod_backend)
    wave = np.array([x[0] for x in filtered_data], dtype=float)
    flux = np.array([x[1] for x in filtered_data], dtype=float)
    
    return wave, flux


def read_fits_file(path: Path):
    """
    .fits reader matching FileSpectrumRepository._read_fits_file, plus support
    for 1D image primary HDU with WCS (CRVAL1/CRPIX1/CDELT1).
    - Looks for extensions SPECTRUM, SPECTRA, FLUX, DATA, or first extension
    - If not found, tries primary HDU as 1D spectrum with header WCS
    - Extracts wavelength/flux from named columns or attributes or WCS
    - Filters to 4000–9000 Å
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        spectrum_data = None
        for ext in ['SPECTRUM', 'SPECTRA', 'FLUX', 'DATA']:
            if ext in hdul:
                spectrum_data = hdul[ext].data
                break
        if spectrum_data is None and len(hdul) > 1:
            spectrum_data = hdul[1].data

        # Table-like extension: wavelength + flux columns/attributes
        if spectrum_data is not None:
            if hasattr(spectrum_data, 'wavelength') and hasattr(spectrum_data, 'flux'):
                wavelength = np.asarray(spectrum_data.wavelength, dtype=float)
                flux = np.asarray(spectrum_data.flux, dtype=float)
            elif hasattr(spectrum_data, 'wave') and hasattr(spectrum_data, 'flux'):
                wavelength = np.asarray(spectrum_data.wave, dtype=float)
                flux = np.asarray(spectrum_data.flux, dtype=float)
            elif spectrum_data.dtype.names and len(spectrum_data.dtype.names) >= 2:
                wavelength = np.asarray(spectrum_data[spectrum_data.dtype.names[0]], dtype=float)
                flux = np.asarray(spectrum_data[spectrum_data.dtype.names[1]], dtype=float)
            else:
                spectrum_data = None  # fall through to primary HDU handling

        # Primary HDU as 1D image with WCS (e.g. IRAF-style spectrum)
        if spectrum_data is None and len(hdul) > 0:
            primary = hdul[0]
            if hasattr(primary, 'data') and primary.data is not None:
                data = np.asarray(primary.data, dtype=float).flatten()
                if data.ndim == 1 and len(data) > 0:
                    h = primary.header
                    crval1 = h.get('CRVAL1')
                    crpix1 = h.get('CRPIX1', 1)
                    cdel1 = h.get('CDELT1')
                    if crval1 is not None and cdel1 is not None:
                        # FITS pixel indices are 1-based
                        wavelength = crval1 + (np.arange(len(data), dtype=float) + 1 - crpix1) * cdel1
                        flux = data
                    else:
                        raise ValueError(f"No spectrum table and no WCS (CRVAL1/CDELT1) in {path}")
                else:
                    raise ValueError(f"No spectrum data found in FITS file {path}")
            else:
                raise ValueError(f"No spectrum data found in FITS file {path}")
        elif spectrum_data is None:
            raise ValueError(f"No spectrum data found in FITS file {path}")

    filtered_mask = (wavelength >= 4000) & (wavelength <= 9000)
    if not np.any(filtered_mask):
        raise ValueError(f"No data in wavelength range 4000-9000 in {path}")

    wave = wavelength[filtered_mask]
    flux = flux[filtered_mask]
    return wave, flux


def main():
    # Change working directory to prod_backend so relative paths in Settings
    # (like "../data/pre_trained_models/...") resolve correctly, matching the app.
    os.chdir(SCRIPT_DIR.parent)

    repo_root = PROJECT_ROOT
    test_dir = repo_root / "test_set" / "test_spectra"
    if not test_dir.is_dir():
        raise SystemExit(f"Directory not found: {test_dir}")

    classifier = DashClassifier()

    # If the model failed to load, DashClassifier logs an error and leaves model as None.
    # In that case, bail out early with a clear message instead of trying to format None values.
    if getattr(classifier, "model", None) is None:
        model_path = getattr(classifier, "model_path", "UNKNOWN")
        raise SystemExit(
            f"Dash model is not loaded. Expected model at: {model_path}\n"
            "Please ensure the PyTorch model and training params exist, then rerun this script."
        )

    files = sorted(
        f
        for f in test_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".lnw", ".dat", ".fits", ".ascii", ".flm", ".csv"}
    )

    if not files:
        print(
            f"No .lnw, .dat, .fits, .ascii, .flm, or .csv files found in {test_dir}"
        )
        return

    for path in files:
        try:
            suf = path.suffix.lower()
            if suf == ".lnw":
                wave, flux = read_lnw_file(path)
            elif suf == ".fits":
                wave, flux = read_fits_file(path)
            elif suf == ".ascii":
                wave, flux = read_ascii_file(path)
            elif suf == ".flm":
                wave, flux = read_flm_file(path)
            elif suf == ".csv":
                wave, flux = read_csv_file(path)
            else:
                wave, flux = read_dat_file(path)
            spectrum = SimpleSpectrum(wave, flux, redshift=0.0, file_name=path.name)

            result = classifier.classify_sync(spectrum) or {}
            best = result.get("best_match") or {}

            sn_type = best.get("type", "Unknown")
            sn_age = best.get("age", "Unknown")
            prob = best.get("probability")
            z = best.get("redshift", 0.0)

            if isinstance(prob, (int, float)):
                prob_str = f"{prob:.4f}"
            else:
                prob_str = "N/A"

            print(
                f"{path.name}: "
                f"type={sn_type}, "
                f"age={sn_age}, "
                f"prob={prob_str}, "
                f"z={z}"
            )
        except Exception as e:
            print(f"{path.name}: ERROR - {e}")


if __name__ == "__main__":
    main()

