#!/usr/bin/env python
"""
Compare TensorFlow and PyTorch DASH model outputs on various spectrum sources.

This script helps identify why outputs differ between the original TF model
and the PyTorch model used in prod_backend.

Data Sources:
1. .lnw files from training_set/templist.txt (SNID format)
2. .dat files from training_set/SLSN/ (two-column format)
3. OSC API spectra (real observed data)

It compares:
1. TF model with TF-style preprocessing (original DASH behavior)
2. PyTorch model with prod_backend preprocessing (current app behavior)
3. PyTorch model with TF-style preprocessing (to isolate preprocessing vs model issues)

Usage:
    cd astrodash-web
    python prod_backend/scripts/compare_tf_pytorch_training_set.py --model_dir zeroZ --num_samples 10
    python prod_backend/scripts/compare_tf_pytorch_training_set.py --model_dir zeroZ --num_samples 20 --verbose

Requirements: Python 3.11, tensorflow, torch, numpy, scipy, requests
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import re

# Paths setup
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_SET_DIR = DATA_DIR / "training_set"
SLSN_DIR = TRAINING_SET_DIR / "SLSN"
MODELS_DIR = DATA_DIR / "pre_trained_models"

# Add prod_backend to path for importing data_processor
sys.path.insert(0, str(SCRIPT_DIR.parent))

# OSC SNe to fetch (well-known Type Ia supernovae)
OSC_SUPERNOVAE = [
    "sn2002er",
    "SN2002bo",
    "SN2001el", 
    "SN2003du",
    "SN2005cf",
    "SN2006X",
    "SN1999aa",
    "SN1999ac",
    "SN2000ca",
    "SN2001ba",
    "SN2001cn",
    "SN2011fe",
    "SN2014J",
    "SN2012fr",
    "SN2013dy",
    "SN2015F",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpectrumData:
    """Container for a single spectrum."""
    filename: str
    sn_type: str
    age: float
    wave: np.ndarray
    flux: np.ndarray
    source: str  # 'lnw', 'slsn_dat', or 'osc'
    redshift: float = 0.0


@dataclass 
class ComparisonResult:
    """Results from comparing TF and PyTorch outputs."""
    spectrum: SpectrumData
    tf_output: np.ndarray
    pytorch_output: np.ndarray
    pytorch_tf_preproc_output: Optional[np.ndarray]
    mean_diff: float
    max_diff: float
    correlation: float
    tf_top1_idx: int
    pytorch_top1_idx: int
    top1_match: bool
    top5_overlap: int


# =============================================================================
# .lnw File Parser (SNID format)
# =============================================================================

def parse_lnw_file(filename: str) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, str, Any]:
    """
    Parse an SNID-format .lnw template file.
    
    Returns:
        wave: Wavelength array
        fluxes: 2D array of fluxes (numAges x nw)
        numAges: Number of age epochs
        ages: Array of ages
        ttype: Supernova type string
        splineInfo: Spline information tuple
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header (line 0)
    header = lines[0].strip().split()
    header = [x for x in header if x]
    numAges = int(header[0])
    nwx = int(header[1])
    w0x = float(header[2])
    w1x = float(header[3])
    mostknots = int(header[4])
    tname = header[5]
    dta = float(header[6])
    ttype = header[7]
    
    # Handle type aliases (matching original DASH)
    if ttype == 'Ia-99aa':
        ttype = 'Ia-91T'
    elif ttype == 'Ia-02cx':
        ttype = 'Iax'
    
    # Parse spline info (lines 1 to mostknots+1)
    nk = np.zeros(numAges)
    fmean = np.zeros(numAges)
    xk = np.zeros((mostknots, numAges))
    yk = np.zeros((mostknots, numAges))
    
    spline_line = lines[1].strip().split()
    spline_line = [x for x in spline_line if x]
    for j in range(numAges):
        nk[j] = float(spline_line[2 * j + 1])
        fmean[j] = float(spline_line[2 * j + 2])
    
    for k in range(mostknots):
        spline_line = lines[2 + k].strip().split()
        spline_line = [x for x in spline_line if x]
        for j in range(numAges):
            xk[k, j] = float(spline_line[2 * j + 1])
            yk[k, j] = float(spline_line[2 * j + 2])
    
    splineInfo = (nk, fmean, xk, yk)
    
    # Parse normalized spectra (after spline info)
    arr = np.loadtxt(filename, skiprows=mostknots + 2)
    ages = arr[0, 1:]  # First row after header, skip wavelength column
    arr = arr[1:]  # Remove ages row
    
    wave = arr[:, 0]
    fluxes = np.zeros((numAges, len(wave)))
    for i in range(numAges):
        fluxes[i] = arr[:, i + 1]
    
    return wave, fluxes, numAges, ages, ttype, splineInfo


# =============================================================================
# .dat File Parser (SLSN two-column format)
# =============================================================================

def parse_slsn_dat_file(filepath: Path) -> Optional[SpectrumData]:
    """
    Parse a SLSN .dat file (two-column wavelength/flux format).
    
    Filename format: SNname.ageinfo.dat
    Examples: PTF12dam.m08.dat (8 days before max), sn2005ap.p06.dat (6 days after)
    """
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        
        wave = data[:, 0]
        flux = data[:, 1]
        
        # Parse age from filename
        basename = filepath.stem  # e.g., "PTF12dam.m08"
        parts = basename.rsplit('.', 1)
        if len(parts) != 2:
            return None
        
        sn_name = parts[0]
        age_str = parts[1]
        
        # Parse age: 'max' = 0, 'm08' = -8, 'p06' = +6
        if age_str == 'max':
            age = 0.0
        elif age_str.startswith('m'):
            age = -float(age_str[1:])
        elif age_str.startswith('p'):
            age = float(age_str[1:])
        else:
            age = 0.0
        
        return SpectrumData(
            filename=filepath.name,
            sn_type="SLSN-I",  # All files in SLSN dir are SLSN-I
            age=age,
            wave=wave,
            flux=flux,
            source='slsn_dat',
            redshift=0.0
        )
    except Exception as e:
        print(f"Error parsing SLSN file {filepath}: {e}")
        return None


def load_slsn_spectra(slsn_dir: Path, num_samples: int = None) -> List[SpectrumData]:
    """Load spectra from SLSN .dat files."""
    spectra = []
    
    if not slsn_dir.exists():
        print(f"Warning: SLSN directory not found: {slsn_dir}")
        return spectra
    
    dat_files = sorted(slsn_dir.glob("*.dat"))
    
    if num_samples:
        dat_files = dat_files[:num_samples]
    
    for filepath in dat_files:
        spectrum = parse_slsn_dat_file(filepath)
        if spectrum is not None:
            # Filter by age range
            if -20 <= spectrum.age <= 50:
                spectra.append(spectrum)
    
    return spectra


# =============================================================================
# OSC API Fetcher (matching prod_backend implementation)
# =============================================================================

def fetch_osc_spectrum(sn_name: str) -> Optional[SpectrumData]:
    """
    Fetch spectrum from the Open Supernova Catalog API.
    Uses the same approach as prod_backend/app/infrastructure/storage/file_spectrum_repository.py
    
    Args:
        sn_name: Supernova name (e.g., "sn2002er", "SN2011fe")
    
    Returns:
        SpectrumData or None if fetch fails
    """
    import requests
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    # Clean up name and convert to uppercase (API expects uppercase)
    clean_name = sn_name.replace(" ", "").upper()
    
    # OSC API endpoint - matching prod_backend format
    base_url = "https://api.astrocats.space"
    url = f"{base_url}/{clean_name}/spectra/time+data"
    
    try:
        response = requests.get(url, verify=False, timeout=30)
        
        if response.status_code != 200:
            print(f"FAILED (HTTP {response.status_code})")
            return None
        
        data = response.json()
        
        if clean_name not in data:
            # Try with original case
            for key in data.keys():
                if key.upper() == clean_name:
                    clean_name = key
                    break
            else:
                print(f"FAILED (name not in response)")
                return None
        
        sn_data = data[clean_name]
        
        # Get spectra - format is: {"spectra": [["time", [[wave, flux], ...]], ...]}
        if 'spectra' not in sn_data or not sn_data['spectra']:
            print(f"FAILED (no spectra)")
            return None
        
        # Get first spectrum
        first_spectrum = sn_data['spectra'][0]
        
        # First element is time, second is the spectrum data
        # Format: ["52512", [["wavelength", "flux"], ...]]
        if len(first_spectrum) < 2:
            print(f"FAILED (invalid spectrum format)")
            return None
        
        time_str = first_spectrum[0]
        spectrum_data = first_spectrum[1]
        
        # Parse spectrum data - it's a list of [wave, flux] pairs
        try:
            arr = np.array(spectrum_data, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                wave = arr[:, 0]
                flux = arr[:, 1]
            else:
                print(f"FAILED (spectrum shape: {arr.shape})")
                return None
        except (ValueError, IndexError) as e:
            print(f"FAILED (parse error: {e})")
            return None
        
        # Validate
        if len(wave) < 100:
            print(f"FAILED (too few points: {len(wave)})")
            return None
        
        if np.any(~np.isfinite(wave)) or np.any(~np.isfinite(flux)):
            print(f"FAILED (non-finite values)")
            return None
        
        # Fetch redshift separately
        redshift = 0.0
        try:
            z_url = f"{base_url}/{clean_name}/redshift"
            z_response = requests.get(z_url, verify=False, timeout=10)
            if z_response.status_code == 200:
                z_data = z_response.json()
                if clean_name in z_data and 'redshift' in z_data[clean_name]:
                    z_list = z_data[clean_name]['redshift']
                    if z_list and len(z_list) > 0:
                        redshift = float(z_list[0]['value'])
        except Exception:
            pass  # Use 0.0 if we can't get redshift
        
        # Fetch claimed type separately
        sn_type = "Unknown"
        try:
            type_url = f"{base_url}/{clean_name}/claimedtype"
            type_response = requests.get(type_url, verify=False, timeout=10)
            if type_response.status_code == 200:
                type_data = type_response.json()
                if clean_name in type_data and 'claimedtype' in type_data[clean_name]:
                    type_list = type_data[clean_name]['claimedtype']
                    if type_list and len(type_list) > 0:
                        sn_type = type_list[0]['value']
        except Exception:
            pass
        
        # Parse time to age (MJD to days relative to max, if possible)
        age = 0.0
        try:
            age = float(time_str)
        except (ValueError, TypeError):
            pass
        
        return SpectrumData(
            filename=f"OSC:{clean_name}",
            sn_type=sn_type,
            age=age,
            wave=wave,
            flux=flux,
            source='osc',
            redshift=redshift
        )
        
    except requests.RequestException as e:
        print(f"FAILED (request error: {e})")
        return None
    except Exception as e:
        print(f"FAILED (error: {e})")
        return None


def load_osc_spectra(sn_names: List[str]) -> List[SpectrumData]:
    """Load spectra from OSC API for given supernova names."""
    spectra = []
    
    print(f"Fetching {len(sn_names)} spectra from OSC API...")
    for i, name in enumerate(sn_names):
        print(f"  [{i+1}/{len(sn_names)}] Fetching {name}...", end=" ", flush=True)
        spectrum = fetch_osc_spectrum(name)
        if spectrum:
            spectra.append(spectrum)
            # fetch_osc_spectrum doesn't print on success, so print here
            print(f"OK ({len(spectrum.wave)} pts, z={spectrum.redshift:.4f}, type={spectrum.sn_type})")
        # If spectrum is None, fetch_osc_spectrum already printed the failure reason
    
    return spectra


# =============================================================================
# Load .lnw Training Spectra
# =============================================================================

def load_lnw_spectra(training_set_dir: Path, templist_path: Path, 
                     num_samples: int = None) -> List[SpectrumData]:
    """Load spectra from training set based on templist.txt."""
    spectra = []
    
    # Read template list
    with open(templist_path, 'r') as f:
        filenames = [line.strip() for line in f if line.strip() and line.strip().endswith('.lnw')]
    
    if num_samples:
        filenames = filenames[:num_samples]
    
    for fname in filenames:
        filepath = training_set_dir / fname
        if not filepath.exists():
            continue
        
        try:
            wave, fluxes, numAges, ages, ttype, _ = parse_lnw_file(str(filepath))
            
            # Take first valid age spectrum for comparison
            for i, age in enumerate(ages):
                if -20 <= age <= 50:  # Valid age range from DASH
                    spectra.append(SpectrumData(
                        filename=fname,
                        sn_type=ttype,
                        age=age,
                        wave=wave,
                        flux=fluxes[i],
                        source='lnw',
                        redshift=0.0
                    ))
                    break  # Just take first valid age per file
                    
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            continue
    
    return spectra


# =============================================================================
# Preprocessing Functions
# =============================================================================

class TFStylePreprocessor:
    """
    Preprocessing matching original DASH TensorFlow implementation.
    This replicates the preprocessing from the original astrodash package.
    """
    
    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points
    
    def log_wavelength(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Bin to log-wavelength grid."""
        dwlog = np.log(self.w1 / self.w0) / self.nw
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0, right=0)
        
        non_zero = np.where(binned_flux != 0)[0]
        if len(non_zero) == 0:
            return wlog, binned_flux, 0, 0
        
        min_idx = non_zero[0]
        max_idx = non_zero[-1]
        
        return wlog, binned_flux, min_idx, max_idx
    
    def continuum_removal(self, wave: np.ndarray, flux: np.ndarray, 
                          min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Remove continuum using spline fitting."""
        from scipy.interpolate import splrep, splev
        
        wave_region = wave[min_idx:max_idx + 1]
        flux_region = flux[min_idx:max_idx + 1]
        
        if len(wave_region) > self.num_spline_points:
            indices = np.linspace(0, len(wave_region) - 1, self.num_spline_points, dtype=int)
            spline = splrep(wave_region[indices], flux_region[indices], k=3)
            continuum = splev(wave_region, spline)
        else:
            continuum = np.full_like(flux_region, np.mean(flux_region))
        
        full_continuum = np.zeros_like(flux)
        full_continuum[min_idx:max_idx + 1] = continuum
        
        return flux - full_continuum, full_continuum
    
    def mean_zero(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """Zero-mean the flux."""
        flux_out = np.copy(flux)
        flux_out[:min_idx] = flux_out[min_idx]
        flux_out[max_idx:] = flux_out[max_idx]
        valid_mean = np.mean(flux_out[min_idx:max_idx + 1])
        return flux_out - valid_mean
    
    def apodize(self, flux: np.ndarray, min_idx: int, max_idx: int, 
                edge_width: int = 50) -> np.ndarray:
        """Apply cosine taper at edges."""
        apodized = np.copy(flux)
        actual_edge = min(edge_width, (max_idx - min_idx) // 4)
        
        if actual_edge > 0:
            for i in range(actual_edge):
                factor = 0.5 * (1 + np.cos(np.pi * i / actual_edge))
                if min_idx + i < len(apodized):
                    apodized[min_idx + i] *= factor
                if max_idx - i >= 0:
                    apodized[max_idx - i] *= factor
        
        return apodized
    
    def normalise(self, flux: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        flux_min, flux_max = np.min(flux), np.max(flux)
        if np.isclose(flux_min, flux_max):
            return np.zeros_like(flux)
        return (flux - flux_min) / (flux_max - flux_min)
    
    def zero_non_overlap(self, flux: np.ndarray, min_idx: int, max_idx: int, 
                         outer_val: float = 0.5) -> np.ndarray:
        """Set regions outside valid range to outer_val."""
        flux_out = np.copy(flux)
        flux_out[:min_idx] = outer_val
        flux_out[max_idx:] = outer_val
        return flux_out
    
    def process(self, wave: np.ndarray, flux: np.ndarray, z: float = 0.0) -> Tuple[np.ndarray, int, int]:
        """Full preprocessing pipeline."""
        # Deredshift
        wave_dereds = wave / (1 + z)
        
        # Normalize first (original DASH normalizes input)
        flux_norm = self.normalise(flux)
        
        # Log wavelength binning
        binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength(wave_dereds, flux_norm)
        
        # Continuum removal
        new_flux, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
        
        # Mean zero
        mean_zero_flux = self.mean_zero(new_flux, min_idx, max_idx)
        
        # Apodize
        apodized = self.apodize(mean_zero_flux, min_idx, max_idx)
        
        # Final normalization
        final_flux = self.normalise(apodized)
        
        # Zero non-overlap
        final_flux = self.zero_non_overlap(final_flux, min_idx, max_idx, 0.5)
        
        return final_flux, min_idx, max_idx


class ProdBackendPreprocessor:
    """
    Preprocessing matching prod_backend/app/infrastructure/ml/data_processor.py
    This is what the current app uses.
    """
    
    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points
    
    def normalise_spectrum(self, flux: np.ndarray) -> np.ndarray:
        """Normalize flux array to [0, 1] range."""
        flux_min, flux_max = np.min(flux), np.max(flux)
        if np.isclose(flux_min, flux_max):
            return np.zeros(len(flux))
        return (flux - flux_min) / (flux_max - flux_min)
    
    def log_wavelength_binning(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Bin flux to log-wavelength grid."""
        dwlog = np.log(self.w1 / self.w0) / self.nw
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0, right=0)
        
        non_zero = np.where(binned_flux != 0)[0]
        if len(non_zero) == 0:
            return wlog, binned_flux, 0, 0
        
        min_idx = non_zero[0]
        max_idx = non_zero[-1]
        
        return wlog, binned_flux, min_idx, max_idx
    
    def continuum_removal(self, wave: np.ndarray, flux: np.ndarray, 
                          min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Remove continuum using spline fitting."""
        from scipy.interpolate import splrep, splev
        
        min_idx = np.clip(min_idx, 0, len(flux) - 1)
        max_idx = np.clip(max_idx, min_idx, len(flux) - 1)
        
        wave_region = wave[min_idx:max_idx + 1]
        flux_region = flux[min_idx:max_idx + 1]
        
        if len(wave_region) > self.num_spline_points:
            indices = np.linspace(0, len(wave_region) - 1, self.num_spline_points, dtype=int)
            spline = splrep(wave_region[indices], flux_region[indices], k=3)
            continuum = splev(wave_region, spline)
        else:
            continuum = np.full_like(flux_region, np.mean(flux_region))
        
        full_continuum = np.zeros_like(flux)
        full_continuum[min_idx:max_idx + 1] = continuum
        
        return flux - full_continuum, full_continuum
    
    def mean_zero(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """Zero-mean the flux array within the specified region."""
        flux_out = np.copy(flux)
        min_idx = np.clip(min_idx, 0, len(flux_out) - 1)
        max_idx = np.clip(max_idx, min_idx, len(flux_out) - 1)
        
        flux_out[:min_idx] = flux_out[min_idx]
        flux_out[max_idx:] = flux_out[max_idx]
        valid_mean = np.mean(flux_out[min_idx:max_idx + 1])
        return flux_out - valid_mean
    
    def apodize(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """Apply apodization to reduce edge effects."""
        apodized = np.copy(flux)
        min_idx = np.clip(min_idx, 0, len(apodized) - 1)
        max_idx = np.clip(max_idx, min_idx, len(apodized) - 1)
        
        edge_width = min(50, (max_idx - min_idx) // 4)
        
        if edge_width > 0:
            for i in range(edge_width):
                factor = 0.5 * (1 + np.cos(np.pi * i / edge_width))
                left_idx = min_idx + i
                if 0 <= left_idx < len(apodized):
                    apodized[left_idx] *= factor
                right_idx = max_idx - i
                if 0 <= right_idx < len(apodized):
                    apodized[right_idx] *= factor
        
        return apodized
    
    def zero_non_overlap_part(self, array: np.ndarray, min_idx: int, max_idx: int, 
                              outer_val: float = 0.5) -> np.ndarray:
        """Set regions outside the valid range to a specified value."""
        sliced = np.copy(array)
        min_idx = np.clip(min_idx, 0, len(sliced) - 1)
        max_idx = np.clip(max_idx, min_idx, len(sliced) - 1)
        sliced[:min_idx] = outer_val
        sliced[max_idx:] = outer_val
        return sliced
    
    def process(self, wave: np.ndarray, flux: np.ndarray, z: float = 0.0) -> Tuple[np.ndarray, int, int]:
        """Full preprocessing pipeline (matching prod_backend)."""
        # Normalize input spectrum
        flux_processed = self.normalise_spectrum(flux)
        
        # Deredshift
        wave_dereds = wave / (1 + z)
        
        # Log wavelength binning
        binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(wave_dereds, flux_processed)
        
        # Continuum removal
        new_flux, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
        
        # Mean zero
        mean_zero_flux = self.mean_zero(new_flux, min_idx, max_idx)
        
        # Apodize
        apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
        
        # Final normalization
        flux_norm = self.normalise_spectrum(apodized_flux)
        
        # Zero non-overlap
        flux_norm = self.zero_non_overlap_part(flux_norm, min_idx, max_idx, 0.5)
        
        return flux_norm, min_idx, max_idx


# =============================================================================
# Model Loading and Inference
# =============================================================================

def load_training_params(params_path: str) -> Dict[str, Any]:
    """Load training parameters from pickle file."""
    with open(params_path, 'rb') as f:
        params = pickle.load(f, encoding='latin1')
    return params


class TensorFlowModel:
    """Wrapper for TensorFlow DASH model."""
    
    def __init__(self, model_path: str):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        
        self.model_path = model_path
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model_path + '.meta')
            self.sess = tf.Session()
            self.saver.restore(self.sess, model_path)
            
            self.x = self.graph.get_tensor_by_name("Placeholder:0")
            self.keep_prob = self.graph.get_tensor_by_name("Placeholder_2:0")
            self.y_conv = self.graph.get_tensor_by_name("Softmax:0")
    
    def predict(self, processed_flux: np.ndarray) -> np.ndarray:
        """Run inference and return softmax probabilities."""
        with self.graph.as_default():
            input_data = processed_flux.reshape(1, -1).astype(np.float32)
            output = self.sess.run(self.y_conv, feed_dict={
                self.x: input_data, 
                self.keep_prob: 1.0
            })
            return output[0]
    
    def close(self):
        """Close TensorFlow session."""
        self.sess.close()


class PyTorchModel:
    """Wrapper for PyTorch DASH model."""
    
    def __init__(self, model_path: str, n_types: int, im_width: int = 32):
        import torch
        import torch.nn as nn
        
        # Define model architecture (from convert_model.py)
        class AstroDashPyTorchNet(nn.Module):
            def __init__(self, n_types, im_width=32):
                super(AstroDashPyTorchNet, self).__init__()
                self.im_width = im_width
                
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=5, padding='same'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
                
                self.layer2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=5, padding='same'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
                
                pooled_size = im_width // 4
                self.fc1 = nn.Linear(64 * pooled_size * pooled_size, 1024)
                self.dropout = nn.Dropout()
                self.output = nn.Linear(1024, n_types)
            
            def forward(self, x):
                x = x.view(-1, 1, self.im_width, self.im_width)
                h_pool1 = self.layer1(x)
                h_pool2 = self.layer2(h_pool1)
                
                # Reshape to match TF flattening order
                h_pool2_transposed = h_pool2.permute(0, 2, 3, 1)
                h_pool2_flat = h_pool2_transposed.reshape(h_pool2.size(0), -1)
                
                h_fc1 = torch.nn.functional.relu(self.fc1(h_pool2_flat))
                h_fc_drop = self.dropout(h_fc1)
                output = self.output(h_fc_drop)
                
                return torch.nn.functional.softmax(output, dim=1)
        
        self.model = AstroDashPyTorchNet(n_types, im_width)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.torch = torch
    
    def predict(self, processed_flux: np.ndarray) -> np.ndarray:
        """Run inference and return softmax probabilities."""
        with self.torch.no_grad():
            input_tensor = self.torch.from_numpy(processed_flux).float().reshape(1, -1)
            output = self.model(input_tensor)
            return output[0].numpy()


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_outputs(tf_output: np.ndarray, pytorch_output: np.ndarray) -> Dict[str, Any]:
    """Compare two probability distributions."""
    diff = np.abs(tf_output - pytorch_output)
    
    tf_top5 = set(np.argsort(tf_output)[-5:])
    pytorch_top5 = set(np.argsort(pytorch_output)[-5:])
    
    return {
        'mean_diff': np.mean(diff),
        'max_diff': np.max(diff),
        'std_diff': np.std(diff),
        'correlation': np.corrcoef(tf_output, pytorch_output)[0, 1] if len(tf_output) > 1 else 1.0,
        'tf_top1': np.argmax(tf_output),
        'pytorch_top1': np.argmax(pytorch_output),
        'top1_match': np.argmax(tf_output) == np.argmax(pytorch_output),
        'top5_overlap': len(tf_top5 & pytorch_top5),
        'tf_top1_prob': tf_output[np.argmax(tf_output)],
        'pytorch_top1_prob': pytorch_output[np.argmax(pytorch_output)],
    }


def compare_preprocessing(tf_preproc: np.ndarray, prod_preproc: np.ndarray) -> Dict[str, Any]:
    """Compare two preprocessing outputs."""
    diff = np.abs(tf_preproc - prod_preproc)
    
    return {
        'mean_diff': np.mean(diff),
        'max_diff': np.max(diff),
        'std_diff': np.std(diff),
        'correlation': np.corrcoef(tf_preproc, prod_preproc)[0, 1] if len(tf_preproc) > 1 else 1.0,
    }


def decode_label(label_idx: int, type_list: List[str], min_age: float, 
                 max_age: float, age_bin_size: float) -> str:
    """Decode a label index to type + age string."""
    num_age_bins = int((max_age - min_age) / age_bin_size) + 1
    type_idx = label_idx // num_age_bins
    age_idx = label_idx % num_age_bins
    
    if type_idx >= len(type_list):
        return f"Unknown type {type_idx}, age bin {age_idx}"
    
    age_start = min_age + age_idx * age_bin_size
    age_end = age_start + age_bin_size
    
    return f"{type_list[type_idx]}: {int(age_start)} to {int(age_end)}"


# =============================================================================
# Main Comparison Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare TF and PyTorch DASH models')
    parser.add_argument('--model_dir', type=str, default='zeroZ', 
                        help='Model subdirectory (zeroZ or agnosticZ)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of spectra per source (lnw/slsn). OSC always uses all 15.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-spectrum results')
    parser.add_argument('--skip_osc', action='store_true',
                        help='Skip OSC API fetching (for offline testing)')
    args = parser.parse_args()
    
    # Paths
    original_tf_dir = MODELS_DIR / "original_dash_models" / args.model_dir
    converted_pytorch_dir = MODELS_DIR / "dash" / args.model_dir
    
    tf_model_path = original_tf_dir / "tensorflow_model.ckpt"
    pytorch_model_path = converted_pytorch_dir / "pytorch_model.pth"
    params_path = original_tf_dir / "training_params.pickle"
    templist_path = TRAINING_SET_DIR / "templist.txt"
    
    # Validate paths
    for path, name in [(tf_model_path.with_suffix('.ckpt.index'), 'TF model'),
                       (pytorch_model_path, 'PyTorch model'),
                       (params_path, 'Training params'),
                       (templist_path, 'Template list')]:
        if not path.exists():
            print(f"Error: {name} not found at {path}")
            sys.exit(1)
    
    # Load parameters
    print("=" * 70)
    print("DASH Model Comparison: TensorFlow vs PyTorch")
    print("=" * 70)
    
    params = load_training_params(str(params_path))
    w0 = params['w0']
    w1 = params['w1']
    nw = params['nw']
    n_types = params['nTypes']
    min_age = params['minAge']
    max_age = params['maxAge']
    age_bin_size = params['ageBinSize']
    type_list = params.get('typeList', [])
    
    num_age_bins = int((max_age - min_age) / age_bin_size) + 1
    total_classes = n_types * num_age_bins
    
    print(f"\nModel Parameters:")
    print(f"  Wavelength range: {w0} - {w1} Å")
    print(f"  Number of bins: {nw}")
    print(f"  Number of SN types: {n_types}")
    print(f"  Age range: {min_age} to {max_age} days")
    print(f"  Age bin size: {age_bin_size} days")
    print(f"  Number of age bins: {num_age_bins}")
    print(f"  Total output classes: {total_classes}")
    
    # Load models
    print("\nLoading models...")
    tf_model = TensorFlowModel(str(tf_model_path))
    pytorch_model = PyTorchModel(str(pytorch_model_path), n_types=total_classes)
    
    # Initialize preprocessors
    tf_preprocessor = TFStylePreprocessor(w0, w1, nw)
    prod_preprocessor = ProdBackendPreprocessor(w0, w1, nw)
    
    # Load spectra from all sources
    print(f"\n{'='*70}")
    print("Loading spectra from multiple sources...")
    print("="*70)
    
    # Calculate samples per source (split num_samples between lnw and slsn)
    lnw_samples = args.num_samples // 2
    slsn_samples = args.num_samples - lnw_samples
    
    # 1. Load .lnw spectra
    print(f"\n[1/3] Loading {lnw_samples} .lnw spectra from templist.txt...")
    lnw_spectra = load_lnw_spectra(TRAINING_SET_DIR, templist_path, lnw_samples)
    print(f"  Loaded {len(lnw_spectra)} .lnw spectra")
    
    # 2. Load SLSN .dat spectra
    print(f"\n[2/3] Loading {slsn_samples} SLSN .dat spectra...")
    slsn_spectra = load_slsn_spectra(SLSN_DIR, slsn_samples)
    print(f"  Loaded {len(slsn_spectra)} SLSN spectra")
    
    # 3. Load OSC spectra (all 15)
    osc_spectra = []
    if not args.skip_osc:
        print(f"\n[3/3] Loading ALL {len(OSC_SUPERNOVAE)} OSC API spectra...")
        osc_spectra = load_osc_spectra(OSC_SUPERNOVAE)
        print(f"  Loaded {len(osc_spectra)} OSC spectra")
    else:
        print("\n[3/3] Skipping OSC API (--skip_osc flag)")
    
    # Combine all spectra
    all_spectra = lnw_spectra + slsn_spectra + osc_spectra
    print(f"\nTotal spectra to compare: {len(all_spectra)}")
    print(f"  - .lnw: {len(lnw_spectra)}")
    print(f"  - SLSN .dat: {len(slsn_spectra)}")
    print(f"  - OSC API: {len(osc_spectra)}")
    
    if not all_spectra:
        print("No spectra to compare!")
        tf_model.close()
        return
    
    # Run comparisons
    print("\n" + "=" * 70)
    print("Running comparisons...")
    print("=" * 70)
    
    results_by_source = {
        'lnw': {'preproc': [], 'same_preproc': [], 'diff_preproc': []},
        'slsn_dat': {'preproc': [], 'same_preproc': [], 'diff_preproc': []},
        'osc': {'preproc': [], 'same_preproc': [], 'diff_preproc': []},
    }
    
    # Helper to get top N predictions
    def get_top_n(output: np.ndarray, n: int = 3) -> List[Tuple[int, float, str]]:
        """Get top N predictions with indices, probabilities, and labels."""
        top_indices = np.argsort(output)[-n:][::-1]
        return [(idx, output[idx], decode_label(idx, type_list, min_age, max_age, age_bin_size)) 
                for idx in top_indices]
    
    for i, spectrum in enumerate(all_spectra):
        source_label = {
            'lnw': 'LNW',
            'slsn_dat': 'SLSN',
            'osc': 'OSC'
        }.get(spectrum.source, spectrum.source.upper())
        
        # Check if this is sn2002er (special detailed output)
        is_sn2002er = 'sn2002er' in spectrum.filename.lower() or 'SN2002ER' in spectrum.filename
        
        if args.verbose or is_sn2002er:
            print(f"\n[{i+1}/{len(all_spectra)}] [{source_label}] {spectrum.filename} ({spectrum.sn_type}, age={spectrum.age:.1f}d, z={spectrum.redshift:.4f})")
        
        # Use redshift for OSC spectra
        z = spectrum.redshift if spectrum.source == 'osc' else 0.0
        
        # Print raw wavelength info BEFORE preprocessing
        wave_min, wave_max = spectrum.wave.min(), spectrum.wave.max()
        wave_dereds_min, wave_dereds_max = wave_min / (1 + z), wave_max / (1 + z)
        model_range_overlap = (wave_dereds_min <= w1) and (wave_dereds_max >= w0)
        
        if args.verbose or is_sn2002er:
            print(f"  Raw spectrum: {len(spectrum.wave)} points")
            print(f"    Observed wavelength range: [{wave_min:.1f}, {wave_max:.1f}] Å")
            print(f"    De-redshifted (z={z:.4f}): [{wave_dereds_min:.1f}, {wave_dereds_max:.1f}] Å")
            print(f"    Model expects: [{w0:.1f}, {w1:.1f}] Å")
            print(f"    Overlap with model range: {model_range_overlap}")
        
        # Skip spectra with no overlap
        if not model_range_overlap:
            if args.verbose or is_sn2002er:
                print(f"  ⚠️  SKIPPING: No wavelength overlap with model range!")
            continue
        
        # Preprocess with both methods
        try:
            tf_preproc, tf_min, tf_max = tf_preprocessor.process(spectrum.wave, spectrum.flux, z=z)
            prod_preproc, prod_min, prod_max = prod_preprocessor.process(spectrum.wave, spectrum.flux, z=z)
        except Exception as e:
            if args.verbose or is_sn2002er:
                print(f"  ERROR in preprocessing: {e}")
            continue
        
        # Check if preprocessing produced valid data
        if tf_min == tf_max == 0:
            if args.verbose or is_sn2002er:
                print(f"  ⚠️  WARNING: Preprocessing produced no valid data (min_idx=max_idx=0)")
        
        # Compare preprocessing
        preproc_comparison = compare_preprocessing(tf_preproc, prod_preproc)
        results_by_source[spectrum.source]['preproc'].append(preproc_comparison)
        
        if args.verbose or is_sn2002er:
            print(f"  Preprocessing comparison:")
            print(f"    TF preproc:   min={tf_preproc.min():.6f}, max={tf_preproc.max():.6f}, mean={tf_preproc.mean():.6f}, std={tf_preproc.std():.6f}")
            print(f"    Prod preproc: min={prod_preproc.min():.6f}, max={prod_preproc.max():.6f}, mean={prod_preproc.mean():.6f}, std={prod_preproc.std():.6f}")
            print(f"    Difference:   mean_diff={preproc_comparison['mean_diff']:.6f}, max_diff={preproc_comparison['max_diff']:.6f}, corr={preproc_comparison['correlation']:.6f}")
            print(f"    TF min/max idx: [{tf_min}, {tf_max}], Prod min/max idx: [{prod_min}, {prod_max}]")
        
        # Run TF model with TF preprocessing
        tf_output = tf_model.predict(tf_preproc)
        
        # Run PyTorch model with TF preprocessing
        pytorch_tf_output = pytorch_model.predict(tf_preproc)
        
        # Run PyTorch model with prod_backend preprocessing
        pytorch_prod_output = pytorch_model.predict(prod_preproc)
        
        # Compare TF vs PyTorch (same preprocessing)
        tf_vs_pytorch_tf = compare_outputs(tf_output, pytorch_tf_output)
        results_by_source[spectrum.source]['same_preproc'].append(tf_vs_pytorch_tf)
        
        # Compare TF vs PyTorch (different preprocessing)
        tf_vs_pytorch_prod = compare_outputs(tf_output, pytorch_prod_output)
        results_by_source[spectrum.source]['diff_preproc'].append(tf_vs_pytorch_prod)
        
        if args.verbose or is_sn2002er:
            tf_label = decode_label(tf_vs_pytorch_tf['tf_top1'], type_list, min_age, max_age, age_bin_size)
            pt_tf_label = decode_label(tf_vs_pytorch_tf['pytorch_top1'], type_list, min_age, max_age, age_bin_size)
            pt_prod_label = decode_label(tf_vs_pytorch_prod['pytorch_top1'], type_list, min_age, max_age, age_bin_size)
            
            print(f"  Model predictions:")
            print(f"    TF model (TF preproc):        {tf_label} ({tf_vs_pytorch_tf['tf_top1_prob']:.4f})")
            print(f"    PyTorch model (TF preproc):   {pt_tf_label} ({tf_vs_pytorch_tf['pytorch_top1_prob']:.4f})")
            print(f"    PyTorch model (prod preproc): {pt_prod_label} ({tf_vs_pytorch_prod['pytorch_top1_prob']:.4f})")
            print(f"  Comparison metrics:")
            print(f"    TF vs PyTorch (same preproc): mean_diff={tf_vs_pytorch_tf['mean_diff']:.6f}, corr={tf_vs_pytorch_tf['correlation']:.6f}, top1_match={tf_vs_pytorch_tf['top1_match']}")
            print(f"    TF vs PyTorch (diff preproc): mean_diff={tf_vs_pytorch_prod['mean_diff']:.6f}, corr={tf_vs_pytorch_prod['correlation']:.6f}, top1_match={tf_vs_pytorch_prod['top1_match']}")
        
        # Special detailed output for sn2002er
        if is_sn2002er:
            print(f"\n  {'='*60}")
            print(f"  DETAILED TOP 3 PREDICTIONS FOR SN2002ER (from OSC)")
            print(f"  {'='*60}")
            
            print(f"\n  📊 Preprocessed Input Comparison:")
            print(f"  ┌{'─'*40}┬{'─'*40}┐")
            print(f"  │ {'TF-style Preprocessing':^38} │ {'Prod Backend Preprocessing':^38} │")
            print(f"  ├{'─'*40}┼{'─'*40}┤")
            print(f"  │ Min:  {tf_preproc.min():>10.6f}               │ Min:  {prod_preproc.min():>10.6f}               │")
            print(f"  │ Max:  {tf_preproc.max():>10.6f}               │ Max:  {prod_preproc.max():>10.6f}               │")
            print(f"  │ Mean: {tf_preproc.mean():>10.6f}               │ Mean: {prod_preproc.mean():>10.6f}               │")
            print(f"  │ Std:  {tf_preproc.std():>10.6f}               │ Std:  {prod_preproc.std():>10.6f}               │")
            print(f"  │ Valid range: [{tf_min:>4}, {tf_max:>4}]           │ Valid range: [{prod_min:>4}, {prod_max:>4}]           │")
            print(f"  └{'─'*40}┴{'─'*40}┘")
            print(f"  Correlation between preprocessings: {preproc_comparison['correlation']:.6f}")
            
            # Top 3 for TF model
            tf_top3 = get_top_n(tf_output, 3)
            print(f"\n  🔷 TensorFlow Model (with TF preprocessing):")
            for rank, (idx, prob, label) in enumerate(tf_top3, 1):
                print(f"     #{rank}: {label} (idx={idx}, prob={prob:.4f})")
            
            # Top 3 for PyTorch model with TF preprocessing
            pt_tf_top3 = get_top_n(pytorch_tf_output, 3)
            print(f"\n  🔶 PyTorch Model (with TF preprocessing):")
            for rank, (idx, prob, label) in enumerate(pt_tf_top3, 1):
                print(f"     #{rank}: {label} (idx={idx}, prob={prob:.4f})")
            
            # Top 3 for PyTorch model with prod_backend preprocessing
            pt_prod_top3 = get_top_n(pytorch_prod_output, 3)
            print(f"\n  🟢 PyTorch Model (with prod_backend preprocessing):")
            for rank, (idx, prob, label) in enumerate(pt_prod_top3, 1):
                print(f"     #{rank}: {label} (idx={idx}, prob={prob:.4f})")
            
            # Show first few values of preprocessed arrays
            print(f"\n  📈 First 10 values of preprocessed flux:")
            print(f"     TF preproc:   {tf_preproc[:10]}")
            print(f"     Prod preproc: {prod_preproc[:10]}")
            print(f"\n  📈 Last 10 values of preprocessed flux:")
            print(f"     TF preproc:   {tf_preproc[-10:]}")
            print(f"     Prod preproc: {prod_preproc[-10:]}")
            print(f"  {'='*60}")
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS BY SOURCE")
    print("=" * 70)
    
    def summarize_source(source_name: str, source_key: str, results: Dict):
        data = results[source_key]
        if not data['preproc']:
            print(f"\n{source_name}: No data")
            return
        
        preproc = data['preproc']
        same = data['same_preproc']
        diff = data['diff_preproc']
        
        print(f"\n{source_name} ({len(preproc)} spectra):")
        print("-" * 50)
        
        # Preprocessing comparison
        preproc_mean = np.mean([c['mean_diff'] for c in preproc])
        preproc_max = np.max([c['max_diff'] for c in preproc])
        preproc_corr = np.mean([c['correlation'] for c in preproc])
        print(f"  Preprocessing (TF vs prod_backend):")
        print(f"    Mean diff: avg={preproc_mean:.6f}, max={preproc_max:.6f}")
        print(f"    Correlation: avg={preproc_corr:.6f}")
        
        # Same preprocessing comparison
        same_mean = np.mean([c['mean_diff'] for c in same])
        same_max = np.max([c['max_diff'] for c in same])
        same_corr = np.mean([c['correlation'] for c in same])
        same_match = sum(c['top1_match'] for c in same)
        same_top5 = np.mean([c['top5_overlap'] for c in same])
        print(f"  Model outputs (SAME preprocessing):")
        print(f"    Mean diff: avg={same_mean:.6f}, max={same_max:.6f}")
        print(f"    Correlation: avg={same_corr:.6f}")
        print(f"    Top-1 match: {same_match}/{len(same)} ({100*same_match/len(same):.1f}%)")
        print(f"    Top-5 overlap: avg={same_top5:.2f}/5")
        
        # Different preprocessing comparison
        diff_mean = np.mean([c['mean_diff'] for c in diff])
        diff_max = np.max([c['max_diff'] for c in diff])
        diff_corr = np.mean([c['correlation'] for c in diff])
        diff_match = sum(c['top1_match'] for c in diff)
        diff_top5 = np.mean([c['top5_overlap'] for c in diff])
        print(f"  Model outputs (DIFFERENT preprocessing):")
        print(f"    Mean diff: avg={diff_mean:.6f}, max={diff_max:.6f}")
        print(f"    Correlation: avg={diff_corr:.6f}")
        print(f"    Top-1 match: {diff_match}/{len(diff)} ({100*diff_match/len(diff):.1f}%)")
        print(f"    Top-5 overlap: avg={diff_top5:.2f}/5")
    
    summarize_source("📁 .LNW Training Files", 'lnw', results_by_source)
    summarize_source("🌟 SLSN .DAT Files", 'slsn_dat', results_by_source)
    summarize_source("🌐 OSC API Spectra", 'osc', results_by_source)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    all_preproc = []
    all_same = []
    all_diff = []
    for source_data in results_by_source.values():
        all_preproc.extend(source_data['preproc'])
        all_same.extend(source_data['same_preproc'])
        all_diff.extend(source_data['diff_preproc'])
    
    if all_preproc:
        print(f"\nTotal spectra compared: {len(all_preproc)}")
        print(f"\nPreprocessing (TF vs prod_backend):")
        print(f"  Correlation: avg={np.mean([c['correlation'] for c in all_preproc]):.6f}")
        
        print(f"\nModel outputs (SAME preprocessing):")
        print(f"  Correlation: avg={np.mean([c['correlation'] for c in all_same]):.6f}")
        print(f"  Top-1 match: {sum(c['top1_match'] for c in all_same)}/{len(all_same)} ({100*sum(c['top1_match'] for c in all_same)/len(all_same):.1f}%)")
        
        print(f"\nModel outputs (DIFFERENT preprocessing):")
        print(f"  Correlation: avg={np.mean([c['correlation'] for c in all_diff]):.6f}")
        print(f"  Top-1 match: {sum(c['top1_match'] for c in all_diff)}/{len(all_diff)} ({100*sum(c['top1_match'] for c in all_diff)/len(all_diff):.1f}%)")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if all_same:
        same_corr = np.mean([c['correlation'] for c in all_same])
        same_match = sum(c['top1_match'] for c in all_same) / len(all_same)
        diff_corr = np.mean([c['correlation'] for c in all_diff])
        diff_match = sum(c['top1_match'] for c in all_diff) / len(all_diff)
        preproc_corr = np.mean([c['correlation'] for c in all_preproc])
        
        if same_corr > 0.9999 and same_match > 0.95:
            print("\n✅ PyTorch model weights are correctly converted from TensorFlow.")
            if diff_corr < same_corr - 0.001 or diff_match < same_match - 0.05:
                print("⚠️  Preprocessing differences are causing some output discrepancies.")
                print("   The prod_backend preprocessing differs from original DASH preprocessing.")
            else:
                print("✅ Preprocessing pipelines produce nearly identical results.")
        elif same_corr < 0.999:
            print("\n❌ PyTorch model outputs differ significantly from TensorFlow even with same preprocessing.")
            print("   This indicates a problem with the weight conversion or model architecture.")
        
        if preproc_corr < 0.99:
            print(f"\n⚠️  Preprocessing outputs differ significantly (corr={preproc_corr:.6f}).")
            print("   Review the preprocessing pipeline for differences.")
    
    # Cleanup
    tf_model.close()
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
