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
    "sn1994s",
    "sn2003du",
    # "sn2002er",
    # "SN2002bo",
    # "SN2001el", 
    # "SN2003du",
    # "SN2005cf",
    # "SN2006X",
    # "SN1999aa",
    # "SN1999ac",
    # "SN2000ca",
    # "SN2001ba",
    # "SN2001cn",
    # "SN2011fe",
    # "SN2014J",
    # "SN2012fr",
    # "SN2013dy",
    # "SN2015F",
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
    Preprocessing matching the original DASH TensorFlow implementation
    used in astrodash_old.PreProcessing.two_column_data + PreProcessSpectrum.
    """

    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13, smooth: int = 6):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points
        self.dwlog = np.log(w1 / w0) / nw
        # Use the same default smoothing as typical astrodash usage
        self.smooth = smooth

    # ---- Low-level helpers mirroring astrodash_old ----

    def _normalise(self, flux: np.ndarray) -> np.ndarray:
        """Match astrodash_old.array_tools.normalise_spectrum."""
        if flux.size == 0 or np.min(flux) == np.max(flux):
            return np.zeros_like(flux)
        return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

    def _limit_wavelength_range(self, wave: np.ndarray, flux: np.ndarray,
                                min_wave: float, max_wave: float) -> np.ndarray:
        """Match astrodash_old.sn_processing.limit_wavelength_range."""
        min_idx = int(np.abs(wave - min_wave).argmin())
        max_idx = int(np.abs(wave - max_wave).argmin())
        flux_out = np.copy(flux)
        flux_out[:min_idx] = 0.0
        flux_out[max_idx:] = 0.0
        return flux_out

    def _two_col_input_spectrum(self, wave: np.ndarray, flux: np.ndarray, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match astrodash_old.preprocessing.ReadSpectrumFile.two_col_input_spectrum,
        assuming wave/flux arrays are already loaded.
        """
        # Deredshift
        wave_new = wave / (1.0 + z)
        # Restrict to [w0, w1]
        mask = (wave_new >= self.w0) & (wave_new < self.w1)
        wave_new = wave_new[mask]
        flux_new = flux[mask]
        if wave_new.size == 0:
            raise ValueError(
                f"Spectrum out of model wavelength range [{self.w0}, {self.w1}] after deredshifting (z={z})."
            )
        # Normalise again
        flux_new = self._normalise(flux_new)
        return wave_new, flux_new

    def _log_wavelength(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Log-wavelength binning. This approximates astrodash_old.PreProcessSpectrum.log_wavelength
        using interpolation onto the same log grid.
        """
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0.0, right=0.0)
        non_zero = np.where(binned_flux != 0.0)[0]
        if non_zero.size == 0:
            return wlog, binned_flux, 0, 0
        min_idx = int(non_zero[0])
        max_idx = int(non_zero[-1])
        return wlog, binned_flux, min_idx, max_idx

    def _continuum_removal(self, wave: np.ndarray, flux: np.ndarray,
                           min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match astrodash_old.PreProcessSpectrum.continuum_removal:
        - shift flux by +1
        - spline-fit continuum in [min_idx, max_idx]
        - divide by continuum
        - normalise (minus 1) and zero outside [min_idx, max_idx]
        """
        from scipy.interpolate import UnivariateSpline

        flux_plus = flux + 1.0
        cont_removed = np.copy(flux_plus)

        continuum = np.zeros_like(flux_plus)
        if max_idx - min_idx > 5:
            spline = UnivariateSpline(
                wave[min_idx:max_idx + 1], flux_plus[min_idx:max_idx + 1], k=3
            )
            spline_wave = np.linspace(wave[min_idx], wave[max_idx],
                                      num=self.num_spline_points, endpoint=True)
            spline_points = spline(spline_wave)
            spline_more = UnivariateSpline(spline_wave, spline_points, k=3)
            spline_points_more = spline_more(wave[min_idx:max_idx + 1])
            continuum[min_idx:max_idx + 1] = spline_points_more
        else:
            # Fallback: flat continuum
            continuum[min_idx:max_idx + 1] = 1.0

        valid = continuum[min_idx:max_idx + 1] != 0
        if np.any(valid):
            cont_removed[min_idx:max_idx + 1][valid] = (
                flux_plus[min_idx:max_idx + 1][valid] / continuum[min_idx:max_idx + 1][valid]
            )

        # Normalise cont_removed - 1 and zero outside region
        cont_removed_norm = self._normalise(cont_removed - 1.0)
        cont_removed_norm[:min_idx] = 0.0
        cont_removed_norm[max_idx + 1:] = 0.0

        return cont_removed_norm, continuum - 1.0

    def _mean_zero(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """Match astrodash_old.PreProcessSpectrum.mean_zero."""
        if max_idx <= min_idx or max_idx >= flux.size:
            return flux
        mean_flux = np.mean(flux[min_idx:max_idx])
        out = flux - mean_flux
        out[:min_idx] = flux[:min_idx]
        out[max_idx + 1:] = flux[max_idx + 1:]
        return out

    def _apodize(self, flux: np.ndarray, min_idx: int, max_idx: int, outer_val: float = 0.0) -> np.ndarray:
        """
        Match astrodash_old.PreProcessSpectrum.apodize (5% cosine bell),
        with optional outer_val and zeroing outside [min_idx, max_idx].
        """
        percent = 0.05
        out = np.copy(flux) - outer_val
        nsquash = int(self.nw * percent)
        for i in range(nsquash):
            if nsquash <= 1:
                break
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1.0 - np.cos(arg))
            if (min_idx + i < self.nw) and (max_idx - i >= 0):
                out[min_idx + i] = factor * out[min_idx + i]
                out[max_idx - i] = factor * out[max_idx - i]
            else:
                break

        if outer_val != 0.0:
            out = out + outer_val
            # zero_non_overlap_part semantics
            out[:min_idx] = outer_val
            out[max_idx + 1:] = outer_val
        return out

    # ---- Public API used by the comparison script ----

    def process(self, wave: np.ndarray, flux: np.ndarray, z: float = 0.0) -> Tuple[np.ndarray, int, int]:
        """
        Full TF-style preprocessing pipeline, mirroring:
        astrodash_old.sn_processing.PreProcessing.two_column_data.
        """
        from scipy.signal import medfilt

        # 1) Initial normalisation and wavelength limiting
        flux_norm = self._normalise(flux)
        flux_limited = self._limit_wavelength_range(wave, flux_norm, self.w0, self.w1)

        # 2) Smoothing with median filter (same kernel logic)
        w_density = (self.w1 - self.w0) / self.nw
        wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
        if wavelength_density <= 0:
            filter_size = 1
        else:
            filter_size = int(w_density / wavelength_density * self.smooth / 2) * 2 + 1
        if filter_size < 1:
            filter_size = 1
        # medfilt requires odd kernel size
        if filter_size % 2 == 0:
            filter_size += 1
        pre_filtered = medfilt(flux_limited, kernel_size=filter_size)

        # 3) Deredshift + clip + re-normalise
        wave_dereds, flux_dereds = self._two_col_input_spectrum(wave, pre_filtered, z)

        # 4) Log-wavelength binning
        binned_wave, binned_flux, min_idx, max_idx = self._log_wavelength(wave_dereds, flux_dereds)

        # Guard against completely empty or pathological spectra
        if min_idx == max_idx == 0 and not np.any(binned_flux):
            return np.full(self.nw, 0.5, dtype=float), 0, 0

        # 5) Continuum removal
        cont_removed, _ = self._continuum_removal(binned_wave, binned_flux, min_idx, max_idx)

        # 6) Mean zero
        mean_zero_flux = self._mean_zero(cont_removed, min_idx, max_idx)

        # 7) Apodize (no outer value at this stage)
        apodized = self._apodize(mean_zero_flux, min_idx, max_idx, outer_val=0.0)

        # 8) Final normalisation and zero_non_overlap_part with outerVal=0.5
        final_flux = self._normalise(apodized)
        final_flux[:min_idx] = 0.5
        final_flux[max_idx + 1:] = 0.5

        return final_flux, min_idx, max_idx


class ProdBackendPreprocessor:
    """
    Preprocessing intended to mirror what the production backend *should*
    be doing for DASH, but reimplemented here with the same logic as the
    original DASH TF pipeline (no delegation to TFStylePreprocessor).
    """

    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13, smooth: int = 6):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points
        self.dwlog = np.log(w1 / w0) / nw
        self.smooth = smooth

    # ---- Helpers modelled after astrodash_old ----

    def _normalise(self, flux: np.ndarray) -> np.ndarray:
        if flux.size == 0 or np.min(flux) == np.max(flux):
            return np.zeros_like(flux)
        return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

    def _limit_wavelength_range(self, wave: np.ndarray, flux: np.ndarray,
                                min_wave: float, max_wave: float) -> np.ndarray:
        min_idx = int(np.abs(wave - min_wave).argmin())
        max_idx = int(np.abs(wave - max_wave).argmin())
        out = np.copy(flux)
        out[:min_idx] = 0.0
        out[max_idx:] = 0.0
        return out

    def _two_col_input_spectrum(self, wave: np.ndarray, flux: np.ndarray, z: float) -> Tuple[np.ndarray, np.ndarray]:
        # Deredshift and clip to [w0, w1], then renormalise
        wave_new = wave / (1.0 + z)
        mask = (wave_new >= self.w0) & (wave_new < self.w1)
        wave_new = wave_new[mask]
        flux_new = flux[mask]
        if wave_new.size == 0:
            raise ValueError(
                f"Spectrum out of wavelength range [{self.w0}, {self.w1}] after deredshifting (z={z})."
            )
        flux_new = self._normalise(flux_new)
        return wave_new, flux_new

    def _log_wavelength_binning(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0.0, right=0.0)
        non_zero = np.where(binned_flux != 0.0)[0]
        if non_zero.size == 0:
            return wlog, binned_flux, 0, 0
        min_idx = int(non_zero[0])
        max_idx = int(non_zero[-1])
        return wlog, binned_flux, min_idx, max_idx

    def _continuum_removal(self, wave: np.ndarray, flux: np.ndarray,
                           min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.interpolate import UnivariateSpline

        flux_plus = flux + 1.0
        cont_removed = np.copy(flux_plus)

        continuum = np.zeros_like(flux_plus)
        if max_idx - min_idx > 5:
            spline = UnivariateSpline(
                wave[min_idx:max_idx + 1], flux_plus[min_idx:max_idx + 1], k=3
            )
            spline_wave = np.linspace(wave[min_idx], wave[max_idx],
                                      num=self.num_spline_points, endpoint=True)
            spline_points = spline(spline_wave)
            spline_more = UnivariateSpline(spline_wave, spline_points, k=3)
            spline_points_more = spline_more(wave[min_idx:max_idx + 1])
            continuum[min_idx:max_idx + 1] = spline_points_more
        else:
            continuum[min_idx:max_idx + 1] = 1.0

        valid = continuum[min_idx:max_idx + 1] != 0
        if np.any(valid):
            cont_removed[min_idx:max_idx + 1][valid] = (
                flux_plus[min_idx:max_idx + 1][valid] / continuum[min_idx:max_idx + 1][valid]
            )

        cont_removed_norm = self._normalise(cont_removed - 1.0)
        cont_removed_norm[:min_idx] = 0.0
        cont_removed_norm[max_idx + 1:] = 0.0

        return cont_removed_norm, continuum - 1.0

    def _mean_zero(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        if max_idx <= min_idx or max_idx >= flux.size:
            return flux
        mean_flux = np.mean(flux[min_idx:max_idx])
        out = flux - mean_flux
        out[:min_idx] = flux[:min_idx]
        out[max_idx + 1:] = flux[max_idx + 1:]
        return out

    def _apodize(self, flux: np.ndarray, min_idx: int, max_idx: int, outer_val: float = 0.0) -> np.ndarray:
        percent = 0.05
        out = np.copy(flux) - outer_val
        nsquash = int(self.nw * percent)
        for i in range(nsquash):
            if nsquash <= 1:
                break
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1.0 - np.cos(arg))
            if (min_idx + i < self.nw) and (max_idx - i >= 0):
                out[min_idx + i] = factor * out[min_idx + i]
                out[max_idx - i] = factor * out[max_idx - i]
            else:
                break

        if outer_val != 0.0:
            out = out + outer_val
            out[:min_idx] = outer_val
            out[max_idx + 1:] = outer_val
        return out

    def _zero_non_overlap_part(self, array: np.ndarray, min_idx: int, max_idx: int,
                               outer_val: float = 0.5) -> np.ndarray:
        sliced = np.copy(array)
        sliced[:min_idx] = outer_val
        sliced[max_idx + 1:] = outer_val
        return sliced

    # ---- Public API (used by the script) ----

    def process(self, wave: np.ndarray, flux: np.ndarray, z: float = 0.0) -> Tuple[np.ndarray, int, int]:
        """
        Full preprocessing pipeline, reimplemented here to mirror the
        original DASH logic (same as TFStylePreprocessor.process, but
        implemented independently).
        """
        from scipy.signal import medfilt

        # 1) Initial normalisation and wavelength limiting
        flux_norm = self._normalise(flux)
        flux_limited = self._limit_wavelength_range(wave, flux_norm, self.w0, self.w1)

        # 2) Smoothing with median filter (same kernel logic)
        w_density = (self.w1 - self.w0) / self.nw
        wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
        if wavelength_density <= 0:
            filter_size = 1
        else:
            filter_size = int(w_density / wavelength_density * self.smooth / 2) * 2 + 1
        if filter_size < 1:
            filter_size = 1
        if filter_size % 2 == 0:
            filter_size += 1
        pre_filtered = medfilt(flux_limited, kernel_size=filter_size)

        # 3) Deredshift + clip + re-normalise
        wave_dereds, flux_dereds = self._two_col_input_spectrum(wave, pre_filtered, z)

        # 4) Log-wavelength binning
        binned_wave, binned_flux, min_idx, max_idx = self._log_wavelength_binning(wave_dereds, flux_dereds)

        if min_idx == max_idx == 0 and not np.any(binned_flux):
            return np.full(self.nw, 0.5, dtype=float), 0, 0

        # 5) Continuum removal
        cont_removed, _ = self._continuum_removal(binned_wave, binned_flux, min_idx, max_idx)

        # 6) Mean zero
        mean_zero_flux = self._mean_zero(cont_removed, min_idx, max_idx)

        # 7) Apodize
        apodized = self._apodize(mean_zero_flux, min_idx, max_idx, outer_val=0.0)

        # 8) Final normalisation and zero_non_overlap_part with outerVal=0.5
        final_flux = self._normalise(apodized)
        final_flux = self._zero_non_overlap_part(final_flux, min_idx, max_idx, outer_val=0.5)

        return final_flux, min_idx, max_idx


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

def run_preprocessor_parity_check(
    params: Dict[str, Any],
    num_samples: int,
    templist_path: Path,
) -> None:
    """
    Quick test: compare TFStylePreprocessor vs DashSpectrumProcessor
    on a small subset of .lnw training spectra.
    """
    # Local import so that sys.path can be configured first
    from app.infrastructure.ml.data_processor import DashSpectrumProcessor

    w0 = params["w0"]
    w1 = params["w1"]
    nw = params["nw"]

    print("\n" + "=" * 70)
    print("DashSpectrumProcessor vs TFStylePreprocessor parity check")
    print("=" * 70)
    print(f"Using w0={w0}, w1={w1}, nw={nw}, num_samples={num_samples}")

    tf_preproc = TFStylePreprocessor(w0, w1, nw)
    dash_preproc = DashSpectrumProcessor(w0, w1, nw)

    lnw_spectra = load_lnw_spectra(TRAINING_SET_DIR, templist_path, num_samples)
    if not lnw_spectra:
        print("No .lnw spectra loaded for parity check.")
        return

    diffs = []
    for i, spec in enumerate(lnw_spectra):
        wave, flux, z = spec.wave, spec.flux, spec.redshift
        tf_flux, tf_min, tf_max = tf_preproc.process(wave, flux, z)
        dash_flux, d_min, d_max, _ = dash_preproc.process(wave, flux, z, smooth=6)

        mean_diff = float(np.mean(np.abs(tf_flux - dash_flux)))
        max_diff = float(np.max(np.abs(tf_flux - dash_flux)))
        corr = (
            float(np.corrcoef(tf_flux, dash_flux)[0, 1])
            if np.std(tf_flux) > 0 and np.std(dash_flux) > 0
            else 1.0
        )
        diffs.append((mean_diff, max_diff, corr))

        print(f"\n[{i+1}/{len(lnw_spectra)}] {spec.filename}")
        print(f"  TF   min/max idx: [{tf_min}, {tf_max}]")
        print(f"  Dash min/max idx: [{d_min}, {d_max}]")
        print(f"  mean_abs_diff={mean_diff:.6e}, max_abs_diff={max_diff:.6e}, corr={corr:.6f}")

    if diffs:
        mean_mean_diff = np.mean([d[0] for d in diffs])
        mean_max_diff = np.mean([d[1] for d in diffs])
        mean_corr = np.mean([d[2] for d in diffs])
        print("\nParity summary over .lnw samples:")
        print(f"  mean(mean_abs_diff) = {mean_mean_diff:.6e}")
        print(f"  mean(max_abs_diff)  = {mean_max_diff:.6e}")
        print(f"  mean(correlation)   = {mean_corr:.6f}")


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
    parser.add_argument('--test_preprocessor_only', action='store_true',
                        help='Run DashSpectrumProcessor vs TFStylePreprocessor parity check on .lnw and exit')
    parser.add_argument('--use_prod_backend_model', action='store_true',
                        help='Use the same model paths as production backend (dash/pytorch_model.pth instead of dash/zeroZ/pytorch_model.pth)')
    args = parser.parse_args()
    
    # Paths - choose based on flag
    original_tf_dir = MODELS_DIR / "original_dash_models" / args.model_dir
    tf_model_path = original_tf_dir / "tensorflow_model.ckpt"

    if args.use_prod_backend_model:
        # PyTorch + params from prod backend (same as web app); TF from original for comparison
        pytorch_model_path = MODELS_DIR / "dash" / "pytorch_model.pth"
        params_path = MODELS_DIR / "dash" / "training_params.pickle"
        print("\n📌 Using PROD BACKEND model (same as web application):")
        print(f"   PyTorch model: {pytorch_model_path}")
        print(f"   Training params: {params_path}")
        print(f"   TF model (for comparison): {tf_model_path}")
    else:
        # All from original comparison paths
        converted_pytorch_dir = MODELS_DIR / "dash" / args.model_dir
        pytorch_model_path = converted_pytorch_dir / "pytorch_model.pth"
        params_path = original_tf_dir / "training_params.pickle"
    
    templist_path = TRAINING_SET_DIR / "templist.txt"
    
    # Validate paths (always require TF model so we can compare prod PyTorch vs original TF)
    required_paths = [
        (tf_model_path.with_suffix('.ckpt.index'), 'TF model'),
        (pytorch_model_path, 'PyTorch model'),
        (params_path, 'Training params'),
        (templist_path, 'Template list')
    ]
    
    for path, name in required_paths:
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

    # Optional: just test preprocessor parity and exit
    if args.test_preprocessor_only:
        run_preprocessor_parity_check(params, args.num_samples, templist_path)
        return
    
    # Load models
    print("\nLoading models...")
    tf_model = TensorFlowModel(str(tf_model_path))
    print("  ✓ TensorFlow model loaded")
    pytorch_model = PyTorchModel(str(pytorch_model_path), n_types=total_classes)
    print("  ✓ PyTorch model loaded")
    
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
        print(f"  Model outputs (SAME preprocessing - TF vs PyTorch):")
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
        print(f"  Model outputs (DIFFERENT preprocessing - TF vs PyTorch):")
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
        
        print(f"\nModel outputs (SAME preprocessing - TF vs PyTorch):")
        print(f"  Correlation: avg={np.mean([c['correlation'] for c in all_same]):.6f}")
        print(f"  Top-1 match: {sum(c['top1_match'] for c in all_same)}/{len(all_same)} ({100*sum(c['top1_match'] for c in all_same)/len(all_same):.1f}%)")
        
        print(f"\nModel outputs (DIFFERENT preprocessing - TF vs PyTorch):")
        print(f"  Correlation: avg={np.mean([c['correlation'] for c in all_diff]):.6f}")
        print(f"  Top-1 match: {sum(c['top1_match'] for c in all_diff)}/{len(all_diff)} ({100*sum(c['top1_match'] for c in all_diff)/len(all_diff):.1f}%)")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if args.use_prod_backend_model:
        print("\n📌 PROD BACKEND mode: PyTorch = web app model; TF = original (zeroZ) for comparison.")
        print("   Probabilities like 0.55 vs 0.9999 for SN2002er show the prod backend model differs from the original TF model.")
    
    preproc_corr = np.mean([c['correlation'] for c in all_preproc]) if all_preproc else 0.0
    if all_same:
        same_corr = np.mean([c['correlation'] for c in all_same])
        same_match = sum(c['top1_match'] for c in all_same) / len(all_same)
        diff_corr = np.mean([c['correlation'] for c in all_diff])
        diff_match = sum(c['top1_match'] for c in all_diff) / len(all_diff)
        
        if same_corr > 0.9999 and same_match > 0.95:
            if not args.use_prod_backend_model:
                print("\n✅ PyTorch model weights are correctly converted from TensorFlow.")
            if diff_corr < same_corr - 0.001 or diff_match < same_match - 0.05:
                if args.use_prod_backend_model:
                    print("   TF (original) vs PyTorch (prod) differ because they are different models.")
                else:
                    print("⚠️  Preprocessing differences are causing some output discrepancies.")
                    print("   The prod_backend preprocessing differs from original DASH preprocessing.")
            else:
                print("✅ Preprocessing pipelines produce nearly identical results.")
        elif same_corr < 0.999 and not args.use_prod_backend_model:
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
