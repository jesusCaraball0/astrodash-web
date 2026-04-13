"""
Duplicated DASH `process()` pipeline with explicit toggles so test eval matches each trained run
regardless of what is commented in prod `data_processor.py`.

Used by dash_save_variant_test_metrics.py and (indirectly) dash_preprocessing_removal_diff_plot.py
via saved `model_performance.json` → `test_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline

import constants as const
from prod_backend.app.shared.utils.validators import validate_spectrum, ValidationError


@dataclass(frozen=True)
class PreprocessFlags:
    """Flags match training ablations: when False, the step is skipped or uses a straight copy."""

    initial_norm: bool = True
    use_medfilt: bool = True
    deredshift: bool = True
    norm_after_slice: bool = True
    use_continuum_removal: bool = True
    continuum_tail_norm: bool = True
    mean_zero: bool = True
    use_apodize: bool = True
    final_norm: bool = True


def _norm(proc, flux: np.ndarray, use: bool) -> np.ndarray:
    if use:
        return proc.normalise_spectrum(flux)
    return np.asarray(flux, dtype=float).copy()


def _continuum_removal_variant(
    proc,
    wave: np.ndarray,
    flux: np.ndarray,
    min_idx: int,
    max_idx: int,
    continuum_tail_norm: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as DashSpectrumProcessor.continuum_removal; `continuum_tail_norm` mirrors ablating the final normalise in that step."""
    min_idx = int(np.clip(min_idx, 0, len(flux) - 1))
    max_idx = int(np.clip(max_idx, min_idx, len(flux) - 1))

    flux_plus = flux + 1.0
    cont_removed = np.copy(flux_plus)

    continuum = np.zeros_like(flux_plus)
    if (max_idx - min_idx) > 5:
        spline = UnivariateSpline(
            wave[min_idx : max_idx + 1], flux_plus[min_idx : max_idx + 1], k=3
        )
        spline_wave = np.linspace(wave[min_idx], wave[max_idx], num=proc.num_spline_points, endpoint=True)
        spline_points = spline(spline_wave)
        spline_more = UnivariateSpline(spline_wave, spline_points, k=3)
        spline_points_more = spline_more(wave[min_idx : max_idx + 1])
        continuum[min_idx : max_idx + 1] = spline_points_more
    else:
        continuum[min_idx : max_idx + 1] = 1.0

    valid = continuum[min_idx : max_idx + 1] != 0
    if np.any(valid):
        cont_removed[min_idx : max_idx + 1][valid] = (
            flux_plus[min_idx : max_idx + 1][valid] / continuum[min_idx : max_idx + 1][valid]
        )

    shifted = cont_removed - 1.0
    if continuum_tail_norm:
        cont_removed_norm = proc.normalise_spectrum(shifted)
    else:
        cont_removed_norm = np.copy(shifted)
    cont_removed_norm[:min_idx] = 0.0
    cont_removed_norm[max_idx + 1 :] = 0.0

    return cont_removed_norm, continuum - 1.0


def process_dash_with_flags(
    wave: np.ndarray,
    flux: np.ndarray,
    z: float,
    flags: PreprocessFlags,
    smooth: int = 0,
    min_wave: Optional[float] = None,
    max_wave: Optional[float] = None,
):
    """
    Return (flux_out, min_idx, max_idx, z) matching DashSpectrumProcessor.process when flags are all True.
    """
    proc = const._PROCESSOR
    validate_spectrum(wave.tolist(), flux.tolist(), z)

    wave = np.asarray(wave, dtype=float).copy()
    flux = np.asarray(flux, dtype=float).copy()
    if wave.size > 1 and wave[0] > wave[-1]:
        perm = np.argsort(wave)
        wave = wave[perm]
        flux = flux[perm]

    flux_norm = _norm(proc, flux, flags.initial_norm)
    effective_min = proc.w0 if min_wave is None else min_wave
    effective_max = proc.w1 if max_wave is None else max_wave
    flux_limited = proc.limit_wavelength_range(wave, flux_norm, effective_min, effective_max)

    effective_smooth = smooth if smooth > 0 else 6
    w_density = (proc.w1 - proc.w0) / proc.nw
    wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
    if wavelength_density <= 0:
        filter_size = 1
    else:
        filter_size = int(w_density / wavelength_density * effective_smooth / 2) * 2 + 1
    if filter_size < 1:
        filter_size = 1
    if filter_size % 2 == 0:
        filter_size += 1
    n_flux = len(flux_limited)
    if filter_size > n_flux:
        filter_size = n_flux if n_flux % 2 == 1 else max(1, n_flux - 1)

    if flags.use_medfilt:
        flux_smoothed = medfilt(flux_limited, kernel_size=filter_size)
    else:
        flux_smoothed = np.asarray(flux_limited, dtype=float).copy()

    if flags.deredshift:
        wave_deredshifted = wave / (1 + z)
        if len(wave_deredshifted) < 2:
            raise ValidationError("Spectrum is out of classification range after deredshifting")
        mask = (wave_deredshifted >= proc.w0) & (wave_deredshifted < proc.w1)
        wave_dereds = wave_deredshifted[mask]
        flux_dereds = flux_smoothed[mask]
        if wave_dereds.size == 0:
            raise ValidationError(
                f"Spectrum out of wavelength range [{proc.w0}, {proc.w1}] after deredshifting"
            )
    else:
        if len(wave) < 2:
            raise ValidationError("Spectrum is out of classification range after deredshifting")
        mask = (wave >= proc.w0) & (wave < proc.w1)
        wave_dereds = wave[mask]
        flux_dereds = flux_smoothed[mask]
        if wave_dereds.size == 0:
            raise ValidationError(
                f"Spectrum out of wavelength range [{proc.w0}, {proc.w1}] after deredshifting"
            )

    flux_dereds = _norm(proc, flux_dereds, flags.norm_after_slice)

    binned_wave, binned_flux, min_idx, max_idx = proc.log_wavelength_binning(wave_dereds, flux_dereds)

    if min_idx == max_idx == 0 and not np.any(binned_flux):
        flat = np.full(proc.nw, proc.DEFAULT_OUTER_VAL, dtype=float)
        return flat, 0, 0, z

    if flags.use_continuum_removal:
        cont_removed, _ = _continuum_removal_variant(
            proc, binned_wave, binned_flux, min_idx, max_idx, flags.continuum_tail_norm
        )
    else:
        cont_removed = np.copy(binned_flux)

    if flags.mean_zero:
        mean_zero_flux = proc.mean_zero(cont_removed, min_idx, max_idx)
    else:
        mean_zero_flux = np.copy(cont_removed)

    if flags.use_apodize:
        apodized_flux = proc.apodize(mean_zero_flux, min_idx, max_idx)
    else:
        apodized_flux = np.copy(mean_zero_flux)

    flux_norm_final = _norm(proc, apodized_flux, flags.final_norm)
    flux_norm_final = proc.zero_non_overlap_part(
        flux_norm_final, min_idx, max_idx, proc.DEFAULT_OUTER_VAL
    )

    return flux_norm_final, min_idx, max_idx, z


def preprocess_spectrum_variant(
    wave: np.ndarray,
    flux: np.ndarray,
    redshift: Optional[float],
    target_length: int,
    flags: PreprocessFlags,
) -> Optional[np.ndarray]:
    """Same contract as helpers.preprocess_spectrum: returns nw-bin flux only; parquet dataset appends z."""
    del target_length  # unused; kept for call compatibility
    try:
        if redshift is None:
            return None
        processed_flux, _, _, _ = process_dash_with_flags(
            wave, flux, float(redshift), flags, smooth=0, min_wave=None, max_wave=None
        )
        return processed_flux.astype(np.float32)
    except Exception:
        return None
