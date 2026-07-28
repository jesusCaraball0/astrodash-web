from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import astropy.units as u
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler

# -----------------------------------------------------------------------------
# Default --outdir (which preprocess bundle: data_z vs data_no_z)
# -----------------------------------------------------------------------------
_WISEREP_DIR = pathlib.Path(__file__).resolve().parent
TEST_ROOT = _WISEREP_DIR / "Test"
USE_REDSHIFT_CORRECTED_DATA = True
DEFAULT_PREPROCESS_OUTDIR = TEST_ROOT / ("data_z" if USE_REDSHIFT_CORRECTED_DATA else "data_no_z")


def make_grid(lam_min: float, lam_max: float, dlam: float) -> Tuple[np.ndarray, np.ndarray]:
    nbins = int(np.floor((lam_max - lam_min) / dlam))
    if nbins < 1:
        raise ValueError(f"Grid has <1 bin. lam_min={lam_min}, lam_max={lam_max}, dlam={dlam}")
    edges = lam_min + dlam * np.arange(nbins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges


def to_angstrom(lam: np.ndarray, wl_units: Any) -> Tuple[np.ndarray, bool]:
    if wl_units is None:
        return lam, False
    try:
        if pd.isna(wl_units):
            return lam, False
    except Exception:
        pass
    s = str(wl_units).strip().lower()
    if s in {"angstrom", "ang", "a", "å", "aa"}:
        return lam, True
    if s == "nm":
        return lam * 10.0, True
    if s in {"micrometre", "micrometer", "um", "µm"}:
        return lam * 1e4, True
    return lam, False


def vacuum_to_air(lam_vac: np.ndarray) -> np.ndarray:
    lam_air = np.asarray(lam_vac, dtype=np.float64).copy()
    good = np.isfinite(lam_air) & (lam_air > 2000.0)
    if np.any(good):
        s2 = (1e4 / lam_air[good]) ** 2
        n = 1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2) + 0.00015998 / (38.9 - s2)
        lam_air[good] = lam_air[good] / n
    return lam_air


def deredshift(lam_obs: np.ndarray, z: float) -> np.ndarray:
    return lam_obs / (1.0 + z)


def parse_redshift(z_raw: Any, z_max: float = 6.0) -> Tuple[Optional[float], str]:
    if z_raw is None:
        return None, "missing_z"
    try:
        if pd.isna(z_raw):
            return None, "missing_z"
    except Exception:
        pass
    s = str(z_raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None, "missing_z"
    try:
        z = float(s)
    except Exception:
        return None, "missing_z"
    if not np.isfinite(z):
        return None, "missing_z"
    if z <= 0.0:
        return None, "nonpositive_z"
    if z > z_max:
        return None, "z_too_high"
    return z, "ok"


def parse_sn_name(meta_row: pd.Series) -> Tuple[Optional[str], str]:
    for col in ["IAU name", "Name", "Obj. Name", "Object", "SN Name", "Internal name/s"]:
        if col in meta_row.index:
            val = meta_row.get(col)
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            s = str(val).strip()
            if s and s.lower() not in {"nan", "none", "unknown", "?"}:
                return s, "ok"
    return None, "missing_sn_name"


def parse_already_deredshifted_warning(row: pd.Series) -> bool:
    remarks = row.get("Remarks", "")
    try:
        if pd.isna(remarks):
            return False
    except Exception:
        pass
    s = str(remarks).strip().lower()
    if not s:
        return False
    return ("de-redshifted" in s and "please check" in s) or ("open supernova catalog" in s and "de-redshifted" in s)


def clean_spectrum(lam: np.ndarray, flx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(lam) & np.isfinite(flx)
    lam, flx = lam[good], flx[good]
    if lam.size < 2:
        return lam, flx
    order = np.argsort(lam)
    lam, flx = lam[order], flx[order]
    uniq = np.concatenate([[True], lam[1:] > lam[:-1]])
    return lam[uniq], flx[uniq]


def resample_to_grid(lam_AA: np.ndarray, flx: np.ndarray, grid_centers: np.ndarray) -> np.ndarray:
    spec = Spectrum(spectral_axis=lam_AA * u.AA, flux=flx * u.dimensionless_unscaled)
    out = FluxConservingResampler()(spec, grid_centers * u.AA)
    f = out.flux.value
    if np.ma.isMaskedArray(f):
        f = np.asarray(f.filled(np.nan), dtype=np.float64)
    else:
        f = np.asarray(f, dtype=np.float64)
    f[grid_centers < lam_AA.min()] = np.nan
    f[grid_centers > lam_AA.max()] = np.nan
    return f


def normalize_median_abs(flux: np.ndarray, min_finite: int = 50, eps: float = 1e-20) -> Tuple[np.ndarray, float, bool]:
    finite_mask = np.isfinite(flux)
    if int(np.sum(finite_mask)) < min_finite:
        return np.full_like(flux, np.nan), np.nan, False
    scale = float(np.median(np.abs(flux[finite_mask])))
    if scale < eps or not np.isfinite(scale):
        return np.full_like(flux, np.nan), np.nan, False
    return flux / scale, scale, True


def make_mask(flux: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(flux), 1.0, 0.0).astype(np.float32)


def fill_nans(flux: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = flux.copy()
    out[~np.isfinite(out)] = fill_value
    return out


def is_kept_type(obj_type: Any) -> Tuple[bool, str]:
    if obj_type is None:
        return False, "missing_obj_type"
    try:
        if pd.isna(obj_type):
            return False, "missing_obj_type"
    except Exception:
        pass
    s = str(obj_type).strip()
    if not s or s.lower() in {"nan", "none", "unknown", "?", "na/unknown"}:
        return False, "missing_obj_type"
    su = s.upper()
    if su.startswith("SN") or su.startswith("SLSN"):
        return True, "ok"
    if su == "KILONOVA":
        return True, "ok"
    if su.startswith("COMPUTED"):
        return True, "ok"
    return False, f"non_target:{s}"


def build_meta_lookup(meta_df: pd.DataFrame) -> pd.DataFrame:
    if "Spec. ID" not in meta_df.columns:
        raise RuntimeError(f"'Spec. ID' missing from metadata. Found: {list(meta_df.columns)}")
    meta = meta_df.copy()
    meta["Spec. ID"] = meta["Spec. ID"].astype(str).str.strip()
    return meta.drop_duplicates(subset=["Spec. ID"], keep="first").set_index("Spec. ID", drop=False)


def compute_overlap_frac(lam_min: float, lam_max: float, grid_min: float, grid_max: float) -> float:
    span = grid_max - grid_min
    return max(0.0, min(lam_max, grid_max) - max(lam_min, grid_min)) / span


def check_resampled(
    flux: np.ndarray,
    lam_in_min: float,
    lam_in_max: float,
    grid_min: float,
    grid_max: float,
    min_overlap_frac: float,
    min_finite_frac: float,
) -> Tuple[bool, str]:
    overlap = compute_overlap_frac(lam_in_min, lam_in_max, grid_min, grid_max)
    if overlap < min_overlap_frac:
        return False, f"low_overlap:{overlap:.3f}"
    finite_frac = float(np.mean(np.isfinite(flux)))
    if finite_frac < min_finite_frac:
        return False, f"low_finite_frac:{finite_frac:.3f}"
    finite_flux = flux[np.isfinite(flux)]
    if len(finite_flux) == 0 or np.all(finite_flux == 0):
        return False, "all_zero_or_empty"
    return True, "ok"


def evaluate_candidate(
    lam_work: np.ndarray,
    flx_raw: np.ndarray,
    grid_centers: np.ndarray,
    grid_min: float,
    grid_max: float,
    min_overlap_frac: float,
    min_finite_frac: float,
) -> Dict[str, Any]:
    lam_min = float(np.min(lam_work))
    lam_max = float(np.max(lam_work))
    overlap = compute_overlap_frac(lam_min, lam_max, grid_min, grid_max)

    try:
        f_resampled = resample_to_grid(lam_work, flx_raw, grid_centers)
    except Exception as e:
        return {
            "resample_ok": False,
            "quality_ok": False,
            "quality_reason": f"resample_fail:{type(e).__name__}:{e}",
            "finite_frac": -1.0,
            "overlap": overlap,
            "lam_work": lam_work,
            "lam_min": lam_min,
            "lam_max": lam_max,
            "f_resampled": None,
            "score": (-1, -1.0, -1.0),
        }

    finite_frac = float(np.mean(np.isfinite(f_resampled)))
    quality_ok, quality_reason = check_resampled(
        f_resampled,
        lam_min,
        lam_max,
        grid_min,
        grid_max,
        min_overlap_frac,
        min_finite_frac,
    )

    return {
        "resample_ok": True,
        "quality_ok": quality_ok,
        "quality_reason": quality_reason,
        "finite_frac": finite_frac,
        "overlap": overlap,
        "lam_work": lam_work,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "f_resampled": f_resampled,
        "score": (int(quality_ok), finite_frac, overlap),
    }


def choose_frame_for_row(
    row: pd.Series,
    lam_AA: np.ndarray,
    flx_raw: np.ndarray,
    z: float,
    grid_centers: np.ndarray,
    grid_min: float,
    grid_max: float,
    min_overlap_frac: float,
    min_finite_frac: float,
) -> Dict[str, Any]:
    warning_flag = parse_already_deredshifted_warning(row)

    cand_der = evaluate_candidate(
        deredshift(lam_AA, z),
        flx_raw,
        grid_centers,
        grid_min,
        grid_max,
        min_overlap_frac,
        min_finite_frac,
    )

    if not warning_flag:
        cand_der["frame_choice"] = "deredshift"
        cand_der["warning_flag"] = False
        cand_der["compare_mode"] = False
        return cand_der

    cand_asis = evaluate_candidate(
        lam_AA,
        flx_raw,
        grid_centers,
        grid_min,
        grid_max,
        min_overlap_frac,
        min_finite_frac,
    )

    if cand_asis["score"] >= cand_der["score"]:
        chosen = cand_asis
        chosen["frame_choice"] = "asis"
    else:
        chosen = cand_der
        chosen["frame_choice"] = "deredshift"

    chosen["warning_flag"] = True
    chosen["compare_mode"] = True
    chosen["asis_overlap"] = cand_asis["overlap"]
    chosen["asis_finite_frac"] = cand_asis["finite_frac"]
    chosen["asis_quality_ok"] = cand_asis["quality_ok"]
    chosen["asis_quality_reason"] = cand_asis["quality_reason"]
    chosen["der_overlap"] = cand_der["overlap"]
    chosen["der_finite_frac"] = cand_der["finite_frac"]
    chosen["der_quality_ok"] = cand_der["quality_ok"]
    chosen["der_quality_reason"] = cand_der["quality_reason"]
    return chosen


def print_progress(i: int, n: int, counts: Dict[str, int], t0: float, every: int, force: bool = False):
    if every <= 0:
        return
    if not force and (i + 1) % every != 0:
        return
    done = i + 1
    pct = 100.0 * done / max(n, 1)
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    skip_z = counts.get("skip_missing_z", 0) + counts.get("skip_nonpositive_z", 0) + counts.get("skip_bad_z", 0)
    print(
        f"[{done}/{n} | {pct:5.1f}%] "
        f"kept={counts['kept']} "
        f"skip_z={skip_z} "
        f"skip_quality={counts.get('skip_quality', 0)} "
        f"skip_units={counts.get('skip_bad_wl_units', 0)} "
        f"skip_norm={counts.get('skip_norm_fail', 0)} "
        f"warn_asis={counts.get('warning_choose_asis', 0)} "
        f"warn_der={counts.get('warning_choose_der', 0)} "
        f"elapsed={elapsed/60.0:.1f}m "
        f"rate={rate:.1f} rows/s"
    )


def plot_example(outdir, grid_centers, lam_orig, flx_orig, f_resampled, f_norm, mask, spec_id, z, frame_choice):
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=False)

    axes[0].plot(lam_orig, flx_orig, lw=0.8, color="steelblue")
    axes[0].set_title(f"1. Input spectrum  (Spec. ID={spec_id}  z={z:.4f}  frame_choice={frame_choice})")
    axes[0].set_ylabel("Raw flux")
    axes[0].set_xlabel("Input wavelength [Å]")

    axes[1].plot(grid_centers, f_resampled, lw=0.9, color="darkorange")
    axes[1].set_title("2. Resampled to working grid")
    axes[1].set_ylabel("Flux")
    axes[1].set_xlabel("Working wavelength [Å]")

    axes[2].plot(grid_centers, f_norm, lw=0.9, color="mediumseagreen")
    axes[2].set_title("3. Median-abs normalized")
    axes[2].set_ylabel("Normalized flux")
    axes[2].set_xlabel("Working wavelength [Å]")

    axes[3].fill_between(grid_centers, 0, mask, step="mid", alpha=0.7, color="mediumpurple")
    axes[3].set_ylim(-0.05, 1.15)
    axes[3].set_title("4. Coverage mask")
    axes[3].set_ylabel("Mask")
    axes[3].set_xlabel("Working wavelength [Å]")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "example_preprocessing.png"), dpi=150)
    plt.close()
    print(f"Saved example_preprocessing.png (Spec. ID={spec_id} z={z:.4f} frame_choice={frame_choice})")


def plot_dataset_stats(outdir, coverage_fracs, norm_scales, lam_in_mins, lam_in_maxs, redshifts, z_max):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    axes[0, 0].hist(coverage_fracs, bins=60, color="steelblue", edgecolor="white", lw=0.3)
    axes[0, 0].axvline(np.median(coverage_fracs), color="red", ls="--", label=f"median={np.median(coverage_fracs):.2f}")
    axes[0, 0].set_xlabel("Coverage fraction")
    axes[0, 0].set_title("Coverage fraction")
    axes[0, 0].legend()

    log_scales = np.log10(norm_scales[norm_scales > 0])
    axes[0, 1].hist(log_scales, bins=60, color="darkorange", edgecolor="white", lw=0.3)
    axes[0, 1].set_xlabel("log10(norm scale)")
    axes[0, 1].set_title("Normalization scale")

    axes[1, 0].hist(lam_in_mins, bins=60, color="mediumseagreen", edgecolor="white", lw=0.3)
    axes[1, 0].set_xlabel("Working λ_min [Å]")
    axes[1, 0].set_title("Blue wavelength limit")

    axes[1, 1].hist(lam_in_maxs, bins=60, color="mediumpurple", edgecolor="white", lw=0.3)
    axes[1, 1].set_xlabel("Working λ_max [Å]")
    axes[1, 1].set_title("Red wavelength limit")

    z_plot = redshifts[np.isfinite(redshifts) & (redshifts > 0) & (redshifts <= z_max)]
    axes[2, 0].hist(z_plot, bins=60, color="firebrick", edgecolor="white", lw=0.3)
    axes[2, 0].set_xlabel("Redshift z")
    axes[2, 0].set_title("Redshift distribution (kept spectra)")

    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dataset_stats.png"), dpi=150)
    plt.close()
    print("Saved dataset_stats.png")


def process(spec_df, meta_lookup, grid_centers, grid_edges, args, example_idx):
    n = len(spec_df)
    n_wave = len(grid_centers)
    grid_min, grid_max = float(grid_centers.min()), float(grid_centers.max())

    flux_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    kept_rows: List[int] = []
    kept_ids: List[str] = []
    meta_out: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    coverage_fracs: List[float] = []
    norm_scales: List[float] = []
    lam_in_mins: List[float] = []
    lam_in_maxs: List[float] = []
    redshifts_kept: List[float] = []

    example_data = None

    counts: Dict[str, int] = {
        "total": n,
        "kept": 0,
        "skip_no_meta": 0,
        "skip_non_target": 0,
        "skip_missing_obj_type": 0,
        "skip_missing_sn_name": 0,
        "skip_missing_z": 0,
        "skip_nonpositive_z": 0,
        "skip_bad_z": 0,
        "skip_bad_wl_units": 0,
        "skip_too_few_points": 0,
        "skip_quality": 0,
        "skip_norm_fail": 0,
        "skip_resample_fail": 0,
        "warning_rows": 0,
        "warning_choose_asis": 0,
        "warning_choose_der": 0,
        "total_clipped_bins": 0,
    }

    t0 = time.time()

    pbar = None
    if getattr(args, "use_tqdm", False):
        try:
            from tqdm import tqdm

            pbar = tqdm(
                range(n),
                desc="Preprocess spectra",
                unit="spec",
                ncols=100,
                mininterval=0.3,
            )
            row_iter = pbar
        except ImportError:
            row_iter = range(n)
            print("[Preprocess] tqdm not installed; pip install tqdm for a progress bar")
    else:
        row_iter = range(n)

    for i in row_iter:
        row = spec_df.iloc[i]

        spec_id_raw = row.get("Spec. ID")
        try:
            spec_id = None if pd.isna(spec_id_raw) else str(spec_id_raw).strip()
        except Exception:
            spec_id = str(spec_id_raw).strip()

        wl_units = row.get("WL Units")
        wl_medium = row.get("WL Medium")
        coeff_raw = row.get("Flux Unit Coefficient")

        if spec_id is None or spec_id not in meta_lookup.index:
            counts["skip_no_meta"] += 1
            report.append({"spec_id": spec_id, "row": i, "status": "skip:no_metadata"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        meta_row = meta_lookup.loc[spec_id]
        obj_type = meta_row.get("Obj. Type")
        z_raw = meta_row.get("Redshift")
        obs_date = meta_row.get("Obs-date") if "Obs-date" in meta_row.index else row.get("Obs-date")
        jd_obs = meta_row.get("JD") if "JD" in meta_row.index else row.get("JD")

        sn_name, sn_flag = parse_sn_name(meta_row)
        if sn_flag != "ok":
            counts["skip_missing_sn_name"] += 1
            report.append({"spec_id": spec_id, "row": i, "status": "skip:missing_sn_name"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        keep, reason = is_kept_type(obj_type)
        if not keep:
            key = "skip_missing_obj_type" if reason == "missing_obj_type" else "skip_non_target"
            counts[key] += 1
            report.append({"spec_id": spec_id, "row": i, "obj_type": obj_type, "status": f"skip:{reason}"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        z, z_flag = parse_redshift(z_raw, z_max=args.z_max)
        if z_flag != "ok":
            if z_flag == "missing_z":
                counts["skip_missing_z"] += 1
            elif z_flag == "nonpositive_z":
                counts["skip_nonpositive_z"] += 1
            elif z_flag == "z_too_high":
                counts["skip_bad_z"] += 1
            report.append({"spec_id": spec_id, "row": i, "z_raw": z_raw, "status": f"skip:{z_flag}"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        try:
            lam_raw = np.asarray(row["wavelength"], dtype=np.float64).ravel()
            flx_raw = np.asarray(row["flux"], dtype=np.float64).ravel()
        except Exception as e:
            counts["skip_too_few_points"] += 1
            report.append({"spec_id": spec_id, "row": i, "status": f"skip:bad_array:{e}"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        lam_AA, ok_units = to_angstrom(lam_raw, wl_units)
        if not ok_units:
            counts["skip_bad_wl_units"] += 1
            report.append({"spec_id": spec_id, "row": i, "wl_units": wl_units, "status": "skip:bad_wl_units"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        if wl_medium is not None:
            try:
                if not pd.isna(wl_medium) and str(wl_medium).strip().lower() == "vacuum":
                    lam_AA = vacuum_to_air(lam_AA)
            except Exception:
                pass

        try:
            c = float(coeff_raw) if (coeff_raw is not None and pd.notna(coeff_raw)) else 1.0
        except Exception:
            c = 1.0
        flx_raw = flx_raw * c

        lam_AA, flx_raw = clean_spectrum(lam_AA, flx_raw)
        if lam_AA.size < 2:
            counts["skip_too_few_points"] += 1
            report.append({"spec_id": spec_id, "row": i, "status": "skip:too_few_points"})
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        frame_eval = choose_frame_for_row(
            row,
            lam_AA,
            flx_raw,
            z,
            grid_centers,
            grid_min,
            grid_max,
            args.min_overlap_frac,
            args.min_finite_frac,
        )

        if frame_eval["warning_flag"]:
            counts["warning_rows"] += 1
            if frame_eval["frame_choice"] == "asis":
                counts["warning_choose_asis"] += 1
            else:
                counts["warning_choose_der"] += 1

        if not frame_eval["resample_ok"]:
            counts["skip_resample_fail"] += 1
            report.append({
                "spec_id": spec_id,
                "row": i,
                "sn_name": sn_name,
                "obs_date": obs_date,
                "jd_obs": jd_obs,
                "frame_choice": frame_eval["frame_choice"],
                "warning_flag": frame_eval["warning_flag"],
                "status": f"skip:{frame_eval['quality_reason']}",
            })
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        lam_work = frame_eval["lam_work"]
        f_resampled = frame_eval["f_resampled"]
        lam_work_min = frame_eval["lam_min"]
        lam_work_max = frame_eval["lam_max"]

        if not frame_eval["quality_ok"]:
            counts["skip_quality"] += 1
            report.append({
                "spec_id": spec_id,
                "row": i,
                "sn_name": sn_name,
                "obs_date": obs_date,
                "jd_obs": jd_obs,
                "frame_choice": frame_eval["frame_choice"],
                "warning_flag": frame_eval["warning_flag"],
                "status": f"skip:{frame_eval['quality_reason']}",
                "lam_work_min": lam_work_min,
                "lam_work_max": lam_work_max,
                "finite_frac": frame_eval["finite_frac"],
                "overlap": frame_eval["overlap"],
            })
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        f_norm, norm_scale, norm_ok = normalize_median_abs(f_resampled, min_finite=args.min_finite_norm)
        if not norm_ok:
            counts["skip_norm_fail"] += 1
            report.append({
                "spec_id": spec_id,
                "row": i,
                "sn_name": sn_name,
                "obs_date": obs_date,
                "jd_obs": jd_obs,
                "frame_choice": frame_eval["frame_choice"],
                "warning_flag": frame_eval["warning_flag"],
                "status": "skip:norm_fail",
            })
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        f_norm = np.where(np.isfinite(f_norm), f_norm, np.nan)
        n_clipped_bins = int(np.sum((f_norm < -50.0) | (f_norm > 50.0)))
        counts["total_clipped_bins"] += n_clipped_bins
        f_norm = np.clip(f_norm, -50.0, 50.0)

        if not np.any(np.isfinite(f_norm)):
            counts["skip_norm_fail"] += 1
            report.append({
                "spec_id": spec_id,
                "row": i,
                "sn_name": sn_name,
                "obs_date": obs_date,
                "jd_obs": jd_obs,
                "frame_choice": frame_eval["frame_choice"],
                "warning_flag": frame_eval["warning_flag"],
                "status": "skip:all_inf_after_norm",
            })
            if args.debug:
                print_progress(i, n, counts, t0, args.print_every)
            continue

        mask = make_mask(f_norm)
        f_filled = fill_nans(f_norm, fill_value=0.0)
        coverage_frac = float(mask.mean())

        flux_list.append(f_filled.astype(np.float32))
        mask_list.append(mask)
        kept_rows.append(i)
        kept_ids.append(spec_id)

        coverage_fracs.append(coverage_frac)
        norm_scales.append(norm_scale)
        lam_in_mins.append(lam_work_min)
        lam_in_maxs.append(lam_work_max)
        redshifts_kept.append(z)

        meta_rec = {
            "spec_id": spec_id,
            "sn_name": sn_name,
            "obj_type": obj_type,
            "obs_date": obs_date,
            "jd_obs": jd_obs,
            "frame_choice": frame_eval["frame_choice"],
            "warning_flag": frame_eval["warning_flag"],
            "dataset_idx": counts["kept"],
            "redshift": z,
            "z_flag": z_flag,
            "norm_scale": norm_scale,
            "coverage_frac": coverage_frac,
            "lam_work_min": lam_work_min,
            "lam_work_max": lam_work_max,
            "n_finite_bins": int(np.sum(np.isfinite(f_norm))),
            "n_clipped_bins": n_clipped_bins,
        }

        if frame_eval["warning_flag"]:
            meta_rec["asis_overlap"] = frame_eval["asis_overlap"]
            meta_rec["asis_finite_frac"] = frame_eval["asis_finite_frac"]
            meta_rec["asis_quality_ok"] = frame_eval["asis_quality_ok"]
            meta_rec["asis_quality_reason"] = frame_eval["asis_quality_reason"]
            meta_rec["der_overlap"] = frame_eval["der_overlap"]
            meta_rec["der_finite_frac"] = frame_eval["der_finite_frac"]
            meta_rec["der_quality_ok"] = frame_eval["der_quality_ok"]
            meta_rec["der_quality_reason"] = frame_eval["der_quality_reason"]

        meta_out.append(meta_rec)
        report.append({**meta_rec, "row": i, "status": "ok"})
        counts["kept"] += 1

        if i == example_idx and example_data is None:
            example_data = {
                "lam_orig": lam_AA.copy(),
                "flx_orig": flx_raw.copy(),
                "f_resampled": f_resampled.copy(),
                "f_norm": f_norm.copy(),
                "mask": mask.copy(),
                "spec_id": spec_id,
                "z": z,
                "frame_choice": frame_eval["frame_choice"],
            }

        if args.debug:
            print_progress(i, n, counts, t0, args.print_every)

        if pbar is not None and (i % 250 == 0 or i == n - 1):
            pbar.set_postfix_str(f"kept={counts['kept']}", refresh=False)

    if pbar is not None:
        pbar.close()

    if args.debug and n > 0:
        print_progress(n - 1, n, counts, t0, args.print_every, force=True)

    N = counts["kept"]
    if N > 0:
        flux_array = np.stack(flux_list).astype(np.float32)
        mask_array = np.stack(mask_list).astype(np.float32)
    else:
        flux_array = np.empty((0, n_wave), dtype=np.float32)
        mask_array = np.empty((0, n_wave), dtype=np.float32)

    return {
        "flux_array": flux_array,
        "mask_array": mask_array,
        "kept_rows": kept_rows,
        "kept_ids": kept_ids,
        "meta_out": meta_out,
        "report": report,
        "counts": counts,
        "example_data": example_data,
        "coverage_fracs": np.array(coverage_fracs, dtype=np.float32),
        "norm_scales": np.array(norm_scales, dtype=np.float64),
        "lam_in_mins": np.array(lam_in_mins, dtype=np.float32),
        "lam_in_maxs": np.array(lam_in_maxs, dtype=np.float32),
        "redshifts": np.array(redshifts_kept, dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description="Preprocess WISeREP spectra onto a common wavelength grid, ready for daep.")

    ap.add_argument("--spectra-parquet", default="/projects/ncsa/caps/uiucsn/tfm/tfm_spectra_data/wiserep/wiserep_spectra.parquet")
    ap.add_argument("--metadata-csv", default="/projects/ncsa/caps/uiucsn/tfm/tfm_spectra_data/wiserep/wiserep_metadata.csv")
    ap.add_argument(
        "--outdir",
        default=str(DEFAULT_PREPROCESS_OUTDIR),
        help="Default: Test/data_z or Test/data_no_z per USE_REDSHIFT_CORRECTED_DATA at top of this file.",
    )

    ap.add_argument("--lam-min", type=float, default=3200.0)
    ap.add_argument("--lam-max", type=float, default=9700.0)
    ap.add_argument("--dlam", type=float, default=2.0)

    ap.add_argument("--min-overlap-frac", type=float, default=0.20)
    ap.add_argument("--min-finite-frac", type=float, default=0.10)
    ap.add_argument("--min-finite-norm", type=int, default=50)

    ap.add_argument("--z-max", type=float, default=6.0)
    ap.add_argument("--example-seed", type=int, default=42)
    ap.add_argument("--example-index", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--print-every", type=int, default=500)
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm bar in process()")

    args = ap.parse_args()
    args.use_tqdm = not args.no_tqdm
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("Loading spectra parquet...")
    spec_df = pd.read_parquet(args.spectra_parquet)
    print(f"{len(spec_df)} rows")

    print("Loading metadata CSV...")
    meta_df = pd.read_csv(args.metadata_csv, low_memory=False, dtype=str)
    print(f"{len(meta_df)} rows")

    for col in ["Spec. ID", "WL Units", "Flux Unit Coefficient", "wavelength", "flux"]:
        if col not in spec_df.columns:
            raise RuntimeError(f"Required column '{col}' missing from spectra parquet.")

    meta_lookup = build_meta_lookup(meta_df)

    grid_centers, grid_edges = make_grid(args.lam_min, args.lam_max, args.dlam)
    print(f"Working grid: {args.lam_min}–{args.lam_max} Å, step={args.dlam} Å, {len(grid_centers)} bins")

    rng = np.random.default_rng(args.example_seed)
    example_idx = int(args.example_index) if args.example_index is not None else int(rng.integers(0, len(spec_df)))

    print("Processing spectra...")
    R = process(spec_df, meta_lookup, grid_centers, grid_edges, args, example_idx)

    flux_array = R["flux_array"]
    mask_array = R["mask_array"]
    counts = R["counts"]
    N = counts["kept"]

    print(f"Saving {N} spectra...")
    np.save(os.path.join(outdir, "wiserep_flux.npy"), flux_array)
    np.save(os.path.join(outdir, "wiserep_mask.npy"), mask_array)
    np.save(os.path.join(outdir, "wiserep_wavelength.npy"), grid_centers.astype(np.float32))

    if N > 0:
        out_df = spec_df.iloc[R["kept_rows"]].copy(deep=True).reset_index(drop=True)
        out_df["common_wave"] = [grid_centers.astype(np.float32)] * N
        out_df["common_flux"] = list(flux_array)
        out_df["common_mask"] = list(mask_array)
        out_df["common_wave_unit"] = "Angstrom_working_frame"
        out_df["norm_scale"] = R["norm_scales"]
        out_df["coverage_frac"] = R["coverage_fracs"]
        out_df["redshift_used"] = R["redshifts"]
        out_df["frame_choice"] = [m["frame_choice"] for m in R["meta_out"]]
        out_df["warning_flag"] = [m["warning_flag"] for m in R["meta_out"]]
        out_df["sn_name_used"] = [m["sn_name"] for m in R["meta_out"]]
    else:
        out_df = spec_df.iloc[0:0].copy()

    out_df.to_parquet(os.path.join(outdir, "wiserep_spectra_processed.parquet"), index=False)
    print(f"Saved wiserep_spectra_processed.parquet ({N} rows)")

    meta_out_df = meta_lookup.loc[R["kept_ids"]].copy().reset_index(drop=True)
    if N > 0:
        meta_out_df["sn_name_used"] = [m["sn_name"] for m in R["meta_out"]]
        meta_out_df["obs_date_used"] = [m["obs_date"] for m in R["meta_out"]]
        meta_out_df["jd_obs_used"] = [m["jd_obs"] for m in R["meta_out"]]
        meta_out_df["frame_choice"] = [m["frame_choice"] for m in R["meta_out"]]
        meta_out_df["warning_flag"] = [m["warning_flag"] for m in R["meta_out"]]
    meta_out_df.to_csv(os.path.join(outdir, "wiserep_metadata_processed.csv"), index=False)
    print(f"Saved wiserep_metadata_processed.csv ({len(meta_out_df)} rows)")

    pd.DataFrame(R["meta_out"]).to_csv(os.path.join(outdir, "wiserep_meta.csv"), index=False)
    pd.DataFrame(R["report"]).to_csv(os.path.join(outdir, "wiserep_report.csv"), index=False)

    np.savez(
        os.path.join(outdir, "wiserep_common_grid.npz"),
        centers=grid_centers,
        edges=grid_edges,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        dlam=args.dlam,
        frame="working",
    )

    np.savez(
        os.path.join(outdir, "wiserep_norm_stats.npz"),
        norm_scales=R["norm_scales"],
        coverage_fracs=R["coverage_fracs"],
        lam_work_mins=R["lam_in_mins"],
        lam_work_maxs=R["lam_in_maxs"],
        redshifts=R["redshifts"],
    )

    summary = {
        "grid": {
            "lam_min": args.lam_min,
            "lam_max": args.lam_max,
            "dlam": args.dlam,
            "n_wave": int(len(grid_centers)),
            "frame": "working",
        },
        "normalization": "median_abs_per_spectrum: flux / median(|flux[finite]|)",
        "continuum_removal": "none",
        "redshift_handling": {
            "default": "lam_work = lam_obs / (1 + z)",
            "warning_rows": "for rows whose Remarks warn they may already be de-redshifted, compare as-is and de-redshifted and keep the better-scoring option",
            "warning_choice_score": "prefer quality_ok, then larger finite fraction after resampling, then larger overlap with working grid",
        },
        "vacuum_to_air": "Morton (2000) formula applied where WL Medium == Vacuum",
        "z_max_cutoff": args.z_max,
        "kept_obj_types": "SN*, SLSN*, Kilonova, Computed-*",
        "filters": {
            "require_sn_name": True,
            "min_overlap_frac": args.min_overlap_frac,
            "min_finite_frac": args.min_finite_frac,
            "min_finite_norm": args.min_finite_norm,
        },
        "counts": counts,
        "output_shape": {"N": N, "L": int(len(grid_centers))},
    }

    with open(os.path.join(outdir, "wiserep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if R["example_data"] is not None:
        ed = R["example_data"]
        plot_example(
            outdir,
            grid_centers,
            ed["lam_orig"],
            ed["flx_orig"],
            ed["f_resampled"],
            ed["f_norm"],
            ed["mask"],
            ed["spec_id"],
            ed["z"],
            ed["frame_choice"],
        )

    if N > 0:
        plot_dataset_stats(
            outdir,
            R["coverage_fracs"],
            R["norm_scales"],
            R["lam_in_mins"],
            R["lam_in_maxs"],
            R["redshifts"],
            args.z_max,
        )
    final_lines = [
        "=" * 64,
        f"Total input rows              : {counts['total']}",
        f"Kept                          : {N} ({100 * N / max(counts['total'], 1):.1f}%)",
        f"warning rows checked          : {counts['warning_rows']}",
        f"warning rows chose as-is      : {counts['warning_choose_asis']}",
        f"warning rows chose deredshift : {counts['warning_choose_der']}",
        f"skip missing SN name          : {counts['skip_missing_sn_name']}",
        f"skip missing z                : {counts['skip_missing_z']}",
        f"skip nonpositive z            : {counts['skip_nonpositive_z']}",
        f"skip bad z (z>{args.z_max})   : {counts['skip_bad_z']}",
        f"skip no metadata              : {counts['skip_no_meta']}",
        f"skip non-target type          : {counts['skip_non_target']}",
        f"skip missing type             : {counts['skip_missing_obj_type']}",
        f"skip bad WL units             : {counts['skip_bad_wl_units']}",
        f"skip too few points           : {counts['skip_too_few_points']}",
        f"skip quality/optical          : {counts['skip_quality']}",
        f"skip norm fail                : {counts['skip_norm_fail']}",
        f"skip resample error           : {counts['skip_resample_fail']}",
        f"total clipped bins            : {counts['total_clipped_bins']}",
    ]
    
    if N > 0:
        cov = R["coverage_fracs"]
        z_k = R["redshifts"]
        final_lines.append(
            f"Coverage frac                 : median={np.median(cov):.2f} min={cov.min():.2f} max={cov.max():.2f}"
        )
        final_lines.append(
            f"Redshift (kept)               : median={np.median(z_k):.4f} max={z_k.max():.4f}"
        )
    
    final_lines.append(f"Output shape                  : ({N}, {len(grid_centers)})")
    final_lines.append(f"Written to                    : {outdir}/")
    final_lines.append("=" * 64)
    
    with open(os.path.join(outdir, "wiserep_final_summary.txt"), "w") as f:
        f.write("\n".join(final_lines) + "\n")

    print("=" * 64)
    print(f"Total input rows              : {counts['total']}")
    print(f"Kept                          : {N} ({100 * N / max(counts['total'], 1):.1f}%)")
    print(f"warning rows checked          : {counts['warning_rows']}")
    print(f"warning rows chose as-is      : {counts['warning_choose_asis']}")
    print(f"warning rows chose deredshift : {counts['warning_choose_der']}")
    print(f"skip missing SN name          : {counts['skip_missing_sn_name']}")
    print(f"skip missing z                : {counts['skip_missing_z']}")
    print(f"skip nonpositive z            : {counts['skip_nonpositive_z']}")
    print(f"skip bad z (z>{args.z_max})   : {counts['skip_bad_z']}")
    print(f"skip no metadata              : {counts['skip_no_meta']}")
    print(f"skip non-target type          : {counts['skip_non_target']}")
    print(f"skip missing type             : {counts['skip_missing_obj_type']}")
    print(f"skip bad WL units             : {counts['skip_bad_wl_units']}")
    print(f"skip too few points           : {counts['skip_too_few_points']}")
    print(f"skip quality/optical          : {counts['skip_quality']}")
    print(f"skip norm fail                : {counts['skip_norm_fail']}")
    print(f"skip resample error           : {counts['skip_resample_fail']}")
    print(f"total clipped bins            : {counts['total_clipped_bins']}")

    if N > 0:
        cov = R["coverage_fracs"]
        print(f"Coverage frac                 : median={np.median(cov):.2f} min={cov.min():.2f} max={cov.max():.2f}")
        z_k = R["redshifts"]
        print(f"Redshift (kept)               : median={np.median(z_k):.4f} max={z_k.max():.4f}")

    print(f"Output shape                  : ({N}, {len(grid_centers)})")
    print(f"Written to                    : {outdir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()