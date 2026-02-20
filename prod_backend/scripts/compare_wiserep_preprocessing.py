#!/usr/bin/env python
"""
Compare TF-style vs PyTorch (backend) preprocessing on WISeREP spectra step-by-step.

Runs the first N files through both pipelines and reports:
1. Final-output comparison (max diff, mean diff, correlation) for each file.
2. Step-by-step comparison for the first few files (or first few that differ).

Run from project root with the main app env (so app.infrastructure.ml is available):
    cd astrodash-web
    python prod_backend/scripts/compare_wiserep_preprocessing.py --limit 1000
    python prod_backend/scripts/compare_wiserep_preprocessing.py --limit 100 --step_detail 5
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Paths: script dir for wiserep_tf_classify, prod_backend for app
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
MODELS_DIR = PROJECT_ROOT / "data" / "pre_trained_models"


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape != b.shape:
        return {"error": "shape_mismatch", "shape_a": str(a.shape), "shape_b": str(b.shape)}
    a = np.asarray(a).astype(float).ravel()
    b = np.asarray(b).astype(float).ravel()
    diff = np.abs(a - b)
    corr = np.nan
    if a.size > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "max_diff": float(np.max(diff)),
        "mean_diff": float(np.mean(diff)),
        "correlation": corr,
    }


def run_tf_steps(
    tf_preproc: Any, wave: np.ndarray, flux: np.ndarray, z: float
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run TF pipeline step-by-step and return dict of intermediates. Returns (steps, error)."""
    from scipy.signal import medfilt

    steps: Dict[str, Any] = {}
    try:
        flux_norm = tf_preproc._normalise(flux)
        flux_limited = tf_preproc._limit_wavelength_range(wave, flux_norm, tf_preproc.w0, tf_preproc.w1)
        steps["step1_flux_limited"] = np.copy(flux_limited)

        w_density = (tf_preproc.w1 - tf_preproc.w0) / tf_preproc.nw
        wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
        if wavelength_density <= 0:
            filter_size = 1
        else:
            filter_size = int(w_density / wavelength_density * tf_preproc.smooth / 2) * 2 + 1
        if filter_size < 1:
            filter_size = 1
        if filter_size % 2 == 0:
            filter_size += 1
        steps["step2_filter_size"] = filter_size
        pre_filtered = medfilt(flux_limited, kernel_size=filter_size)
        steps["step2_flux_smoothed"] = np.copy(pre_filtered)

        wave_dereds, flux_dereds = tf_preproc._two_col_input_spectrum(wave, pre_filtered, z)
        steps["step3_flux_dereds"] = np.copy(flux_dereds)
        steps["step3_wave_dereds"] = np.copy(wave_dereds)

        binned_wave, binned_flux, min_idx, max_idx = tf_preproc._log_wavelength(wave_dereds, flux_dereds)
        steps["step4_binned_flux"] = np.copy(binned_flux)

        if min_idx == max_idx == 0 and not np.any(binned_flux):
            steps["step5_cont_removed"] = np.full(tf_preproc.nw, 0.5, dtype=float)
            steps["step6_mean_zero"] = steps["step5_cont_removed"].copy()
            steps["step7_apodized"] = steps["step5_cont_removed"].copy()
            steps["step8_final"] = steps["step5_cont_removed"].copy()
            return steps, None

        cont_removed, _ = tf_preproc._continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
        steps["step5_cont_removed"] = np.copy(cont_removed)
        mean_zero_flux = tf_preproc._mean_zero(cont_removed, min_idx, max_idx)
        steps["step6_mean_zero"] = np.copy(mean_zero_flux)
        apodized = tf_preproc._apodize(mean_zero_flux, min_idx, max_idx, outer_val=0.0)
        steps["step7_apodized"] = np.copy(apodized)
        final_flux = tf_preproc._normalise(apodized)
        final_flux = np.copy(final_flux)
        final_flux[:min_idx] = 0.5
        final_flux[max_idx + 1 :] = 0.5
        steps["step8_final"] = final_flux
        return steps, None
    except Exception as e:
        return steps, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TF vs PyTorch preprocessing on WISeREP")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of files to compare")
    parser.add_argument(
        "--step_detail",
        type=int,
        default=3,
        help="Number of files for which to run step-by-step comparison (default 3)",
    )
    parser.add_argument(
        "--differing_only",
        action="store_true",
        help="Run step-by-step only on files where final output differs (corr < 0.99)",
    )
    parser.add_argument(
        "--diagnose",
        type=int,
        default=0,
        metavar="N",
        help="After run, diagnose first N differing files with Backend (0,0): print backend step intermediates and exit",
    )
    args = parser.parse_args()

    # Standalone imports (no app)
    from wiserep_tf_classify import (
        load_spectrum,
        iter_spectrum_files,
        load_metadata_filename_to_redshift,
        TFStylePreprocessor,
        load_training_params,
    )
    # App imports (requires app env)
    from app.infrastructure.ml.data_processor import DashSpectrumProcessor
    from app.shared.utils.validators import ValidationError

    metadata_path = WISEREP_DIR / "wiserep_metadata.csv"
    spectra_dir = WISEREP_DIR / "wiserep_data_noSEDM"
    params_path = MODELS_DIR / "dash" / "zeroZ" / "training_params.pickle"
    if not params_path.exists():
        print(f"Error: params not found at {params_path}")
        sys.exit(1)

    params = load_training_params(str(params_path))
    w0 = params["w0"]
    w1 = params["w1"]
    nw = params["nw"]

    tf_preproc = TFStylePreprocessor(w0, w1, nw)
    backend_processor = DashSpectrumProcessor(w0, w1, nw)

    filename_to_z = load_metadata_filename_to_redshift(metadata_path)
    exts = (".flm", ".ascii", ".dat")

    print("Compare TF vs PyTorch preprocessing on WISeREP")
    print("=" * 60)
    print(f"Params: w0={w0}, w1={w1}, nw={nw}")
    print(f"Limit: {args.limit} files, step_detail: {args.step_detail} files")
    print()

    n_ok = 0
    n_diff = 0
    n_skip = 0
    n_err_tf = 0
    n_err_backend = 0
    differing_files: List[Tuple[str, float, float, float, Optional[tuple], Path]] = []  # (fname, max_d, mean_d, corr, (tf_min, tf_max, be_min, be_max) or None, filepath)
    step_detail_done = 0
    step_detail_max = args.step_detail

    for filepath in iter_spectrum_files(spectra_dir, exts):
        if n_ok + n_diff + n_skip >= args.limit:
            break

        fname = filepath.name
        z = filename_to_z.get(fname, 0.0)
        pair = load_spectrum(filepath)
        if pair is None:
            n_skip += 1
            continue
        wave, flux = pair
        if len(wave) < 100 or np.ptp(flux) == 0:
            n_skip += 1
            continue

        # Ensure ascending wavelength so both pipelines' index-based step 1 keep the same interval
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave.size > 1 and wave[0] > wave[-1]:
            perm = np.argsort(wave)
            wave = wave[perm]
            flux = flux[perm]

        # Run both pipelines (final output only)
        try:
            tf_out, tf_min, tf_max = tf_preproc.process(wave, flux, z=z)
        except Exception as e:
            n_err_tf += 1
            n_skip += 1
            continue
        try:
            backend_out, be_min, be_max, _ = backend_processor.process(wave, flux, z)
        except ValidationError as e:
            n_err_backend += 1
            n_skip += 1
            continue
        except Exception as e:
            n_err_backend += 1
            n_skip += 1
            continue

        cmp = _compare_arrays(tf_out, backend_out)
        if "error" in cmp:
            n_diff += 1
            differing_files.append((fname, -1, -1, float("nan"), None, filepath))
            continue
        max_d, mean_d, corr = cmp["max_diff"], cmp["mean_diff"], cmp["correlation"]
        # Treat nan correlation as differing; threshold on max_diff to catch 0.5-padding mismatch
        is_differing = (corr < 0.99 if np.isfinite(corr) else True) or max_d > 0.01
        if is_differing:
            n_diff += 1
            # Store (fname, max_d, mean_d, corr, (tf_min, tf_max, be_min, be_max), filepath) for diagnostics
            differing_files.append((fname, max_d, mean_d, corr, (tf_min, tf_max, be_min, be_max), filepath))
        else:
            n_ok += 1

        # Step-by-step for first step_detail_max files (or first step_detail_max differing if --differing_only)
        do_step_detail = (
            step_detail_done < step_detail_max
            and ("error" not in cmp)
            and ((args.differing_only and is_differing) or (not args.differing_only))
        )
        if do_step_detail:
            step_detail_done += 1
            print(f"\n--- Step-by-step for file {step_detail_done}/{step_detail_max}: {fname} ---")
            tf_steps, tf_err = run_tf_steps(tf_preproc, wave, flux, z)
            if tf_err:
                print(f"  TF steps error: {tf_err}")
            else:
                try:
                    be_steps, be_final, _, _, _ = backend_processor.process_with_steps(wave, flux, z)
                except Exception as e:
                    print(f"  Backend process_with_steps error: {e}")
                else:
                    for key in ["step1_flux_limited", "step2_flux_smoothed", "step4_binned_flux",
                               "step5_cont_removed", "step6_mean_zero", "step7_apodized", "step8_final"]:
                        if key in tf_steps and key in be_steps:
                            c = _compare_arrays(tf_steps[key], be_steps[key])
                            if "error" in c:
                                print(f"  {key}: {c}")
                            else:
                                print(f"  {key}: max_diff={c['max_diff']:.6f} mean_diff={c['mean_diff']:.6f} corr={c['correlation']:.6f}")
                    if "step2_filter_size" in tf_steps and "step2_filter_size" in be_steps:
                        tf_fs = tf_steps["step2_filter_size"]
                        be_fs = be_steps["step2_filter_size"]
                        if tf_fs != be_fs:
                            print(f"  step2_filter_size: TF={tf_fs} vs Backend={be_fs}  <-- DIFFERENT")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {n_ok + n_diff} (ok: {n_ok}, differing: {n_diff}, skipped: {n_skip})")
    print(f"TF errors: {n_err_tf}, Backend errors: {n_err_backend}")
    if differing_files:
        print(f"\nFirst 20 differing files (corr < 0.99 or max_diff > 0.01):")
        for t in differing_files[:20]:
            fname, max_d, mean_d, corr, indices = t[0], t[1], t[2], t[3], t[4] if len(t) > 4 else None
            if max_d < 0:
                print(f"  {fname} (shape/error)")
            else:
                corr_str = f"{corr:.6f}" if np.isfinite(corr) else "nan"
                print(f"  {fname}  max_diff={max_d:.6f} mean_diff={mean_d:.6f} corr={corr_str}")
                if indices is not None:
                    tf_min, tf_max, be_min, be_max = indices
                    print(f"    -> TF (min_idx,max_idx)=({tf_min},{tf_max})  Backend (min_idx,max_idx)=({be_min},{be_max})")
        # Diagnose: how many have different (min_idx, max_idx)?
        n_region_diff = sum(1 for t in differing_files if len(t) > 4 and t[4] is not None and (t[4][0] != t[4][2] or t[4][1] != t[4][3]))
        if n_region_diff > 0:
            print(f"\n  Diagnostic: {n_region_diff} differing files have (min_idx, max_idx) different between TF and Backend.")

    # Diagnose first N differing files that have Backend (0,0)
    if args.diagnose > 0 and differing_files:
        backend_zeros = [t for t in differing_files if len(t) >= 6 and t[4] is not None and t[4][2] == 0 and t[4][3] == 0]
        for idx, t in enumerate(backend_zeros[: args.diagnose]):
            fname, _max_d, _mean_d, _corr, _indices, filepath = t[0], t[1], t[2], t[3], t[4], t[5]
            z = filename_to_z.get(fname, 0.0)
            pair = load_spectrum(filepath)
            if pair is None:
                print(f"\n[Diagnose {idx+1}] {fname}: load_spectrum returned None")
                continue
            wave, flux = pair
            wave = np.asarray(wave, dtype=float)
            flux = np.asarray(flux, dtype=float)
            if wave.size > 1 and wave[0] > wave[-1]:
                perm = np.argsort(wave)
                wave = wave[perm]
                flux = flux[perm]
            print(f"\n[Diagnose {idx+1}] {fname}  z={z}")
            print(f"  wave: len={len(wave)}  range=[{wave[0]:.1f}, {wave[-1]:.1f}]  ascending={bool(wave.size <= 1 or wave[0] <= wave[-1])}")
            try:
                steps, _final, _mi, _ma, _z = backend_processor.process_with_steps(wave, flux, z)
            except Exception as e:
                print(f"  backend process_with_steps error: {e}")
                continue
            fl = steps["step1_flux_limited"]
            wave_dered = wave / (1.0 + z)
            mask = (wave_dered >= w0) & (wave_dered < w1)
            fl_masked = fl[mask]
            print(f"  step1 flux_limited: nnz={np.count_nonzero(fl)}  mask size={np.count_nonzero(mask)}  flux_limited[mask] nnz={np.count_nonzero(fl_masked)}  min={np.min(fl_masked) if fl_masked.size else 'n/a'}  max={np.max(fl_masked) if fl_masked.size else 'n/a'}")
            fd = steps["step3_flux_dereds"]
            print(f"  step3 flux_dereds: len={len(fd)}  min={np.min(fd):.6f}  max={np.max(fd):.6f}  ptp={np.ptp(fd):.6e}  constant={np.ptp(fd) < 1e-12}")
            bf = steps["step4_binned_flux"]
            print(f"  step4 binned_flux: nnz={np.count_nonzero(bf)}")
    print()


if __name__ == "__main__":
    main()
