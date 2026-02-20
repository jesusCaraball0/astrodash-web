#!/usr/bin/env python
"""
Classify WISeREP spectra using the original TensorFlow DASH model and
TF-style preprocessing. Fully standalone: no imports from app or other scripts.
Same logic as compare_tf_pytorch_training_set.py for TF model + TFStylePreprocessor.

Use this to compare with wiserep_dash_classify.py (PyTorch) and see whether
the Ic bias is from the model or the data.

Requirements: numpy, scipy, tensorflow (compat.v1). Run from project root.

Usage:
    cd astrodash-web
    python prod_backend/scripts/wiserep_tf_classify.py
    python prod_backend/scripts/wiserep_tf_classify.py --model_dir zeroZ --limit 500

TF model and params must exist under:
  data/pre_trained_models/original_dash_models/<model_dir>/
    tensorflow_model.ckpt (and .ckpt.index, .ckpt.meta, etc.)
    training_params.pickle
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Paths (no app or other script imports)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
MODELS_DIR = PROJECT_ROOT / "data" / "pre_trained_models"
OUT_FILENAME = "wiserep_tf_results.csv"
OUT_PATH = WISEREP_DIR / OUT_FILENAME
DEFAULT_TF_MODEL_DIR = "zeroZ"
MAX_FILES_PER_RUN_DEFAULT = 1000


# -----------------------------------------------------------------------------
# WISeREP data loading (standalone copy)
# -----------------------------------------------------------------------------
def load_metadata_filename_to_redshift(metadata_path: Path) -> Dict[str, float]:
    """Load wiserep_metadata.csv: filename -> redshift (first occurrence)."""
    filename_to_z: Dict[str, float] = {}
    if not metadata_path.exists():
        return filename_to_z
    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if "Ascii file" not in (reader.fieldnames or []):
            return filename_to_z
        for row in reader:
            fname = (row.get("Ascii file") or "").strip()
            if not fname or fname in filename_to_z:
                continue
            raw_z = (row.get("Redshift") or "").strip()
            try:
                z = float(raw_z) if raw_z else 0.0
            except ValueError:
                z = 0.0
            filename_to_z[fname] = z
    return filename_to_z


def _ensure_wave_flux_order(col0: np.ndarray, col1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wave, flux) with wave in typical Å range."""
    m0, m1 = float(np.nanmean(col0)), float(np.nanmean(col1))
    wave_like_lo, wave_like_hi = 500.0, 50000.0
    col0_like_wave = wave_like_lo <= m0 <= wave_like_hi
    col1_like_wave = wave_like_lo <= m1 <= wave_like_hi
    if col0_like_wave and not col1_like_wave:
        return col0, col1
    if col1_like_wave and not col0_like_wave:
        return col1, col0
    return col0, col1


def _is_numeric_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def parse_flm(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wave, flux = _ensure_wave_flux_order(data[:, 0], data[:, 1])
        return wave, flux
    except Exception:
        return None


def parse_ascii(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            no_comment = [l for l in f if not l.strip().startswith("#")]
        start = None
        for i, line in enumerate(no_comment):
            if _is_numeric_line(line):
                start = i
                break
        if start is None:
            return None
        data = np.loadtxt(no_comment[start:], delimiter=None)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wave, flux = _ensure_wave_flux_order(
            data[:, 0].astype(float), data[:, 1].astype(float)
        )
        return wave, flux
    except Exception:
        return None


def parse_dat(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wave, flux = _ensure_wave_flux_order(data[:, 0], data[:, 1])
        return wave, flux
    except Exception:
        return None


def iter_spectrum_files(spectra_dir: Path, exts: Tuple[str, ...]):
    seen: Set[str] = set()
    for ext in exts:
        for p in spectra_dir.glob(f"*{ext}"):
            if p.name not in seen:
                seen.add(p.name)
                yield p


def load_spectrum(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    suf = filepath.suffix.lower()
    if suf == ".flm":
        return parse_flm(filepath)
    if suf == ".ascii":
        return parse_ascii(filepath)
    if suf == ".dat":
        return parse_dat(filepath)
    return None


# -----------------------------------------------------------------------------
# TF model and preprocessing (standalone copy from compare_tf_pytorch_training_set)
# -----------------------------------------------------------------------------
def load_training_params(params_path: str) -> Dict[str, Any]:
    with open(params_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


class TensorFlowModel:
    """Wrapper for TensorFlow DASH model."""

    def __init__(self, model_path: str):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        self.model_path = model_path
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model_path + ".meta")
            self.sess = tf.Session()
            self.saver.restore(self.sess, model_path)
            self.x = self.graph.get_tensor_by_name("Placeholder:0")
            self.keep_prob = self.graph.get_tensor_by_name("Placeholder_2:0")
            self.y_conv = self.graph.get_tensor_by_name("Softmax:0")

    def predict(self, processed_flux: np.ndarray) -> np.ndarray:
        with self.graph.as_default():
            input_data = processed_flux.reshape(1, -1).astype(np.float32)
            output = self.sess.run(
                self.y_conv,
                feed_dict={self.x: input_data, self.keep_prob: 1.0},
            )
            return output[0]

    def close(self) -> None:
        self.sess.close()


class TFStylePreprocessor:
    """TF-style preprocessing matching original DASH (astrodash_old)."""

    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13, smooth: int = 6):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points
        self.dwlog = np.log(w1 / w0) / nw
        self.smooth = smooth

    def _normalise(self, flux: np.ndarray) -> np.ndarray:
        if flux.size == 0 or np.min(flux) == np.max(flux):
            return np.zeros_like(flux)
        return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

    def _limit_wavelength_range(
        self, wave: np.ndarray, flux: np.ndarray, min_wave: float, max_wave: float
    ) -> np.ndarray:
        min_idx = int(np.abs(wave - min_wave).argmin())
        max_idx = int(np.abs(wave - max_wave).argmin())
        flux_out = np.copy(flux)
        flux_out[:min_idx] = 0.0
        flux_out[max_idx:] = 0.0
        return flux_out

    def _two_col_input_spectrum(
        self, wave: np.ndarray, flux: np.ndarray, z: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        wave_new = wave / (1.0 + z)
        mask = (wave_new >= self.w0) & (wave_new < self.w1)
        wave_new = wave_new[mask]
        flux_new = flux[mask]
        if wave_new.size == 0:
            raise ValueError(
                f"Spectrum out of model wavelength range [{self.w0}, {self.w1}] after deredshifting (z={z})."
            )
        flux_new = self._normalise(flux_new)
        return wave_new, flux_new

    def _log_wavelength(
        self, wave: np.ndarray, flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0.0, right=0.0)
        non_zero = np.where(binned_flux != 0.0)[0]
        if non_zero.size == 0:
            return wlog, binned_flux, 0, 0
        min_idx = int(non_zero[0])
        max_idx = int(non_zero[-1])
        return wlog, binned_flux, min_idx, max_idx

    def _continuum_removal(
        self, wave: np.ndarray, flux: np.ndarray, min_idx: int, max_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.interpolate import UnivariateSpline

        flux_plus = flux + 1.0
        cont_removed = np.copy(flux_plus)
        continuum = np.zeros_like(flux_plus)
        if max_idx - min_idx > 5:
            spline = UnivariateSpline(
                wave[min_idx : max_idx + 1], flux_plus[min_idx : max_idx + 1], k=3
            )
            spline_wave = np.linspace(
                wave[min_idx], wave[max_idx], num=self.num_spline_points, endpoint=True
            )
            spline_points = spline(spline_wave)
            spline_more = UnivariateSpline(spline_wave, spline_points, k=3)
            continuum[min_idx : max_idx + 1] = spline_more(wave[min_idx : max_idx + 1])
        else:
            continuum[min_idx : max_idx + 1] = 1.0
        valid = continuum[min_idx : max_idx + 1] != 0
        if np.any(valid):
            cont_removed[min_idx : max_idx + 1][valid] = (
                flux_plus[min_idx : max_idx + 1][valid]
                / continuum[min_idx : max_idx + 1][valid]
            )
        cont_removed_norm = self._normalise(cont_removed - 1.0)
        cont_removed_norm[:min_idx] = 0.0
        cont_removed_norm[max_idx + 1 :] = 0.0
        return cont_removed_norm, continuum - 1.0

    def _mean_zero(self, flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        if max_idx <= min_idx or max_idx >= flux.size:
            return flux
        mean_flux = np.mean(flux[min_idx:max_idx])
        out = flux - mean_flux
        out[:min_idx] = flux[:min_idx]
        out[max_idx + 1 :] = flux[max_idx + 1 :]
        return out

    def _apodize(
        self, flux: np.ndarray, min_idx: int, max_idx: int, outer_val: float = 0.0
    ) -> np.ndarray:
        percent = 0.05
        out = np.copy(flux) - outer_val
        nsquash = max(1, int(self.nw * percent))
        for i in range(nsquash):
            if nsquash <= 1:
                break
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1.0 - np.cos(arg))
            if (min_idx + i < self.nw) and (max_idx - i >= 0):
                out[min_idx + i] = factor * out[min_idx + i]
                out[max_idx - i] = factor * out[max_idx - i]
        if outer_val != 0.0:
            out = out + outer_val
            out[:min_idx] = outer_val
            out[max_idx + 1 :] = outer_val
        return out

    def process(
        self, wave: np.ndarray, flux: np.ndarray, z: float = 0.0
    ) -> Tuple[np.ndarray, int, int]:
        from scipy.signal import medfilt

        flux_norm = self._normalise(flux)
        flux_limited = self._limit_wavelength_range(wave, flux_norm, self.w0, self.w1)
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
        wave_dereds, flux_dereds = self._two_col_input_spectrum(wave, pre_filtered, z)
        binned_wave, binned_flux, min_idx, max_idx = self._log_wavelength(
            wave_dereds, flux_dereds
        )
        if min_idx == max_idx == 0 and not np.any(binned_flux):
            return np.full(self.nw, 0.5, dtype=float), 0, 0
        cont_removed, _ = self._continuum_removal(
            binned_wave, binned_flux, min_idx, max_idx
        )
        mean_zero_flux = self._mean_zero(cont_removed, min_idx, max_idx)
        apodized = self._apodize(mean_zero_flux, min_idx, max_idx, outer_val=0.0)
        final_flux = self._normalise(apodized)
        final_flux[:min_idx] = 0.5
        final_flux[max_idx + 1 :] = 0.5
        return final_flux, min_idx, max_idx


def decode_label(
    label_idx: int,
    type_list: List[str],
    min_age: float,
    max_age: float,
    age_bin_size: float,
) -> str:
    num_age_bins = int((max_age - min_age) / age_bin_size) + 1
    type_idx = label_idx // num_age_bins
    age_idx = label_idx % num_age_bins
    if type_idx >= len(type_list):
        return f"Unknown type {type_idx}, age bin {age_idx}"
    age_start = min_age + age_idx * age_bin_size
    age_end = age_start + age_bin_size
    return f"{type_list[type_idx]}: {int(age_start)} to {int(age_end)}"


def decode_label_to_type_age(
    label_idx: int,
    type_list: List[str],
    min_age: float,
    max_age: float,
    age_bin_size: float,
) -> Tuple[str, str]:
    full = decode_label(label_idx, type_list, min_age, max_age, age_bin_size)
    if ": " in full:
        type_part, age_part = full.split(": ", 1)
        return type_part.strip(), age_part.strip()
    return full, ""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify WISeREP spectra with original TensorFlow DASH model (standalone)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_TF_MODEL_DIR,
        help=f"Subdir under original_dash_models (default: {DEFAULT_TF_MODEL_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_FILES_PER_RUN_DEFAULT,
        help=f"Max new files per run (default: {MAX_FILES_PER_RUN_DEFAULT}, 0 = no limit)",
    )
    args = parser.parse_args()

    wiserep_dir = WISEREP_DIR.resolve()
    models_dir = MODELS_DIR.resolve()
    metadata_path = wiserep_dir / "wiserep_metadata.csv"
    spectra_dir = wiserep_dir / "wiserep_data_noSEDM"
    out_path = OUT_PATH.resolve()

    tf_dir = models_dir / "original_dash_models" / args.model_dir
    tf_model_path = tf_dir / "tensorflow_model.ckpt"
    params_path = tf_dir / "training_params.pickle"

    for p, name in [
        (tf_model_path.with_suffix(".ckpt.index"), "TF model"),
        (params_path, "Training params"),
        (spectra_dir, "WISeREP spectra dir"),
    ]:
        if not p.exists():
            print(f"Error: {name} not found at {p}")
            sys.exit(1)

    params = load_training_params(str(params_path))
    w0 = params["w0"]
    w1 = params["w1"]
    nw = params["nw"]
    min_age = params["minAge"]
    max_age = params["maxAge"]
    age_bin_size = params["ageBinSize"]
    type_list = params.get("typeList", [])

    print("WISeREP classification with TensorFlow DASH model (standalone)")
    print("=" * 60)
    print(f"TF model: {tf_model_path}")
    print(f"Params:  {params_path}")
    print(f"Output:  {out_path}")
    print(f"Wavelength range: {w0} - {w1} Å, nw={nw}")

    tf_model = TensorFlowModel(str(tf_model_path))
    preprocessor = TFStylePreprocessor(w0, w1, nw)

    filename_to_z = load_metadata_filename_to_redshift(metadata_path)
    print(f"Metadata: {len(filename_to_z)} filename -> redshift entries")

    exts = (".flm", ".ascii", ".dat")
    done_filenames: Set[str] = set()
    if out_path.exists():
        try:
            with open(out_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done_filenames.add((row.get("filename") or "").strip())
            print(f"Resuming: {len(done_filenames)} rows already in output")
        except Exception:
            done_filenames = set()

    limit = args.limit if args.limit > 0 else None
    if limit is not None:
        print(f"Processing at most {limit} new files this run.")
    else:
        print("Processing all files (no limit).")

    fieldnames = ["filename", "redshift", "best_type", "best_age", "probability", "error"]
    errors = 0
    skipped = 0
    classified = 0
    written_this_run = 0
    mode = "a" if done_filenames else "w"

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not done_filenames:
            w.writeheader()
        f.flush()

        def write_row(
            fname: str,
            z: float,
            best_type: str = "",
            best_age: str = "",
            probability: Any = "",
            error: str = "",
        ) -> None:
            w.writerow({
                "filename": fname,
                "redshift": z,
                "best_type": best_type,
                "best_age": best_age,
                "probability": probability,
                "error": error,
            })
            f.flush()

        for filepath in iter_spectrum_files(spectra_dir, exts):
            if limit is not None and written_this_run >= limit:
                print(f"Reached limit of {limit} new files. Re-run to continue.")
                break

            fname = filepath.name
            if fname in done_filenames:
                continue

            z = filename_to_z.get(fname, 0.0)

            pair = load_spectrum(filepath)
            if pair is None:
                write_row(fname, z, error="load_failed")
                errors += 1
                written_this_run += 1
                continue

            wave, flux = pair
            if len(wave) < 100 or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
                write_row(fname, z, error="invalid_spectrum")
                skipped += 1
                written_this_run += 1
                continue

            if np.ptp(flux) == 0:
                write_row(fname, z, error="constant_flux")
                skipped += 1
                written_this_run += 1
                continue

            wave_dereds_min = float(np.min(wave)) / (1.0 + z)
            wave_dereds_max = float(np.max(wave)) / (1.0 + z)
            if wave_dereds_min > w1 or wave_dereds_max < w0:
                write_row(fname, z, error="no_wavelength_overlap")
                skipped += 1
                written_this_run += 1
                continue

            try:
                processed_flux, _, _ = preprocessor.process(wave, flux, z=z)
            except Exception as e:
                write_row(fname, z, error=str(e))
                errors += 1
                written_this_run += 1
                continue

            if processed_flux is None or processed_flux.size == 0:
                write_row(fname, z, error="preprocess_empty")
                errors += 1
                written_this_run += 1
                continue

            try:
                probs = tf_model.predict(processed_flux)
            except Exception as e:
                write_row(fname, z, error=f"predict_error: {e}")
                errors += 1
                written_this_run += 1
                continue

            top1_idx = int(np.argmax(probs))
            prob = float(probs[top1_idx])
            best_type, best_age = decode_label_to_type_age(
                top1_idx, type_list, min_age, max_age, age_bin_size
            )

            print(f"TF: {fname} -> {best_type} ({best_age}) prob={prob:.4f}")
            write_row(fname, z, best_type=best_type, best_age=best_age, probability=prob)
            classified += 1
            written_this_run += 1

    try:
        tf_model.close()
    except Exception:
        pass

    print(f"Done. Classified: {classified}, errors: {errors}, skipped: {skipped}.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
