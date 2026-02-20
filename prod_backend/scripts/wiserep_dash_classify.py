"""
Load WISeREP raw spectrum files and classify them with the backend DASH model.

- Ascii: skips # lines and any leading non-numeric header (e.g. "WAVE FLUX").
- Constant flux: only skips when flux has zero range (no isclose), to avoid false skips.
- Resume: if the output CSV already exists, re-run skips those filenames and appends.
- Memory: MAX_FILES_PER_RUN=2000 limits new rows per run (re-run to continue); no full file list;
  wave/flux freed before classify; gc every 200 rows.

Usage:
    cd astrodash-web
    python prod_backend/scripts/wiserep_dash_classify.py

Requires: numpy; run from project root so app imports resolve.
"""

import gc
import sys
import csv
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Fix paths to get imports working
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Config
WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
MODELS_DIR = PROJECT_ROOT / "data" / "pre_trained_models"
OUT_FILENAME = "wiserep_dash_results.csv"
# Output CSV is always under data/wiserep/
OUT_PATH = WISEREP_DIR / OUT_FILENAME
# Process at most this many NEW files per run (then re-run to continue). None = no limit.
# Use a limit (e.g. 2000) to avoid OOM when the system kills the process on long runs.
MAX_FILES_PER_RUN: Optional[int] = 5000

# Backend imports
from app.domain.models.spectrum import Spectrum
from app.config.settings import Settings
from app.infrastructure.ml.classifiers.dash_classifier import DashClassifier
from app.shared.utils.validators import ValidationError


def load_metadata_filename_to_redshift(metadata_path: Path) -> Dict[str, float]:
    """
    Load wiserep_metadata.csv and build a map: filename -> redshift.
    Uses first occurrence per 'Ascii file' (column name in CSV).
    """
    filename_to_z: Dict[str, float] = {}
    if not metadata_path.exists():
        return filename_to_z

    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if "Ascii file" not in reader.fieldnames:
            return filename_to_z
        for row in reader:
            fname = (row.get("Ascii file") or "").strip()
            if not fname:
                continue
            if fname in filename_to_z:
                continue
            raw_z = (row.get("Redshift") or "").strip()
            try:
                z = float(raw_z) if raw_z else 0.0
            except ValueError:
                z = 0.0
            filename_to_z[fname] = z

    return filename_to_z


def _ensure_wave_flux_order(col0: np.ndarray, col1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (wave, flux) with wave in typical Å range (e.g. 2000–15000).
    If col0 looks like wavelength and col1 like flux, return (col0, col1).
    If col1 looks like wavelength and col0 like flux, return (col1, col0).
    Otherwise return (col0, col1).
    """
    m0, m1 = float(np.nanmean(col0)), float(np.nanmean(col1))
    # Wavelength in Å: typically 2000–15000; flux: often 1e-20–1e-12 or small
    wave_like_lo, wave_like_hi = 500.0, 50000.0
    col0_like_wave = wave_like_lo <= m0 <= wave_like_hi
    col1_like_wave = wave_like_lo <= m1 <= wave_like_hi
    if col0_like_wave and not col1_like_wave:
        return col0, col1
    if col1_like_wave and not col0_like_wave:
        return col1, col0
    return col0, col1


def parse_flm(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Three-column: wavelength, flux, flux_err. Use flux only."""
    try:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wave, flux = _ensure_wave_flux_order(data[:, 0], data[:, 1])
        return wave, flux
    except Exception:
        return None


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


def parse_ascii(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse ascii spectrum. Common formats:
    - No header: first line is "wave\tflux" (e.g. 3914.50	2.37708 or 3097.47	5.23e-16).
    - With # header: skip lines starting with #, then data.
    - With text header (no #): skip leading non-numeric lines (e.g. "WAVE FLUX"), then data.
    Data is WAVE FLUX [optional FLUX_ERR]; delimiter=None accepts tabs and spaces.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            no_comment = [l for l in f if not l.strip().startswith("#")]
        # Skip leading non-numeric lines (e.g. "WAVE FLUX" header without #)
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
    """Two-column: wavelength, flux."""
    try:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wave, flux = _ensure_wave_flux_order(data[:, 0], data[:, 1])
        return wave, flux
    except Exception:
        return None


def iter_spectrum_files(spectra_dir: Path, exts: Tuple[str, ...]):
    """Yield unique spectrum file paths without building a full list (saves memory)."""
    seen: Set[str] = set()
    for ext in exts:
        for p in spectra_dir.glob(f"*{ext}"):
            if p.name not in seen:
                seen.add(p.name)
                yield p


def load_spectrum(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load wave, flux from a WISeREP spectrum file by extension."""
    suf = filepath.suffix.lower()
    if suf == ".flm":
        return parse_flm(filepath)
    elif suf == ".ascii":
        return parse_ascii(filepath)
    elif suf == ".dat":
        return parse_dat(filepath)
    return None


def main() -> None:
    wiserep_dir = WISEREP_DIR.resolve()
    models_dir = MODELS_DIR.resolve()
    metadata_path = wiserep_dir / "wiserep_metadata.csv"
    spectra_dir = wiserep_dir / "wiserep_data_noSEDM"
    out_path = OUT_PATH.resolve()

    model_path = models_dir / "dash" / "zeroZ" / "pytorch_model.pth"
    params_path = models_dir / "dash" / "zeroZ" / "training_params.pickle"

    # DASH classifier from backend
    config = Settings(
        dash_model_path=str(model_path),
        dash_training_params_path=str(params_path),
    )
    classifier = DashClassifier(config=config)

    filename_to_z = load_metadata_filename_to_redshift(metadata_path)
    print(f"Loaded metadata: {len(filename_to_z)} filename -> redshift entries")

    exts = (".flm", ".ascii", ".dat")
    # Resume: skip files already in existing output (so re-run continues after OOM kill)
    done_filenames: Set[str] = set()
    if out_path.exists():
        try:
            with open(out_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done_filenames.add((row.get("filename") or "").strip())
            print(f"Resuming: {len(done_filenames)} rows already in output")
        except Exception:
            done_filenames = set()

    limit = MAX_FILES_PER_RUN
    if limit is not None:
        print(f"Processing at most {limit} new files per run (re-run to continue)")
    else:
        print("Processing all files (no limit)")

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

        for idx, filepath in enumerate(iter_spectrum_files(spectra_dir, exts)):
            if limit is not None and written_this_run >= limit:
                print(f"Reached limit of {limit} new files this run. Re-run to continue.")
                break
            fname = filepath.name
            if fname in done_filenames:
                continue
            z = filename_to_z.get(fname, 0.0)

            def write_row(best_type: str = "", best_age: str = "", probability: Any = "", error: str = "") -> None:
                w.writerow({
                    "filename": fname,
                    "redshift": z,
                    "best_type": best_type,
                    "best_age": best_age,
                    "probability": probability,
                    "error": error,
                })
                f.flush()

            pair = load_spectrum(filepath)
            if pair is None:
                write_row(error="load_failed")
                errors += 1
                written_this_run += 1
                continue

            wave, flux = pair
            if len(wave) < 100 or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
                write_row(error="invalid_spectrum")
                skipped += 1
                written_this_run += 1
                continue

            # Skip only strictly constant flux (no signal)
            if np.ptp(flux) == 0:
                write_row(error="constant_flux")
                skipped += 1
                written_this_run += 1
                continue

            print(f"Loaded: {fname}")
            spectrum = Spectrum(
                x=wave.tolist(),
                y=flux.tolist(),
                redshift=z,
                file_name=fname,
            )
            del wave, flux  # free arrays before classifier (reduces peak memory)

            try:
                result = classifier.classify_sync(spectrum)
            except (ValidationError, Exception) as e:
                write_row(error=str(e))
                errors += 1
                written_this_run += 1
                continue

            if not result:
                write_row(error="classification_failed")
                errors += 1
                written_this_run += 1
                continue

            best = result.get("best_match", {})
            best_type = best.get("type", "")
            best_age = best.get("age", "")
            prob = best.get("probability", 0.0)
            print(f"Classified: {fname} -> {best_type} ({best_age}) prob={prob:.4f}")
            write_row(best_type=best_type, best_age=best_age, probability=prob)
            classified += 1
            written_this_run += 1

            if written_this_run % 200 == 0:
                gc.collect()

    print(f"Done. Classified: {classified}, errors: {errors}, skipped: {skipped}.")
    print(f"Output: {out_path}")


# -----------------------------------------------------------------------------
# Performance metrics and confusion matrix (from results CSV + optional metadata)
# -----------------------------------------------------------------------------

# 6-class canonical set: Ia, Ib, Ic, IIb, IIn, II (+ Other for unmapped)
# Matches wiserep_perf_analysis.py; SLSN-I -> Ic.
_CANONICAL_ORDER = ("Ia", "Ib", "Ic", "IIb", "IIn", "II", "Other")

# DASH type (prediction) -> canonical
_DASH_TYPE_TO_CANONICAL: Dict[str, str] = {
    "Ia-norm": "Ia", "Ia-91T": "Ia", "Ia-91bg": "Ia", "Ia-02cx": "Ia", "Ia-pec": "Ia",
    "Ia-csm": "IIn",
    "Ib-norm": "Ib", "Ib-pec": "Ib",
    "Ic-norm": "Ic", "Ic-pec": "Ic", "Ic-broad": "Ic",
    "IIb": "IIb",
    "IIn": "IIn", "Ibn": "IIn",
    "IIP": "II", "IIL": "II", "II-pec": "II",
}

# Catalog (Obj. Type) -> canonical; SLSN-I -> Ic
_CATALOG_TO_CANONICAL: Dict[str, str] = {
    "SN Ia": "Ia", "SNIa": "Ia", "Ia": "Ia",
    "SN Ia-91T-like": "Ia", "SN Ia-91bg-like": "Ia", "SN Ia-pec": "Ia",
    "SN Ia-02cx": "Ia", "SN Ia-Ca-rich": "Ia", "Computed-Ia": "Ia", "SN Ia-SC": "Ia",
    "SN Ia-CSM": "IIn", "SN Ia-csm": "IIn",
    "SN Ib": "Ib", "SNIb": "Ib", "Ib": "Ib",
    "SN Ib-pec": "Ib", "SN Ib/c": "Ib", "SN Ib/c ": "Ib", "SN Ibc": "Ib", "Ibc": "Ib",
    "SN Ib-Ca-rich": "Ib", "SN Ib/c-Ca-rich": "Ib",
    "SN Ic": "Ic", "SNIc": "Ic", "Ic": "Ic",
    "SN Ic-BL": "Ic", "SN Ic-pec": "Ic", "SN Ic-Ca-rich": "Ic",
    "SLSN-I": "Ic", "SLSNe-I": "Ic",
    "SN IIb": "IIb", "SNIIb": "IIb", "IIb": "IIb", "Computed-IIb": "IIb",
    "SN IIn": "IIn", "SNIIn": "IIn", "IIn": "IIn",
    "SN IIn-pec": "IIn", "SN Ibn": "IIn", "Ibn": "IIn", "SN Ibn/Icn": "IIn",
    "SN II": "II", "SNII": "II", "II": "II",
    "SN IIP": "II", "SNIIP": "II", "IIP": "II", "Computed-IIP": "II",
    "SN IIL": "II", "SNIIL": "II", "IIL": "II",
    "SN II-pec": "II", "II-pec": "II",
}


def _normalize_type(label: str, from_catalog: bool) -> str:
    """Map label to 6-class canonical: Ia, Ib, Ic, IIb, IIn, II, or Other."""
    if not label or not str(label).strip():
        return "Other"
    raw = str(label).strip()
    if from_catalog:
        out = _CATALOG_TO_CANONICAL.get(raw) or _CATALOG_TO_CANONICAL.get(raw.replace(" ", ""))
        return out if out else ("Other" if raw not in _CANONICAL_ORDER else raw)
    if raw in _DASH_TYPE_TO_CANONICAL:
        return _DASH_TYPE_TO_CANONICAL[raw]
    if raw in _CATALOG_TO_CANONICAL:
        return _CATALOG_TO_CANONICAL[raw]
    base = raw.split("-")[0].split(" ")[0]
    return _DASH_TYPE_TO_CANONICAL.get(base) or _CATALOG_TO_CANONICAL.get(base) or (base if base in _CANONICAL_ORDER else "Other")


def compute_metrics_and_confusion_matrix(
    results_csv_path: str | Path,
    metadata_csv_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load the results CSV (and optionally metadata for catalog types), normalize
    catalog vs predicted types, and compute performance metrics and confusion matrix.

    - If the results CSV has a 'catalog_type' column, it is used as ground truth.
    - Otherwise, pass metadata_csv_path (wiserep_metadata.csv); catalog type will
      be joined by filename (Ascii file -> Obj. Type).

    Returns a dict with: confusion_matrix, labels, accuracy, macro_f1, per_class_metrics,
    n_samples, n_skipped.

    Example:
        from pathlib import Path
        out = compute_metrics_and_confusion_matrix(
            Path("data/wiserep/wiserep_dash_results.csv"),
            metadata_csv_path=Path("data/wiserep/wiserep_metadata.csv"),
        )
    """
    results_csv_path = Path(results_csv_path)
    if not results_csv_path.exists():
        return {"error": f"Results CSV not found: {results_csv_path}"}

    rows: List[Dict[str, str]] = []
    with open(results_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"error": "Results CSV is empty", "n_samples": 0}

    has_catalog_col = "catalog_type" in (rows[0].keys() if rows else set())
    filename_to_catalog: Dict[str, str] = {}
    if not has_catalog_col and metadata_csv_path:
        metadata_path = Path(metadata_csv_path)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8", errors="replace") as mf:
                for row in csv.DictReader(mf):
                    fname = (row.get("Ascii file") or "").strip()
                    if fname and fname not in filename_to_catalog:
                        filename_to_catalog[fname] = (row.get("Obj. Type") or "").strip()

    pairs: List[Tuple[str, str]] = []
    skipped = 0
    n_other = 0
    other_catalog_raw: List[str] = []
    other_pred_raw: List[str] = []
    for r in rows:
        best_type = (r.get("best_type") or "").strip()
        error = (r.get("error") or "").strip()
        if not best_type or error:
            skipped += 1
            continue
        if has_catalog_col:
            catalog_type = (r.get("catalog_type") or "").strip()
        else:
            catalog_type = filename_to_catalog.get((r.get("filename") or "").strip(), "")
        if not catalog_type:
            skipped += 1
            continue
        true_label = _normalize_type(catalog_type, from_catalog=True)
        pred_label = _normalize_type(best_type, from_catalog=False)
        if true_label == "Other" or pred_label == "Other":
            n_other += 1
            if true_label == "Other":
                other_catalog_raw.append(catalog_type)
            if pred_label == "Other":
                other_pred_raw.append(best_type)
            continue
        pairs.append((true_label, pred_label))

    if not pairs:
        return {
            "error": "No rows with both catalog type and prediction (add catalog_type or pass metadata_csv_path)",
            "n_samples": len(rows),
            "n_skipped": skipped,
        }

    # Confusion matrix: rows = true (catalog), cols = pred
    all_true = sorted(set(t for t, _ in pairs))
    all_pred = sorted(set(p for _, p in pairs))
    labels = sorted(set(all_true) | set(all_pred))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for true_label, pred_label in pairs:
        i = label_to_idx.get(true_label, label_to_idx.get("Other", 0))
        j = label_to_idx.get(pred_label, label_to_idx.get("Other", 0))
        cm[i][j] += 1

    # Accuracy
    correct = sum(1 for t, p in pairs if t == p)
    accuracy = correct / len(pairs)

    # Per-class precision, recall, F1
    per_class: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        idx = label_to_idx[lbl]
        tp = cm[idx][idx]
        pred_sum = sum(cm[i][idx] for i in range(n))
        true_sum = sum(cm[idx][j] for j in range(n))
        prec = tp / pred_sum if pred_sum else 0.0
        rec = tp / true_sum if true_sum else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": true_sum}

    f1s = [per_class[l]["f1"] for l in labels if per_class[l]["support"] > 0]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Pretty-print
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS (6 classes: Ia, Ib, Ic, IIb, IIn, II)")
    print("=" * 60)
    print(f"Samples used: {len(pairs)}  (skipped: {skipped}, excluded (Other): {n_other})")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    # Row-normalized percentages (% of true class)
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, 100.0 * cm_arr / row_sums, 0.0)

    print("\nConfusion matrix (% of true class, rows = true catalog, cols = predicted):")
    print(f"Labels: {labels}")
    col_w = 6
    hdr = "".join(f"{p:>{col_w}}" for p in labels)
    print(f"{'':>{col_w}}" + hdr)
    for i, lbl in enumerate(labels):
        row_str = "".join(f"{cm_pct[i][j]:>{col_w}.1f}" for j in range(n))
        print(f"{lbl:>{col_w}}" + row_str)
    print("\nPer-class metrics:")
    for lbl in labels:
        pc = per_class[lbl]
        if pc["support"] == 0:
            continue
        print(f"  {lbl}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} support={int(pc['support'])}")
    if n_other > 0:
        cat_counts = Counter(other_catalog_raw)
        pred_counts = Counter(other_pred_raw)
        print(f"\nExcluded from matrix/metrics: {n_other} rows had true and/or predicted label 'Other'.")
        print(f"  Catalog types that mapped to Other ({len(other_catalog_raw)} occurrences):")
        for raw, count in sorted(cat_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {count:5d}: {raw!r}")
        print(f"  Predictions that mapped to Other ({len(other_pred_raw)} occurrences):")
        for raw, count in sorted(pred_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {count:5d}: {raw!r}")
    print("=" * 60 + "\n")

    return {
        "confusion_matrix": cm,
        "labels": labels,
        "label_to_idx": label_to_idx,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class,
        "n_samples": len(pairs),
        "n_skipped": skipped,
        "n_other": n_other,
        "other_catalog_raw": other_catalog_raw,
        "other_pred_raw": other_pred_raw,
    }


if __name__ == "__main__":
    main()
