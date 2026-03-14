#!/usr/bin/env python
"""
Map rows in wiserep_validation_set.parquet back to source spectrum files so you can
exclude those files from training and avoid contamination.

Strategies (in order):
1. If wiserep_spectra.parquet exists and has a 'source_file' (or similar) column and
   the same row order/IDs, use it to resolve validation rows.
2. Otherwise, build a fingerprint index from all files in wiserep_data_noSEDM, then
   match each validation row by (len(wave), w[0], w[-1], flux[0], flux[-1]) to find
   the source file.

Output: writes data/wiserep/wiserep_validation_source_files.txt (one filename per line,
in the same order as validation parquet rows) and optionally a parquet with an added
source_file column.

Usage (from repo root, with astrodash env):
    python prod_backend/scripts/map_validation_parquet_to_sources.py
    python prod_backend/scripts/map_validation_parquet_to_sources.py --write-parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
VALIDATION_PQ = WISEREP_DIR / "wiserep_validation_set.parquet"
FULL_PQ = WISEREP_DIR / "wiserep_spectra.parquet"
SPECTRA_DIR = WISEREP_DIR / "wiserep_data_noSEDM"
OUT_TXT = WISEREP_DIR / "wiserep_validation_source_files.txt"


def _fingerprint(wave: np.ndarray, flux: np.ndarray) -> tuple:
    """Stable fingerprint for matching: length and a few key values."""
    n = len(wave)
    if n == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    w0, w1 = float(wave[0]), float(wave[-1])
    f0, f1 = float(flux[0]), float(flux[-1])
    return (n, w0, w1, f0, f1)


def _parse_row_wave_flux(wavelength, flux):
    """Convert parquet cell (list or str) to numpy arrays."""
    if hasattr(wavelength, "__iter__") and not isinstance(wavelength, str):
        w = np.asarray(wavelength, dtype=float)
    else:
        import ast
        w = np.asarray(ast.literal_eval(wavelength), dtype=float)
    if hasattr(flux, "__iter__") and not isinstance(flux, str):
        f = np.asarray(flux, dtype=float)
    else:
        import ast
        f = np.asarray(ast.literal_eval(flux), dtype=float)
    return w, f


def load_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (wave, flux) from a WISeREP file. Returns None on failure."""
    suf = filepath.suffix.lower()
    try:
        if suf == ".flm":
            data = np.loadtxt(filepath)
            if data.ndim != 2 or data.shape[1] < 2:
                return None
            return data[:, 0].astype(float), data[:, 1].astype(float)
        if suf == ".ascii":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = [l for l in f if not l.strip().startswith("#")]
            start = None
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        start = i
                        break
                    except ValueError:
                        continue
            if start is None:
                return None
            data = np.loadtxt(lines[start:], delimiter=None)
            if data.ndim != 2 or data.shape[1] < 2:
                return None
            return data[:, 0].astype(float), data[:, 1].astype(float)
        if suf == ".dat":
            data = np.loadtxt(filepath)
            if data.ndim != 2 or data.shape[1] < 2:
                return None
            return data[:, 0].astype(float), data[:, 1].astype(float)
        return None
    except Exception:
        return None


def build_fingerprint_index(spectra_dir: Path, exts: tuple[str, ...] = (".flm", ".ascii", ".dat")):
    """Build map fingerprint -> list of (filename, wave, flux) for collision resolution."""
    index: dict[tuple, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    count = 0
    for ext in exts:
        for f in spectra_dir.glob(f"*{ext}"):
            if not f.is_file():
                continue
            out = load_spectrum(f)
            if out is None:
                continue
            wave, flux = out
            fp = _fingerprint(wave, flux)
            if fp not in index:
                index[fp] = []
            index[fp].append((f.name, wave, flux))
            count += 1
            if count % 5000 == 0:
                print(f"  Indexed {count} files ...", flush=True)
    print(f"  Indexed {count} files total.", flush=True)
    return index


def match_validation_to_sources(
    val_path: Path,
    index: dict,
    out_txt: Path,
    write_parquet_path: Path | None = None,
) -> list[str] | None:
    """Match each validation row to a source file; write out_txt and optionally add source_file to parquet."""
    import pandas as pd

    df = pd.read_parquet(val_path)
    if "wavelength" not in df.columns or "flux" not in df.columns:
        print("Validation parquet missing 'wavelength' or 'flux' column.", file=sys.stderr)
        return None

    source_files: list[str] = []
    matched = 0
    for i in range(len(df)):
        row = df.iloc[i]
        w, f = _parse_row_wave_flux(row["wavelength"], row["flux"])
        fp = _fingerprint(w, f)
        candidates = index.get(fp, [])
        chosen = None
        if len(candidates) == 1:
            chosen = candidates[0][0]
        elif len(candidates) > 1:
            # Resolve by closer match on full vectors (compare first 50 and last 50 points)
            best_name, best_err = None, np.inf
            for name, cw, cf in candidates:
                n = min(len(w), len(cw), 50)
                err = (
                    np.abs(w[:n] - cw[:n]).sum()
                    + np.abs(f[:n] - cf[:n]).sum()
                    + np.abs(w[-n:] - cw[-n:]).sum()
                    + np.abs(f[-n:] - cf[-n:]).sum()
                )
                if err < best_err:
                    best_err = err
                    best_name = name
            chosen = best_name
        if chosen is not None:
            matched += 1
        source_files.append(chosen if chosen else "")
    print(f"Matched {matched} / {len(df)} validation rows to source files.")
    if matched < len(df):
        print("Unmatched rows will have an empty string in source_file.", file=sys.stderr)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for name in source_files:
            f.write(name + "\n")
    print(f"Wrote {out_txt} ({len(source_files)} lines).")

    if write_parquet_path is not None:
        df = df.copy()
        df["source_file"] = source_files
        df.to_parquet(write_parquet_path, index=False)
        print(f"Wrote parquet with source_file column to {write_parquet_path}.")

    return source_files


def try_full_parquet_mapping(val_path: Path, full_path: Path) -> list[str] | None:
    """If full parquet has source_file and we can match rows, return list of source files for validation rows."""
    import pandas as pd

    if not full_path.exists():
        return None
    full = pd.read_parquet(full_path)
    if "source_file" not in full.columns and "filename" not in full.columns:
        return None
    src_col = "source_file" if "source_file" in full.columns else "filename"
    val = pd.read_parquet(val_path)

    # If same row count and fingerprint match on first row, assume same order (e.g. val = full.sample(...))
    if len(full) == len(val) and "wavelength" in full.columns and "flux" in full.columns:
        v0_w, v0_f = _parse_row_wave_flux(val.iloc[0]["wavelength"], val.iloc[0]["flux"])
        f0_w, f0_f = _parse_row_wave_flux(full.iloc[0]["wavelength"], full.iloc[0]["flux"])
        if _fingerprint(v0_w, v0_f) == _fingerprint(f0_w, f0_f):
            return full[src_col].astype(str).tolist()

    # Else match by (label, Redshift, wavelength length)
    val_src = []
    for i in range(len(val)):
        r = val.iloc[i]
        w, _ = _parse_row_wave_flux(r["wavelength"], r["flux"])
        nw = len(w)
        label = r.get("label", r.get("label_raw", ""))
        z = r.get("Redshift", np.nan)
        cand = full[(full["label"].astype(str) == str(label))] if "label" in full.columns else full
        if "Redshift" in full.columns:
            cand = cand[np.isclose(cand["Redshift"], z, rtol=1e-5)]
        chosen = ""
        for _, full_row in cand.iterrows():
            ww = full_row["wavelength"]
            nfull = len(ww) if hasattr(ww, "__len__") and not isinstance(ww, str) else 0
            if nfull == nw:
                chosen = str(full_row[src_col])
                break
        val_src.append(chosen)
    if sum(1 for s in val_src if s) < len(val_src) // 2:
        return None
    return val_src


def main():
    parser = argparse.ArgumentParser(
        description="Map validation parquet rows back to source spectrum filenames"
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write a new parquet with source_file column (default: only write .txt list)",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=None,
        help="Path for output parquet with source_file (default: wiserep_validation_set_with_sources.parquet)",
    )
    args = parser.parse_args()

    if not VALIDATION_PQ.exists():
        print(f"Not found: {VALIDATION_PQ}", file=sys.stderr)
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("pandas and pyarrow required.", file=sys.stderr)
        sys.exit(1)

    # Strategy 1: full parquet with source_file
    source_files = try_full_parquet_mapping(VALIDATION_PQ, FULL_PQ)
    if source_files is not None and sum(1 for s in source_files if s) == len(source_files):
        print("Resolved source files from full wiserep_spectra.parquet.")
        OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_TXT, "w") as f:
            for name in source_files:
                f.write(str(name) + "\n")
        print(f"Wrote {OUT_TXT}.")
        if args.write_parquet:
            out_pq = Path(args.out_parquet) if args.out_parquet else WISEREP_DIR / "wiserep_validation_set_with_sources.parquet"
            if not out_pq.is_absolute():
                out_pq = PROJECT_ROOT / out_pq
            df = pd.read_parquet(VALIDATION_PQ)
            df["source_file"] = source_files
            df.to_parquet(out_pq, index=False)
            print(f"Wrote {out_pq}.")
        return

    # Strategy 2: fingerprint index from raw files
    if not SPECTRA_DIR.is_dir():
        print(f"Spectra dir not found: {SPECTRA_DIR}. Cannot build fingerprint index.", file=sys.stderr)
        sys.exit(1)
    print("Building fingerprint index from spectrum files (this may take a few minutes)...")
    index = build_fingerprint_index(SPECTRA_DIR)
    out_pq = None
    if args.write_parquet:
        out_pq = Path(args.out_parquet) if args.out_parquet else WISEREP_DIR / "wiserep_validation_set_with_sources.parquet"
        if not out_pq.is_absolute():
            out_pq = PROJECT_ROOT / out_pq
    match_validation_to_sources(VALIDATION_PQ, index, OUT_TXT, write_parquet_path=out_pq)
    print()
    print("Use the list in", OUT_TXT, "to exclude these files from training (e.g. filter wiserep file list by this set).")


if __name__ == "__main__":
    main()
