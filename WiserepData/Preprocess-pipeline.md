# WISeREP Preprocessing for DAEP Models (`Preprocess.py`)

`WiserepData/Preprocess.py` is the offline preprocessing pipeline for all DAEP / latent models. It converts raw WISeREP spectra into fixed-length arrays consumed by `TwinsClassifier_Wiserep.py`, `TwinsTrain_Wiserep.py`, and related scripts.

**Input:** WISeREP parquet spectra + metadata CSV  
**Output:** NumPy arrays on a common wavelength grid (`Test/data_z` or `Test/data_no_z`)

---

## Pipeline overview

1. Raw spectrum (variable λ, variable length)
2. Filter & validate metadata
3. Unit conversion & optional de-redshift
4. Resample to common grid (3250 bins)
5. Normalize & build coverage mask
6. Write `wiserep_flux.npy`, `wiserep_mask.npy`, `wiserep_wavelength.npy`

---

## Step 0: Configuration

- `USE_REDSHIFT_CORRECTED_DATA = True` → default output: `WiserepData/Test/data_z`
- `USE_REDSHIFT_CORRECTED_DATA = False` → default output: `WiserepData/Test/data_no_z`
- Working grid defaults:
  - λ min: **3200 Å**
  - λ max: **9700 Å**
  - Step: **2 Å**
  - Bins: **3250** (`(9700 − 3200) / 2`)

---

## Step 1: Load data

- Read `wiserep_spectra.parquet` (one row per spectrum: `wavelength`, `flux`, `Spec. ID`, units, etc.)
- Read `wiserep_metadata.csv` (object type, redshift, SN name, observation date, etc.)
- Build metadata lookup indexed by `Spec. ID` (deduplicated)

---

## Step 2: Build common wavelength grid

- `make_grid()` creates bin edges from 3200–9700 Å in 2 Å steps
- Bin centers (midpoints) are the resampling targets
- Every kept spectrum is mapped to the same 3250-length vector

---

## Step 3: Per-spectrum filtering

Spectra are rejected before signal processing if any check fails:

| Check | Skip reason |
|-------|-------------|
| `Spec. ID` missing from metadata | `skip:no_metadata` |
| No SN name in metadata | `skip:missing_sn_name` |
| Object type not SN/SLSN/Kilonova/Computed | `skip:non_target` or `skip:missing_obj_type` |
| Missing or invalid redshift | `skip:missing_z`, `skip:nonpositive_z`, or `skip:bad_z` |
| Unrecognized wavelength units | `skip:bad_wl_units` |
| Fewer than 2 wavelength points | `skip:too_few_points` |

**Kept object types:** `SN*`, `SLSN*`, `KILONOVA`, `COMPUTED-*`

**Redshift rules:** must be finite, `0 < z ≤ 6` (default `z_max`)

---

## Step 4: Unit conversion & flux scaling

1. **`to_angstrom()`** — convert wavelength to Å (nm × 10, µm × 10⁴)
2. **`vacuum_to_air()`** — if `WL Medium == "Vacuum"`, apply Morton (2000) refractive-index correction
3. **Flux coefficient** — multiply flux by `Flux Unit Coefficient` (default 1.0)
4. **`clean_spectrum()`** — remove NaN/inf, sort by λ, drop duplicate wavelengths

---

## Step 5: Redshift / frame choice

**Default (`data_z`):** de-redshift to rest frame:

```
λ_work = λ_obs / (1 + z)
```

**Ambiguous metadata:** if `Remarks` warns the spectrum may already be de-redshifted (e.g. *"de-redshifted, please check"*):

1. Evaluate both **as-is** and **de-redshifted** versions
2. Resample each to the grid
3. Keep the better option by score: `(quality_ok, finite_frac, overlap)`

This avoids double de-redshifting when WISeREP metadata is inconsistent.

---

## Step 6: Resample to common grid

- **`resample_to_grid()`** uses Specutils `FluxConservingResampler`
- Input: irregular (λ, flux) in the working frame
- Output: flux at each of 3250 grid centers
- Bins outside the spectrum's λ range → **NaN**

---

## Step 7: Quality checks

Rejected if resampled spectrum is too sparse:

| Threshold | Default |
|-----------|---------|
| Grid overlap with 3200–9700 Å | ≥ **20%** |
| Finite bins after resample | ≥ **10%** |
| Non-zero flux | required |

---

## Step 8: Normalization

- **`normalize_median_abs()`:** `flux_norm = flux / median(|flux[finite]|)`
- Requires ≥ **50** finite bins
- **No continuum removal** (unlike DASH preprocessing)
- Clip normalized flux to **[-50, 50]**
- NaN bins remain NaN until masking

---

## Step 9: Mask & fill

1. **`make_mask()`** — 1.0 where flux is finite, 0.0 elsewhere
2. **`fill_nans()`** — replace NaN with **0.0** in the flux array (model input)

**Model receives:**
- `flux` — normalized values, 0 outside coverage
- `mask` — indicates which bins are real data

---

## Step 10: Output files

Written to `--outdir` (default `Test/data_z`):

| File | Shape / contents |
|------|------------------|
| `wiserep_flux.npy` | `(N, 3250)` float32 normalized flux |
| `wiserep_mask.npy` | `(N, 3250)` float32 coverage mask |
| `wiserep_wavelength.npy` | `(3250,)` grid centers [Å] |
| `wiserep_spectra_processed.parquet` | Per-row processed spectra + metadata |
| `wiserep_metadata_processed.csv` | Metadata for kept spectra |
| `wiserep_meta.csv` | Per-spectrum preprocessing diagnostics |
| `wiserep_report.csv` | Keep/skip log for every input row |
| `wiserep_summary.json` | Full config + skip counts |
| `example_preprocessing.png` | 4-panel visualization of one spectrum |
| `dataset_stats.png` | Coverage, norm scales, λ limits, z distribution |

---

## Example preprocessing figure (`example_preprocessing.png`)

1. **Raw** — (λ_obs, flux) after unit conversion
2. **Resampled** — onto 3200–9700 Å working grid
3. **Normalized** — median-abs normalized flux
4. **Mask** — coverage (1 = valid bin)

---

## How DAEP training uses these outputs

`TwinsClassifier_Wiserep.py` loads:

- `wiserep_flux.npy` — model input
- `wiserep_mask.npy` — valid bins for loss/metrics
- `wiserep_wavelength.npy` — grid definition
- `wiserep_metadata_processed.csv` — labels, SN names, train/val/test splits

The DAEP autoencoder trains on 3250-bin vectors; classifier heads train on latent representations from the encoder.

---

## DAEP vs DASH preprocessing

| | **Preprocess.py (DAEP)** | **DASH pipeline** |
|--|--------------------------|-------------------|
| Grid | 3200–9700 Å, linear 2 Å, **3250 bins** | 3500–10000 Å, log-spaced, **1024 bins** |
| Continuum removal | None | Spline fit |
| Smoothing | None | Adaptive median |
| Normalization | `median(|flux|)` | Median + mean-zero + apodization |
| Redshift | `λ/(1+z)` rest frame (`data_z`) | Same idea, different grid |

---

## Run command

```bash
python WiserepData/Preprocess.py \
  --spectra-parquet <path/to/wiserep_spectra.parquet> \
  --metadata-csv <path/to/wiserep_metadata.csv> \
  --outdir WiserepData/Test/data_z
```

Optional flags: `--lam-min`, `--lam-max`, `--dlam`, `--min-overlap-frac`, `--min-finite-frac`, `--z-max`, `--debug`
