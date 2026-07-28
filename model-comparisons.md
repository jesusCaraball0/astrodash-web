# AstroDASH 2.0 — Model reference for paper writing

This document is a **models-only** writer's guide for the AstroDASH 2.0 AAS Letter. It covers training data, preprocessing, architectures, evaluation, spectral twins, and reproducibility. **Ignore web/deployment concerns** — those live in a separate repository.

Use this document for paper sections: **Data & Preprocessing**, **Model Suite**, **Model Evaluation**, **Spectral Twins (interpretability)**, **Limitations**, and **Reproducibility**.

---

## Scientific context (for Introduction / Abstract)

- Rapid supernova (SN) spectral classification is essential for time-domain astrophysics: identifying Type Ia SNe as standardizable candles, tracing chemical enrichment, and probing stellar death.
- Classical workflows (SNID, Superfit) rely on expert template matching, local installation, and manual interpretation — difficult to scale as discovery rates rise (Vera C. Rubin Observatory / LSST era).
- **AstroDASH** (v1) pioneered deep neural network classification of SN spectra with learned features rather than hand-crafted template correlation.
- Related ML work to cite alongside DASH: ABC SN, SNID SAGE, and the original DASH paper/model.
- AstroDASH 2.0 **centralizes multiple classifiers** behind a common preprocessing and inference interface, lowering friction between model development and usable classification — especially important as spectroscopic follow-up volume grows.
- All models in this repo are trained/evaluated primarily on **WISeREP** public spectra.

---

## Model suite at a glance

| Model | Architecture family | Input | Classes | Redshift | Trained in this repo? | Checkpoint location |
|---|---|---|---|---|---|---|
| **Original DASH** | 2D CNN (`AstroDashPyTorchNet`) | 1024 DASH-preprocessed flux bins → reshaped 32×32 | **102** fine-grained type×age bins | Required for de-redshift in preprocessing; **not** an explicit input feature | Pre-trained (converted TF→PyTorch) | `data/pre_trained_models/dash/zeroZ/` |
| **Dash 1D CNN** | 3× Conv1d + 2 FC | 1025-d = 1024 flux bins + `z` | 5 canonical SN classes | Optional pair (+z / −z) | Yes — `zmodel_training/dash_retrain.py` | `data/pre_trained_models/daep_comparison_z/` and `daep_comparison_noz/` |
| **DAEP classifier (no diffusion)** | Perceiver-style transceiver encoder + mean-pool MLP head | flux (3250), λ (3250), phase (1), mask (3250) | 5 classes | Optional pair (+z / −z) | Yes — `WiserepData/TwinsClassifier_Wiserep.py` | `WiserepData/Test/daep_comparison_init/` (+ `_noz`) |
| **DAEP classifier (diffusion / latent)** | Frozen DAEP encoder latents + MLP head | 8192-d latent (64×128 bottleneck) | 5 classes | Optional pair (+z / −z) | Yes — encoder in `TwinsTrain_Wiserep.py`, head in `train_latent.py` | `WiserepData/Test/daep_comparison/` (+ `_noz`) |
| **Transformer classifier** | Cross-attention transformer encoder | 1024 flux + 1024 λ embeddings + redshift token | **5 classes** (different label strings) | Used as explicit embedding | Pre-trained weights bundled | `data/pre_trained_models/transformer/TF_wiserep_v6.pt` |

**Not in the six-model architecture comparison plot** but part of the deployed model family: Original DASH and Transformer use different label spaces (see below).

---

## Training and evaluation data

### Source

| Property | Value |
|---|---|
| **Catalog** | [WISeREP](https://www.wiserep.org/) — public repository of SN spectra |
| **Raw formats** | Parquet (`wiserep_spectra.parquet`) + metadata CSV (`wiserep_metadata.csv`); Dash retrain also uses per-spectrum ASCII under `data/wiserep/wiserep_data_noSEDM/` |
| **Object filter** | SN*, SLSN*, Kilonova, Computed-* types; require valid IAU/SN name |
| **Redshift** | Required for `data_z` bundle; `data_no_z` keeps observed-frame spectra |

### Dataset sizes (approximate)

| Data universe | Preprocessed rows | After 5-class label filter (classifiers) | Notes |
|---|---:|---:|---|
| WISeREP offline (`data_z`) | 40,867 | 32,941 (train 25,879 / val 3,106 / test 3,956) | DAEP models |
| WISeREP offline (`data_no_z`) | 40,908 | 32,954 (train 26,345 / val 3,079 / test 3,530) | DAEP −z models |
| WISeREP ASCII + Dash split | 45,193 listed | ~33–42k effective per split after load failures | Dash 1D CNN |

**Important:** The architecture comparison mixes **two data universes** (preprocessed WISeREP ~33k vs full ASCII ~45k). Cross-group comparisons (DAEP vs Dash) confound architecture with data volume, wavelength grid, and preprocessing — fair comparisons are **within-group** (e.g. +z vs −z for the same model).

### Class distribution (5-class canonical labels)

**WISeREP preprocessed (`data_z`, all labeled rows before split):**

| Class | Count | Fraction |
|---|---:|---:|
| SN Ia | 22,961 | 56.9% |
| SN II | 9,129 | 22.6% |
| SN Ib/c | 4,857 | 12.0% |
| SN IIn | 2,137 | 5.3% |
| SLSN-I | 1,288 | 3.2% |

**Dash IAU test split (`daep_compatible_split.json`, n≈4,190 effective):**

| Class | Fraction |
|---|---:|
| SN Ia | 52.5% |
| SN II | 26.0% |
| SN Ib/c | 14.2% |
| SN IIn | 5.6% |
| SLSN-I | 1.7% |

Use these for a **dataset distribution pie chart** figure.

### Train / validation / test splits

| Models | Split method | Seed | Key constraint |
|---|---|---|---|
| DAEP / latent classifiers | IAU-level 80/10/10 | `IAU_SPLIT_SEED = 0` | No IAU object in more than one split |
| Dash 1D CNN | IAU-level 80/10/10 | seed 0 | `data/wiserep/daep_compatible_split.json` |
| DAEP encoder pretraining (`TwinsTrain_Wiserep.py`) | Random 90/10 row split | configurable | **Not** IAU-based; no held-out test — latents exported for all spectra afterward |

Implementation: `WiserepData/iau_train_val_test_split.py`.

### Redshift in the data

- `data_z`: spectra de-redshifted to rest frame (λ/(1+z)); valid z required at preprocess time; median z ≈ 0.05 for kept spectra.
- `data_no_z`: resampled in **observed frame** only.
- For WISeREP rows whose metadata warns they may already be de-redshifted, preprocessing compares as-is vs de-redshifted and keeps the higher-quality option (`Preprocess.py`).

---

## Classification labels

### Five-class canonical set (DAEP + Dash 1D CNN retrain)

| Output class | Source subtypes (examples) |
|---|---|
| **SN Ia** | SN Ia, Ia-91T, Ia-91bg, Ia-pec, Ia-CSM, Iax, Computed-Ia, … |
| **SN Ib/c** | SN Ib, Ic, Ibn, Icn, Ic-BL, … |
| **SN II** | SN II, IIP, IIL, IIb, II-pec, Computed-IIP/IIb, … |
| **SN IIn** | SN IIn, IIn-pec |
| **SLSN-I** | SLSN-I, SLSN-II, SLSN-R (all mapped to SLSN-I) |

- **Label source column:** WISeREP `Obj. Type`, collapsed via `LABEL_MAP` in `zmodel_training/constants.py` and `WiserepData/TwinsClassifier_Wiserep.py`.
- **Output:** 5 logits → softmax → class probabilities; argmax for hard label.
- **Loss:** cross-entropy with inverse-frequency class weights on training split.

### Original DASH label space (102 classes)

- **17 SN subtypes** × **6 age bins** = 102 softmax outputs.
- Subtypes (`typeList`): Ia-norm, Ia-91T, Ia-91bg, Ia-csm, Ia-02cx, Ia-pec, Ib-norm, Ibn, IIb, Ib-pec, Ic-norm, Ic-broad, Ic-pec, IIP, IIL, IIn, II-pec.
- Age bins: −20 to 50 days in ~4-day steps → labels like `"Ia-norm: 2 to 6"`.
- Post-processing combines type×age probabilities via `combined_prob()` into a single best type + age + reliability flag.
- Parameters: `data/pre_trained_models/dash/zeroZ/training_params.pickle`.

### Transformer label space (5 classes, different strings)

| Index | Label in model |
|---|---|
| 0 | Ia |
| 1 | IIn |
| 2 | SLSNe-I |
| 3 | II |
| 4 | Ib/c |

Configured in `prod_backend/app/config/settings.py` (`label_mapping`). Same scientific categories but **different naming** than the canonical 5-class set.

---

## Preprocessing

Two distinct pipelines feed different models. A **raw vs preprocessed spectrum** figure can show: (left) uploaded ASCII spectrum, (right) DASH-preprocessed 1024-bin vector or WISeREP 3250-bin grid.

### DASH pipeline (`DashSpectrumProcessor`)

Used by: Original DASH, Dash 1D CNN retrain, Transformer (flux path uses `interpolate_to_1024`).

| Step | Operation |
|---|---|
| 1 | Median normalization of flux |
| 2 | Wavelength range limit (3500–10000 Å rest-frame bounds applied to observed λ before de-redshift) |
| 3 | Adaptive median smoothing (kernel sized from spectral sampling density; default smooth factor 6) |
| 4 | De-redshift: λ → λ/(1+z), restrict to [3500, 10000) Å, re-normalize |
| 5 | Log-λ binning to **1024** bins between 3500–10000 Å |
| 6 | Continuum removal (spline fit, DASH semantics) |
| 7 | Mean-zero over valid spectral region |
| 8 | Apodization: 5% cosine bell at edges |
| 9 | Final median normalization; pad out-of-range bins to 0.5 |

**No-redshift variant:** skip step 4; restrict observed frame to [w0, w1]; redshift feature set to 0 (Dash 1D CNN) or redshift embedding uses z=0 (Transformer).

**Grid:** 3500–10000 Å, 1024 log-spaced bins (`nw=1024`).

### WISeREP offline pipeline (`WiserepData/Preprocess.py`)

Used by: all DAEP / latent models.

| Step | Operation |
|---|---|
| 1 | Load parquet spectra + metadata; filter SN/SLSN types |
| 2 | Unit conversion → Å; vacuum→air if needed; flux coefficient |
| 3 | De-redshift (`data_z`) or observed frame (`data_no_z`) |
| 4 | Quality: ≥20% grid overlap, ≥10% finite bins after resampling |
| 5 | Flux-conserving resample (Specutils) to **3250 linear bins**, 3200–9700 Å @ 2 Å |
| 6 | Median-abs normalization; NaN outside coverage; clip to [−50, 50] |
| 7 | Emit `wiserep_flux.npy`, `wiserep_mask.npy`, `wiserep_wavelength.npy`, metadata CSV |

**No continuum removal** in WISeREP offline preprocess (unlike DASH).

**Example preprocessing figure:** `WiserepData/Preprocess.py` writes `example_preprocessing.png` (raw → resampled → normalized → mask).

### Preprocessing ablation study (Dash 1D CNN)

Script: `zmodel_training/dash_preprocessing_removal_diff_plot.py`.

Trains/evaluates Dash models with individual steps removed vs full pipeline:

| Variant removed | Effect (qualitative) |
|---|---|
| Continuum removal | Largest per-class recall drop for some types |
| Median filtering | Moderate degradation |
| Apodization | Moderate |
| Initial normalization | Moderate |
| De-redshifting | Severe when z known |
| All normalization | Severe |

Outputs: `data/pre_trained_models/dash_wiserep/models/preprocessing_removal_difference_vs_full.png` (+ `_no_redshift` variant).

**Paper figure:** "Performance vs preprocessing steps removed."

---

## Model architectures (detail)

### 1. Original DASH

| Property | Detail |
|---|---|
| **Architecture** | 2D CNN: Conv2d(1→32)→Pool→Conv2d(32→64)→Pool → FC(4096→1024)→Dropout→Linear(1024→102) |
| **Input** | 1024-d preprocessed flux reshaped to **32×32×1** "image" |
| **Redshift** | Used only in preprocessing (de-redshift); not concatenated to input |
| **Output** | 102 type×age probabilities; combined to best type + age |
| **Training data** | Original DASH training set (not WISeREP retrain in this repo) |
| **Weights** | `data/pre_trained_models/dash/zeroZ/pytorch_model.pth` (converted from TensorFlow) |
| **Code** | `prod_backend/app/infrastructure/ml/classifiers/dash_classifier.py`, `architectures.py` (`AstroDashPyTorchNet`) |

### 2. Dash 1D CNN (WISeREP retrain)

| Property | Detail |
|---|---|
| **Architecture** | 3× Conv1d blocks (32→64→128 channels, MaxPool ×4 each) → FC(256) → Dropout → 5 logits |
| **Input** | **1025-d** = 1024 DASH-preprocessed bins + redshift scalar (+z) or 0 (−z) |
| **Training** | 50 epochs, batch 64, lr 2e-5, early stop patience 5, seed per `iter_*` folder |
| **Code** | `zmodel_training/dash_retrain.py` (`DashCNN1D`) |

### 3. DAEP classifier (no diffusion)

| Property | Detail |
|---|---|
| **Encoder** | Perceiver-style `spectraTransceiverEncoder` (cross-attention transceiver) inside `Daepaggregator` |
| **Encoder config** (`WiserepData/Test/cfg_used.json`) | bottleneck 64×128, model_dim 192, 6 heads, 4 layers, ff_dim 384, cross-attn only (selfattn=false), concat=true |
| **Classifier head** | Mean-pool encoder output (8192-d) → MLP (GELU, dropout) → 5 logits |
| **Inputs** | flux, wavelength grid, phase (days), validity mask per spectrum |
| **Training** | 150 epochs, batch 16, lr 4e-5, early stop 15; encoder trained end-to-end with head |
| **Code** | `WiserepData/TwinsClassifier_Wiserep.py`, `TwinsModel_Wiserep.py` |

### 4. DAEP classifier (diffusion / latent)

| Property | Detail |
|---|---|
| **Stage 1** | DAEP autoencoder + diffusion score model trained on all preprocessed spectra (`TwinsTrain_Wiserep.py`; denoising loss, 90/10 random val) |
| **Stage 2** | Export frozen latents `latent_raw_z.npz` with shape **(N, 64, 128)** per spectrum |
| **Classifier** | Flatten latent → MLP head (single hidden layer + GELU in `train_latent.py`) → 5 logits; **encoder frozen** |
| **Diffusion vs no-diffusion pair** | Same preprocessing; difference is frozen diffusion-trained latents vs end-to-end encoder+head |
| **Code** | `WiserepData/train_latent.py`, `TwinsTrain_Wiserep.py` |

### 5. Transformer classifier

| Property | Detail |
|---|---|
| **Architecture** | Learnable bottleneck tokens + cross-attention to flux/λ/redshift context; 6 transformer blocks; adaptive avg pool → classifier MLP |
| **Hyperparameters** | bottleneck_length=1, model_dim=128, 4 heads, 6 layers, ff_dim=256, dropout=0.1, selfattn=false |
| **Input** | 1024 flux values + 1024 wavelength values (interpolated) + redshift scalar as sinusoidal MLP embedding |
| **Output** | 5 logits (Ia, IIn, SLSNe-I, II, Ib/c) |
| **Weights** | `data/pre_trained_models/transformer/TF_wiserep_v6.pt` |
| **Code** | `prod_backend/app/infrastructure/ml/classifiers/transformer_classifier.py`, `architectures.py` (`spectraTransformerEncoder`) |

---

## Model evaluation and comparison

### Metrics reported

| Metric | Definition | Primary script |
|---|---|---|
| **Accuracy** | Correct / total on split | `model_performance.json` (per run, **validation** split) |
| **Micro-AUC** | One-vs-rest ROC, micro-averaged over all class pairs | `WiserepData/roc_architecture_comparison.py` (**test** split) |
| **Micro-F1** | F1 with micro averaging | same |
| **Confusion matrix** | Rows=true, cols=predicted | per-run JSON + eval scripts |
| **Per-class recall** | From confusion matrix diagonal | in `model_performance.json` → `per_class` |

**Ensemble protocol:** Multiple training seeds (`iter_0` … `iter_9`); deduplicate identical checkpoints by MD5; report mean ± std across unique runs.

### Test-set results — six-model architecture comparison

Produced by `WiserepData/roc_architecture_comparison.py` on each model's **IAU-held-out test split**. Plot: `WiserepData/Test/architecture_comparison_micro.png`.

| Model | Redshift? | Micro-F1 (test) | Micro-AUC (test) | Unique runs |
|---|---|---:|---:|---:|
| DAEP (diffusion) | Yes | **0.900 ± 0.015** | **0.988 ± 0.004** | 10 |
| DAEP (no diffusion) | Yes | 0.808 ± 0.007 | 0.962 ± 0.004 | 5 |
| DAEP (no diffusion) | No | 0.735 ± 0.023 | 0.941 ± 0.007 | 5 |
| DAEP (diffusion) | No | 0.848 ± 0.006 | 0.976 ± 0.001 | 10 |
| Dash 1D CNN | Yes | 0.853 ± 0.003 | 0.980 ± 0.001 | 10 |
| Dash 1D CNN | No | 0.806 ± 0.007 | 0.967 ± 0.002 | 10 |

**Ranking (micro-AUC, +z):** DAEP diffusion > Dash 1D CNN > DAEP no-diffusion.

### Validation accuracy (from `model_performance.json`, for reference)

| Model | Val accuracy mean ± std (%) |
|---|---:|
| DAEP diffusion +z | 87.5 ± 1.3 |
| DAEP no-diffusion +z | 81.6 ± 1.8 |
| DAEP no-diffusion −z | 73.2 ± 2.3 |
| DAEP diffusion −z | 83.6 ± 0.9 |
| Dash 1D CNN +z | 87.8 ± 0.4 |
| Dash 1D CNN −z | 85.3 ± 0.7 |

### Effect of redshift (within-model pairs)

| Model family | Δ micro-F1 (+z minus −z) | Δ micro-AUC |
|---|---:|---:|
| DAEP diffusion | +0.053 | +0.012 |
| DAEP no-diffusion | +0.073 | +0.021 |
| Dash 1D CNN | +0.047 | +0.013 |

**Takeaway for paper:** Providing redshift consistently improves discrimination; no-redshift mode is harder (observed-frame features, lines shifted). Worst hit: DAEP no-diffusion −z (F1 0.735). Diffusion latents partially compensate without z (F1 0.848 vs 0.735).

### Per-class difficulty (typical across models)

From validation confusion matrices — classes with lowest recall:

1. **SLSN-I** (~50–80% depending on model/redshift) — fewest training examples (~3% of data).
2. **SN IIn** — confused with SN II and Ib/c; narrow-line features sensitive to preprocessing and z.
3. **SN Ia** — highest recall (~90–95% with redshift).
4. **SN Ib/c vs SN II** — main off-diagonal confusion.

### Inference time

Not systematically benchmarked in this repo's evaluation scripts. Order-of-magnitude expectations for paper discussion (CPU, single spectrum, approximate):

- Dash 1D CNN / Original DASH: fastest (milliseconds).
- Transformer: moderate.
- DAEP end-to-end: slower (encoder forward pass over 3250 bins).
- DAEP latent: fast head-only if latents precomputed; full path requires encoder pass.

*Recommend measuring on target deployment hardware if a specific number is needed for the Letter.*

### Suggested comparison table for the paper

| Model | Redshift | Test micro-AUC | Test micro-F1 | Val acc (%) | Input dim | Grid |
|---|---|---:|---:|---:|---:|---|
| DAEP (diffusion) | Yes | 0.988 | 0.900 | 87.5 | 8192 latent | 3250 @ 2Å |
| Dash 1D CNN | Yes | 0.980 | 0.853 | 87.8 | 1025 | 1024 log |
| DAEP (diffusion) | No | 0.976 | 0.848 | 83.6 | 8192 latent | 3250 @ 2Å |
| Dash 1D CNN | No | 0.967 | 0.806 | 85.3 | 1025 | 1024 log |
| DAEP (no diffusion) | Yes | 0.962 | 0.808 | 81.6 | 3250+aux | 3250 @ 2Å |
| DAEP (no diffusion) | No | 0.941 | 0.735 | 73.2 | 3250+aux | 3250 @ 2Å |

---

## Spectral twins (interpretability)

### Purpose

Spectral **twins** are nearest-neighbor spectra in a learned embedding space. They help users interpret a classification by showing historically similar observed SNe (by flux morphology in latent space), analogous to "this spectrum looks like SN 20xxabc at a similar phase."

### How twins are computed

Implementation: `WiserepData/BestTwins.py` (plots) and DAEP latent export from `TwinsTrain_Wiserep.py`.

1. **Encoder:** DAEP autoencoder trained on WISeREP preprocessed spectra (`data_z` by default).
2. **Embedding:** Per-spectrum latent tensor **z** with shape **(64, 128)** from `latent_raw_z.npz`.
3. **Normalization:** Unit-normalize each of the 64 latent positions along the 128-d dimension.
4. **Similarity:** Flatten to 8192-d, compute cosine similarity between all pairs; distance = **1 − cosine similarity** (averaged over sequence length L=64).
5. **Neighbors:** Precompute k=50 nearest neighbors per spectrum.
6. **Twin selection:** For a query spectrum, return nearest neighbor(s); optionally restrict to same `Obj. Type`, exclude same SN name, require unique twin SN.

### Database

- **Training library:** WISeREP spectra that passed preprocessing (~40k), with DAEP embeddings from the encoder checkpoint in `WiserepData/Test/data_z/Output/`.
- **Metadata:** IAU name, observation date, redshift, classified type — used for plot labels and same-type filtering.

### Physical meaning

Twins are similar in **learned flux+wavelength+phase representation**, not raw χ² template matching. Similar twins often share broad morphology (e.g. Si II 6150 for Ia, Hα for II, narrow emission for IIn) but may differ in phase, redshift, or noise. Color twins by `Obj. Type` in figures to show whether the model groups physically related spectra.

**Paper figure:** overlay query + twin flux on common wavelength grid (`BestTwins.py` output PNGs).

---

## Adding a new model (developer interface)

For the "Advanced Workflows" section — model development is **decoupled** from serving:

1. Train model locally on custom or public data.
2. Implement the classifier interface: preprocessing requirements, input shape, class labels, softmax output format (see `prod_backend/app/infrastructure/ml/classifiers/base.py` and existing classifiers).
3. Export weights (.pth / TorchScript) + `class_mapping` JSON + `input_shape` spec.
4. Contact maintainers to register in the model registry (user-upload workflow exists but currently requires developer review).

Key design principle: **separate model development from interaction** — new models can be swapped without changing the classification API contract.

---

## Limitations (model-focused)

| Limitation | Detail |
|---|---|
| Probabilistic outputs | All models output softmax probabilities, not definitive types; low-confidence predictions common for ambiguous spectra |
| Class coverage | Limited to 5 broad classes (or DASH 102 fine bins); rare types (TDE, Kilonova, Ibn) not in training set |
| Redshift sensitivity | No-z modes degrade 5–7 F1 points; wrong input z hurts de-redshifted models |
| Data bias | WISeREP over-represents SN Ia (~57%); SLSN-I under-represented |
| Preprocessing mismatch | User spectrum sampling/units must be handled; failure modes: insufficient wavelength overlap, bad units |
| Cross-model comparison | DAEP and Dash trained on different subsets/grids — do not over-interpret small AUC differences across groups |
| Twins cost | Full pairwise latent distance matrix is O(N²); production uses precomputed neighbors / caching |
| Original DASH age bins | 102-class outputs require age×type combination logic; not directly comparable to 5-class metrics |

---

## Reproducibility and artifacts

| Artifact | Location |
|---|---|
| Architecture comparison ROC plot | `WiserepData/Test/architecture_comparison_micro.png` |
| Per-run metrics + confusion matrices | `*/iter_*/model_performance.json` under each checkpoint root |
| DAEP config used | `WiserepData/Test/cfg_used.json` |
| WISeREP preprocess outputs | `WiserepData/Test/data_z/`, `data_no_z/` |
| Dash IAU splits | `data/wiserep/daep_compatible_split.json` |
| Label maps | `zmodel_training/constants.py`, `WiserepData/TwinsClassifier_Wiserep.py` |
| Preprocessing ablation plots | `data/pre_trained_models/dash_wiserep/models/preprocessing_removal_difference_vs_full*.png` |
| Example preprocess figure | `WiserepData/Test/data_z/example_preprocessing.png` (generated by `Preprocess.py`) |
| Twin example plots | generated by `WiserepData/BestTwins.py` |

### Key scripts

| Task | Script |
|---|---|
| WISeREP offline preprocess | `WiserepData/Preprocess.py` |
| DAEP encoder pretrain | `WiserepData/TwinsTrain_Wiserep.py` |
| DAEP classifier (no diffusion) | `WiserepData/TwinsClassifier_Wiserep.py` |
| DAEP latent classifier (diffusion) | `WiserepData/train_latent.py` |
| Dash 1D CNN retrain | `zmodel_training/dash_retrain.py` |
| Architecture comparison | `WiserepData/roc_architecture_comparison.py` |
| Twin plots | `WiserepData/BestTwins.py` |
| Preprocessing ablation | `zmodel_training/dash_preprocessing_removal_diff_plot.py` |
| IAU splits | `WiserepData/iau_train_val_test_split.py`, `zmodel_training/create_wiserep_splits_by_iau.py` |

### License / weights availability

- Code in this repository: check repo `LICENSE` for software terms.
- Model weights: stored under `data/pre_trained_models/` and `WiserepData/Test/` (large binary checkpoints; suitable for archived release / DOI with data availability statement).
- Training data: WISeREP is publicly accessible; processed bundles can be regenerated from `Preprocess.py`.

---

## Figures checklist (models only)

| Figure | Source |
|---|---|
| Model performance comparison (micro-AUC ROC) | `WiserepData/Test/architecture_comparison_micro.png` |
| Raw vs preprocessed spectrum | `Preprocess.py` → `example_preprocessing.png`; or DASH processor before/after |
| Dataset class distribution pie chart | Counts in **Class distribution** section above |
| Model architecture comparison table | **Model suite at a glance** + **Suggested comparison table** |
| Preprocessing steps diagram | DASH 9-step table + WISeREP 7-step table |
| Performance vs preprocessing removed | `preprocessing_removal_difference_vs_full.png` |
| Spectral twins overlay | `BestTwins.py` output |
| Per-model confusion matrices | `model_performance.json` → `confusion_matrix_raw` |
| DAEP architecture schematic | Perceiver transceiver encoder diagram (cite Perceiver-diffusion-autoencoder / DAEP paper) |

---

## Prior work to cite (models context)

- **DASH** — original deep learning SN classifier (Gluschke et al.).
- **SNID / Superfit** — template-matching baselines.
- **ABC SN, SNID SAGE** — other ML classification approaches.
- **WISeREP** — training data source (Yaron & Gal-Yam).
- **Perceiver-diffusion-autoencoder (DAEP)** — encoder architecture for DAEP models.
- **LSST/Rubin Observatory** — motivation for scalable classification.

---

## Quick takeaways for Conclusion (models angle)

1. AstroDASH 2.0 bundles **multiple complementary classifiers** (legacy DASH, retrained 1D CNN, DAEP variants, transformer) on a shared WISeREP training foundation.
2. **Best test performance:** DAEP diffusion + redshift (micro-AUC 0.988, micro-F1 0.90).
3. **Redshift matters:** always provide z when available; no-z modes are supported but less accurate.
4. **Spectral twins** in DAEP latent space provide interpretability beyond a single class label.
5. The repo **decouples model development from serving** — researchers can train custom models and plug them into the same interface.
6. Fair benchmarking requires matching **data split, preprocessing, and label space** — the six-model comparison is rigorous within the 5-class WISeREP framework but Original DASH and Transformer use distinct output spaces.
