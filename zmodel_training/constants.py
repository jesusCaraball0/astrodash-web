import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_prod_backend = PROJECT_ROOT / "prod_backend"
for _path in (str(PROJECT_ROOT), str(_prod_backend)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from prod_backend.app.config.settings import get_settings
from prod_backend.app.infrastructure.ml.data_processor import DashSpectrumProcessor
from prod_backend.app.infrastructure.storage.file_spectrum_repository import FileSpectrumRepository

RUN_ID = "03_25_26_no_redshift"
HAS_REDSHIFT = False

WISEREP_DIR = PROJECT_ROOT / "data" / "wiserep"
SPECTRA_DIR = WISEREP_DIR / "wiserep_data_noSEDM"
METADATA_CSV = WISEREP_DIR / "wiserep_metadata.csv"
# Splits JSONs: 80/10/10 for single-run train/val/test, 90/10 for k-fold (train+test only)
SPLITS_JSON_80_10_10 = WISEREP_DIR / "wiserep_splits_by_iau_80_10_10.json"
SPLITS_JSON_90_10 = WISEREP_DIR / "wiserep_splits_by_iau_90_10.json"
OUT_DIR = PROJECT_ROOT / "data" / "pre_trained_models" / "dash_wiserep" / "models" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20
VAL_EVERY = 1
NUM_WORKERS = 0
SEED = 10

# 5-class label mapping (from Lanqing)
LABEL_MAP: Dict[str, str] = {
    # Type Ia
    "SN Ia": "SN Ia",
    "SN Ia-CSM": "SN Ia",
    "SN Ia-91T-like": "SN Ia",
    "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia",
    "SN Ia-pec": "SN Ia",
    "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia",
    "Computed-Ia": "SN Ia",
    # Type Ib/c (all Ib and Ic variants)
    "SN Ib": "SN Ib/c",
    "SN Ic": "SN Ib/c",
    "SN Ib/c": "SN Ib/c",
    "SN Ib-Ca-rich": "SN Ib/c",
    "SN Ib-pec": "SN Ib/c",
    "SN Ibn": "SN Ib/c",
    "SN Ic-BL": "SN Ib/c",
    "SN Ic-Ca-rich": "SN Ib/c",
    "SN Ic-pec": "SN Ib/c",
    "SN Icn": "SN Ib/c",
    "SN Ib/c-Ca-rich": "SN Ib/c",
    "SN Ibn/Icn": "SN Ib/c",
    # Type II (NOT including IIn)
    "SN II": "SN II",
    "SN IIP": "SN II",
    "SN IIL": "SN II",
    "SN II-pec": "SN II",
    "SN IIb": "SN II",
    "Computed-IIP": "SN II",
    "Computed-IIb": "SN II",
    # Type IIn (separate)
    "SN IIn": "SN IIn",
    "SN IIn-pec": "SN IIn",
    # SLSN
    "SLSN-I": "SLSN-I",
    "SLSN-II": "SLSN-I",
    "SLSN-R": "SLSN-I",
}

# Default output filenames per mode (can override with --output); same paths as split constants
DEFAULT_OUTPUT_80_10_10 = SPLITS_JSON_80_10_10
DEFAULT_OUTPUT_90_10 = SPLITS_JSON_90_10

# 5-class label mapping (must match dash_retrain / eval)
LABEL_MAP: dict[str, str] = {
    "SN Ia": "SN Ia", "SN Ia-CSM": "SN Ia", "SN Ia-91T-like": "SN Ia", "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia", "SN Ia-pec": "SN Ia", "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia", "Computed-Ia": "SN Ia",
    "SN Ib": "SN Ib/c", "SN Ic": "SN Ib/c", "SN Ib/c": "SN Ib/c", "SN Ib-Ca-rich": "SN Ib/c",
    "SN Ib-pec": "SN Ib/c", "SN Ibn": "SN Ib/c", "SN Ic-BL": "SN Ib/c", "SN Ic-Ca-rich": "SN Ib/c",
    "SN Ic-pec": "SN Ib/c", "SN Icn": "SN Ib/c", "SN Ib/c-Ca-rich": "SN Ib/c", "SN Ibn/Icn": "SN Ib/c",
    "SN II": "SN II", "SN IIP": "SN II", "SN IIL": "SN II", "SN II-pec": "SN II", "SN IIb": "SN II",
    "Computed-IIP": "SN II", "Computed-IIb": "SN II",
    "SN IIn": "SN IIn", "SN IIn-pec": "SN IIn",
    "SLSN-I": "SLSN-I", "SLSN-II": "SLSN-I", "SLSN-R": "SLSN-I",
}

CLASS_NAMES = ["SN Ia", "SN Ib/c", "SN II", "SN IIn", "SLSN-I"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# match backend DASH preprocessing params
_SETTINGS = get_settings()
# Model input = spectrum bins (nw) + 1 redshift feature
TARGET_LENGTH = _SETTINGS.nw + 1
WAVE_MIN, WAVE_MAX = _SETTINGS.w0, _SETTINGS.w1

# global DashSpectrumProcessor (outputs nw bins; we append z to get TARGET_LENGTH)
_PROCESSOR = DashSpectrumProcessor(WAVE_MIN, WAVE_MAX, _SETTINGS.nw)
_FILE_REPO = FileSpectrumRepository(_SETTINGS)

