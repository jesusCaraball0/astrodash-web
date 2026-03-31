#!/usr/bin/env python
"""
Bayesian hyperparameter search for the DASH CNN (80/10/10, single run).

Uses Optuna with a TPE sampler (Tree-structured Parzen Estimator)
sequential model-based optimization, the usual Bayesian optimization stack in Optuna.

Usage:
    python zmodel_training/dash_tune_hyperparams.py

"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import optuna

import constants as const
import helpers
from dash_retrain import DashCNN1D, train

_MODELS_BASE = const.OUT_DIR.parent

# study configs
STUDY_DIR = _MODELS_BASE / f"{const.RUN_ID}_hparam_tune"
N_TRIALS = 25
OPTUNA_STORAGE_URL = ""
OPTUNA_STUDY_NAME = "dash_wiserep_no_z_tune"
BASE_SEED = 42


def _merge_results(study_dir: Path, row: Dict[str, Any]) -> None:
    path = study_dir / "tune_results.json"
    data: Dict[str, Any] = {"trials": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    trials: List[Dict[str, Any]] = data.get("trials", [])
    key = row.get("trial_id")
    trials = [t for t in trials if t.get("trial_id") != key]
    trials.append(row)
    data["trials"] = trials
    path.write_text(json.dumps(data, indent=2))


def suggest_config(trial: Any, base_seed: int) -> Dict[str, Any]:
    """Search space"""
    return {
        "trial_id": f"optuna_trial_{trial.number}",
        "epochs": trial.suggest_int("epochs", 40, 80, step=10),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 5, 8),
        "seed": base_seed + trial.number,
        "val_every": trial.suggest_categorical("val_every", [1, 2]),
        "plateau_factor": trial.suggest_float("plateau_factor", 0.2, 0.8),
        "plateau_patience": trial.suggest_int("plateau_patience", 1, 6),
        "plateau_mode": "min",
        "plateau_min_lr": trial.suggest_float("plateau_min_lr", 1e-8, 1e-4, log=True),
        "plateau_threshold": trial.suggest_float("plateau_threshold", 1e-6, 1e-2, log=True),
        "plateau_threshold_mode": "rel",
        "plateau_cooldown": trial.suggest_int("plateau_cooldown", 0, 3),
    }


def run_trial(cfg: Dict[str, Any], study_dir: Path, device) -> Optional[float]:
    tid = cfg["trial_id"]
    trial_out = study_dir / tid
    trial_out.mkdir(parents=True, exist_ok=True)

    helpers.set_seed(int(cfg["seed"]))

    splits = helpers.load_json(const.SPLITS_JSON_80_10_10)
    metadata = helpers.load_metadata(const.METADATA_CSV)
    train_f, val_f = list(splits["train"]), list(splits["val"])
    bs = int(cfg["batch_size"])

    class_weights = helpers.compute_class_weights_from_filenames(train_f, metadata)
    train_loader = helpers.make_loader(
        train_f, metadata, const.HAS_REDSHIFT, device, shuffle=True, batch_size=bs
    )
    val_loader = helpers.make_loader(
        val_f, metadata, const.HAS_REDSHIFT, device, batch_size=bs
    )

    with open(trial_out / "class_mapping.json", "w") as f:
        json.dump({n: i for n, i in const.CLASS_TO_IDX.items()}, f, indent=2)

    model = DashCNN1D(
        input_length=const.TARGET_LENGTH, num_classes=const.NUM_CLASSES
    ).to(device)

    train(
        model,
        train_loader,
        val_loader,
        device,
        class_weights,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        patience=int(cfg["early_stop_patience"]),
        val_every=int(cfg.get("val_every", 1)),
        out_dir=trial_out,
        plateau_factor=float(cfg["plateau_factor"]),
        plateau_patience=int(cfg["plateau_patience"]),
        plateau_mode=str(cfg["plateau_mode"]),
        plateau_min_lr=float(cfg["plateau_min_lr"]),
        plateau_threshold=float(cfg["plateau_threshold"]),
        plateau_threshold_mode=str(cfg["plateau_threshold_mode"]),
        plateau_cooldown=int(cfg["plateau_cooldown"]),
    )

    perf_path = trial_out / "model_performance.json"
    perf = json.loads(perf_path.read_text()) if perf_path.exists() else {}
    cum = perf.get("cumulative", {})
    val_loss = cum.get("loss")

    training_config = {
        "run_id": const.RUN_ID,
        "trial_id": tid,
        "study_dir": str(study_dir),
        "has_redshift": const.HAS_REDSHIFT,
        "target_length": const.TARGET_LENGTH,
        "wave_min": const.WAVE_MIN,
        "wave_max": const.WAVE_MAX,
        "num_classes": const.NUM_CLASSES,
        "class_names": const.CLASS_NAMES,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "patience": cfg["early_stop_patience"],
        "seed": cfg["seed"],
        "val_every": cfg.get("val_every", 1),
        "plateau_factor": cfg["plateau_factor"],
        "plateau_patience": cfg["plateau_patience"],
        "plateau_mode": cfg["plateau_mode"],
        "plateau_min_lr": cfg["plateau_min_lr"],
        "plateau_threshold": cfg["plateau_threshold"],
        "plateau_threshold_mode": cfg["plateau_threshold_mode"],
        "plateau_cooldown": cfg["plateau_cooldown"],
        "class_weights": class_weights.tolist(),
        "splits_file": str(const.SPLITS_JSON_80_10_10),
        "k_fold": 1,
        "best_epoch": perf.get("best_epoch"),
        "best_val_loss": val_loss,
        "best_val_accuracy_pct": cum.get("accuracy_pct"),
    }
    with open(trial_out / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    _merge_results(
        study_dir,
        {
            "trial_id": tid,
            "best_epoch": perf.get("best_epoch"),
            "best_val_loss": val_loss,
            "best_val_accuracy_pct": cum.get("accuracy_pct"),
            "params": {k: v for k, v in cfg.items() if k != "trial_id"},
        },
    )
    print(f"Trial {tid} done → {trial_out}  val_loss={val_loss}")
    return float(val_loss) if val_loss is not None else None


def _sqlite_storage_url(study_dir: Path) -> str:
    db = study_dir.resolve() / "optuna.sqlite3"
    return "sqlite:///" + str(db).replace("\\", "/")


def main() -> None:

    study_dir = STUDY_DIR
    helpers.require(const.SPLITS_JSON_80_10_10, "Splits file")
    device = helpers.get_device()
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = OPTUNA_STORAGE_URL or _sqlite_storage_url(study_dir)

    print(f"Device: {device}")
    print(f"Study dir: {study_dir}")
    print(f"Running {N_TRIALS} Optuna trials (TPE sampler); storage: {storage}")

    # TPE = Tree-structured Parzen Estimator
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, base_seed=BASE_SEED)
        loss = run_trial(cfg, study_dir, device)
        return float("inf") if loss is None else loss

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial_number": best.number,
        "n_trials_in_study": len(study.trials),
    }
    (study_dir / "optuna_best.json").write_text(json.dumps(summary, indent=2))
    print("\nBest trial:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
