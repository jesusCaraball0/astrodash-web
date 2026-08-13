"""
Embedding-quality artifacts: linear probe table, cosine kNN acc/purity, delta CM.

1) Linear probe (sklearn LogisticRegression on frozen flattened z)
2) Cosine kNN test accuracy + neighbor purity
3) Delta confusion matrix: Original MLP ensemble − best other ensemble

Usage:
  PYTHONPATH=WiserepData python WiserepData/latent_embedding_quality.py
  PYTHONPATH=WiserepData python WiserepData/latent_embedding_quality.py \\
    --sources original,1024d2,five_try,six_try --k 15
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = WISEREP_DIR.parent
if str(WISEREP_DIR) not in sys.path:
    sys.path.insert(0, str(WISEREP_DIR))

from train_latent import (  # noqa: E402
    HENNA_ROOT,
    LABEL_COLUMN,
    load_assignment_indices_from_dir,
    load_latent_and_meta,
    normalize_latent_meta,
    resolve_latent_npz,
    split_and_filter,
)
from TwinsClassifier_Wiserep import CLASS_NAMES, NUM_CLASSES, row_class_idx  # noqa: E402
from latent_plots import (  # noqa: E402
    _col_normalize_cm,
    _row_normalize_cm,
    discover_run_dirs,
)

TRY_NAMES = ("second_try", "third_try", "four_try", "five_try", "six_try")
TRY_TITLE = {
    "second_try": "_2",
    "third_try": "_3",
    "four_try": "_4",
    "five_try": "_5",
    "six_try": "_6",
}
PREFERRED_SPLIT_SEEDS = (36, 73, 149, 257)

TWINSANITY_LATENT_DIR = WISEREP_DIR / "Test" / "twinsanity_latents_40867"

# Ensemble roots for MLP delta-CM / optional MLP columns.
ENSEMBLE_SPECS: list[tuple[str, str, pathlib.Path, str]] = [
    ("original", "Original", WISEREP_DIR / "Test" / "daep_comparison_legacy_unique", "legacy"),
    ("1024d2", "1024d2", WISEREP_DIR / "Test" / "daep_comparison", "underscore"),
    ("second_try", "_2", WISEREP_DIR / "Test" / "daep_comparison_second_try", "split"),
    ("third_try", "_3", WISEREP_DIR / "Test" / "daep_comparison_third_try", "split"),
    ("four_try", "_4", WISEREP_DIR / "Test" / "daep_comparison_four_try", "split"),
    ("five_try", "_5", WISEREP_DIR / "Test" / "daep_comparison_five_try", "split"),
    ("six_try", "_6", WISEREP_DIR / "Test" / "daep_comparison_six_try", "split"),
]


@dataclass
class SourceBundle:
    key: str
    title: str
    z: np.ndarray
    meta: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    latent_path: pathlib.Path


def _try_title(key: str) -> str:
    return TRY_TITLE.get(key, key)


def _pick_try_subdir(try_root: pathlib.Path, prefer_seed: int) -> pathlib.Path:
    subdirs = [p for p in sorted(try_root.iterdir()) if p.is_dir()]
    seed_order = (prefer_seed, *[s for s in PREFERRED_SPLIT_SEEDS if s != prefer_seed])
    for seed in seed_order:
        pat = re.compile(rf"^Dered{seed}(_\d+)?$", re.IGNORECASE)
        for p in subdirs:
            if pat.fullmatch(p.name):
                return p
    for p in subdirs:
        if p.name.lower().startswith("dered"):
            return p
    for p in subdirs:
        try:
            resolve_latent_npz(p)
            return p
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No latent dirs under {try_root}")


def _labels(meta: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return np.asarray([row_class_idx(meta, int(i), LABEL_COLUMN) for i in idx], dtype=np.int64)


def _flatten(z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(z[idx], dtype=np.float32).reshape(len(idx), -1)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def resolve_latent_source(
    key: str,
    *,
    henna_root: pathlib.Path,
    prefer_seed: int,
    original_dir: pathlib.Path,
) -> SourceBundle:
    key = key.strip().lower().replace("-", "_")
    if key in ("original", "twinsanity", "legacy"):
        latent_dir = original_dir.resolve()
        meta, z, latent_npz, _ = load_latent_and_meta(latent_dir)
        try:
            tr, va, te, src = load_assignment_indices_from_dir(latent_dir, meta)
            tr, va, te = split_and_filter(meta, train_idx=tr, val_idx=va, test_idx=te, split_tag=src)
        except FileNotFoundError:
            tr, va, te = split_and_filter(meta)
        return SourceBundle("original", "Original", z, meta, tr, va, te, latent_npz)

    if key in ("1024d2", "henna_matched", "henna1024d2"):
        latent_dir = (henna_root / "1024d2").resolve()
        meta, z, latent_npz, _ = load_latent_and_meta(latent_dir)
        tr, va, te, src = load_assignment_indices_from_dir(latent_dir, meta)
        tr, va, te = split_and_filter(meta, train_idx=tr, val_idx=va, test_idx=te, split_tag=src)
        return SourceBundle("1024d2", "1024d2", z, meta, tr, va, te, latent_npz)

    if key in TRY_NAMES or key.endswith("_try"):
        sub = _pick_try_subdir(henna_root / key, prefer_seed).resolve()
        meta, z, latent_npz, _ = load_latent_and_meta(sub)
        tr, va, te, src = load_assignment_indices_from_dir(sub, meta)
        tr, va, te = split_and_filter(meta, train_idx=tr, val_idx=va, test_idx=te, split_tag=src)
        return SourceBundle(key, _try_title(key), z, meta, tr, va, te, latent_npz)

    latent_dir = pathlib.Path(key).expanduser().resolve()
    meta, z, latent_npz, _ = load_latent_and_meta(latent_dir)
    tr, va, te, src = load_assignment_indices_from_dir(latent_dir, meta)
    tr, va, te = split_and_filter(meta, train_idx=tr, val_idx=va, test_idx=te, split_tag=src)
    return SourceBundle(latent_dir.name, latent_dir.name, z, meta, tr, va, te, latent_npz)


def metrics_from_cm(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    total = float(cm.sum())
    acc = float(cm.trace() / total) if total else 0.0
    prec = np.zeros(NUM_CLASSES, dtype=np.float64)
    rec = np.zeros(NUM_CLASSES, dtype=np.float64)
    f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        support = cm[c].sum()
        pred = cm[:, c].sum()
        p = float(tp / pred) if pred > 0 else 0.0
        r = float(tp / support) if support > 0 else 0.0
        prec[c] = p
        rec[c] = r
        f1[c] = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
    return {
        "acc": acc,
        "macro_f1": float(f1.mean()),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
    }


def cm_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < NUM_CLASSES and 0 <= int(p) < NUM_CLASSES:
            cm[int(t), int(p)] += 1.0
    return cm


def run_linear_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xe = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    cm = cm_from_preds(y_test, pred)
    out = metrics_from_cm(cm)
    out["y_pred"] = pred
    return out


def run_cosine_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int,
) -> dict:
    Xtr = _l2_normalize(X_train)
    Xte = _l2_normalize(X_test)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(Xtr)
    _, idx = nn.kneighbors(Xte, return_distance=True)
    neigh_y = y_train[idx]  # (n_test, k)

    # Majority vote (stable: lowest class index on ties via bincount).
    preds = np.empty(len(y_test), dtype=np.int64)
    for i in range(len(y_test)):
        preds[i] = int(np.bincount(neigh_y[i], minlength=NUM_CLASSES).argmax())

    purity = (neigh_y == y_test[:, None]).mean(axis=1)
    per_class_purity = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        m = y_test == c
        per_class_purity[c] = float(purity[m].mean()) if np.any(m) else float("nan")

    cm = cm_from_preds(y_test, preds)
    out = metrics_from_cm(cm)
    out.update(
        {
            "purity": float(purity.mean()),
            "per_class_purity": per_class_purity,
            "k": k,
            "y_pred": preds,
        }
    )
    return out


def load_ensemble_cms(root: pathlib.Path, run_style: str) -> list[np.ndarray]:
    if not root.is_dir():
        return []
    try:
        runs = discover_run_dirs(root, run_style=run_style)
    except FileNotFoundError:
        return []
    cms: list[np.ndarray] = []
    for run in runs:
        perf = run / "model_performance.json"
        if not perf.is_file():
            continue
        payload = json.loads(perf.read_text(encoding="utf-8"))
        cms.append(np.asarray(payload["confusion_matrix_raw"], dtype=np.float64))
    return cms


def summarize_ensemble(cms: list[np.ndarray]) -> dict | None:
    if not cms:
        return None
    mets = [metrics_from_cm(cm) for cm in cms]
    accs = np.asarray([m["acc"] for m in mets])
    f1s = np.asarray([m["macro_f1"] for m in mets])
    prec = np.mean([m["precision"] for m in mets], axis=0)
    rec = np.mean([m["recall"] for m in mets], axis=0)
    # Mean of normalized CMs (latent_plots style).
    recall_stack = np.stack([_row_normalize_cm(cm) * 100.0 for cm in cms], axis=0)
    prec_stack = np.stack([_col_normalize_cm(cm) * 100.0 for cm in cms], axis=0)
    return {
        "n_runs": len(cms),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std()),
        "precision": prec,
        "recall": rec,
        "cm_recall_pct": recall_stack.mean(axis=0),
        "cm_precision_pct": prec_stack.mean(axis=0),
        "cm_recall_std": recall_stack.std(axis=0),
        "cm_precision_std": prec_stack.std(axis=0),
    }


def _fmt_pct(mean: float, std: float | None = None, digits: int = 1) -> str:
    if std is None:
        return f"{100.0 * mean:.{digits}f}"
    return f"{100.0 * mean:.{digits}f}±{100.0 * std:.{digits}f}"


def plot_probe_knn_tables(
    rows: list[dict],
    *,
    k: int,
    out_path: pathlib.Path,
) -> None:
    """Two stacked tables: linear probe + cosine kNN."""
    probe_cols = [
        "Source",
        "Acc",
        "Macro-F1",
        "IIn P",
        "IIn R",
        "SLSN-I P",
        "SLSN-I R",
    ]
    knn_cols = [
        "Source",
        f"kNN@{k} Acc",
        f"Purity@{k}",
        "IIn Pur",
        "SLSN-I Pur",
        "IIn R",
        "SLSN-I R",
    ]

    probe_cells: list[list[str]] = []
    knn_cells: list[list[str]] = []
    for r in rows:
        iin = CLASS_NAMES.index("SN IIn")
        sl = CLASS_NAMES.index("SLSN-I")
        probe_cells.append(
            [
                r["title"],
                _fmt_pct(r["probe"]["acc"]),
                _fmt_pct(r["probe"]["macro_f1"]),
                _fmt_pct(r["probe"]["precision"][iin]),
                _fmt_pct(r["probe"]["recall"][iin]),
                _fmt_pct(r["probe"]["precision"][sl]),
                _fmt_pct(r["probe"]["recall"][sl]),
            ]
        )
        knn_cells.append(
            [
                r["title"],
                _fmt_pct(r["knn"]["acc"]),
                _fmt_pct(r["knn"]["purity"]),
                _fmt_pct(r["knn"]["per_class_purity"][iin]),
                _fmt_pct(r["knn"]["per_class_purity"][sl]),
                _fmt_pct(r["knn"]["recall"][iin]),
                _fmt_pct(r["knn"]["recall"][sl]),
            ]
        )

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 4.8 + 0.35 * len(rows)))
    for ax, title, colnames, cells in (
        (
            axes[0],
            "Linear probe on frozen latents (LogReg, balanced CE)",
            probe_cols,
            probe_cells,
        ),
        (
            axes[1],
            f"Cosine kNN on frozen latents (train→test, k={k})",
            knn_cols,
            knn_cells,
        ),
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=8)
        table = ax.table(
            cellText=cells,
            colLabels=colnames,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.15, 1.45)
        # Header styling + highlight best Acc column.
        for j in range(len(colnames)):
            table[0, j].set_facecolor("#2F4A6D")
            table[0, j].set_text_props(color="white", weight="bold")
        # Bold first data column (source names).
        for i in range(1, len(cells) + 1):
            table[i, 0].set_text_props(weight="bold")
            if i % 2 == 0:
                for j in range(len(colnames)):
                    table[i, j].set_facecolor("#F3F5F8")
        # Highlight max accuracy row in Acc column (index 1).
        acc_vals = [float(c[1].split("±")[0]) for c in cells]
        best_i = int(np.argmax(acc_vals)) + 1
        table[best_i, 1].set_facecolor("#D9EAD3")

    fig.suptitle("Embedding quality probes", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_ensemble_table(ensembles: list[dict], out_path: pathlib.Path) -> None:
    cols = ["Source", "N", "Acc", "Macro-F1", "IIn P", "IIn R", "SLSN-I P", "SLSN-I R"]
    cells: list[list[str]] = []
    iin = CLASS_NAMES.index("SN IIn")
    sl = CLASS_NAMES.index("SLSN-I")
    for e in ensembles:
        cells.append(
            [
                e["title"],
                str(e["n_runs"]),
                _fmt_pct(e["acc_mean"], e["acc_std"]),
                _fmt_pct(e["macro_f1_mean"], e["macro_f1_std"]),
                _fmt_pct(e["precision"][iin]),
                _fmt_pct(e["recall"][iin]),
                _fmt_pct(e["precision"][sl]),
                _fmt_pct(e["recall"][sl]),
            ]
        )

    fig, ax = plt.subplots(figsize=(11.5, 2.2 + 0.4 * len(cells)))
    ax.axis("off")
    ax.set_title("MLP latent-classifier ensembles (existing runs)", fontsize=12, pad=8)
    table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.5)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2F4A6D")
        table[0, j].set_text_props(color="white", weight="bold")
    for i in range(1, len(cells) + 1):
        table[i, 0].set_text_props(weight="bold")
        if i % 2 == 0:
            for j in range(len(cols)):
                table[i, j].set_facecolor("#F3F5F8")
    acc_vals = [float(c[2].split("±")[0]) for c in cells]
    best_i = int(np.argmax(acc_vals)) + 1
    table[best_i, 2].set_facecolor("#D9EAD3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_cm(
    original: dict,
    other: dict,
    *,
    out_path: pathlib.Path,
) -> None:
    d_prec = original["cm_precision_pct"] - other["cm_precision_pct"]
    d_rec = original["cm_recall_pct"] - other["cm_recall_pct"]
    lim = float(max(np.abs(d_prec).max(), np.abs(d_rec).max(), 1.0))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, mat, title in (
        (axes[0], d_prec, "Δ Precision (pp)"),
        (axes[1], d_rec, "Δ Recall (pp)"),
    ):
        im = ax.imshow(mat, cmap="RdBu", norm=norm, origin="upper", aspect="equal")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    o_acc = 100.0 * original["acc_mean"]
    o_std = 100.0 * original["acc_std"]
    b_acc = 100.0 * other["acc_mean"]
    b_std = 100.0 * other["acc_std"]
    fig.suptitle(
        f"Δ CM = {original['title']} − {other['title']}  |  "
        f"{original['title']} {o_acc:.1f}±{o_std:.1f}%   vs   "
        f"{other['title']} {b_acc:.1f}±{b_std:.1f}%",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def default_sources() -> list[str]:
    names = ["original", "1024d2"]
    for t in TRY_NAMES:
        if (HENNA_ROOT / t).is_dir():
            names.append(t)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe table + cosine kNN + delta CM artifacts.")
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated: original,1024d2,second_try,...,six_try",
    )
    parser.add_argument("--prefer-seed", type=int, default=36)
    parser.add_argument("--henna-root", type=pathlib.Path, default=HENNA_ROOT)
    parser.add_argument(
        "--original-dir",
        type=pathlib.Path,
        default=TWINSANITY_LATENT_DIR,
        help="Twinsanity latent dir (meta + npz + assignment).",
    )
    parser.add_argument("--k", type=int, default=15, help="k for cosine kNN / purity.")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=WISEREP_DIR / "Test" / "plots" / "latent_embedding_quality",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip linear probe / kNN (only MLP table + delta CM).",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_keys = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else default_sources()
    )

    # --- MLP ensemble table + delta CM (from saved runs) ---
    ensembles: list[dict] = []
    for key, title, root, style in ENSEMBLE_SPECS:
        cms = load_ensemble_cms(root, style)
        summary = summarize_ensemble(cms)
        if summary is None:
            print(f"[mlp] skip {title}: no runs under {root}", flush=True)
            continue
        summary["key"] = key
        summary["title"] = title
        ensembles.append(summary)
        print(
            f"[mlp] {title}: n={summary['n_runs']} "
            f"acc={100*summary['acc_mean']:.1f}±{100*summary['acc_std']:.1f}% "
            f"macroF1={100*summary['macro_f1_mean']:.1f}±{100*summary['macro_f1_std']:.1f}%",
            flush=True,
        )

    if ensembles:
        mlp_table_path = out_dir / "mlp_ensemble_probe_table.png"
        plot_mlp_ensemble_table(ensembles, mlp_table_path)
        print(f"[save] {mlp_table_path}", flush=True)

        original = next((e for e in ensembles if e["key"] == "original"), None)
        others = [e for e in ensembles if e["key"] != "original"]
        if original is not None and others:
            best = max(others, key=lambda e: e["acc_mean"])
            safe = re.sub(r"[^\w.\-]+", "_", best["title"]).strip("_")
            delta_path = out_dir / f"delta_cm_Original_minus_{safe}.png"
            plot_delta_cm(original, best, out_path=delta_path)
            print(f"[delta] best of rest = {best['title']} ({100*best['acc_mean']:.1f}%)", flush=True)
            print(f"[save] {delta_path}", flush=True)

    if args.skip_probe:
        print("Done (skipped linear probe / kNN).", flush=True)
        return

    # --- Fresh linear probe + cosine kNN ---
    rows: list[dict] = []
    records: list[dict] = []
    for key in source_keys:
        print(f"\n=== source {key} ===", flush=True)
        bundle = resolve_latent_source(
            key,
            henna_root=args.henna_root.expanduser().resolve(),
            prefer_seed=args.prefer_seed,
            original_dir=args.original_dir.expanduser().resolve(),
        )
        y_tr = _labels(bundle.meta, bundle.train_idx)
        y_te = _labels(bundle.meta, bundle.test_idx)
        assert np.all(y_tr >= 0) and np.all(y_te >= 0)
        X_tr = _flatten(bundle.z, bundle.train_idx)
        X_te = _flatten(bundle.z, bundle.test_idx)
        print(
            f"[data] {bundle.title}: z={tuple(bundle.z.shape)} "
            f"train={len(bundle.train_idx)} test={len(bundle.test_idx)} feat={X_tr.shape[1]}",
            flush=True,
        )

        print("[probe] fitting LogReg…", flush=True)
        probe = run_linear_probe(X_tr, y_tr, X_te, y_te)
        print(
            f"[probe] acc={100*probe['acc']:.1f}% macroF1={100*probe['macro_f1']:.1f}%",
            flush=True,
        )

        print(f"[knn] cosine k={args.k}…", flush=True)
        knn = run_cosine_knn(X_tr, y_tr, X_te, y_te, k=args.k)
        print(
            f"[knn] acc={100*knn['acc']:.1f}% purity={100*knn['purity']:.1f}%",
            flush=True,
        )

        rows.append({"title": bundle.title, "key": bundle.key, "probe": probe, "knn": knn})
        iin = CLASS_NAMES.index("SN IIn")
        sl = CLASS_NAMES.index("SLSN-I")
        records.append(
            {
                "source": bundle.title,
                "key": bundle.key,
                "z_shape": str(tuple(bundle.z.shape)),
                "feat_dim": int(X_tr.shape[1]),
                "n_train": int(len(bundle.train_idx)),
                "n_test": int(len(bundle.test_idx)),
                "probe_acc": probe["acc"],
                "probe_macro_f1": probe["macro_f1"],
                "probe_iin_p": probe["precision"][iin],
                "probe_iin_r": probe["recall"][iin],
                "probe_slsn_p": probe["precision"][sl],
                "probe_slsn_r": probe["recall"][sl],
                "knn_k": args.k,
                "knn_acc": knn["acc"],
                "knn_purity": knn["purity"],
                "knn_iin_purity": knn["per_class_purity"][iin],
                "knn_slsn_purity": knn["per_class_purity"][sl],
                "knn_iin_r": knn["recall"][iin],
                "knn_slsn_r": knn["recall"][sl],
            }
        )

    table_path = out_dir / "linear_probe_and_cosine_knn_table.png"
    plot_probe_knn_tables(rows, k=args.k, out_path=table_path)
    print(f"[save] {table_path}", flush=True)

    csv_path = out_dir / "linear_probe_and_cosine_knn_metrics.csv"
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
    print(f"[save] {csv_path}", flush=True)

    # Delta CM from linear-probe CMs (Original vs best other by probe acc).
    if rows:
        original_row = next((r for r in rows if r["key"] == "original"), None)
        other_rows = [r for r in rows if r["key"] != "original"]
        if original_row is not None and other_rows:
            best = max(other_rows, key=lambda r: r["probe"]["acc"])
            # Build summarize_ensemble-like dicts from single probe CMs.
            def _as_ens(r: dict) -> dict:
                cm = r["probe"]["cm"]
                return {
                    "title": r["title"],
                    "acc_mean": r["probe"]["acc"],
                    "acc_std": 0.0,
                    "cm_precision_pct": _col_normalize_cm(cm) * 100.0,
                    "cm_recall_pct": _row_normalize_cm(cm) * 100.0,
                }

            safe = re.sub(r"[^\w.\-]+", "_", best["title"]).strip("_")
            delta_probe_path = out_dir / f"delta_cm_probe_Original_minus_{safe}.png"
            plot_delta_cm(_as_ens(original_row), _as_ens(best), out_path=delta_probe_path)
            print(
                f"[delta-probe] best of rest = {best['title']} "
                f"(probe acc {100*best['probe']['acc']:.1f}%)",
                flush=True,
            )
            print(f"[save] {delta_probe_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
