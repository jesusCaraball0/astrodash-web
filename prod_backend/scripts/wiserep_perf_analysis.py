"""
Run performance metrics and confusion matrix for WISeREP results (PyTorch or TF).
Standalone: no imports from app or wiserep_dash_classify.

Usage (from project root):
    python prod_backend/scripts/wiserep_perf_analysis.py
    python prod_backend/scripts/wiserep_perf_analysis.py --results_csv data/wiserep/wiserep_tf_results.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 6-class canonical set: Ia, Ib, Ic, IIb, IIn, II (+ Other for unmapped)
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

# Catalog (Obj. Type) -> canonical; SLSN-I mapped to Ic
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
    # DASH prediction: use explicit DASH mapping first, then catalog map, then base
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
    """Load results CSV + optional metadata, normalize types, compute metrics and confusion matrix."""
    results_csv_path = Path(results_csv_path)
    if not results_csv_path.exists():
        return {"error": f"Results CSV not found: {results_csv_path}"}
    rows: List[Dict[str, str]] = []
    with open(results_csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
        catalog_type = (r.get("catalog_type") or "").strip() if has_catalog_col else filename_to_catalog.get((r.get("filename") or "").strip(), "")
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
        return {"error": "No rows with both catalog type and prediction (add catalog_type or pass metadata_csv_path)", "n_samples": len(rows), "n_skipped": skipped}
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
    correct = sum(1 for t, p in pairs if t == p)
    accuracy = correct / len(pairs)
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
    print("(Diagonal = recall R for each class; P = precision is column-based, not shown in matrix.)")
    print(f"Labels: {labels}")
    col_w = 6
    hdr = "".join(f"{p:>{col_w}}" for p in labels)
    print(f"{'':>{col_w}}" + hdr)
    for i, lbl in enumerate(labels):
        row_str = "".join(f"{cm_pct[i][j]:>{col_w}.1f}" for j in range(n))
        print(f"{lbl:>{col_w}}" + row_str)
    print("\nPer-class metrics (P=precision, R=recall; R = diagonal in matrix above):")
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
    return {"confusion_matrix": cm, "labels": labels, "label_to_idx": label_to_idx, "accuracy": accuracy, "macro_f1": macro_f1, "per_class_metrics": per_class, "n_samples": len(pairs), "n_skipped": skipped, "n_other": n_other, "other_catalog_raw": other_catalog_raw, "other_pred_raw": other_pred_raw}


def plot_confusion_matrix(cm: list, labels: list, title: str = "Confusion matrix (True vs Predicted)") -> None:
    """Plot confusion matrix as a heatmap with row-normalized percentages (% of true class)."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.6)))
    im = ax.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_pct[i, j]
            text_str = f"{val:.1f}" if row_sums[i, 0] > 0 else "—"
            text = ax.text(j, i, text_str, ha="center", va="center", color="black" if val < 50 else "white", fontsize=9)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (catalog)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WISeREP classification metrics and confusion matrix")
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Path to results CSV (default: data/wiserep/wiserep_dash_results.csv)",
    )
    args = parser.parse_args()
    results_path = Path(args.results_csv) if args.results_csv else (PROJECT_ROOT / "data" / "wiserep" / "wiserep_dash_results.csv")
    results_path = results_path.resolve()
    metadata_path = PROJECT_ROOT / "data" / "wiserep" / "wiserep_metadata.csv"

    out = compute_metrics_and_confusion_matrix(
        str(results_path),
        metadata_csv_path=str(metadata_path),
    )

    if "error" in out:
        print("Result:", out)
    else:
        print("\nReturned dict keys:", list(out.keys()))
        print("Accuracy:", out.get("accuracy"))
        print("Macro F1:", out.get("macro_f1"))
        print("Labels:", out.get("labels"))
        print("n_samples:", out.get("n_samples"), "| n_skipped:", out.get("n_skipped"))
        print("\nPer-class metrics:", out.get("per_class_metrics"))

        # Plot confusion matrix
        plot_confusion_matrix(
            out["confusion_matrix"],
            out["labels"],
            title="WISeREP DASH classification (True catalog vs Predicted)",
        )
