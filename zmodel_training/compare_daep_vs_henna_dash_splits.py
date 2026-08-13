#!/usr/bin/env python3
"""
Print (and save) dataset / split / class-mix diagnostics comparing
the original DAEP-matched Dash ensemble data vs the Henna-matched retrain.

Usage:
  conda activate astrodash
  python zmodel_training/compare_daep_vs_henna_dash_splits.py
  python zmodel_training/compare_daep_vs_henna_dash_splits.py --out path/to/report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Stdlib-only: mirror constants.LABEL_MAP / CLASS_NAMES (no prod_backend import).
LABEL_MAP: dict[str, str] = {
    "SN Ia": "SN Ia",
    "SN Ia-CSM": "SN Ia",
    "SN Ia-91T-like": "SN Ia",
    "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia",
    "SN Ia-pec": "SN Ia",
    "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia",
    "Computed-Ia": "SN Ia",
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
    "SN II": "SN II",
    "SN IIP": "SN II",
    "SN IIL": "SN II",
    "SN II-pec": "SN II",
    "SN IIb": "SN II",
    "Computed-IIP": "SN II",
    "Computed-IIb": "SN II",
    "SN IIn": "SN IIn",
    "SN IIn-pec": "SN IIn",
    "SLSN-I": "SLSN-I",
    "SLSN-II": "SLSN-I",
    "SLSN-R": "SLSN-I",
}
ORDER = ["SN Ia", "SN Ib/c", "SN II", "SN IIn", "SLSN-I"]

DEFAULT_OUT = (
    PROJECT_ROOT
    / "data"
    / "pre_trained_models"
    / "daep_vs_henna_dash_split_comparison.txt"
)

SETS = [
    {
        "name": "daep",
        "split_json": PROJECT_ROOT / "data" / "wiserep" / "daep_matched_split_z.json",
        "meta_csv": PROJECT_ROOT
        / "WiserepData"
        / "Test"
        / "data_z"
        / "wiserep_metadata_processed.csv",
        "train_cfg": PROJECT_ROOT
        / "data"
        / "pre_trained_models"
        / "daep_comparison_z"
        / "iter_0"
        / "training_config.json",
    },
    {
        "name": "henna",
        "split_json": PROJECT_ROOT / "data" / "wiserep" / "henna_matched_split_z.json",
        "meta_csv": PROJECT_ROOT
        / "data"
        / "wiserep_henna"
        / "deredshifted"
        / "wiserep_metadata_processed.csv",
        "train_cfg": PROJECT_ROOT
        / "data"
        / "pre_trained_models"
        / "henna_matched_comparison_z"
        / "iter_0"
        / "training_config.json",
    },
]


def canon(raw: str) -> str | None:
    t = (raw or "").strip()
    if not t:
        return None
    return LABEL_MAP.get(t) or LABEL_MAP.get(t.replace(" ", ""))


def load_meta(csv_path: Path) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            fname = (row.get("Ascii file") or "").strip()
            if not fname or fname in info:
                continue
            info[fname] = {
                "type": (row.get("Obj. Type") or "").strip(),
                "z": (row.get("Redshift") or "").strip(),
                "iau": (row.get("IAU name") or row.get("sn_name_used") or "").strip(),
            }
    return info


def zstats(zs: list[float]) -> dict[str, float]:
    if not zs:
        return {}
    zs = sorted(zs)
    n = len(zs)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round(p / 100 * (n - 1)))))
        return zs[i]

    return {
        "n": float(n),
        "mean": sum(zs) / n,
        "median": pct(50),
        "p90": pct(90),
        "max": zs[-1],
    }


def ensemble_val_acc(out_root: Path) -> tuple[float, float, int] | None:
    accs: list[float] = []
    for p in sorted(out_root.glob("iter_*/model_performance.json")):
        perf = json.loads(p.read_text(encoding="utf-8"))
        accs.append(float(perf["cumulative"]["accuracy_pct"]))
    if not accs:
        return None
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    return mean, var**0.5, len(accs)


def summarize_split(
    lines: list[str],
    name: str,
    split_json: Path,
    meta_csv: Path,
    train_cfg: Path,
) -> None:
    d = json.loads(split_json.read_text(encoding="utf-8"))
    meta = load_meta(meta_csv)
    cfg = json.loads(train_cfg.read_text(encoding="utf-8")) if train_cfg.is_file() else {}
    weights = cfg.get("class_weights")

    lines.append(f"======== {name} ========")
    lines.append(f"split_json: {split_json}")
    lines.append(f"meta_csv:   {meta_csv}")
    lines.append(f"split_method: {d.get('split_method')}")
    lines.append(f"counts: {d.get('counts')}")
    lines.append(f"unique_iau: {d.get('unique_iau')}")
    if weights is not None:
        lines.append(
            f"class_weights (iter_0 train): {[round(float(x), 3) for x in weights]}"
        )
    lines.append("")

    for split in ("train", "val", "test"):
        files = list(d.get(split, []))
        c: Counter[str] = Counter()
        zs: list[float] = []
        iau_by = {lab: set() for lab in ORDER}
        raw_by_canon: dict[str, Counter[str]] = {lab: Counter() for lab in ORDER}
        missing = unmapped = 0
        for f in files:
            m = meta.get(f)
            if m is None:
                missing += 1
                continue
            lab = canon(m["type"])
            if lab is None:
                unmapped += 1
                continue
            c[lab] += 1
            raw_by_canon[lab][m["type"]] += 1
            if m["iau"]:
                iau_by[lab].add(m["iau"])
            try:
                zs.append(float(m["z"]))
            except ValueError:
                pass
        total = sum(c.values())
        lines.append(
            f"{split}: listed={len(files)} mapped={total} "
            f"missing_meta={missing} unmapped={unmapped}"
        )
        for i, lab in enumerate(ORDER):
            w = f"{float(weights[i]):.2f}" if weights is not None else "n/a"
            lines.append(
                f"  {lab:8s}: spectra={c[lab]:5d} ({100.0 * c[lab] / max(total, 1):5.1f}%)  "
                f"unique_iau={len(iau_by[lab]):4d}  w={w}"
            )
        st = zstats(zs)
        if st:
            lines.append(
                f"  redshift mean={st['mean']:.4f} med={st['median']:.4f} "
                f"p90={st['p90']:.4f} max={st['max']:.4f}"
            )
        if split == "test":
            lines.append("  raw Obj.Type mix (rare / Ib/c buckets):")
            for rare in ("SN IIn", "SLSN-I", "SN Ib/c"):
                lines.append(f"    {rare}: {dict(raw_by_canon[rare])}")
        lines.append("")


def overlap_report(lines: list[str]) -> None:
    daep = json.loads(SETS[0]["split_json"].read_text(encoding="utf-8"))
    henna = json.loads(SETS[1]["split_json"].read_text(encoding="utf-8"))

    lines.append("======== overlap (ascii filenames) ========")
    daep_test = set(daep["test"])
    henna_test = set(henna["test"])
    inter = daep_test & henna_test
    union = daep_test | henna_test
    lines.append(
        f"test: daep={len(daep_test)} henna={len(henna_test)} "
        f"intersect={len(inter)} jaccard={len(inter) / max(len(union), 1):.3f}"
    )

    daep_all: set[str] = set()
    henna_all: set[str] = set()
    for s in ("train", "val", "test"):
        daep_all.update(daep[s])
        henna_all.update(henna[s])
    inter_all = daep_all & henna_all
    union_all = daep_all | henna_all
    lines.append(
        f"corpus: daep={len(daep_all)} henna={len(henna_all)} "
        f"intersect={len(inter_all)} jaccard={len(inter_all) / max(len(union_all), 1):.3f} "
        f"daep_only={len(daep_all - henna_all)} henna_only={len(henna_all - daep_all)}"
    )
    lines.append(
        f"daep test in henna corpus: {len(daep_test & henna_all)}/{len(daep_test)}"
    )
    lines.append(
        f"henna test in daep corpus: {len(henna_test & daep_all)}/{len(henna_test)}"
    )
    lines.append("")


def val_acc_notes(lines: list[str]) -> None:
    lines.append("======== validation accuracy (iter_*/model_performance.json) ========")
    for spec in SETS:
        root = spec["train_cfg"].parent.parent
        stats = ensemble_val_acc(root)
        if stats is None:
            lines.append(f"  {spec['name']}: no model_performance.json under {root}")
        else:
            mean, std, n = stats
            lines.append(f"  {spec['name']}: val acc {mean:.2f} ± {std:.2f}%  (n={n} runs)")
    lines.append("")


def build_report() -> str:
    lines: list[str] = []
    lines.append("DAEP-matched vs Henna-matched Dash 1D CNN (+z) split comparison")
    lines.append(f"project: {PROJECT_ROOT}")
    lines.append("")
    for spec in SETS:
        if not spec["split_json"].is_file():
            lines.append(f"MISSING split: {spec['split_json']}")
            continue
        if not spec["meta_csv"].is_file():
            lines.append(f"MISSING meta: {spec['meta_csv']}")
            continue
        summarize_split(
            lines,
            spec["name"],
            spec["split_json"],
            spec["meta_csv"],
            spec["train_cfg"],
        )
    overlap_report(lines)
    val_acc_notes(lines)
    lines.append("Same model / hypers; differences are data-driven:")
    lines.append(
        "  - sample list / dedup (WiserepData/Test/data_z vs wiserep_henna/deredshifted)"
    )
    lines.append(
        "  - IAU 80/10/10 split membership (same recipe+seed, different rows)"
    )
    lines.append("  - class_weights recomputed from train")
    lines.append("  - test class priors, rare counts, subtype mix, redshift mix")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DAEP-matched vs Henna-matched Dash split statistics."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Write report here (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print only; do not write a .txt file.",
    )
    args = parser.parse_args()

    report = build_report()
    print(report, end="")

    if not args.no_write:
        out = args.out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
