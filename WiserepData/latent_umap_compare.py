"""
2D UMAP of frozen latent spaces: Twinsanity original vs Henna try embeddings.

Each panel is an independent UMAP (spaces are not comparable in shared coords).
Points are colored by the 5-class taxonomy used by the latent MLP.

Usage:
  PYTHONPATH=WiserepData NUMBA_CACHE_DIR=/tmp/numba_cache \\
    python WiserepData/latent_umap_compare.py

  PYTHONPATH=WiserepData NUMBA_CACHE_DIR=/tmp/numba_cache \\
    python WiserepData/latent_umap_compare.py \\
      --sources original,five_try,six_try \\
      --max-points 10000 \\
      --out-dir WiserepData/Test/plots/latent_umap

  # Point at Twinsanity files explicitly if not found via defaults:
  python WiserepData/latent_umap_compare.py \\
    --original-npz /path/to/latent_raw_z.npz \\
    --original-meta /path/to/wiserep_metadata_processed.csv
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from dataclasses import dataclass

# Avoid numba/umap cache failures in sandboxed / odd cwd environments.
os.environ.setdefault("NUMBA_CACHE_DIR", str(pathlib.Path("/tmp") / "numba_cache_umap"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = WISEREP_DIR.parent
if str(WISEREP_DIR) not in sys.path:
    sys.path.insert(0, str(WISEREP_DIR))

from train_latent import (  # noqa: E402
    HENNA_ROOT,
    LABEL_COLUMN,
    load_latent_and_meta,
    normalize_latent_meta,
    resolve_latent_npz,
)
from TwinsClassifier_Wiserep import CLASS_NAMES, row_class_idx  # noqa: E402

TRY_NAMES = ("second_try", "third_try", "four_try", "five_try", "six_try")
TRY_TITLE = {
    "second_try": "_2",
    "third_try": "_3",
    "four_try": "_4",
    "five_try": "_5",
    "six_try": "_6",
}
PREFERRED_SPLIT_SEEDS = (36, 73, 149, 257)

# Colorblind-friendly categorical palette (shared across panels).
CLASS_COLORS = {
    "SN Ia": "#0072B2",
    "SN Ib/c": "#E69F00",
    "SN II": "#009E73",
    "SN IIn": "#D55E00",
    "SLSN-I": "#CC79A7",
}

ORIGINAL_NPZ_CANDIDATES = (
    pathlib.Path("/Users/jesuscaraball0/code/personal_code/Twinsanity/WiserepData/Test/latent_raw_z.npz"),
    pathlib.Path.home() / "Downloads" / "latent_raw_z.npz",
    _PROJECT_ROOT / "data" / "twinsanity" / "latent_raw_z.npz",
)
ORIGINAL_META_CANDIDATES = (
    pathlib.Path(
        "/Users/jesuscaraball0/code/personal_code/Twinsanity/WiserepData/Test/"
        "wiserep_metadata_processed.csv"
    ),
    pathlib.Path.home() / "Downloads" / "wiserep_metadata_processed.csv",
    _PROJECT_ROOT / "data" / "twinsanity" / "wiserep_metadata_processed.csv",
)


@dataclass(frozen=True)
class LatentSource:
    name: str
    title: str
    z: np.ndarray
    meta: pd.DataFrame
    latent_path: pathlib.Path
    meta_path: pathlib.Path


def _first_existing(paths: tuple[pathlib.Path, ...]) -> pathlib.Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def resolve_original_paths(
    npz: pathlib.Path | None,
    meta: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    npz_path = pathlib.Path(npz).expanduser() if npz else _first_existing(ORIGINAL_NPZ_CANDIDATES)
    meta_path = pathlib.Path(meta).expanduser() if meta else _first_existing(ORIGINAL_META_CANDIDATES)
    if npz_path is None or not npz_path.is_file():
        tried = "\n  ".join(str(p) for p in ORIGINAL_NPZ_CANDIDATES)
        raise FileNotFoundError(
            "Could not find Twinsanity latent_raw_z.npz. Pass --original-npz.\n"
            f"Tried:\n  {tried}"
        )
    if meta_path is None or not meta_path.is_file():
        tried = "\n  ".join(str(p) for p in ORIGINAL_META_CANDIDATES)
        raise FileNotFoundError(
            "Could not find Twinsanity wiserep_metadata_processed.csv. Pass --original-meta.\n"
            f"Tried:\n  {tried}"
        )
    return npz_path.resolve(), meta_path.resolve()


def _try_panel_title(try_key: str) -> str:
    if try_key in TRY_TITLE:
        return TRY_TITLE[try_key]
    m = re.fullmatch(r"(?:.*_)?(\d+)_try", try_key)
    if m:
        return f"_{m.group(1)}"
    m = re.search(r"(\d+)$", try_key)
    if m:
        return f"_{m.group(1)}"
    return try_key


def _pick_try_subdir(try_root: pathlib.Path, prefer_seed: int) -> pathlib.Path:
    if not try_root.is_dir():
        raise FileNotFoundError(f"Missing try root: {try_root}")

    subdirs = [p for p in sorted(try_root.iterdir()) if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirs under {try_root}")

    # Prefer Dered{seed}_* then any Dered*_*, else first child with latents.
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


def load_from_npz_meta(
    name: str,
    title: str,
    latent_npz: pathlib.Path,
    meta_csv: pathlib.Path,
) -> LatentSource:
    z = np.load(latent_npz)["z"].astype(np.float32, copy=False)
    meta = normalize_latent_meta(pd.read_csv(meta_csv, low_memory=False))
    if z.ndim != 3:
        raise ValueError(f"{latent_npz}: expected (N, L, D), got {z.shape}")
    if len(meta) != z.shape[0]:
        raise ValueError(
            f"meta rows ({len(meta)}) != latent rows ({z.shape[0]}) "
            f"for {latent_npz.name} / {meta_csv.name}"
        )
    if LABEL_COLUMN not in meta.columns:
        raise KeyError(f"{meta_csv} missing {LABEL_COLUMN!r} after normalize")
    return LatentSource(
        name=name,
        title=title,
        z=z,
        meta=meta,
        latent_path=latent_npz,
        meta_path=meta_csv,
    )


def load_from_latent_dir(name: str, title: str, latent_dir: pathlib.Path) -> LatentSource:
    meta, z, latent_npz, meta_csv = load_latent_and_meta(latent_dir)
    return LatentSource(
        name=name,
        title=title,
        z=z,
        meta=meta,
        latent_path=latent_npz,
        meta_path=meta_csv,
    )


def resolve_sources(
    source_names: list[str],
    *,
    original_npz: pathlib.Path | None,
    original_meta: pathlib.Path | None,
    prefer_seed: int,
    henna_root: pathlib.Path,
) -> list[LatentSource]:
    out: list[LatentSource] = []
    for raw in source_names:
        name = raw.strip()
        key = name.lower().replace("-", "_")
        if key in ("original", "twinsanity", "legacy"):
            npz, meta = resolve_original_paths(original_npz, original_meta)
            src = load_from_npz_meta(
                "original",
                "Original",
                npz,
                meta,
            )
        elif key in ("1024d2", "henna_matched", "henna1024d2"):
            src = load_from_latent_dir(
                "1024d2",
                "1024d2",
                (henna_root / "1024d2").resolve(),
            )
        elif key in TRY_NAMES or key.endswith("_try"):
            try_root = henna_root / key
            sub = _pick_try_subdir(try_root, prefer_seed)
            src = load_from_latent_dir(key, _try_panel_title(key), sub.resolve())
        else:
            # Treat as a path to a latent directory.
            latent_dir = pathlib.Path(name).expanduser().resolve()
            if not latent_dir.is_dir():
                raise FileNotFoundError(f"Unknown source {name!r} (not a try name or directory)")
            src = load_from_latent_dir(latent_dir.name, latent_dir.name, latent_dir)
        out.append(src)
        print(
            f"[load] {src.name}: z={tuple(src.z.shape)} flat={int(np.prod(src.z.shape[1:]))} "
            f"npz={src.latent_path.name}",
            flush=True,
        )
    return out


def mapped_class_labels(meta: pd.DataFrame) -> np.ndarray:
    """Return class index per row; -1 if unmapped."""
    n = len(meta)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        y[i] = row_class_idx(meta, i, LABEL_COLUMN)
    return y


def flatten_latents(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float32).reshape(z.shape[0], -1)


def subsample_indices(
    y: np.ndarray,
    *,
    max_points: int | None,
    seed: int,
    balanced: bool,
) -> np.ndarray:
    mapped = np.flatnonzero(y >= 0)
    if max_points is None or mapped.size <= max_points:
        return mapped

    rng = np.random.default_rng(seed)
    if not balanced:
        return rng.choice(mapped, size=max_points, replace=False)

    # Roughly equal per class, leftover to largest remaining pools.
    classes = [c for c in range(len(CLASS_NAMES)) if np.any(y[mapped] == c)]
    if not classes:
        return mapped[:0]
    per = max(1, max_points // len(classes))
    chosen: list[np.ndarray] = []
    for c in classes:
        idx_c = mapped[y[mapped] == c]
        take = min(per, idx_c.size)
        chosen.append(rng.choice(idx_c, size=take, replace=False))
    sel = np.concatenate(chosen)
    if sel.size < max_points:
        remaining = np.setdiff1d(mapped, sel, assume_unique=False)
        extra = min(max_points - sel.size, remaining.size)
        if extra:
            sel = np.concatenate([sel, rng.choice(remaining, size=extra, replace=False)])
    if sel.size > max_points:
        sel = rng.choice(sel, size=max_points, replace=False)
    return np.sort(sel)


def embed_umap(
    X: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    pca_dims: int | None,
    seed: int,
) -> np.ndarray:
    import umap

    X = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if pca_dims is not None and pca_dims > 0 and Xs.shape[1] > pca_dims:
        n_comp = min(pca_dims, Xs.shape[0] - 1, Xs.shape[1])
        Xs = PCA(n_components=n_comp, random_state=seed).fit_transform(Xs)
        print(f"  PCA -> {Xs.shape[1]}-d before UMAP", flush=True)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(Xs)


def plot_umap_panel(
    ax,
    xy: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    subtitle: str,
    point_size: float,
    alpha: float,
) -> None:
    order = list(range(len(CLASS_NAMES)))  # draw rarer classes later? keep fixed order
    # Draw abundant classes first so rare ones sit on top.
    counts = {c: int(np.sum(y == c)) for c in order}
    draw_order = sorted(order, key=lambda c: counts[c], reverse=True)
    for c in draw_order:
        mask = y == c
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            c=CLASS_COLORS[CLASS_NAMES[c]],
            alpha=alpha,
            linewidths=0,
            label=f"{CLASS_NAMES[c]} (n={counts[c]})",
            rasterized=True,
        )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=2.0, framealpha=0.9)


def default_source_names(*, all_tries: bool, include_1024d2: bool) -> list[str]:
    names = ["original"]
    if include_1024d2:
        names.append("1024d2")
    if all_tries:
        names.extend([t for t in TRY_NAMES if (HENNA_ROOT / t).is_dir()])
    else:
        for t in ("five_try", "six_try"):
            if (HENNA_ROOT / t).is_dir():
                names.append(t)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D UMAP comparison of Twinsanity original vs Henna try latent spaces."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated sources: original,1024d2,second_try,...,six_try "
            "and/or paths to latent dirs. Default: original + five_try + six_try."
        ),
    )
    parser.add_argument(
        "--all-tries",
        action="store_true",
        help="Include every existing *_try under data/wiserep_henna (with original).",
    )
    parser.add_argument(
        "--include-1024d2",
        action="store_true",
        help="Also include Henna 1024d2 matched baseline.",
    )
    parser.add_argument("--original-npz", type=pathlib.Path, default=None)
    parser.add_argument("--original-meta", type=pathlib.Path, default=None)
    parser.add_argument(
        "--prefer-seed",
        type=int,
        default=36,
        help="Preferred Dered{seed}_* folder inside each try (default: 36).",
    )
    parser.add_argument(
        "--henna-root",
        type=pathlib.Path,
        default=HENNA_ROOT,
        help="Root containing try folders / 1024d2.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=WISEREP_DIR / "Test" / "plots" / "latent_umap",
        help="Output directory for PNGs.",
    )
    parser.add_argument("--max-points", type=int, default=12000, help="Subsample cap (None=all).")
    parser.add_argument(
        "--no-subsample",
        action="store_true",
        help="Use all mapped-class points (slow for ~40k × 8192-d).",
    )
    parser.add_argument(
        "--balanced-subsample",
        action="store_true",
        help="Subsample roughly equally across classes (helps rare SLSN-I / IIn).",
    )
    parser.add_argument("--pca-dims", type=int, default=50, help="PCA dims before UMAP (0=skip).")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip multi-panel grid; only write per-source PNGs.",
    )
    args = parser.parse_args()

    if args.sources:
        source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        source_names = default_source_names(
            all_tries=args.all_tries,
            include_1024d2=args.include_1024d2,
        )

    max_points = None if args.no_subsample else args.max_points
    pca_dims = None if args.pca_dims <= 0 else args.pca_dims
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_sources(
        source_names,
        original_npz=args.original_npz,
        original_meta=args.original_meta,
        prefer_seed=args.prefer_seed,
        henna_root=args.henna_root.expanduser().resolve(),
    )

    embeddings: list[tuple[LatentSource, np.ndarray, np.ndarray]] = []
    for src in sources:
        y_all = mapped_class_labels(src.meta)
        idx = subsample_indices(
            y_all,
            max_points=max_points,
            seed=args.seed,
            balanced=args.balanced_subsample,
        )
        if idx.size == 0:
            raise RuntimeError(f"No mapped-class rows for {src.name}")
        X = flatten_latents(src.z[idx])
        y = y_all[idx]
        print(
            f"[umap] {src.name}: points={idx.size}/{len(y_all)} mapped, "
            f"feat={X.shape[1]}, classes={ {CLASS_NAMES[c]: int(np.sum(y == c)) for c in range(len(CLASS_NAMES))} }",
            flush=True,
        )
        xy = embed_umap(
            X,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            pca_dims=pca_dims,
            seed=args.seed,
        )
        embeddings.append((src, xy, y))

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        n, L, D = src.z.shape
        subtitle = f"z=[{n}, {L}, {D}]"
        plot_umap_panel(
            ax,
            xy,
            y,
            title=src.title,
            subtitle=subtitle,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        fig.tight_layout()
        single_path = out_dir / f"umap_{src.name}.png"
        fig.savefig(single_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {single_path}", flush=True)

    if not args.no_combined and embeddings:
        n_panels = len(embeddings)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.8 * ncols, 5.2 * nrows),
            squeeze=False,
        )
        for i, (src, xy, y) in enumerate(embeddings):
            ax = axes[i // ncols][i % ncols]
            n, L, D = src.z.shape
            subtitle = f"z=[{n}, {L}, {D}]"
            plot_umap_panel(
                ax,
                xy,
                y,
                title=src.title,
                subtitle=subtitle,
                point_size=args.point_size,
                alpha=args.alpha,
            )
        for j in range(n_panels, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(
            "Frozen latent UMAP by class (independent fit per backbone)",
            fontsize=13,
            y=1.01,
        )
        fig.tight_layout()
        combined = out_dir / "umap_compare.png"
        fig.savefig(combined, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {combined}", flush=True)

    # Small legend-only reference (useful for slides).
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CLASS_COLORS[name],
            markersize=10,
            label=name,
        )
        for name in CLASS_NAMES
    ]
    ax.legend(handles=handles, loc="center", title="Class", frameon=True)
    legend_path = out_dir / "umap_class_legend.png"
    fig.savefig(legend_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {legend_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
