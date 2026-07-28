#BestSNTwinsPlot

import argparse
import pathlib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Default --data_dir / --run_dir (twin plots; edit independently of train scripts)
# -----------------------------------------------------------------------------
_WISEREP_DIR = pathlib.Path(__file__).resolve().parent
TEST_ROOT = _WISEREP_DIR / "Test"
USE_REDSHIFT_CORRECTED_DATA = True
_DEFAULT_WISEREP_DATA_DIR = TEST_ROOT / ("data_z" if USE_REDSHIFT_CORRECTED_DATA else "data_no_z")

PINK_1 = "#ff4fa3"
PINK_2 = "#ff9ecf"


def clean_text_value(v):
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "<na>", "unknown", "?"}:
        return ""
    return s


def canonicalize_name(x):
    s = clean_text_value(x).lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "").replace("-", "")
    return s


def infer_name_column(meta):
    candidates = [
        "sn_name_used",
        "IAU name",
        "sn_name",
        "SN Name",
        "Name",
        "Obj. Name",
        "Object Name",
        "Target Name",
        "Target",
        "Object",
        "objname",
        "name",
        "sn",
        "Internal name/s",
    ]
    lower_map = {c.lower(): c for c in meta.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in meta.columns:
        cl = c.lower()
        if "name" in cl or "object" in cl or "target" in cl or cl == "sn":
            return c
    return None


def infer_type_column(meta):
    candidates = [
        "Obj. Type",
        "Type",
        "type",
        "SN Type",
        "Classification",
        "class",
        "label",
        "canonical_type",
    ]
    lower_map = {c.lower(): c for c in meta.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in meta.columns:
        cl = c.lower()
        if "type" in cl or "class" in cl:
            return c
    return None


def infer_obsdate_column(meta):
    preferred = [
        "Obs-date",
        "Obs Date",
        "Observation Date",
        "obs_date",
        "obsdate",
        "Date Obs",
        "date_obs",
        "DATE-OBS",
        "date-obs",
        "UT Date",
        "ut_date",
    ]
    lower_map = {c.lower(): c for c in meta.columns}
    for c in preferred:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in meta.columns:
        cl = c.lower().strip()
        if ("obs" in cl and "date" in cl) or cl in {"date-obs", "date_obs"}:
            return c
    return None


def normalize_type_string(x):
    s = clean_text_value(x)
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return s.lower()


def matches_type(x, wanted):
    sx = normalize_type_string(x)
    sw = normalize_type_string(wanted)
    return sx == sw or sw in sx


def looks_like_iau_sn_name(s):
    s = s.strip()
    return bool(re.match(r"^(SN|AT)\s+[A-Za-z0-9][A-Za-z0-9\-]*$", s))


def get_display_names(meta):
    primary_cols = ["sn_name_used", "IAU name"]
    fallback_cols = [
        "sn_name",
        "Name",
        "Obj. Name",
        "Object",
        "SN Name",
        "Internal name/s",
    ]

    names = []
    for i, row in meta.iterrows():
        chosen = ""
        for c in primary_cols:
            if c in meta.columns:
                val = clean_text_value(row.get(c))
                if val and looks_like_iau_sn_name(val):
                    chosen = val
                    break
        if not chosen:
            for c in primary_cols + fallback_cols:
                if c in meta.columns:
                    val = clean_text_value(row.get(c))
                    if val:
                        chosen = val
                        break
        if not chosen and "Spec. ID" in meta.columns:
            sid = clean_text_value(row.get("Spec. ID"))
            if sid:
                chosen = f"Spec. ID {sid}"
        if not chosen:
            chosen = f"obs_{i}"
        names.append(chosen)
    return names


def format_obsdate_value(v):
    if pd.isna(v):
        return "obs=NA"
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.notna(ts):
            return f"obs={ts.strftime('%Y-%m-%d')}"
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return "obs=NA"
    return f"obs={s}"


def get_obsdate_text(row, obsdate_col):
    if obsdate_col is None or obsdate_col not in row.index:
        return "obs=NA"
    return format_obsdate_value(row[obsdate_col])


def load_bundle(data_dir, run_dir):
    data_dir = pathlib.Path(data_dir).resolve()
    run_dir = pathlib.Path(run_dir).resolve()

    meta = pd.read_csv(data_dir / "wiserep_metadata_processed.csv")
    flux = np.load(data_dir / "wiserep_flux.npy").astype(np.float32)
    mask = np.load(data_dir / "wiserep_mask.npy").astype(bool)
    wavelength = np.load(data_dir / "wiserep_wavelength.npy").astype(np.float32)

    z = np.load(run_dir / "latent_raw_z.npz")["z"].astype(np.float32)

    if z.ndim != 3:
        raise ValueError(f"Expected latent_raw_z.npz['z'] to have shape (N, L, D), got {z.shape}")

    assert len(meta) == z.shape[0], f"meta rows ({len(meta)}) != latent rows ({z.shape[0]})"
    assert flux.shape[0] == z.shape[0], f"flux rows ({flux.shape[0]}) != latent rows ({z.shape[0]})"

    return meta.reset_index(drop=True), wavelength, flux, mask, z


def compute_full_latent_distance(z):
    z = np.asarray(z, dtype=np.float32)
    n, L, D = z.shape
    norms = np.maximum(np.linalg.norm(z, axis=2, keepdims=True), 1e-12)
    z_unit = z / norms
    X = z_unit.reshape(n, L * D)
    sim = np.clip((X @ X.T) / float(L), -1.0, 1.0)
    dist = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_neighbors_from_distance(D, k=50):
    n = D.shape[0]
    k = max(2, min(int(k), n))
    order = np.argsort(D, axis=1)[:, :k]
    dists = np.take_along_axis(D, order, axis=1)
    return dists.astype(np.float32), order.astype(np.int32)


def find_indices_for_type(meta, type_col, wanted_type):
    if type_col is None or type_col not in meta.columns:
        return np.array([], dtype=np.int64)
    keep = meta[type_col].map(lambda x: matches_type(x, wanted_type)).to_numpy()
    return np.where(keep)[0].astype(np.int64)


def find_indices_for_target_names(meta, display_names, targets):
    wanted = {canonicalize_name(x) for x in targets}
    keep = np.array([canonicalize_name(x) in wanted for x in display_names], dtype=bool)
    return np.where(keep)[0].astype(np.int64)


def find_indices_for_name_and_obsdate(meta, display_names, obsdate_col, target_name, target_obsdate):
    if obsdate_col is None or obsdate_col not in meta.columns:
        return np.array([], dtype=np.int64)

    target_name_c = canonicalize_name(target_name)
    obs_series = pd.to_datetime(meta[obsdate_col], errors="coerce")
    target_obs = pd.to_datetime(target_obsdate, errors="coerce")
    if pd.isna(target_obs):
        return np.array([], dtype=np.int64)

    keep_name = np.array([canonicalize_name(x) == target_name_c for x in display_names], dtype=bool)
    keep_date = obs_series.dt.strftime("%Y-%m-%d") == target_obs.strftime("%Y-%m-%d")
    keep = keep_name & keep_date.to_numpy()
    return np.where(keep)[0].astype(np.int64)


def get_neighbor_list(i, nbrs, dists, display_names, exclude_same_sn=True, unique_twin_sn=False):
    key_i = canonicalize_name(display_names[i])
    out = []
    seen = set()

    for t in range(nbrs.shape[1]):
        j = int(nbrs[i, t])
        if j == i:
            continue
        key_j = canonicalize_name(display_names[j])
        if exclude_same_sn and key_j == key_i:
            continue
        if unique_twin_sn:
            if key_j in seen:
                continue
            seen.add(key_j)
        out.append((j, float(dists[i, t])))
    return out


def choose_best_pair_from_neighbors(indices, nbrs, dists, display_names, exclude_same_sn=True, unique_twin_sn=False):
    idx_set = set(int(x) for x in indices.tolist())
    best = None
    best_d = np.inf

    for i in indices:
        neighbors = get_neighbor_list(
            int(i),
            nbrs=nbrs,
            dists=dists,
            display_names=display_names,
            exclude_same_sn=exclude_same_sn,
            unique_twin_sn=unique_twin_sn,
        )
        for j, d in neighbors:
            if j not in idx_set:
                continue
            if d < best_d:
                best_d = d
                best = {"idx1": int(i), "idx2": int(j), "dist": float(d)}
            break

    return best


def choose_best_twin_for_target_from_neighbors(target_idx, candidate_indices, nbrs, dists, display_names, exclude_same_sn=True, unique_twin_sn=False):
    idx_set = set(int(x) for x in candidate_indices.tolist())
    neighbors = get_neighbor_list(
        int(target_idx),
        nbrs=nbrs,
        dists=dists,
        display_names=display_names,
        exclude_same_sn=exclude_same_sn,
        unique_twin_sn=unique_twin_sn,
    )
    for j, d in neighbors:
        if j in idx_set:
            return {"idx1": int(target_idx), "idx2": int(j), "dist": float(d)}
    return None


def plot_pair(pair, wave, flux, mask, meta, display_names, type_col, obsdate_col, outpath, title_prefix):
    i = pair["idx1"]
    j = pair["idx2"]

    y1 = np.array(flux[i], dtype=float)
    y2 = np.array(flux[j], dtype=float)

    m1 = np.array(mask[i], dtype=bool)
    m2 = np.array(mask[j], dtype=bool)

    y1[~m1] = np.nan
    y2[~m2] = np.nan

    row1 = meta.loc[i]
    row2 = meta.loc[j]

    t1 = str(row1[type_col]) if type_col is not None and type_col in meta.columns else "NA"
    t2 = str(row2[type_col]) if type_col is not None and type_col in meta.columns else "NA"

    n1 = display_names[i]
    n2 = display_names[j]

    d1 = get_obsdate_text(row1, obsdate_col)
    d2 = get_obsdate_text(row2, obsdate_col)

    z1 = clean_text_value(row1["Redshift"]) if "Redshift" in meta.columns else ""
    z2 = clean_text_value(row2["Redshift"]) if "Redshift" in meta.columns else ""

    if z1:
        try:
            z1 = f"z={float(z1):.6f}"
        except Exception:
            z1 = f"z={z1}"
    else:
        z1 = "z=NA"

    if z2:
        try:
            z2 = f"z={float(z2):.6f}"
        except Exception:
            z2 = f"z={z2}"
    else:
        z2 = "z=NA"

    plt.rc('font', family='serif')
    plt.figure(figsize=(11, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    plt.minorticks_on()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tick_params(
        which='major',
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction='in',
        length=10
    )
    plt.tick_params(
        which='minor',
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction='in',
        length=5
    )

    plt.plot(wave, y1, lw=2.4, color=PINK_1, label=f"{n1} | {t1} | {z1} | {d1}")
    plt.plot(wave, y2, lw=2.4, color=PINK_2, label=f"{n2} | {t2} | {z2} | {d2}")

    plt.xlabel("Wavelength (Å)", fontsize=20)
    plt.ylabel("Relative Flux", fontsize=20)
    plt.title(f"{title_prefix}", fontsize=20)
    plt.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, facecolor="white")
    plt.close()


def parse_list_arg(s):
    if s is None:
        return []
    parts = [x.strip() for x in str(s).split(",")]
    return [x for x in parts if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(_DEFAULT_WISEREP_DATA_DIR))
    ap.add_argument("--run_dir", type=str, default=str(_DEFAULT_WISEREP_DATA_DIR / "Output"))
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--mode", type=str, choices=["types", "targets", "obsdate", "row"], default="types")
    ap.add_argument("--types", type=str, default="Ia,IIn")
    ap.add_argument("--targets", type=str, default=None)
    ap.add_argument("--target_name", type=str, default=None)
    ap.add_argument("--target_obsdate", type=str, default=None)
    ap.add_argument("--target_row", type=int, default=None)
    ap.add_argument("--n_plots", type=int, default=None)

    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--exclude_same_sn", action="store_true")
    ap.add_argument("--unique_twin_sn", action="store_true")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    meta, wave, flux, mask, z = load_bundle(args.data_dir, args.run_dir)
    D = compute_full_latent_distance(z)
    dists, nbrs = compute_neighbors_from_distance(D, k=args.k)

    display_names = get_display_names(meta)
    type_col = infer_type_column(meta)
    obsdate_col = infer_obsdate_column(meta)

    print("type_col   =", type_col)
    print("obsdate_col=", obsdate_col)
    print("latent shape =", z.shape)
    print("distance matrix shape =", D.shape)

    made = 0

    if args.mode == "types":
        types = parse_list_arg(args.types)
        if args.n_plots is not None:
            types = types[:args.n_plots]

        for wanted_type in types:
            idx = find_indices_for_type(meta, type_col, wanted_type)
            pair = choose_best_pair_from_neighbors(
                indices=idx,
                nbrs=nbrs,
                dists=dists,
                display_names=display_names,
                exclude_same_sn=args.exclude_same_sn,
                unique_twin_sn=args.unique_twin_sn,
            )

            if pair is None:
                print(f"Could not find a valid twin pair for type {wanted_type}")
                continue

            safe_type = re.sub(r"[^A-Za-z0-9]+", "_", wanted_type)
            outpath = outdir / f"best_twin_pair_{safe_type}.png"

            plot_pair(
                pair=pair,
                wave=wave,
                flux=flux,
                mask=mask,
                meta=meta,
                display_names=display_names,
                type_col=type_col,
                obsdate_col=obsdate_col,
                outpath=outpath,
                title_prefix=f"Best twin pair within type {wanted_type}",
            )

            print(f"saved {outpath}")
            print(pair)
            made += 1

    elif args.mode == "targets":
        targets = parse_list_arg(args.targets)
        if len(targets) == 0:
            raise ValueError("In targets mode, pass --targets 'SNname1,SNname2,...'")
        if args.n_plots is not None:
            targets = targets[:args.n_plots]

        for target in targets:
            target_idx_all = find_indices_for_target_names(meta, display_names, [target])
            if len(target_idx_all) == 0:
                print(f"Could not find target {target}")
                continue

            target_idx = int(target_idx_all[0])

            if type_col is not None and type_col in meta.columns:
                target_type = meta.loc[target_idx, type_col]
                candidate_indices = find_indices_for_type(meta, type_col, str(target_type))
                title_prefix = f"Best twin for target {display_names[target_idx]} within type {target_type}"
            else:
                candidate_indices = np.arange(len(meta), dtype=np.int64)
                title_prefix = f"Best twin for target {display_names[target_idx]}"

            pair = choose_best_twin_for_target_from_neighbors(
                target_idx=target_idx,
                candidate_indices=candidate_indices,
                nbrs=nbrs,
                dists=dists,
                display_names=display_names,
                exclude_same_sn=args.exclude_same_sn,
                unique_twin_sn=args.unique_twin_sn,
            )

            if pair is None:
                print(f"Could not find a valid twin for target {target}")
                continue

            safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(target))
            outpath = outdir / f"target_twin_{safe_name}.png"

            plot_pair(
                pair=pair,
                wave=wave,
                flux=flux,
                mask=mask,
                meta=meta,
                display_names=display_names,
                type_col=type_col,
                obsdate_col=obsdate_col,
                outpath=outpath,
                title_prefix=title_prefix,
            )

            print(f"saved {outpath}")
            print(pair)
            made += 1

    elif args.mode == "obsdate":
        if args.target_name is None or args.target_obsdate is None:
            raise ValueError("In obsdate mode, pass both --target_name and --target_obsdate")

        idxs = find_indices_for_name_and_obsdate(
            meta=meta,
            display_names=display_names,
            obsdate_col=obsdate_col,
            target_name=args.target_name,
            target_obsdate=args.target_obsdate,
        )
        if len(idxs) == 0:
            raise ValueError(f"No observation found for {args.target_name} on {args.target_obsdate}")

        target_idx = int(idxs[0])

        if type_col is not None and type_col in meta.columns:
            target_type = meta.loc[target_idx, type_col]
            candidate_indices = find_indices_for_type(meta, type_col, str(target_type))
            title_prefix = f"Best twin for {display_names[target_idx]} on {args.target_obsdate} within type {target_type}"
        else:
            candidate_indices = np.arange(len(meta), dtype=np.int64)
            title_prefix = f"Best twin for {display_names[target_idx]} on {args.target_obsdate}"

        pair = choose_best_twin_for_target_from_neighbors(
            target_idx=target_idx,
            candidate_indices=candidate_indices,
            nbrs=nbrs,
            dists=dists,
            display_names=display_names,
            exclude_same_sn=args.exclude_same_sn,
            unique_twin_sn=args.unique_twin_sn,
        )

        if pair is None:
            raise RuntimeError("Could not find a valid twin for the requested observation")

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(args.target_name))
        safe_date = re.sub(r"[^A-Za-z0-9]+", "_", str(args.target_obsdate))
        outpath = outdir / f"target_twin_{safe_name}_{safe_date}.png"

        plot_pair(
            pair=pair,
            wave=wave,
            flux=flux,
            mask=mask,
            meta=meta,
            display_names=display_names,
            type_col=type_col,
            obsdate_col=obsdate_col,
            outpath=outpath,
            title_prefix=title_prefix,
        )

        print(f"saved {outpath}")
        print(pair)
        made += 1

    elif args.mode == "row":
        if args.target_row is None:
            raise ValueError("In row mode, pass --target_row")
        if args.target_row < 0 or args.target_row >= len(meta):
            raise ValueError(f"target_row {args.target_row} is out of range for dataset of length {len(meta)}")

        target_idx = int(args.target_row)

        if type_col is not None and type_col in meta.columns:
            target_type = meta.loc[target_idx, type_col]
            candidate_indices = find_indices_for_type(meta, type_col, str(target_type))
            title_prefix = f"Best twin for row {target_idx} ({display_names[target_idx]}) within type {target_type}"
        else:
            candidate_indices = np.arange(len(meta), dtype=np.int64)
            title_prefix = f"Best twin for row {target_idx} ({display_names[target_idx]})"

        pair = choose_best_twin_for_target_from_neighbors(
            target_idx=target_idx,
            candidate_indices=candidate_indices,
            nbrs=nbrs,
            dists=dists,
            display_names=display_names,
            exclude_same_sn=args.exclude_same_sn,
            unique_twin_sn=args.unique_twin_sn,
        )

        if pair is None:
            raise RuntimeError("Could not find a valid twin for the requested row")

        outpath = outdir / f"target_twin_row_{target_idx}.png"

        plot_pair(
            pair=pair,
            wave=wave,
            flux=flux,
            mask=mask,
            meta=meta,
            display_names=display_names,
            type_col=type_col,
            obsdate_col=obsdate_col,
            outpath=outpath,
            title_prefix=title_prefix,
        )

        print(f"saved {outpath}")
        print(pair)
        made += 1

    print(f"made {made} plot(s)")


if __name__ == "__main__":
    main()