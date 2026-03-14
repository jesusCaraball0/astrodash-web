#!/usr/bin/env python
"""
Unpack and inspect data/wiserep/wiserep_validation_set.parquet.
Run from repo root with an env that has pandas (e.g. conda activate astrodash):
    python prod_backend/scripts/inspect_wiserep_validation_parquet.py
    python prod_backend/scripts/inspect_wiserep_validation_parquet.py --csv data/wiserep/wiserep_validation_set.csv

To map validation rows back to source files (to avoid train/val contamination), run:
    python prod_backend/scripts/map_validation_parquet_to_sources.py
    python prod_backend/scripts/map_validation_parquet_to_sources.py --write-parquet
That writes data/wiserep/wiserep_validation_source_files.txt (one filename per row).
When creating future validation parquets, include a 'source_file' column (one filename per row)
so you can exclude those files from training without running the mapper.
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Inspect (and optionally export) wiserep validation set parquet")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="If set, export parquet to this CSV path (e.g. data/wiserep/wiserep_validation_set.csv)",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    parquet_path = PROJECT_ROOT / "data" / "wiserep" / "wiserep_validation_set.parquet"
    if not parquet_path.exists():
        print(f"Not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    print("Shape:", df.shape)
    print()
    print("Columns:", list(df.columns))
    print()
    print("Dtypes:")
    print(df.dtypes.to_string())
    print()
    print("First 5 rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(df.head(5).to_string())
    print()
    print("Describe (numeric):")
    print(df.describe(include="all").to_string())

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print()
        print(f"Exported to {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
