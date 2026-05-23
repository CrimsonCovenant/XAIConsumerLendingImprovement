"""Build yearly processed parquet files for the entire 2008–2024 range.

For each year:
  1. Loads raw CSV via the unified preprocessor (handles schema break).
  2. Writes data/processed/processed_{year}.parquet  (modeling features + target)
  3. Writes data/processed/protected_{year}.parquet  (protected attrs, row-aligned)
  4. Writes data/processed/_manifest_{year}.json      (audit metadata)
"""
import sys
import os
import json
import time
import logging
import argparse

# Project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
import pandas as pd

from src.features.preprocessing import preprocess_year
from src.utils.reproducibility import file_sha256, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build yearly processed parquets")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output files already exist")
    parser.add_argument("--years", type=int, nargs="*",
                        help="Specific years to process (default: all from config)")
    args = parser.parse_args()

    seed_everything(42)

    with open("config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    years = args.years if args.years else config["pipeline"]["years"]
    out_dir = config["data"]["processed"]
    raw_dir = config["data"]["raw"]
    os.makedirs(out_dir, exist_ok=True)

    for year in years:
        parquet_path = os.path.join(out_dir, f"processed_{year}.parquet")
        protected_path = os.path.join(out_dir, f"protected_{year}.parquet")
        manifest_path = os.path.join(out_dir, f"_manifest_{year}.json")

        # Skip if already built (unless --force)
        if not args.force and os.path.exists(parquet_path) and os.path.exists(protected_path):
            logger.info("Year %d already processed, skipping. Use --force to rebuild.", year)
            continue

        # Check raw file exists
        raw_path = os.path.join(raw_dir, f"year_{year}.csv")
        if not os.path.exists(raw_path):
            logger.warning("Raw file not found for %d at %s, skipping.", year, raw_path)
            continue

        t0 = time.time()
        logger.info("Processing year %d%s", year, " (FORCE REBUILD)" if args.force else "")

        try:
            df_model, df_protected, meta = preprocess_year(year, config)
        except Exception as e:
            logger.error("Failed to preprocess year %d: %s", year, e)
            continue

        # Save parquet files
        df_model.to_parquet(parquet_path, index=False, engine="pyarrow")
        df_protected.to_parquet(protected_path, index=False, engine="pyarrow")

        wall_clock = time.time() - t0

        # Build manifest
        manifest = {
            **meta,
            "input_file": raw_path,
            "input_sha256": file_sha256(raw_path) if os.path.getsize(raw_path) < 5e9 else "skipped-large-file",
            "output_processed": parquet_path,
            "output_protected": protected_path,
            "wall_clock_s": round(wall_clock, 1),
            "target_values": df_model["loan_status"].value_counts().to_dict(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(
            "Year %d done in %.1fs — %d rows (%d features), saved to %s",
            year, wall_clock, len(df_model),
            len(meta["numeric_features"]) + len(meta["categorical_features"]),
            parquet_path,
        )

    logger.info("All years processed.")

if __name__ == "__main__":
    main()
