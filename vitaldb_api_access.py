"""
vitaldb_api_access.py

Fetches VitalDB open-dataset tables directly from the public API — no bulk
download required. Produces a single merged "raw" dataset (clinical +
preop labs) ready to hand to preprocessing_pipeline.py.

API reference: https://api.vitaldb.net/
    GET https://api.vitaldb.net/cases  -> clinical/perioperative parameters (1 row/case)
    GET https://api.vitaldb.net/labs   -> lab results (long format)
    GET https://api.vitaldb.net/trks   -> per-case track catalogue (for waveform/vitals access)
    GET https://api.vitaldb.net/{tid}  -> a single track's time-series data

NOTE: Column names for the labs/cases endpoints are based on VitalDB's
published schema at the time of writing. Before running this for real,
open https://api.vitaldb.net/cases and https://api.vitaldb.net/labs once
in a browser (or `pd.read_csv(...).columns`) to confirm exact column names
haven't changed, and adjust PREOP_LAB_NAMES / merge keys if needed.

Usage:
    python vitaldb_api_access.py --out data/raw/vitaldb_raw.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CASES_URL = "https://api.vitaldb.net/cases"
LABS_URL = "https://api.vitaldb.net/labs"
TRACKS_URL = "https://api.vitaldb.net/trks"

# Preop lab test names (VitalDB 'name' field in the labs table) we want as features.
# NOTE: VitalDB's `cases` table already ships with several preop labs built in
# (preop_hb, preop_na, preop_k, preop_ast, preop_alt, preop_cr, preop_plt, ...).
# get_preop_labs() below deliberately drops any pivoted column that already
# exists in `cases` to avoid duplicate-column merge collisions (pandas would
# otherwise silently suffix them as `_x`/`_y`, which breaks every downstream
# column-name lookup in preprocessing_pipeline.py).
PREOP_LAB_NAMES = ["hb", "plt", "wbc", "cr", "gluc", "na", "k", "ast", "alt"]


def fetch_cases() -> pd.DataFrame:
    """Download the clinical/perioperative parameters table (one row per case)."""
    logger.info("Fetching case-level clinical data from %s", CASES_URL)
    df = pd.read_csv(CASES_URL)
    logger.info("Fetched %d cases, %d columns", len(df), df.shape[1])
    return df


def fetch_labs() -> pd.DataFrame:
    """Download the lab-results table (long format: caseid, dt, name, result)."""
    logger.info("Fetching lab results from %s", LABS_URL)
    df = pd.read_csv(LABS_URL)
    logger.info("Fetched %d lab records", len(df))
    return df


def fetch_track_list() -> pd.DataFrame:
    """Download the per-case track catalogue (caseid, tname, tid).
    Only needed if you later want to sample intraoperative vitals/waveforms."""
    logger.info("Fetching track list from %s", TRACKS_URL)
    df = pd.read_csv(TRACKS_URL)
    return df


def get_preop_labs(labs_df: pd.DataFrame, lab_names=PREOP_LAB_NAMES) -> pd.DataFrame:
    """
    Reduce the long-format labs table to one row per case, with the most
    recent PRE-operative (dt <= 0, i.e. before casestart) value for each
    requested lab, pivoted into columns named 'preop_<lab>'.
    """
    filtered = labs_df[labs_df["name"].isin(lab_names)].copy()
    preop = filtered[filtered["dt"] <= 0]
    preop = preop.sort_values("dt").drop_duplicates(subset=["caseid", "name"], keep="last")
    pivot = preop.pivot(index="caseid", columns="name", values="result")
    pivot.columns = [f"preop_{c}" for c in pivot.columns]
    return pivot.reset_index()


def build_raw_dataset(save_path: str = None) -> pd.DataFrame:
    """
    Fetch + merge clinical parameters and preop labs into a single raw
    dataset. This is the DataFrame that preprocessing_pipeline.py expects
    as input.

    Any pivoted lab column whose name already exists in `cases` is dropped
    before merging (cases' own preop_* fields are kept as the source of
    truth) — this avoids pandas silently creating `_x`/`_y` suffixed
    duplicate columns, which would otherwise break every downstream
    column-name lookup in the preprocessing pipeline.
    """
    cases = fetch_cases()
    labs = fetch_labs()
    preop_labs = get_preop_labs(labs)

    colliding = [c for c in preop_labs.columns if c != "caseid" and c in cases.columns]
    if colliding:
        logger.info(
            "Dropping %d lab columns already present in cases (keeping cases' own values): %s",
            len(colliding), colliding,
        )
        preop_labs = preop_labs.drop(columns=colliding)

    raw = cases.merge(preop_labs, on="caseid", how="left")
    logger.info("Merged raw dataset shape: %s", raw.shape)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(save_path, index=False)
        logger.info("Saved raw dataset to %s", save_path)

    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Fetch VitalDB open dataset via API (no bulk download)."
    )
    parser.add_argument(
        "--out",
        default="data/raw/vitaldb_raw.csv",
        help="Where to save the merged raw CSV (default: data/raw/vitaldb_raw.csv)",
    )
    args = parser.parse_args()
    build_raw_dataset(save_path=args.out)


if __name__ == "__main__":
    main()