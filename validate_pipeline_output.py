"""
validate_pipeline_output.py

Validates the OUTPUT of preprocessing_pipeline.py against a set of concrete
data-quality checks. This closes out the last required task in the project
guideline ("Validate processed data quality") — it's a proper module, not
a bare row-count comparison.

Checks performed:
    1. Schema validation      -> ID column present & unique, no fully-null
                                  columns, numeric columns are actually numeric
    2. Completeness            -> no missing values remain post-imputation
    3. Imputation plausibility -> MICE can technically extrapolate outside
                                  the clinical bounds we defined (it's a
                                  regression-based estimate, not a lookup) —
                                  check imputed values still fall within
                                  clinically plausible ranges, on the
                                  PRE-SCALING data (values checked before
                                  RobustScaler is applied)
    4. Scaling sanity          -> RobustScaler should center each scaled
                                  column's median near 0 and its IQR near 1;
                                  confirm that actually happened
    5. Distribution comparison -> before/after mean, median and skew per
                                  numeric column, so you can visually/
                                  numerically justify your scaling choice

Produces a pass/fail summary dict and an optional Markdown report file
(handy to drop straight into your project README or portfolio writeup).

Usage:
    python validate_pipeline_output.py --input data/raw/vitaldb_raw.csv \
                                        --report-out reports/quality_report.md
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_pipeline import (
    CLINICAL_BOUNDS,
    ENGINEERED_NUMERIC_COLS,
    ID_COL,
    NUMERIC_COLS,
    build_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, id_col: str = ID_COL) -> dict:
    issues = []

    if id_col not in df.columns:
        issues.append(f"Missing required ID column '{id_col}'")
    elif df[id_col].duplicated().any():
        issues.append(f"'{id_col}' contains {df[id_col].duplicated().sum()} duplicate values after processing")

    fully_null_cols = [c for c in df.columns if df[c].isna().all()]
    if fully_null_cols:
        issues.append(f"Columns entirely null: {fully_null_cols}")

    non_numeric_expected = [c for c in NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
                             if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_expected:
        issues.append(f"Expected-numeric columns are not numeric dtype: {non_numeric_expected}")

    return {"check": "schema", "passed": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# 2. Completeness
# ---------------------------------------------------------------------------

def validate_no_missing_values(df: pd.DataFrame, allowed_missing_cols: list = None) -> dict:
    allowed_missing_cols = allowed_missing_cols or []
    check_cols = [c for c in df.columns if c not in allowed_missing_cols]
    missing_counts = df[check_cols].isna().sum()
    offending = missing_counts[missing_counts > 0]

    issues = [f"'{col}': {count} missing values remain" for col, count in offending.items()]
    return {"check": "completeness", "passed": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# 3. Imputation plausibility (checked BEFORE scaling)
# ---------------------------------------------------------------------------

def validate_imputed_values_plausible(pipeline, raw_df: pd.DataFrame, bounds: dict = None) -> dict:
    """
    MICE (IterativeImputer) fits a regression per column and can, in
    principle, extrapolate a value outside the clinically plausible range
    we defined — unlike KNN, which only ever returns values seen in real
    neighbors. This check catches that: it looks at the pipeline's output
    right BEFORE the scaling step (using pipeline slicing, which reuses the
    already-fitted transformers) and re-checks every bounded column against
    CLINICAL_BOUNDS.
    """
    bounds = bounds or CLINICAL_BOUNDS
    issues = []

    # Pipeline steps are: ..., "clip_imputed", "feature_engineering", "encode",
    # "scale", "select". Slicing off the last TWO steps ("select", "scale")
    # gives the post-clip, pre-scaling data these bounds are defined in.
    pre_scale_df = pipeline[:-2].transform(raw_df)

    for col, (lo, hi) in bounds.items():
        if col in pre_scale_df.columns:
            violations = ~pre_scale_df[col].between(lo, hi)
            n_violations = int(violations.sum())
            if n_violations > 0:
                issues.append(
                    f"'{col}': {n_violations} imputed/retained values fall outside "
                    f"clinically plausible range [{lo}, {hi}]"
                )

    return {"check": "imputation_plausibility", "passed": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# 4. Scaling sanity
# ---------------------------------------------------------------------------

def validate_scaling_properties(processed_df: pd.DataFrame, cols: list = None, tolerance: float = 0.5) -> dict:
    """
    RobustScaler centers each column's median to ~0 and scales by IQR to
    ~1. If a column's post-scaling median/IQR is far from that, something
    is off (e.g. a column was scaled twice, or wasn't scaled at all).
    """
    cols = cols or (NUMERIC_COLS + ENGINEERED_NUMERIC_COLS)
    issues = []

    for col in cols:
        if col not in processed_df.columns:
            continue
        median = processed_df[col].median()
        q75, q25 = processed_df[col].quantile([0.75, 0.25])
        iqr = q75 - q25

        if abs(median) > tolerance:
            issues.append(f"'{col}': post-scaling median is {median:.3f}, expected ~0")
        if iqr != 0 and abs(iqr - 1) > tolerance:
            issues.append(f"'{col}': post-scaling IQR is {iqr:.3f}, expected ~1")

    return {"check": "scaling_sanity", "passed": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# 5. Distribution comparison (informational, not pass/fail)
# ---------------------------------------------------------------------------

def compare_distributions(raw_df: pd.DataFrame, processed_df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    cols = cols or NUMERIC_COLS
    rows = []
    for col in cols:
        if col not in raw_df.columns or col not in processed_df.columns:
            continue
        rows.append({
            "column": col,
            "mean_before": raw_df[col].mean(),
            "mean_after": processed_df[col].mean(),
            "median_before": raw_df[col].median(),
            "median_after": processed_df[col].median(),
            "skew_before": raw_df[col].skew(),
            "skew_after": processed_df[col].skew(),
        })
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_quality_report(pipeline, raw_df: pd.DataFrame, processed_df: pd.DataFrame,
                             report_out: str = None) -> dict:
    checks = [
        validate_schema(processed_df),
        validate_no_missing_values(processed_df),
        validate_imputed_values_plausible(pipeline, raw_df),
        validate_scaling_properties(processed_df),
    ]
    distribution_report = compare_distributions(raw_df, processed_df)

    overall_passed = all(c["passed"] for c in checks)
    summary = {
        "overall_passed": overall_passed,
        "checks": checks,
        "distribution_comparison": distribution_report.to_dict(orient="records"),
    }

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        logger.info("[%s] %s", status, c["check"])
        for issue in c["issues"]:
            logger.warning("    -> %s", issue)

    if report_out:
        _write_markdown_report(summary, distribution_report, report_out)

    return summary


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Hand-rolled Markdown table builder — avoids depending on the optional
    `tabulate` package that pandas.DataFrame.to_markdown() requires (which
    isn't installed by default and shouldn't be a hard requirement just to
    write a report).
    """
    if df.empty:
        return "_No data available._"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def _write_markdown_report(summary: dict, distribution_report: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Validation Report", ""]
    lines.append(f"**Overall result: {'PASS' if summary['overall_passed'] else 'FAIL'}**\n")

    for c in summary["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"## {c['check']} — {status}")
        if c["issues"]:
            for issue in c["issues"]:
                lines.append(f"- {issue}")
        else:
            lines.append("- No issues found.")
        lines.append("")

    lines.append("## Distribution comparison (before -> after)")
    lines.append(_dataframe_to_markdown_table(distribution_report))
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote quality report to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate preprocessing pipeline output quality.")
    parser.add_argument("--input", default="data/raw/vitaldb_raw.csv",
                         help="Raw CSV produced by vitaldb_api_access.py")
    parser.add_argument("--report-out", default="reports/quality_report.md",
                         help="Where to save the Markdown quality report")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    pipeline = build_pipeline(imputation_strategy="mice")
    processed_df = pipeline.fit_transform(raw_df)

    summary = generate_quality_report(pipeline, raw_df, processed_df, report_out=args.report_out)

    if not summary["overall_passed"]:
        raise SystemExit("Data quality validation FAILED — see report for details.")
    logger.info("Data quality validation PASSED.")


if __name__ == "__main__":
    main()