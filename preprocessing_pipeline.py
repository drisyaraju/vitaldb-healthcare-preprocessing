"""
preprocessing_pipeline.py

Reusable, sklearn-compatible preprocessing pipeline for the VitalDB
clinical/perioperative dataset (produced by vitaldb_api_access.py), or any
similarly-shaped healthcare tabular data.

Pipeline steps (in order):
    1. Duplicate removal
    2. Outlier detection & handling  -> clinical plausibility bounds
    3. Missing value imputation      -> MICE (IterativeImputer) as the
                                         production method; KNN is kept
                                         available purely as a comparison
                                         baseline (see compare_imputers())
    4. Feature engineering           -> BMI, durations, abnormal-lab-count
    5. Categorical encoding          -> ordinal (ASA) + frequency encoding
                                         (deliberately NOT one-hot)
    6. Normalization                 -> RobustScaler

Every step is a proper scikit-learn Transformer (fit/transform), so the
whole thing is a single Pipeline.fit_transform(df) call and can be
re-applied to new/incoming data with .transform() alone.

Usage:
    python preprocessing_pipeline.py --input data/raw/vitaldb_raw.csv \
                                      --output data/processed/vitaldb_processed.csv

    # To also run the MICE-vs-KNN imputation comparison report:
    python preprocessing_pipeline.py --input data/raw/vitaldb_raw.csv \
                                      --output data/processed/vitaldb_processed.csv \
                                      --compare-imputers
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required to unlock IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — clinically-informed settings. Edit here, not inside the classes.
# ---------------------------------------------------------------------------

ID_COL = "caseid"

NUMERIC_COLS = [
    "age", "height", "weight",
    "preop_hb", "preop_plt", "preop_cr", "preop_gluc",
    "preop_na", "preop_k", "preop_ast", "preop_alt", "preop_wbc",
]

CATEGORICAL_NOMINAL_COLS = ["sex", "department", "optype", "approach", "position"]
CATEGORICAL_ORDINAL_COLS = ["asa"]  # ASA physical status: 1 (healthy) -> 5 (moribund)
ORDINAL_ORDER = [[1, 2, 3, 4, 5]]

# Clinically plausible ranges (min, max) for adult patients. Values outside
# these are treated as implausible/erroneous, not just "statistically rare" —
# grounded in standard adult reference ranges, not an arbitrary z-score cutoff.
CLINICAL_BOUNDS = {
    "age": (0, 110),
    "height": (100, 220),        # cm
    "weight": (20, 250),         # kg
    "preop_hb": (3, 20),         # g/dL
    "preop_plt": (5, 1000),      # x10^3/uL
    "preop_cr": (0.1, 15),       # mg/dL
    "preop_gluc": (30, 600),     # mg/dL
    "preop_na": (100, 170),      # mmol/L
    "preop_k": (1.5, 9),         # mmol/L
    "preop_wbc": (0.5, 50),      # x10^3/uL
}

ENGINEERED_NUMERIC_COLS = [
    "bmi", "surgery_duration_min", "anesthesia_duration_min", "induction_to_incision_min",
]


# ---------------------------------------------------------------------------
# 1. Duplicate handling
# ---------------------------------------------------------------------------

class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Drop exact duplicate rows, and duplicate case IDs (keeping the first).

    Reasoning: a naive full-row hash catches ingestion errors, but a
    patient with two genuinely separate surgeries is NOT a duplicate —
    dedup on the case identifier, not the patient identity.
    """

    def __init__(self, id_col=ID_COL):
        self.id_col = id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        before = len(X)
        X = X.drop_duplicates()
        if self.id_col in X.columns:
            X = X.drop_duplicates(subset=[self.id_col], keep="first")
        logger.info("DuplicateRemover: %d -> %d rows", before, len(X))
        return X


# ---------------------------------------------------------------------------
# 2. Outlier detection & handling — clinical plausibility bounds
# ---------------------------------------------------------------------------

class ClinicalOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Flags values outside clinically plausible bounds as missing (NaN)
    rather than dropping the row.

    Reasoning: a generic z-score/IQR rule would flag a genuinely critical
    (but real) patient value as noise and discard it. Using literature-based
    reference ranges instead distinguishes IMPOSSIBLE values (sensor/entry
    error -> should be treated as missing) from EXTREME-BUT-REAL values
    (a genuinely sick patient -> clinically important, must be kept).
    Flagging-as-missing (rather than row deletion) also means the outlier
    goes through the same imputation step as any other missing value,
    keeping the rest of that patient's record intact.
    """

    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else CLINICAL_BOUNDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        flagged_total = 0
        for col, (lo, hi) in self.bounds.items():
            if col in X.columns:
                mask = ~X[col].between(lo, hi) & X[col].notna()
                flagged_total += int(mask.sum())
                X.loc[mask, col] = np.nan
        logger.info("ClinicalOutlierHandler: flagged %d implausible values as missing", flagged_total)
        return X


# ---------------------------------------------------------------------------
# 3. Missing value imputation — MICE (primary) with KNN kept as a comparison
# ---------------------------------------------------------------------------

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes numeric columns using MICE (Multiple Imputation by Chained
    Equations, via sklearn's IterativeImputer) and mode-imputes categorical
    columns. Also adds a `<col>_was_missing` indicator per imputed numeric
    field, since missingness itself can be informative in EHR data (e.g. a
    lab simply wasn't ordered because it wasn't clinically indicated).

    Why MICE over a single KNN pass:
    KNN imputation borrows values from the k most *similar* patients on the
    other numeric columns, which is a reasonable heuristic but is a single
    deterministic pass — it doesn't model uncertainty and can propagate
    noise from an unrepresentative neighborhood. MICE instead models each
    missing column as a function of *all* other columns via chained
    regression, iterating until the estimates stabilize, which is standard
    practice in clinical/epidemiological statistics for exactly this kind
    of correlated, mixed-missingness lab data. See compare_imputers() below
    for a head-to-head accuracy comparison against KNN on this dataset.
    """

    def __init__(self, numeric_cols=None, categorical_cols=None,
                 strategy="mice", n_neighbors=5, max_iter=10, random_state=42):
        self.numeric_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLS
        self.categorical_cols = categorical_cols if categorical_cols is not None else (CATEGORICAL_NOMINAL_COLS + CATEGORICAL_ORDINAL_COLS)
        self.strategy = strategy  # "mice" or "knn"
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state

    def _make_imputer(self):
        # keep_empty_features=True is required: by default sklearn's imputers
        # silently DROP a column that is entirely NaN, which shrinks the
        # output's column count and breaks reassignment back into the
        # DataFrame (X[cols] = imputer.transform(X[cols])). Keeping the
        # feature (filled with 0, or the column mean if any values exist)
        # preserves the schema even in that edge case.
        if self.strategy == "mice":
            return IterativeImputer(max_iter=self.max_iter, random_state=self.random_state,
                                     keep_empty_features=True)
        elif self.strategy == "knn":
            return KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
        raise ValueError(f"Unknown strategy '{self.strategy}', expected 'mice' or 'knn'")

    def fit(self, X, y=None):
        self._fit_cols = [c for c in self.numeric_cols if c in X.columns]
        self.imputer_ = self._make_imputer()
        if self._fit_cols:
            # .to_numpy(copy=True) forces a writable array: pandas' Copy-on-Write
            # mode can otherwise hand sklearn a read-only view, which crashes
            # IterativeImputer's internal empty-feature handling.
            self.imputer_.fit(X[self._fit_cols].to_numpy(copy=True, dtype=float))
        self._modes = {
            c: X[c].mode(dropna=True).iloc[0]
            for c in self.categorical_cols
            if c in X.columns and X[c].notna().any()
        }
        return self

    def transform(self, X):
        X = X.copy()
        for c in self._fit_cols:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        if self._fit_cols:
            imputed = self.imputer_.transform(X[self._fit_cols].to_numpy(copy=True, dtype=float))
            X[self._fit_cols] = imputed
        for c, mode_val in self._modes.items():
            X[c] = X[c].fillna(mode_val)
        logger.info(
            "MissingValueImputer[%s]: imputed %d numeric cols, %d categorical cols",
            self.strategy, len(self._fit_cols), len(self._modes),
        )
        return X


class ImputationBoundsClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric columns to their clinical bounds AFTER imputation.

    Why this is needed: MICE (IterativeImputer) fits a regression per
    column and can, in principle, extrapolate an estimate slightly outside
    a physiologically valid range — unlike KNN, which only ever copies a
    real observed neighbor's value. Values that were already within bounds
    (the vast majority of the data) are left untouched; this only affects
    the rare regression-extrapolated estimate, and is what makes MICE
    imputation safe to use alongside a strict plausibility validation
    check downstream.
    """

    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else CLINICAL_BOUNDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        clipped_total = 0
        for col, (lo, hi) in self.bounds.items():
            if col in X.columns:
                before = X[col].copy()
                X[col] = X[col].clip(lower=lo, upper=hi)
                clipped_total += int((before != X[col]).sum())
        logger.info("ImputationBoundsClipper: clipped %d values back into plausible range", clipped_total)
        return X


def compare_imputers(df: pd.DataFrame, numeric_cols=None, missing_frac=0.15, random_state=42) -> pd.DataFrame:
    """
    Head-to-head comparison of MICE vs. KNN imputation accuracy.

    Method: take rows that are already fully observed for `numeric_cols`,
    artificially mask `missing_frac` of their values (so we know ground
    truth), impute with each method, and compute RMSE per column against
    the true values. This is the evidence you cite in your README/report
    for *why* MICE was chosen over KNN (or vice versa), instead of just
    asserting it.

    Returns a DataFrame of per-column RMSE for each method.
    """
    numeric_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLS
    cols = [c for c in numeric_cols if c in df.columns]
    complete = df[cols].dropna().copy()

    if len(complete) < 30:
        logger.warning("compare_imputers: fewer than 30 fully-observed rows (%d) — "
                        "results may be unreliable.", len(complete))

    rng = np.random.default_rng(random_state)
    masked = complete.copy()
    true_values = {}
    for col in cols:
        n_mask = max(1, int(len(masked) * missing_frac))
        idx = rng.choice(masked.index, size=n_mask, replace=False)
        true_values[col] = masked.loc[idx, col].copy()
        masked.loc[idx, col] = np.nan

    results = {}
    for strategy, imputer in [
        ("mice", IterativeImputer(max_iter=10, random_state=random_state)),
        ("knn", KNNImputer(n_neighbors=5)),
    ]:
        imputed = pd.DataFrame(imputer.fit_transform(masked), columns=cols, index=masked.index)
        rmses = {}
        for col in cols:
            idx = true_values[col].index
            rmses[col] = float(np.sqrt(np.mean((imputed.loc[idx, col] - true_values[col]) ** 2)))
        results[strategy] = rmses

    report = pd.DataFrame(results)
    report["better_method"] = report.idxmin(axis=1)
    logger.info("Imputer comparison (lower RMSE = better):\n%s", report)
    return report


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives clinically meaningful features from raw/imputed columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if {"height", "weight"}.issubset(X.columns):
            X["bmi"] = X["weight"] / ((X["height"] / 100) ** 2)

        for start, end, new_col in [
            ("opstart", "opend", "surgery_duration_min"),
            ("anestart", "aneend", "anesthesia_duration_min"),
        ]:
            if {start, end}.issubset(X.columns):
                X[new_col] = (X[end] - X[start]) / 60.0

        if {"anestart", "opstart"}.issubset(X.columns):
            X["induction_to_incision_min"] = (X["opstart"] - X["anestart"]) / 60.0

        lab_cols = [c for c in NUMERIC_COLS if c.startswith("preop_") and c in X.columns]
        if lab_cols:
            abnormal = pd.DataFrame(index=X.index)
            for c in lab_cols:
                lo, hi = CLINICAL_BOUNDS.get(c, (-np.inf, np.inf))
                abnormal[c] = ~X[c].between(lo, hi)
            X["abnormal_preop_lab_count"] = abnormal.sum(axis=1)

        logger.info("FeatureEngineer: engineered features added, new shape %s", X.shape)
        return X


# ---------------------------------------------------------------------------
# 5. Categorical encoding — ordinal + frequency encoding (NOT one-hot)
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each nominal category by its observed frequency (proportion)
    in the training data, learned during fit(). Unseen categories at
    transform-time are mapped to 0.

    Why frequency encoding instead of one-hot:
    Fields like `optype`/`department` are moderate-to-high cardinality.
    One-hot would explode into dozens of sparse binary columns, inflating
    dimensionality without adding much signal per column. Frequency
    encoding compresses each category into a single informative numeric
    column (how common is this category?), keeps the pipeline compact and
    reusable on new data, and — unlike target encoding — needs no outcome
    label, so it can't leak information and works cleanly in an
    unsupervised preprocessing pipeline.
    """

    def __init__(self, cols=None):
        self.cols = cols if cols is not None else CATEGORICAL_NOMINAL_COLS

    def fit(self, X, y=None):
        self._present = [c for c in self.cols if c in X.columns]
        self.freq_maps_ = {
            c: X[c].value_counts(normalize=True, dropna=True).to_dict()
            for c in self._present
        }
        return self

    def transform(self, X):
        X = X.copy()
        for c in self._present:
            X[f"{c}_freq"] = X[c].map(self.freq_maps_[c]).fillna(0.0)
        X = X.drop(columns=self._present)
        logger.info("FrequencyEncoder: frequency-encoded %d nominal columns", len(self._present))
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Combines ordinal encoding (for clinically ORDERED categories, e.g. ASA
    class, where the order itself carries meaning) with frequency encoding
    (for unordered nominal categories) — deliberately avoiding one-hot.
    """

    def __init__(self, ordinal_cols=None, ordinal_order=None, nominal_cols=None):
        self.ordinal_cols = ordinal_cols if ordinal_cols is not None else CATEGORICAL_ORDINAL_COLS
        self.ordinal_order = ordinal_order if ordinal_order is not None else ORDINAL_ORDER
        self.nominal_cols = nominal_cols if nominal_cols is not None else CATEGORICAL_NOMINAL_COLS

    def fit(self, X, y=None):
        self._ordinal_present = [c for c in self.ordinal_cols if c in X.columns]
        if self._ordinal_present:
            self.ordinal_encoder_ = OrdinalEncoder(
                categories=self.ordinal_order,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.ordinal_encoder_.fit(X[self._ordinal_present])

        self.freq_encoder_ = FrequencyEncoder(cols=self.nominal_cols)
        self.freq_encoder_.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        if self._ordinal_present:
            X[self._ordinal_present] = self.ordinal_encoder_.transform(X[self._ordinal_present])
        X = self.freq_encoder_.transform(X)
        logger.info("CategoricalEncoder: encoded %d ordinal columns + frequency-encoded nominal columns",
                    len(self._ordinal_present))
        return X


# ---------------------------------------------------------------------------
# 6. Normalization — RobustScaler
# ---------------------------------------------------------------------------

class RobustNumericScaler(BaseEstimator, TransformerMixin):
    """
    Scales numeric columns using RobustScaler (median + IQR) rather than
    StandardScaler (mean + std).

    Why RobustScaler is the apt choice here: preop labs (creatinine,
    glucose, platelets, etc.) are right-skewed with genuine extreme values
    from critically ill patients — these are real, clinically meaningful
    data points, not noise, and must NOT be treated as outliers-to-remove.
    StandardScaler's mean/std are themselves pulled around by exactly these
    extreme-but-real values, which would compress the "normal" bulk of the
    distribution and distort the scaled feature space. RobustScaler's
    median/IQR are far less sensitive to that skew, so the resulting scaled
    features better represent typical patients while still keeping the
    extreme (clinically important) values distinguishable rather than
    warping everything else around them.
    """

    def __init__(self, cols=None):
        self.cols = cols if cols is not None else (NUMERIC_COLS + ENGINEERED_NUMERIC_COLS)

    def fit(self, X, y=None):
        self._present = [c for c in self.cols if c in X.columns]
        self.scaler_ = RobustScaler()
        if self._present:
            self.scaler_.fit(X[self._present])
        return self

    def transform(self, X):
        X = X.copy()
        if self._present:
            X[self._present] = self.scaler_.transform(X[self._present])
        logger.info("RobustNumericScaler: scaled %d columns", len(self._present))
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects the pipeline's intentional final feature set, dropping raw
    VitalDB columns we never chose to process (e.g. `cormack`, `tubesize`,
    `iv1`/`iv2`, `intraop_ebl`, `preop_ph`, ...).

    Why this step exists: without it, every raw column VitalDB happens to
    ship rides along untouched to the final output — including columns
    this pipeline was never designed to clean, which still carry their
    original missing values. That's not a data-quality bug in the
    pipeline itself, but it does make "no missing values remain" validation
    meaningless, since it's checking columns we never touched. Explicitly
    selecting ID + engineered/imputed/encoded columns makes the pipeline's
    output an intentional, complete feature set rather than an accidental
    passthrough of the entire raw schema.
    """

    def __init__(self, id_col=None, numeric_cols=None, engineered_cols=None,
                 ordinal_cols=None):
        self.id_col = id_col if id_col is not None else ID_COL
        self.numeric_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLS
        self.engineered_cols = engineered_cols if engineered_cols is not None else ENGINEERED_NUMERIC_COLS
        self.ordinal_cols = ordinal_cols if ordinal_cols is not None else CATEGORICAL_ORDINAL_COLS

    def fit(self, X, y=None):
        explicit = set([self.id_col] + self.numeric_cols + self.engineered_cols + self.ordinal_cols)
        derived = [
            c for c in X.columns
            if c.endswith("_was_missing") or c.endswith("_freq")
            or c == "abnormal_preop_lab_count"
        ]
        self._keep = [c for c in X.columns if c in explicit or c in derived]
        return self

    def transform(self, X):
        X = X[[c for c in self._keep if c in X.columns]].copy()
        logger.info("FeatureSelector: kept %d intentional feature columns (dropped the rest)", X.shape[1])
        return X


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_pipeline(imputation_strategy="mice") -> Pipeline:
    """
    Step order matters:
      1. Remove duplicates first (cheap; avoids processing redundant rows)
      2. Flag implausible values as missing (must happen BEFORE imputation)
      3. Impute missing values (MICE by default; pass strategy='knn' to compare)
      4. Clip any imputed value that landed outside clinical bounds
         (MICE can extrapolate; this guarantees plausibility afterward)
      5. Engineer features (BMI, durations, abnormal-lab-count) from clean data
      6. Encode categoricals (ordinal + frequency, no one-hot)
      7. Scale numeric features last (RobustScaler), once data is clean and complete
      8. Select the final intentional feature set (drops raw columns this
         pipeline never chose to process, e.g. equipment/catheter codes)
    """
    return Pipeline(steps=[
        ("duplicates", DuplicateRemover()),
        ("outliers", ClinicalOutlierHandler()),
        ("impute", MissingValueImputer(strategy=imputation_strategy)),
        ("clip_imputed", ImputationBoundsClipper()),
        ("feature_engineering", FeatureEngineer()),
        ("encode", CategoricalEncoder()),
        ("scale", RobustNumericScaler()),
        ("select", FeatureSelector()),
    ])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data_quality(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Before/after data-quality report — a lightweight sanity check that
    the pipeline actually improved the data rather than just changed it."""
    report = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "cols_before": df_before.shape[1],
        "cols_after": df_after.shape[1],
        "missing_pct_before": round(df_before.isna().mean().mean() * 100, 2),
        "missing_pct_after": round(df_after.isna().mean().mean() * 100, 2),
        "duplicate_rows_after": int(df_after.duplicated().sum()),
    }
    logger.info("Data quality report: %s", report)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the VitalDB preprocessing pipeline.")
    parser.add_argument(
        "--input", default="data/raw/vitaldb_raw.csv",
        help="Raw CSV produced by vitaldb_api_access.py",
    )
    parser.add_argument(
        "--output", default="data/processed/vitaldb_processed.csv",
        help="Where to save the processed CSV",
    )
    parser.add_argument(
        "--compare-imputers", action="store_true",
        help="Run the MICE-vs-KNN imputation accuracy comparison and print the report.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)

    if args.compare_imputers:
        compare_imputers(raw)

    pipeline = build_pipeline(imputation_strategy="mice")
    processed = pipeline.fit_transform(raw)

    validate_data_quality(raw, processed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    logger.info("Saved processed dataset to %s", args.output)


if __name__ == "__main__":
    main()