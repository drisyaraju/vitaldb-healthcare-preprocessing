"""
test_pipeline.py

Unit tests for preprocessing_pipeline.py and validate_pipeline_output.py.
Uses small synthetic DataFrames (not the live VitalDB API) so tests run
fast, offline, and deterministically.

Run with:
    pytest test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing_pipeline import (
    CLINICAL_BOUNDS,
    CategoricalEncoder,
    ClinicalOutlierHandler,
    DuplicateRemover,
    FeatureEngineer,
    FrequencyEncoder,
    ImputationBoundsClipper,
    MissingValueImputer,
    RobustNumericScaler,
    build_pipeline,
    compare_imputers,
)
from validate_pipeline_output import (
    validate_imputed_values_plausible,
    validate_no_missing_values,
    validate_schema,
    validate_scaling_properties,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small, deliberately messy synthetic dataset mirroring the real schema."""
    return pd.DataFrame({
        "caseid": [1, 2, 3, 4, 5, 5],           # note: case 5 duplicated
        "age": [45, 60, 999, 30, 55, 55],        # 999 is an implausible outlier
        "height": [170, 165, 180, np.nan, 175, 175],
        "weight": [70, 60, 85, 55, np.nan, np.nan],
        "sex": ["M", "F", "M", "F", "M", "M"],
        "asa": [1, 2, 3, 1, 2, 2],
        "department": ["General", "Ortho", "General", "Cardiac", "General", "General"],
        "preop_hb": [13.5, 12.0, np.nan, 14.0, 11.5, 11.5],
        "preop_cr": [0.9, 1.1, 0.8, np.nan, 999, 999],  # 999 implausible
        "opstart": [0, 0, 0, 0, 0, 0],
        "opend": [3600, 5400, 1800, 7200, 2700, 2700],
        "anestart": [-600, -900, -300, -1200, -600, -600],
        "aneend": [4000, 5800, 2000, 7500, 3000, 3000],
    })


# ---------------------------------------------------------------------------
# DuplicateRemover
# ---------------------------------------------------------------------------

class TestDuplicateRemover:

    def test_removes_duplicate_caseids(self, sample_df):
        out = DuplicateRemover().fit_transform(sample_df)
        assert out["caseid"].is_unique
        assert len(out) == 5  # one duplicate row removed

    def test_no_duplicates_is_noop(self):
        df = pd.DataFrame({"caseid": [1, 2, 3], "age": [30, 40, 50]})
        out = DuplicateRemover().fit_transform(df)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# ClinicalOutlierHandler
# ---------------------------------------------------------------------------

class TestClinicalOutlierHandler:

    def test_flags_implausible_values_as_nan(self, sample_df):
        out = ClinicalOutlierHandler().fit_transform(sample_df)
        assert out.loc[out["age"].isna()].shape[0] >= 1  # age=999 flagged
        assert out.loc[2, "age"] is np.nan or pd.isna(out.loc[2, "age"])
        assert pd.isna(out.loc[4, "preop_cr"])  # cr=999 flagged

    def test_leaves_plausible_values_untouched(self, sample_df):
        out = ClinicalOutlierHandler().fit_transform(sample_df)
        assert out.loc[0, "age"] == 45  # untouched, within bounds

    def test_empty_bounds_dict_is_noop(self, sample_df):
        out = ClinicalOutlierHandler(bounds={}).fit_transform(sample_df)
        pd.testing.assert_frame_equal(out, sample_df)


# ---------------------------------------------------------------------------
# MissingValueImputer (MICE + KNN)
# ---------------------------------------------------------------------------

class TestMissingValueImputer:

    @pytest.mark.parametrize("strategy", ["mice", "knn"])
    def test_fills_all_missing_numeric_values(self, sample_df, strategy):
        imputer = MissingValueImputer(strategy=strategy)
        out = imputer.fit_transform(sample_df)
        for col in imputer._fit_cols:
            assert out[col].isna().sum() == 0

    def test_adds_was_missing_indicator(self, sample_df):
        out = MissingValueImputer(strategy="knn").fit_transform(sample_df)
        assert "weight_was_missing" in out.columns
        assert out.loc[4, "weight_was_missing"] == 1
        assert out.loc[0, "weight_was_missing"] == 0

    def test_invalid_strategy_raises(self, sample_df):
        with pytest.raises(ValueError):
            MissingValueImputer(strategy="bogus").fit(sample_df)

    def test_handles_column_with_single_unique_value(self):
        df = pd.DataFrame({
            "age": [40, 40, 40, np.nan],
            "height": [170, 170, 170, 170],
        })
        out = MissingValueImputer(numeric_cols=["age", "height"], strategy="knn").fit_transform(df)
        assert out["age"].isna().sum() == 0


# ---------------------------------------------------------------------------
# ImputationBoundsClipper
# ---------------------------------------------------------------------------

class TestImputationBoundsClipper:

    def test_clips_values_outside_bounds(self):
        df = pd.DataFrame({"weight": [15, 70, 300]})  # 15 and 300 outside [20, 250]
        out = ImputationBoundsClipper(bounds={"weight": (20, 250)}).fit_transform(df)
        assert out["weight"].tolist() == [20, 70, 250]

    def test_leaves_in_bounds_values_untouched(self):
        df = pd.DataFrame({"age": [30, 45, 60]})
        out = ImputationBoundsClipper(bounds={"age": (0, 110)}).fit_transform(df)
        pd.testing.assert_series_equal(out["age"], df["age"])

    def test_missing_column_skipped_gracefully(self):
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        out = ImputationBoundsClipper().fit_transform(df)
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# compare_imputers
# ---------------------------------------------------------------------------

class TestCompareImputers:

    def test_returns_rmse_report_with_winner_column(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "age": rng.normal(50, 10, 200),
            "height": rng.normal(170, 8, 200),
            "weight": rng.normal(70, 12, 200),
        })
        report = compare_imputers(df, numeric_cols=["age", "height", "weight"], missing_frac=0.2)
        assert "better_method" in report.columns
        assert set(report["better_method"].unique()).issubset({"mice", "knn"})
        assert (report[["mice", "knn"]] >= 0).all().all()  # RMSE can't be negative


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class TestFeatureEngineer:

    def test_computes_bmi(self, sample_df):
        clean = sample_df.fillna({"height": 170, "weight": 70})
        out = FeatureEngineer().fit_transform(clean)
        expected_bmi = 70 / (1.70 ** 2)
        assert abs(out.loc[0, "bmi"] - expected_bmi) < 0.01

    def test_computes_durations(self, sample_df):
        out = FeatureEngineer().fit_transform(sample_df)
        assert out.loc[0, "surgery_duration_min"] == pytest.approx(60.0)
        assert "anesthesia_duration_min" in out.columns
        assert "induction_to_incision_min" in out.columns

    def test_missing_source_columns_skipped_gracefully(self):
        df = pd.DataFrame({"age": [30, 40]})  # no height/weight/timing cols at all
        out = FeatureEngineer().fit_transform(df)
        assert "bmi" not in out.columns  # silently skipped, no crash


# ---------------------------------------------------------------------------
# FrequencyEncoder / CategoricalEncoder
# ---------------------------------------------------------------------------

class TestFrequencyEncoder:

    def test_encodes_by_observed_frequency(self, sample_df):
        enc = FrequencyEncoder(cols=["department"]).fit(sample_df)
        out = enc.transform(sample_df)
        assert "department_freq" in out.columns
        assert "department" not in out.columns
        # "General" appears 4/6 times (indices 0, 2, 4, 5) -> freq 0.667
        general_rows = sample_df["department"] == "General"
        assert out.loc[general_rows, "department_freq"].iloc[0] == pytest.approx(4 / 6)

    def test_unseen_category_maps_to_zero(self, sample_df):
        enc = FrequencyEncoder(cols=["department"]).fit(sample_df)
        new_df = pd.DataFrame({"department": ["NeverSeenBefore"]})
        out = enc.transform(new_df)
        assert out.loc[0, "department_freq"] == 0.0

    def test_single_unique_value_column(self):
        df = pd.DataFrame({"sex": ["M", "M", "M"]})
        out = FrequencyEncoder(cols=["sex"]).fit_transform(df)
        assert (out["sex_freq"] == 1.0).all()


class TestCategoricalEncoder:

    def test_ordinal_preserves_clinical_order(self, sample_df):
        out = CategoricalEncoder().fit_transform(sample_df)
        # asa=1 should encode to a lower value than asa=3
        asa1_encoded = out.loc[sample_df["asa"] == 1, "asa"].iloc[0]
        asa3_encoded = out.loc[sample_df["asa"] == 3, "asa"].iloc[0]
        assert asa1_encoded < asa3_encoded

    def test_no_onehot_columns_created(self, sample_df):
        out = CategoricalEncoder().fit_transform(sample_df)
        onehot_style_cols = [c for c in out.columns if c.startswith("sex_") and c != "sex_freq"]
        assert onehot_style_cols == []  # confirms we're not accidentally one-hot encoding


# ---------------------------------------------------------------------------
# RobustNumericScaler
# ---------------------------------------------------------------------------

class TestRobustNumericScaler:

    def test_scaled_median_near_zero(self):
        df = pd.DataFrame({"age": [20, 30, 40, 50, 60, 200]})  # includes a real extreme value
        out = RobustNumericScaler(cols=["age"]).fit_transform(df)
        assert abs(out["age"].median()) < 0.5

    def test_missing_column_skipped_gracefully(self):
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        out = RobustNumericScaler(cols=["age"]).fit_transform(df)
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# Full pipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_end_to_end_runs_without_error(self, sample_df):
        pipeline = build_pipeline(imputation_strategy="knn")  # knn is faster for tests
        out = pipeline.fit_transform(sample_df)
        assert len(out) <= len(sample_df)  # duplicates removed
        assert out.isna().sum().sum() == 0  # nothing left missing

    def test_pipeline_handles_all_null_column_without_crashing(self, sample_df):
        df = sample_df.copy()
        df["preop_hb"] = np.nan  # entire column null
        pipeline = build_pipeline(imputation_strategy="knn")
        out = pipeline.fit_transform(df)  # should not raise
        assert out is not None


# ---------------------------------------------------------------------------
# Validation module
# ---------------------------------------------------------------------------

class TestValidationModule:

    def test_validate_schema_flags_duplicate_ids(self):
        df = pd.DataFrame({"caseid": [1, 1, 2], "age": [30, 30, 40]})
        result = validate_schema(df)
        assert result["passed"] is False
        assert any("duplicate" in issue for issue in result["issues"])

    def test_validate_no_missing_values_passes_on_clean_data(self):
        df = pd.DataFrame({"age": [30, 40, 50]})
        result = validate_no_missing_values(df)
        assert result["passed"] is True

    def test_validate_no_missing_values_fails_on_dirty_data(self):
        df = pd.DataFrame({"age": [30, np.nan, 50]})
        result = validate_no_missing_values(df)
        assert result["passed"] is False

    def test_validate_scaling_properties_passes_for_correctly_scaled_data(self):
        df = pd.DataFrame({"age": [20, 30, 40, 50, 60]})
        scaled = RobustNumericScaler(cols=["age"]).fit_transform(df)
        result = validate_scaling_properties(scaled, cols=["age"])
        assert result["passed"] is True

    def test_full_pipeline_passes_imputation_plausibility_check(self, sample_df):
        pipeline = build_pipeline(imputation_strategy="knn")
        pipeline.fit(sample_df)
        result = validate_imputed_values_plausible(pipeline, sample_df)
        # KNN only ever copies real observed values, so it can't produce
        # an out-of-bounds value here once outliers have been flagged first.
        assert result["passed"] is True