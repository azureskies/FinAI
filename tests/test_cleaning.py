"""Tests for data.processors.cleaning.DataCleaner."""

import numpy as np
import pandas as pd
import pytest

from data.processors.cleaning import DataCleaner


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def cleaner():
    """Return a DataCleaner instance."""
    return DataCleaner()


@pytest.fixture
def clean_ohlcv():
    """Return a clean OHLCV DataFrame with no issues."""
    np.random.seed(42)
    n = 50
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.1,
            "high": price + np.abs(np.random.randn(n)) * 0.5,
            "low": price - np.abs(np.random.randn(n)) * 0.5,
            "close": price,
            "volume": np.random.randint(1000, 50000, size=n).astype(float),
        },
        index=dates,
    )


# ------------------------------------------------------------------ #
#  validate_price
# ------------------------------------------------------------------ #

class TestValidatePrice:
    """Tests for DataCleaner.validate_price."""

    def test_clean_data_returns_is_clean_true(self, cleaner, clean_ohlcv):
        """Clean data should produce a report with is_clean=True."""
        _, report = cleaner.validate_price(clean_ohlcv)
        assert report["is_clean"] is True
        assert report["total_rows"] == len(clean_ohlcv)

    def test_missing_columns_raises(self, cleaner):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5]})
        with pytest.raises(ValueError, match="Missing required columns"):
            cleaner.validate_price(df)

    def test_detects_missing_values(self, cleaner, clean_ohlcv):
        """NaN values in OHLCV columns should be reported."""
        df = clean_ohlcv.copy()
        df.iloc[3, df.columns.get_loc("close")] = np.nan
        df.iloc[7, df.columns.get_loc("volume")] = np.nan
        _, report = cleaner.validate_price(df)
        assert "missing_values" in report["issues"]

    def test_swaps_high_low(self, cleaner, clean_ohlcv):
        """Rows where high < low should be swapped in the output."""
        df = clean_ohlcv.copy()
        idx = 5
        original_high = df.iloc[idx]["high"]
        original_low = df.iloc[idx]["low"]
        # Force high < low
        df.iloc[idx, df.columns.get_loc("high")] = original_low - 1
        df.iloc[idx, df.columns.get_loc("low")] = original_high + 1

        cleaned, report = cleaner.validate_price(df)
        assert "high_lt_low" in report["issues"]
        # After fix, high >= low
        assert cleaned.iloc[idx]["high"] >= cleaned.iloc[idx]["low"]

    def test_detects_non_positive_volume(self, cleaner, clean_ohlcv):
        """Zero or negative volume rows should be flagged."""
        df = clean_ohlcv.copy()
        df.iloc[2, df.columns.get_loc("volume")] = 0
        df.iloc[4, df.columns.get_loc("volume")] = -100
        _, report = cleaner.validate_price(df)
        assert report["issues"]["non_positive_volume"] == 2

    def test_detects_over_daily_limit(self, cleaner, clean_ohlcv):
        """Price changes exceeding 10% should be flagged."""
        df = clean_ohlcv.copy()
        # Inject a 20% jump
        df.iloc[10, df.columns.get_loc("close")] = df.iloc[9]["close"] * 1.25
        _, report = cleaner.validate_price(df)
        assert "over_daily_limit" in report["issues"]

    def test_does_not_mutate_original(self, cleaner, clean_ohlcv):
        """validate_price should not mutate the input DataFrame."""
        original = clean_ohlcv.copy()
        cleaner.validate_price(clean_ohlcv)
        pd.testing.assert_frame_equal(clean_ohlcv, original)


# ------------------------------------------------------------------ #
#  fill_missing
# ------------------------------------------------------------------ #

class TestFillMissing:
    """Tests for DataCleaner.fill_missing."""

    def test_ffill(self, cleaner, clean_ohlcv):
        """Forward fill should propagate previous values."""
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        filled = cleaner.fill_missing(df, method="ffill")
        assert not np.isnan(filled.iloc[5]["close"])
        assert filled.iloc[5]["close"] == df.iloc[4]["close"]

    def test_bfill(self, cleaner, clean_ohlcv):
        """Backward fill should propagate next values."""
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        filled = cleaner.fill_missing(df, method="bfill")
        assert not np.isnan(filled.iloc[5]["close"])
        assert filled.iloc[5]["close"] == clean_ohlcv.iloc[6]["close"]

    def test_unsupported_method_raises(self, cleaner, clean_ohlcv):
        """Unsupported fill method should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported fill method"):
            cleaner.fill_missing(clean_ohlcv, method="linear")

    def test_no_missing_is_noop(self, cleaner, clean_ohlcv):
        """When no missing values exist, output should equal input."""
        filled = cleaner.fill_missing(clean_ohlcv)
        pd.testing.assert_frame_equal(filled, clean_ohlcv)

    def test_leading_nan_remains_with_ffill(self, cleaner, clean_ohlcv):
        """Leading NaN cannot be forward-filled and should remain NaN."""
        df = clean_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("close")] = np.nan
        filled = cleaner.fill_missing(df, method="ffill")
        assert np.isnan(filled.iloc[0]["close"])


# ------------------------------------------------------------------ #
#  detect_outliers
# ------------------------------------------------------------------ #

class TestDetectOutliers:
    """Tests for DataCleaner.detect_outliers."""

    def test_no_outliers_in_normal_data(self, cleaner, clean_ohlcv):
        """Normal data should produce no outliers with default n_std=4."""
        outliers = cleaner.detect_outliers(clean_ohlcv, "close")
        assert outliers.sum() == 0

    def test_detects_extreme_value(self, cleaner, clean_ohlcv):
        """An extreme value should be flagged as an outlier."""
        df = clean_ohlcv.copy()
        mean = df["close"].mean()
        std = df["close"].std()
        df.iloc[10, df.columns.get_loc("close")] = mean + 10 * std
        outliers = cleaner.detect_outliers(df, "close")
        assert outliers.iloc[10] is True or outliers.iloc[10]

    def test_zero_std_returns_all_false(self, cleaner):
        """Constant column (std=0) should produce no outliers."""
        df = pd.DataFrame({"x": [5.0] * 20})
        outliers = cleaner.detect_outliers(df, "x")
        assert outliers.sum() == 0

    def test_custom_n_std(self, cleaner, clean_ohlcv):
        """Lower n_std should catch more outliers."""
        outliers_strict = cleaner.detect_outliers(clean_ohlcv, "close", n_std=1.0)
        outliers_loose = cleaner.detect_outliers(clean_ohlcv, "close", n_std=4.0)
        assert outliers_strict.sum() >= outliers_loose.sum()

    def test_returns_boolean_series(self, cleaner, clean_ohlcv):
        """Return type should be a boolean pd.Series."""
        outliers = cleaner.detect_outliers(clean_ohlcv, "close")
        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool


# ------------------------------------------------------------------ #
#  generate_quality_report
# ------------------------------------------------------------------ #

class TestGenerateQualityReport:
    """Tests for DataCleaner.generate_quality_report."""

    def test_report_structure(self, cleaner, clean_ohlcv):
        """Report should contain all expected keys."""
        report = cleaner.generate_quality_report(clean_ohlcv)
        assert "rows" in report
        assert "columns" in report
        assert "date_range" in report
        assert "completeness" in report

    def test_completeness_is_one_for_clean_data(self, cleaner, clean_ohlcv):
        """Full data should have completeness == 1.0."""
        report = cleaner.generate_quality_report(clean_ohlcv)
        assert report["completeness"] == 1.0

    def test_completeness_decreases_with_nan(self, cleaner, clean_ohlcv):
        """Adding NaNs should lower the completeness score."""
        df = clean_ohlcv.copy()
        df.iloc[0:5, df.columns.get_loc("close")] = np.nan
        report = cleaner.generate_quality_report(df)
        assert report["completeness"] < 1.0

    def test_empty_dataframe(self, cleaner):
        """Empty DataFrame should still produce a valid report."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index.name = "date"
        report = cleaner.generate_quality_report(df)
        assert report["rows"] == 0
        assert report["completeness"] == 1.0  # 0/0 -> 1.0

    def test_numeric_stats_present(self, cleaner, clean_ohlcv):
        """Report should contain numeric_stats for numeric columns."""
        report = cleaner.generate_quality_report(clean_ohlcv)
        assert "numeric_stats" in report
        assert "close" in report["numeric_stats"]
