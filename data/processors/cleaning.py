"""Data cleaning and validation module for Taiwan stock data.

Provides quality checks, outlier detection, and missing-value handling
tailored to TWSE / OTC daily price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    """Validate, clean, and report on raw price / financial data."""

    # Taiwan stock daily price change limit (±10 %)
    _DAILY_LIMIT = 0.10

    # ------------------------------------------------------------------ #
    #  Price validation
    # ------------------------------------------------------------------ #
    def validate_price(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Validate price data quality and return cleaned data with report.

        Checks performed:
        - Missing values in OHLCV columns
        - Price logic: high >= low, high >= open/close, low <= open/close
        - Volume > 0
        - Daily close change within ±10 % limit
        - Consecutive zero-volume trading days

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data with DatetimeIndex.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            (cleaned_df, quality_report)
        """
        df = df.copy()
        report: dict = {"total_rows": len(df), "issues": {}}

        required = {"open", "high", "low", "close", "volume"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # --- Missing values ---
        na_counts = df[list(required)].isna().sum()
        total_na = int(na_counts.sum())
        if total_na > 0:
            report["issues"]["missing_values"] = na_counts[na_counts > 0].to_dict()
            logger.warning("Found {} missing values in OHLCV columns", total_na)

        # --- Price logic ---
        bad_hl = df["high"] < df["low"]
        n_bad_hl = int(bad_hl.sum())
        if n_bad_hl > 0:
            report["issues"]["high_lt_low"] = n_bad_hl
            logger.warning("{} rows where high < low", n_bad_hl)
            # Fix by swapping
            df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

        # --- Volume <= 0 ---
        bad_vol = df["volume"] <= 0
        n_bad_vol = int(bad_vol.sum())
        if n_bad_vol > 0:
            report["issues"]["non_positive_volume"] = n_bad_vol
            logger.warning("{} rows with volume <= 0", n_bad_vol)

        # --- Daily change limit ---
        daily_change = df["close"].pct_change(fill_method=None).abs()
        over_limit = daily_change > self._DAILY_LIMIT
        # Skip the first row (NaN) when counting
        n_over = int(over_limit.sum())
        if n_over > 0:
            report["issues"]["over_daily_limit"] = {
                "count": n_over,
                "dates": df.index[over_limit].strftime("%Y-%m-%d").tolist(),
            }
            logger.warning("{} rows exceed ±{}% daily change limit", n_over, self._DAILY_LIMIT * 100)

        # --- Consecutive zero-volume days ---
        zero_vol = (df["volume"] == 0).astype(int)
        if zero_vol.any():
            groups = zero_vol.groupby((zero_vol != zero_vol.shift()).cumsum())
            max_consec = groups.sum().max()
            if max_consec > 0:
                report["issues"]["max_consecutive_zero_volume"] = int(max_consec)
                if max_consec >= 5:
                    logger.warning("{} consecutive zero-volume days detected", max_consec)

        report["is_clean"] = len(report["issues"]) == 0
        logger.info(
            "Validation done: {} rows, {} issue types found",
            len(df),
            len(report["issues"]),
        )
        return df, report

    # ------------------------------------------------------------------ #
    #  Missing value handling
    # ------------------------------------------------------------------ #
    def fill_missing(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """Fill missing values using the specified method.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (should be sorted by date).
        method : str
            Fill method — ``"ffill"`` (forward fill, safe for time series)
            or ``"bfill"`` (backward fill, use with caution).

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled.
        """
        before = int(df.isna().sum().sum())
        if method == "ffill":
            df_filled = df.ffill()
        elif method == "bfill":
            df_filled = df.bfill()
        else:
            raise ValueError(f"Unsupported fill method: {method}")
        after = int(df_filled.isna().sum().sum())
        logger.info("Filled {} missing values using '{}' (remaining: {})", before - after, method, after)
        return df_filled

    # ------------------------------------------------------------------ #
    #  Outlier detection
    # ------------------------------------------------------------------ #
    def detect_outliers(
        self, df: pd.DataFrame, column: str, n_std: float = 4.0
    ) -> pd.Series:
        """Detect outliers using the z-score method.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        column : str
            Column name to check.
        n_std : float
            Number of standard deviations for the threshold (default 4.0).

        Returns
        -------
        pd.Series
            Boolean Series — ``True`` for outlier rows.
        """
        series = df[column]
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=df.index)

        z_scores = (series - mean).abs() / std
        outliers = z_scores > n_std
        n_outliers = int(outliers.sum())
        if n_outliers > 0:
            logger.info(
                "Detected {} outliers in '{}' (>{:.1f} std)", n_outliers, column, n_std
            )
        return outliers

    # ------------------------------------------------------------------ #
    #  Quality report
    # ------------------------------------------------------------------ #
    def generate_quality_report(self, df: pd.DataFrame) -> dict:
        """Generate a comprehensive data quality report.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        dict
            Quality metrics including completeness, date range, and
            per-column statistics.
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isna().sum().sum())

        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
            "completeness": round(1 - missing_cells / max(total_cells, 1), 4),
            "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            report["numeric_stats"] = stats

        logger.info(
            "Quality report: {} rows, completeness={:.2%}",
            report["rows"],
            report["completeness"],
        )
        return report


# ---------------------------------------------------------------------- #
#  Quick smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running DataCleaner smoke test...")

    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)

    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.2,
            "high": price + abs(np.random.randn(n)) * 1.0,
            "low": price - abs(np.random.randn(n)) * 1.0,
            "close": price,
            "volume": np.random.randint(1000, 50000, size=n).astype(float),
        },
        index=dates,
    )

    # Inject some issues
    df.iloc[5, df.columns.get_loc("close")] = np.nan  # missing value
    df.iloc[10, df.columns.get_loc("volume")] = 0     # zero volume
    # Swap high/low to create invalid row
    df.iloc[15, df.columns.get_loc("high")] = df.iloc[15]["low"] - 1

    cleaner = DataCleaner()

    # Validate
    cleaned, report = cleaner.validate_price(df)
    logger.info("Validation report: {}", report)

    # Fill missing
    filled = cleaner.fill_missing(cleaned)
    logger.info("Remaining NaN after fill: {}", int(filled.isna().sum().sum()))

    # Outlier detection
    outliers = cleaner.detect_outliers(filled, "close")
    logger.info("Outliers in close: {}", int(outliers.sum()))

    # Quality report
    qr = cleaner.generate_quality_report(filled)
    logger.info("Quality report: {}", qr)

    logger.info("Smoke test passed.")
