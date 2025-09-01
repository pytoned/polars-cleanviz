from __future__ import annotations
import polars as pl
from great_tables import GT
from typing import Union
import numpy as np
import time
import sys

# Optional scipy imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


def xray(
    df: pl.DataFrame,
    *,
    great_tables: bool = True,
    expanded: bool = False,
    percentiles: list[float] | None = None,
    outlier_method: str = "iqr",
    outlier_bounds: list[float] | None = None,
    corr_target: str | None = None,
    normality_test: str = "shapiro",
    uniformity_test: str = "ks",
    missing_threshold: float = 0.3,
    constant_threshold: float = 0.99,
    skew_threshold: float = 2.0,
    kurtosis_threshold: float = 7.0,
    outlier_threshold: float = 0.05,
    shakiness_threshold: int = 2,
    decimals: int = 2,
    sep_mark: str = ",",
    dec_mark: str = ".",
    compact: bool = False,
    pattern: str | None = None,
    locale: str | None = None
) -> Union[GT, pl.DataFrame]:
    """
    X-ray your data: comprehensive statistical analysis with quality assessment.
    
    This function provides deep insight into DataFrame structure and quality,
    revealing hidden issues, statistical properties, and data health indicators.
    Perfect for exploratory data analysis and data quality assessment.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to summarize.
    great_tables : bool, default True
        Whether to return a formatted Great Tables object (True) or standard
        Polars DataFrame output (False).
    expanded : bool, default False
        If True, shows all available statistics. If False, shows only essential
        metrics: dtype, count, mean, min, Q25, median, Q75, max, IQR, 
        pct_missing, n_outliers.
    percentiles : list[float] | None, optional
        Custom percentiles to calculate. Default: [0.25, 0.5, 0.75].
        Example: [0.1, 0.25, 0.5, 0.75, 0.9] for additional quantiles.
    outlier_method : str, default "iqr"
        Method for outlier detection:
        - "iqr": Interquartile range method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - "percentile": Use custom percentile bounds from outlier_bounds
        - "zscore": Z-score method (mean ± 3*std)
    outlier_bounds : list[float] | None, optional
        Custom percentile bounds for outlier detection when method="percentile".
        Example: [0.05, 0.95] for 5th and 95th percentiles.
    corr_target : str | None, optional
        Target column for correlation analysis. Must be a numeric column.
        Shows correlation between each numeric column and the target.
    normality_test : str, default "shapiro"
        Statistical test for normality (requires scipy):
        - "shapiro": Shapiro-Wilk test (good for small/medium samples)
        - "anderson": Anderson-Darling test (sensitive to tail behavior)
        - "ks": Kolmogorov-Smirnov test vs normal distribution
    uniformity_test : str, default "ks"
        Statistical test for uniformity (requires scipy):
        - "ks": Kolmogorov-Smirnov test vs uniform distribution
        - "chi2": Chi-square goodness of fit test
    missing_threshold : float, default 0.3
        Threshold for flagging high missingness (0.3 = 30%).
    constant_threshold : float, default 0.99
        Threshold for flagging quasi-constant columns (0.99 = 99% same value).
    skew_threshold : float, default 2.0
        Threshold for flagging extreme skewness.
    kurtosis_threshold : float, default 7.0
        Threshold for flagging high kurtosis (fat tails).
    outlier_threshold : float, default 0.05
        Threshold for flagging outlier-heavy columns (0.05 = 5%).
    shakiness_threshold : int, default 2
        Minimum score to flag column as "shaky" for parametric models.
    decimals : int, default 2
        Number of decimal places for numeric formatting in Great Tables output.
    sep_mark : str, default ","
        Thousands separator mark for numeric formatting (e.g., "1,000").
    dec_mark : str, default "."
        Decimal mark for numeric formatting (e.g., "1.23").
    compact : bool, default False
        If True, large numbers are auto-scaled with suffixes (e.g., "10K", "1.5M").
    pattern : str | None, optional
        Text pattern for decorating formatted values (e.g., "[{x}]").
    locale : str | None, optional
        Locale ID for region-specific formatting (e.g., "en", "fr", "de-AT").
        Overrides sep_mark and dec_mark when specified.

    Returns
    -------
    Union[GT, pl.DataFrame]
        Either a Great Tables object (if great_tables=True) or a Polars DataFrame
        (if great_tables=False) containing the comprehensive summary statistics.

    Examples
    --------
    Basic data X-ray (minimal metrics):
    
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({
    ...     'price': [100, 200, 150, 300, 250],
    ...     'volume': [1000, 1500, 1200, 2000, 1800],
    ...     'rating': [4.5, 3.8, 4.2, 4.9, 4.1]
    ... })
    >>> table = plc.xray(df)
    >>> table.show()
    
    Comprehensive data analysis:
    
    >>> full_xray = plc.xray(df, expanded=True)
    
    With correlation and custom tests:
    
    >>> advanced_xray = plc.xray(
    ...     df, 
    ...     expanded=True,
    ...     corr_target='price',
    ...     normality_test='anderson',
    ...     percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ... )
    
    Custom quality thresholds:
    
    >>> quality_xray = plc.xray(
    ...     df,
    ...     missing_threshold=0.2,  # Flag >20% missing
    ...     skew_threshold=1.5,     # Flag |skew| > 1.5
    ...     shakiness_threshold=1   # Flag any quality issue
    ... )
    
    Advanced formatting options:
    
    >>> formatted_xray = plc.xray(
    ...     df,
    ...     decimals=3,            # 3 decimal places
    ...     compact=True,          # Use "10K" instead of "10,000"
    ...     locale="fr",           # French formatting
    ...     pattern="({x})"        # Wrap values in parentheses
    ... )

    Notes
    -----
    The shakiness score combines multiple data quality indicators:
    - High missingness (> threshold)
    - Constant/quasi-constant values
    - Extreme skewness
    - High outlier percentage
    - Failed normality tests
    
    Columns with shakiness_score >= shakiness_threshold are flagged as "⚠ SHAKY"
    for potential issues with parametric statistical models.
    
    Statistical tests require scipy. If not available, tests are skipped
    with informative messages.
    """
    # Start timing for performance measurement
    start_time = time.perf_counter()
    
    # Calculate DataFrame memory usage
    memory_usage_bytes = df.estimated_size("bytes")
    memory_usage_mb = memory_usage_bytes / 1024 / 1024
    
    # Set default percentiles
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    
    # Validate parameters
    if outlier_method not in ["iqr", "percentile", "zscore"]:
        raise ValueError("outlier_method must be 'iqr', 'percentile', or 'zscore'")
    
    if outlier_method == "percentile" and not outlier_bounds:
        raise ValueError("outlier_bounds must be provided when outlier_method='percentile'")
    
    if outlier_bounds and len(outlier_bounds) != 2:
        raise ValueError("outlier_bounds must be a list of exactly 2 values")
    
    if normality_test not in ["shapiro", "anderson", "ks"]:
        raise ValueError("normality_test must be 'shapiro', 'anderson', or 'ks'")
    
    if uniformity_test not in ["ks", "chi2"]:
        raise ValueError("uniformity_test must be 'ks' or 'chi2'")
    
    # Validate correlation target
    if corr_target:
        if corr_target not in df.columns:
            raise ValueError(f"Target column '{corr_target}' not found in DataFrame")
        target_dtype = df.select(pl.col(corr_target)).dtypes[0]
        if not target_dtype.is_numeric():
            raise ValueError(f"Target column '{corr_target}' must be numeric, got {target_dtype}")
    
    # Get all columns for analysis (both numeric and non-numeric)
    all_cols = df.columns
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    
    # Calculate comprehensive statistics for all columns
    stats_data = []
    
    for col in all_cols:
        dtype = df.select(pl.col(col)).dtypes[0]
        is_numeric = dtype.is_numeric()
        series = df[col]
        series_clean = series.drop_nulls()
        n_total = len(series)
        n_valid = len(series_clean)
        n_missing = series.null_count()
        pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0
        
        # Initialize column stats
        col_stats = {
            'Column': col,
            'Dtype': str(dtype),
            'Count': n_valid,
            'N_Missing': n_missing,
            'Pct_Missing': round(pct_missing, 2)
        }
        
        # Basic counts and ratios
        n_unique = series.n_unique()
        col_stats['N_Unique'] = n_unique
        col_stats['Uniqueness_Ratio'] = round(n_unique / n_total, 4) if n_total > 0 else 0
        
        if is_numeric and n_valid > 0:
            # Numeric-specific statistics
            
            # Basic descriptive stats
            quantile_stats = _calculate_quantiles(series_clean, percentiles)
            col_stats.update(quantile_stats)
            
            mean_val = float(series_clean.mean())
            std_val = float(series_clean.std()) if n_valid > 1 else None
            min_val = float(series_clean.min())
            max_val = float(series_clean.max())
            
            col_stats['Mean'] = round(mean_val, 3)
            col_stats['Std'] = round(std_val, 3) if std_val is not None else None
            col_stats['Min'] = round(min_val, 3)
            col_stats['Max'] = round(max_val, 3)
            
            # IQR
            if 0.25 in percentiles and 0.75 in percentiles:
                q25 = quantile_stats.get('Q25')
                q75 = quantile_stats.get('Q75') 
                if q25 is not None and q75 is not None:
                    col_stats['IQR'] = round(q75 - q25, 3)
            
            # Zero/positive/negative counts
            n_zero = int((series_clean == 0).sum())
            n_pos = int((series_clean > 0).sum())
            n_neg = int((series_clean < 0).sum())
            
            col_stats['N_Zero'] = n_zero
            col_stats['Pct_Zero'] = round(n_zero / n_valid * 100, 2) if n_valid > 0 else 0
            col_stats['Pct_Pos'] = round(n_pos / n_valid * 100, 2) if n_valid > 0 else 0
            col_stats['Pct_Neg'] = round(n_neg / n_valid * 100, 2) if n_valid > 0 else 0
            
            # Advanced statistics (expanded mode)
            if expanded:
                # MAD (Median Absolute Deviation)
                if n_valid > 0:
                    median_val = float(series_clean.median())
                    mad = (series_clean - median_val).abs().median()
                    col_stats['MAD'] = round(float(mad), 3)
                
                # Skewness and Kurtosis
                if n_valid > 2:
                    skew_val = float(series_clean.skew())
                    col_stats['Skewness'] = round(skew_val, 3)
                    
                    # Calculate kurtosis (excess kurtosis)
                    kurt_val = _calculate_kurtosis(series_clean.to_numpy())
                    col_stats['Kurtosis'] = round(kurt_val, 3)
                
                # Optimal dtype suggestion
                col_stats['Opt_Dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                
                # Statistical tests (if scipy available)
                if SCIPY_AVAILABLE and n_valid > 3:
                    normality_result = _test_normality(series_clean.to_numpy(), normality_test)
                    col_stats['Normality_Test'] = normality_result
                    
                    uniformity_result = _test_uniformity(series_clean.to_numpy(), uniformity_test)
                    col_stats['Uniformity_Test'] = uniformity_result
            
            # Outlier detection
            n_outliers = _count_outliers(series_clean, outlier_method, outlier_bounds)
            pct_outliers = (n_outliers / n_valid * 100) if n_valid > 0 else 0
            col_stats['N_Outliers'] = n_outliers
            col_stats['Pct_Outliers'] = round(pct_outliers, 2)
            
            # Correlation with target
            if corr_target and col != corr_target:
                try:
                    corr_val = df.select([pl.corr(corr_target, col)]).item()
                    col_stats['Correlation'] = round(corr_val, 3) if corr_val is not None else None
                except:
                    col_stats['Correlation'] = None
        
        else:
            # Non-numeric columns - set numeric stats to None/0
            for stat in ['Mean', 'Std', 'Min', 'Max', 'IQR', 'N_Zero', 'Pct_Zero', 
                        'Pct_Pos', 'Pct_Neg', 'N_Outliers', 'Pct_Outliers']:
                col_stats[stat] = None if stat in ['Mean', 'Std', 'IQR'] else 0
            
            # For quantiles, set based on percentiles
            for p in percentiles:
                label = _percentile_to_label(p)
                col_stats[label] = None
            
            if expanded:
                col_stats['MAD'] = None
                col_stats['Skewness'] = None  
                col_stats['Kurtosis'] = None
                col_stats['Opt_Dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                col_stats['Normality_Test'] = "N/A (non-numeric)"
                col_stats['Uniformity_Test'] = "N/A (non-numeric)"
        
        # Calculate shakiness score
        shakiness_score = _calculate_shakiness_score(
            col_stats, missing_threshold, constant_threshold, 
            skew_threshold, kurtosis_threshold, outlier_threshold
        )
        col_stats['Shakiness_Score'] = shakiness_score
        col_stats['Quality_Flag'] = "⚠ SHAKY" if shakiness_score >= shakiness_threshold else "✓ OK"
        
        stats_data.append(col_stats)
    
    # Create DataFrame
    summary_df = pl.DataFrame(stats_data)
    
    # Return standard DataFrame if great_tables=False
    if not great_tables:
        return summary_df
    
    # Build Great Tables object
    if expanded:
        # Full statistics mode
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return _build_expanded_gt_table(summary_df, df.height, df.width, memory_usage_mb, execution_time_ms, corr_target, percentiles, decimals, sep_mark, dec_mark, compact, pattern, locale)
    else:
        # Minimal mode - only essential columns
        essential_cols = ['Column', 'Dtype', 'Count', 'Mean', 'Min', 'Median', 'Max', 
                         'IQR', 'Pct_Missing', 'N_Outliers', 'Quality_Flag']
        
        # Add quantile columns that exist
        quantile_cols = []
        for p in percentiles:
            label = _percentile_to_label(p)
            if label in summary_df.columns and label not in essential_cols:
                quantile_cols.append(label)
        
        if corr_target:
            essential_cols.append('Correlation')
        
        # Select only essential columns that exist
        available_cols = [c for c in essential_cols if c in summary_df.columns]
        available_cols += quantile_cols
        
        minimal_df = summary_df.select(available_cols)
        
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return _build_minimal_gt_table(minimal_df, df.height, df.width, memory_usage_mb, execution_time_ms, corr_target, decimals, sep_mark, dec_mark, compact, pattern, locale)


# Helper Functions

def _percentile_to_label(p: float) -> str:
    """Convert percentile float to column label."""
    if p == 0.25:
        return "Q25"
    elif p == 0.5:
        return "Median"
    elif p == 0.75:
        return "Q75"
    else:
        return f"P{int(p*100)}"


def _calculate_quantiles(series: pl.Series, percentiles: list[float]) -> dict:
    """Calculate quantiles for a series."""
    quantile_stats = {}
    for p in percentiles:
        label = _percentile_to_label(p)
        if len(series) > 0:
            val = float(series.quantile(p))
            quantile_stats[label] = round(val, 3)
        else:
            quantile_stats[label] = None
    return quantile_stats


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis (kurtosis - 3)."""
    if len(data) < 4:
        return np.nan
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return np.nan
    
    # Calculate fourth moment
    fourth_moment = np.mean((data - mean) ** 4)
    kurtosis = fourth_moment / (std ** 4)
    
    # Return excess kurtosis (subtract 3 for normal distribution)
    return kurtosis - 3


def _suggest_optimal_dtype(series: pl.Series, current_dtype) -> str:
    """Suggest optimal data type for a series."""
    if len(series) == 0:
        return str(current_dtype)
    
    # Check if boolean
    unique_vals = set(series.unique().to_list())
    if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
        return "Bool"
    
    if current_dtype.is_integer():
        min_val = int(series.min())
        max_val = int(series.max())
        
        # Check if fits in smaller integer types
        if -128 <= min_val <= 127 and -128 <= max_val <= 127:
            return "Int8"
        elif -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767:
            return "Int16"
        elif -2147483648 <= min_val <= 2147483647:
            return "Int32"
        else:
            return "Int64"
    
    elif current_dtype.is_float():
        # For floats, check if values could be represented as integers
        if all(x.is_integer() for x in series.to_list() if not np.isnan(x)):
            # Could be integer, suggest Int64
            return "Int64"
        else:
            # Check precision requirements
            return "Float32"  # Usually sufficient for most data
    
    elif current_dtype == pl.String:
        # Check if categorical would be better
        n_unique = series.n_unique()
        n_total = len(series)
        if n_unique / n_total < 0.5:  # Less than 50% unique
            return "Categorical"
        return "String"
    
    return str(current_dtype)


def _test_normality(data: np.ndarray, test_type: str) -> str:
    """Perform normality test and return formatted result."""
    if not SCIPY_AVAILABLE:
        return "N/A (scipy not available)"
    
    if len(data) < 3:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "shapiro":
            if len(data) > 5000:
                # Shapiro-Wilk test not reliable for large samples
                return "N/A (sample too large)"
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
            
        elif test_type == "anderson":
            result = stats.anderson(data, dist='norm')
            # Anderson-Darling uses critical values, not p-values
            # Use 5% significance level (index 2 in critical_values)
            is_normal = result.statistic < result.critical_values[2]
            p_value = None  # A-D doesn't give exact p-value
            test_name = "Anderson-Darling"
            
        elif test_type == "ks":
            # Lilliefors test (KS test with estimated parameters)
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            test_name = "Kolmogorov-Smirnov"
        
        # Format result
        if test_type == "anderson":
            result_str = "NORMAL" if is_normal else "NON-NORMAL"
            return f"{result_str} ({test_name})"
        else:
            alpha = 0.05
            is_normal = p_value > alpha
            result_str = "NORMAL" if is_normal else "NON-NORMAL"
            return f"{result_str} (p={p_value:.3f})"
            
    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"


def _test_uniformity(data: np.ndarray, test_type: str) -> str:
    """Perform uniformity test and return formatted result."""
    if not SCIPY_AVAILABLE:
        return "N/A (scipy not available)"
    
    if len(data) < 5:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "ks":
            # KS test against uniform distribution
            min_val, max_val = np.min(data), np.max(data)
            if min_val == max_val:
                return "N/A (constant data)"
            
            # Normalize to [0,1] for uniform test
            normalized = (data - min_val) / (max_val - min_val)
            stat, p_value = stats.kstest(normalized, 'uniform')
            test_name = "KS"
            
        elif test_type == "chi2":
            # Chi-square goodness of fit test
            # Create bins and expected frequencies
            n_bins = min(10, int(np.sqrt(len(data))))
            observed, bin_edges = np.histogram(data, bins=n_bins)
            expected = np.full(n_bins, len(data) / n_bins)
            
            # Remove bins with very low expected frequency
            mask = expected >= 5
            if np.sum(mask) < 2:
                return "N/A (insufficient bins)"
            
            stat, p_value = stats.chisquare(observed[mask], expected[mask])
            test_name = "Chi-square"
        
        # Format result
        alpha = 0.05
        is_uniform = p_value > alpha
        result_str = "UNIFORM" if is_uniform else "NON-UNIFORM"
        return f"{result_str} (p={p_value:.3f})"
        
    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"


def _calculate_shakiness_score(
    col_stats: dict, 
    missing_threshold: float,
    constant_threshold: float,
    skew_threshold: float,
    kurtosis_threshold: float,
    outlier_threshold: float
) -> int:
    """Calculate shakiness score based on data quality indicators."""
    score = 0
    
    # High missingness
    if col_stats.get('Pct_Missing', 0) > missing_threshold * 100:
        score += 1
    
    # Constant/quasi-constant
    uniqueness_ratio = col_stats.get('Uniqueness_Ratio', 1)
    if uniqueness_ratio == 0 or col_stats.get('N_Unique', 0) == 1:
        score += 1
    elif uniqueness_ratio < (1 - constant_threshold):  # Mode percentage > threshold
        score += 1
    
    # ID-like (too many unique values)
    if uniqueness_ratio > 0.95:  # More than 95% unique
        score += 1
    
    # Extreme skewness
    skewness = col_stats.get('Skewness')
    if skewness is not None and abs(skewness) > skew_threshold:
        score += 1
    
    # High kurtosis
    kurtosis = col_stats.get('Kurtosis')
    if kurtosis is not None and abs(kurtosis) > kurtosis_threshold:
        score += 1
    
    # Outlier-heavy
    pct_outliers = col_stats.get('Pct_Outliers', 0)
    if pct_outliers > outlier_threshold * 100:
        score += 1
    
    # Failed normality test
    normality_test = col_stats.get('Normality_Test', '')
    if 'NON-NORMAL' in normality_test:
        score += 1
    
    return score


def _count_outliers(series: pl.Series, method: str, bounds: list[float] | None) -> int:
    """Count outliers in a series using specified method."""
    if len(series) == 0:
        return 0
    
    if method == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
    elif method == "percentile":
        lower_bound = series.quantile(bounds[0])
        upper_bound = series.quantile(bounds[1])
        
    elif method == "zscore":
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return 0
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
    
    # Count values outside bounds using boolean indexing
    mask = (series < lower_bound) | (series > upper_bound)
    return int(mask.sum())


def _build_minimal_gt_table(
    summary_df: pl.DataFrame, 
    n_rows: int,
    n_cols: int,
    memory_mb: float,
    execution_ms: float,
    corr_target: str | None,
    decimals: int,
    sep_mark: str,
    dec_mark: str,
    compact: bool,
    pattern: str | None,
    locale: str | None
) -> GT:
    """Build minimal Great Tables object."""
    # Determine column organization
    basic_cols = ["Dtype", "Count", "Mean", "Min", "Q25", "Median", "Q75", "Max"]
    essential_cols = ["IQR", "Pct_Missing", "N_Outliers"]
    quality_cols = ["Quality_Flag"]
    
    # Filter to existing columns
    basic_cols = [c for c in basic_cols if c in summary_df.columns]
    essential_cols = [c for c in essential_cols if c in summary_df.columns] 
    quality_cols = [c for c in quality_cols if c in summary_df.columns]
    
    gt_table = (
        GT(summary_df)
        .tab_header(
            title="🔬 DataFrame X-ray",
            subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({memory_mb:.1f} MB in memory) - X-rayed in {execution_ms:.0f} ms"
        )
        .tab_spanner(label="Basic Statistics", columns=basic_cols)
        .tab_spanner(label="Key Metrics", columns=essential_cols)
        .tab_spanner(label="Quality", columns=quality_cols)
        .fmt_integer(columns=["Count", "N_Outliers"], sep_mark=sep_mark, locale=locale)
        .fmt_number(
            columns=[c for c in ["Mean", "Min", "Q25", "Median", "Q75", "Max", "IQR"] if c in summary_df.columns], 
            decimals=decimals, 
            sep_mark=sep_mark, 
            dec_mark=dec_mark,
            compact=compact,
            pattern=pattern,
            locale=locale
        )
        .fmt_number(columns=["Pct_Missing"], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
        .cols_align(align="center", columns=basic_cols + essential_cols + quality_cols)
        .cols_align(align="left", columns=["Column"])
        .tab_options(
            table_font_size="13px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef"
        )
    )
    
    # Add correlation if specified
    if corr_target and "Correlation" in summary_df.columns:
        gt_table = (
            gt_table
            .tab_spanner(label=f"Correlation with '{corr_target}'", columns=["Correlation"])
            .fmt_number(columns=["Correlation"], decimals=3, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
            .cols_align(align="center", columns=["Correlation"])
        )
    
    return gt_table


def _build_expanded_gt_table(
    summary_df: pl.DataFrame, 
    n_rows: int,
    n_cols: int,
    memory_mb: float,
    execution_ms: float,
    corr_target: str | None, 
    percentiles: list[float],
    decimals: int,
    sep_mark: str,
    dec_mark: str,
    compact: bool,
    pattern: str | None,
    locale: str | None
) -> GT:
    """Build expanded Great Tables object with all statistics."""
    # Organize columns by category
    basic_cols = ["Dtype", "Count", "Mean", "Std", "Min", "Max"]
    quantile_cols = [_percentile_to_label(p) for p in percentiles if _percentile_to_label(p) in summary_df.columns]
    
    distribution_cols = ["IQR", "Skewness", "Kurtosis", "MAD"]
    count_cols = ["N_Missing", "Pct_Missing", "N_Unique", "Uniqueness_Ratio", "N_Zero", "Pct_Zero", "Pct_Pos", "Pct_Neg"]
    outlier_cols = ["N_Outliers", "Pct_Outliers"]
    test_cols = ["Normality_Test", "Uniformity_Test"]
    quality_cols = ["Opt_Dtype", "Shakiness_Score", "Quality_Flag"]
    
    # Filter to existing columns
    basic_cols = [c for c in basic_cols if c in summary_df.columns]
    distribution_cols = [c for c in distribution_cols if c in summary_df.columns]
    count_cols = [c for c in count_cols if c in summary_df.columns]
    outlier_cols = [c for c in outlier_cols if c in summary_df.columns]
    test_cols = [c for c in test_cols if c in summary_df.columns]
    quality_cols = [c for c in quality_cols if c in summary_df.columns]
    
    gt_table = (
        GT(summary_df)
        .tab_header(
            title="🔬 Comprehensive DataFrame X-ray",
            subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({memory_mb:.1f} MB in memory) - X-rayed in {execution_ms:.0f} ms • Deep Quality Analysis"
        )
        .tab_spanner(label="Basic Statistics", columns=basic_cols)
        .tab_spanner(label="Quantiles", columns=quantile_cols)
        .tab_spanner(label="Distribution", columns=distribution_cols)
        .tab_spanner(label="Counts & Ratios", columns=count_cols)
        .tab_spanner(label="Outliers", columns=outlier_cols)
        .tab_spanner(label="Statistical Tests", columns=test_cols)
        .tab_spanner(label="Quality Assessment", columns=quality_cols)
        .fmt_integer(columns=["Count", "N_Missing", "N_Unique", "N_Zero", "N_Outliers", "Shakiness_Score"], sep_mark=sep_mark, locale=locale)
        .fmt_number(
            columns=["Mean", "Std", "Min", "Max", "IQR", "MAD"] + quantile_cols, 
            decimals=decimals, 
            sep_mark=sep_mark, 
            dec_mark=dec_mark,
            compact=compact,
            pattern=pattern,
            locale=locale
        )
        .fmt_number(columns=["Skewness", "Kurtosis"], decimals=3, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
        .fmt_number(columns=["Pct_Missing", "Pct_Zero", "Pct_Pos", "Pct_Neg", "Pct_Outliers"], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
        .fmt_number(columns=["Uniqueness_Ratio"], decimals=4, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
        .cols_align(align="center", columns=basic_cols + quantile_cols + distribution_cols + count_cols + outlier_cols + ["Shakiness_Score"])
        .cols_align(align="left", columns=["Column", "Opt_Dtype", "Quality_Flag"] + test_cols)
        .tab_options(
            table_font_size="12px",
            heading_background_color="#f8f9fa", 
            column_labels_background_color="#e9ecef"
        )
    )
    
    # Add correlation if specified
    if corr_target and "Correlation" in summary_df.columns:
        gt_table = (
            gt_table
            .tab_spanner(label=f"Correlation with '{corr_target}'", columns=["Correlation"])
            .fmt_number(columns=["Correlation"], decimals=3, sep_mark=sep_mark, dec_mark=dec_mark, locale=locale)
            .cols_align(align="center", columns=["Correlation"])
        )
    
    return gt_table