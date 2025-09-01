from __future__ import annotations
import polars as pl
from great_tables import GT
from typing import Union


def summary(
    df: pl.DataFrame,
    *,
    great_tables: bool = True,
    outlier_method: str = "iqr",
    outlier_bounds: list[float] | None = None,
    corr_target: str | None = None
) -> Union[GT, pl.DataFrame]:
    """
    Generate an enhanced summary table of DataFrame statistics.
    
    This function creates a comprehensive statistical summary with outlier detection,
    skewness analysis, and optional correlation analysis, beautifully formatted
    with Great Tables or returned as standard Polars output.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to summarize.
    great_tables : bool, default True
        Whether to return a formatted Great Tables object (True) or standard
        Polars DataFrame output (False).
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

    Returns
    -------
    Union[GT, pl.DataFrame]
        Either a Great Tables object (if great_tables=True) or a Polars DataFrame
        (if great_tables=False) containing the enhanced summary statistics.

    Examples
    --------
    Basic usage with Great Tables formatting:
    
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({
    ...     'price': [100, 200, 150, 300, 250],
    ...     'volume': [1000, 1500, 1200, 2000, 1800],
    ...     'rating': [4.5, 3.8, 4.2, 4.9, 4.1]
    ... })
    >>> table = plc.summary(df)
    >>> table.show()  # Display in notebook or browser
    
    Standard Polars output:
    
    >>> standard_summary = plc.summary(df, great_tables=False)
    >>> print(standard_summary)
    
    With correlation analysis:
    
    >>> corr_summary = plc.summary(df, corr_target='price')
    
    Custom outlier detection:
    
    >>> custom_summary = plc.summary(df, outlier_method='percentile', 
    ...                              outlier_bounds=[0.1, 0.9])
    """
    # Validate parameters
    if outlier_method not in ["iqr", "percentile", "zscore"]:
        raise ValueError("outlier_method must be 'iqr', 'percentile', or 'zscore'")
    
    if outlier_method == "percentile" and not outlier_bounds:
        raise ValueError("outlier_bounds must be provided when outlier_method='percentile'")
    
    if outlier_bounds and len(outlier_bounds) != 2:
        raise ValueError("outlier_bounds must be a list of exactly 2 values")
    
    # Validate correlation target
    if corr_target:
        if corr_target not in df.columns:
            raise ValueError(f"Target column '{corr_target}' not found in DataFrame")
        target_dtype = df.select(pl.col(corr_target)).dtypes[0]
        if not target_dtype.is_numeric():
            raise ValueError(f"Target column '{corr_target}' must be numeric, got {target_dtype}")
    
    # Get only numeric columns for analysis
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    
    if not numeric_cols:
        # Handle case with no numeric columns - return basic info
        if great_tables:
            basic_info = []
            for col in df.columns:
                basic_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Count': len(df[col]),
                    'Unique': df[col].n_unique(),
                    'Missing': df[col].null_count()
                })
            
            summary_df = pl.DataFrame(basic_info)
            return GT(summary_df).tab_header(
                title="📊 Data Summary",
                subtitle=f"Dataset: {df.height:,} rows × {df.width} columns (no numeric columns)"
            )
        else:
            # Return standard Polars-style output
            return df.select([pl.col(c).dtype.name().alias(c) for c in df.columns])
    
    # If great_tables=False, return enhanced standard describe
    if not great_tables:
        df_numeric = df.select(numeric_cols)
        describe_df = df_numeric.describe()
        
        # Add enhanced statistics to standard describe
        enhanced_stats = []
        for col in numeric_cols:
            series = df_numeric[col].drop_nulls()
            
            # Calculate IQR
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            
            # Calculate skewness using Polars native method
            skew_val = series.skew() if len(series) > 2 else None
            
            # Calculate outliers
            n_outliers = _count_outliers(series, outlier_method, outlier_bounds)
            
            # Calculate correlation if target specified
            corr_val = None
            if corr_target and col != corr_target:
                try:
                    corr_val = df.select([pl.corr(corr_target, col)]).item()
                except:
                    corr_val = None
            
            enhanced_stats.append({
                'statistic': 'iqr',
                col: iqr
            })
            enhanced_stats.append({
                'statistic': 'skewness', 
                col: skew_val
            })
            enhanced_stats.append({
                'statistic': 'n_outliers',
                col: n_outliers
            })
            if corr_val is not None:
                enhanced_stats.append({
                    'statistic': f'corr_with_{corr_target}',
                    col: corr_val
                })
        
        # Combine with original describe
        enhanced_df = pl.DataFrame(enhanced_stats)
        return pl.concat([describe_df, enhanced_df])
    
    # Great Tables formatting
    df_numeric = df.select(numeric_cols)
    describe_df = df_numeric.describe()
    
    # Transform to enhanced format for display
    stats_data = []
    
    for col in numeric_cols:
        series = df_numeric[col].drop_nulls()
        
        # Basic stats from describe
        col_stats = {
            'Column': col,
            'Count': int(describe_df.filter(pl.col('statistic') == 'count')[col].item()),
            'Mean': float(describe_df.filter(pl.col('statistic') == 'mean')[col].item()),
            'Std': float(describe_df.filter(pl.col('statistic') == 'std')[col].item()),
            'Min': float(describe_df.filter(pl.col('statistic') == 'min')[col].item()),
            'Q25': float(describe_df.filter(pl.col('statistic') == '25%')[col].item()),
            'Median': float(describe_df.filter(pl.col('statistic') == '50%')[col].item()),
            'Q75': float(describe_df.filter(pl.col('statistic') == '75%')[col].item()),
            'Max': float(describe_df.filter(pl.col('statistic') == 'max')[col].item()),
        }
        
        # Enhanced statistics
        # IQR (Inter Quartile Range)
        col_stats['IQR'] = col_stats['Q75'] - col_stats['Q25']
        
        # Skewness using Polars native method
        col_stats['Skewness'] = float(series.skew()) if len(series) > 2 else None
        
        # Outlier count
        col_stats['N_Outliers'] = _count_outliers(series, outlier_method, outlier_bounds)
        
        # Correlation with target if specified
        if corr_target and col != corr_target:
            try:
                corr_val = df.select([pl.corr(corr_target, col)]).item()
                col_stats['Correlation'] = round(corr_val, 3) if corr_val is not None else None
            except:
                col_stats['Correlation'] = None
        
        stats_data.append(col_stats)
    
    # Create DataFrame for Great Tables
    summary_df = pl.DataFrame(stats_data)
    
    # Build enhanced Great Tables object
    basic_cols = ["Count", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"]
    enhanced_cols = ["IQR", "Skewness", "N_Outliers"]
    
    gt_table = (
        GT(summary_df)
        .tab_header(
            title="📊 Enhanced Data Summary",
            subtitle=f"Dataset: {df.height:,} rows × {len(numeric_cols)} numeric columns"
        )
        .tab_spanner(
            label="Basic Statistics",
            columns=basic_cols
        )
        .tab_spanner(
            label="Enhanced Statistics", 
            columns=enhanced_cols
        )
        .fmt_integer(columns=["Count", "N_Outliers"])
        .fmt_number(columns=["Mean", "Std", "Min", "Q25", "Median", "Q75", "Max", "IQR"], decimals=2)
        .fmt_number(columns=["Skewness"], decimals=3)
        .tab_options(
            table_font_size="13px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef",
            table_border_top_style="hidden",
            table_border_bottom_style="hidden"
        )
        .cols_align(align="center", columns=basic_cols + enhanced_cols)
        .cols_align(align="left", columns=["Column"])
    )
    
    # Add correlation spanner if target specified
    if corr_target:
        gt_table = (
            gt_table
            .tab_spanner(
                label=f"Correlation with '{corr_target}'",
                columns=["Correlation"]
            )
            .fmt_number(columns=["Correlation"], decimals=3)
            .cols_align(align="center", columns=["Correlation"])
        )
    
    return gt_table


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