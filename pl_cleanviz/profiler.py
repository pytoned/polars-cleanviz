from __future__ import annotations
import polars as pl
from great_tables import GT


def summary(df: pl.DataFrame) -> GT:
    """
    Generate a beautiful summary table of DataFrame statistics using Great Tables.
    
    This function creates a clean, formatted version of Polars' .describe() method
    with professional styling and improved readability.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to summarize.

    Returns
    -------
    GT
        A Great Tables object containing the formatted summary statistics.

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({
    ...     'price': [100, 200, 150, 300, 250],
    ...     'volume': [1000, 1500, 1200, 2000, 1800],
    ...     'rating': [4.5, 3.8, 4.2, 4.9, 4.1]
    ... })
    >>> table = plc.summary(df)
    >>> table.show()  # Display in notebook or browser
    """
    # Get only numeric columns for .describe()
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    
    if not numeric_cols:
        # Handle case with no numeric columns
        basic_info = []
        for col in df.columns:
            basic_info.append({
                'statistic': col,
                'type': str(df[col].dtype),
                'count': len(df[col]),
                'unique': df[col].n_unique(),
                'missing': df[col].null_count()
            })
        
        summary_df = pl.DataFrame(basic_info)
        
        gt_table = (
            GT(summary_df)
            .tab_header(
                title="📊 Data Summary",
                subtitle=f"Dataset: {df.height:,} rows × {df.width} columns (no numeric columns)"
            )
            .fmt_integer(columns=["count", "unique", "missing"])
        )
        
        return gt_table
    
    # Use Polars .describe() for numeric columns
    df_numeric = df.select(numeric_cols)
    describe_df = df_numeric.describe()
    
    # Transform to a better format for display
    # Transpose so columns become rows
    stats_data = []
    
    for col in numeric_cols:
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
        stats_data.append(col_stats)
    
    # Create DataFrame for Great Tables
    summary_df = pl.DataFrame(stats_data)
    
    # Build beautiful Great Tables object
    gt_table = (
        GT(summary_df)
        .tab_header(
            title="📊 Data Summary",
            subtitle=f"Dataset: {df.height:,} rows × {len(numeric_cols)} numeric columns"
        )
        .tab_spanner(
            label="Descriptive Statistics",
            columns=["Count", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"]
        )
        .fmt_integer(columns=["Count"])
        .fmt_number(columns=["Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"], decimals=2)
        .tab_options(
            table_font_size="14px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef",
            table_border_top_style="hidden",
            table_border_bottom_style="hidden"
        )
        .cols_align(
            align="center",
            columns=["Count", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"]
        )
        .cols_align(
            align="left", 
            columns=["Column"]
        )
    )
    
    return gt_table