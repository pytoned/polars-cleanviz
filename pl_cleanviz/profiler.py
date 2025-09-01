
from __future__ import annotations
import polars as pl
import numpy as np
from great_tables import GT, nanoplot_options



def quick_profile(
    df: pl.DataFrame,
    *,
    corr_target: str | None = None,
) -> GT:
    """
    Generate an enhanced data profiling table using Great Tables.
    
    This function creates a comprehensive statistical summary that goes beyond
    Polars' basic .describe() method, including:
    - Enhanced statistics (skewness, unique values, etc.)
    - Histogram nanoplots for each column
    - Correlation bars relative to a target column (if specified)
    - Beautiful formatting with Great Tables

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to profile.
    corr_target : str | None, optional
        Target column for correlation analysis. If provided, shows correlation
        bars for each numeric column relative to this target.

    Returns
    -------
    GT
        A Great Tables object containing the enhanced profile report.

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({
    ...     'price': [100, 200, 150, 300, 250],
    ...     'volume': [1000, 1500, 1200, 2000, 1800],
    ...     'category': ['A', 'B', 'A', 'C', 'B']
    ... })
    >>> table = plc.quick_profile(df, corr_target='price')
    >>> table.show()  # Display in notebook or browser
    """
    # Collect comprehensive statistics for each column
    profile_data = []
    
    # Calculate correlations if target specified
    correlations = {}
    if corr_target and corr_target in df.columns:
        target_series = df.select(pl.col(corr_target))
        if target_series.dtypes[0].is_numeric():
            numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() and c != corr_target]
            for col in numeric_cols:
                try:
                    corr_val = df.select([pl.corr(corr_target, col)]).item()
                    correlations[col] = corr_val if corr_val is not None else 0.0
                except:
                    correlations[col] = 0.0
    
    for col in df.columns:
        dtype = df[col].dtype
        series = df[col]
        
        # Basic info
        row_data = {
            'Column': col,
            'Type': str(dtype),
            'Count': len(series),
            'Missing': series.null_count(),
            'Missing %': round((series.null_count() / len(series)) * 100, 1),
            'Unique': series.n_unique(),
        }
        
        # Type-specific statistics
        if dtype.is_numeric():
            # Numeric statistics
            non_null_series = series.drop_nulls()
            if len(non_null_series) > 0:
                stats = non_null_series.describe()
                row_data.update({
                    'Mean': round(float(stats.filter(pl.col('statistic') == 'mean')['value'].item()), 3),
                    'Std': round(float(stats.filter(pl.col('statistic') == 'std')['value'].item()), 3),
                    'Min': float(stats.filter(pl.col('statistic') == 'min')['value'].item()),
                    'Q25': float(stats.filter(pl.col('statistic') == '25%')['value'].item()),
                    'Median': float(stats.filter(pl.col('statistic') == '50%')['value'].item()),
                    'Q75': float(stats.filter(pl.col('statistic') == '75%')['value'].item()),
                    'Max': float(stats.filter(pl.col('statistic') == 'max')['value'].item()),
                })
                
                # Calculate skewness
                values = non_null_series.to_numpy()
                if len(values) > 2 and np.std(values) > 0:
                    skewness = float(pl.DataFrame({'x': values}).select(
                        ((pl.col('x') - pl.col('x').mean()) ** 3).mean() / (pl.col('x').std() ** 3)
                    ).item())
                    row_data['Skewness'] = round(skewness, 3)
                else:
                    row_data['Skewness'] = 0.0
                
                # Create histogram data for nanoplot
                hist_values = values
                row_data['Histogram'] = list(hist_values[:50])  # Limit for nanoplot
                
                # Add correlation if target specified
                if corr_target and col in correlations:
                    row_data['Correlation'] = round(correlations[col], 3)
                    # Create correlation bar data (single value between -1 and 1)
                    row_data['Corr_Bar'] = [correlations[col]]
                
            else:
                # Handle empty numeric columns
                for key in ['Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Skewness']:
                    row_data[key] = None
                row_data['Histogram'] = []
                if corr_target and col != corr_target:
                    row_data['Correlation'] = None
                    row_data['Corr_Bar'] = [0]
        
        else:
            # Non-numeric columns - fill with None/empty for numeric-only columns
            for key in ['Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Skewness']:
                row_data[key] = None
            
            # For categorical/string columns, show most frequent values
            if dtype in (pl.String, pl.Utf8, pl.Categorical):
                try:
                    value_counts = series.value_counts(sort=True).head(3)
                    top_values = value_counts.get_column(col).to_list()
                    row_data['Top_Values'] = ', '.join([str(v) for v in top_values])
                except:
                    row_data['Top_Values'] = ''
            else:
                row_data['Top_Values'] = ''
                
            row_data['Histogram'] = []
            if corr_target:
                row_data['Correlation'] = None  
                row_data['Corr_Bar'] = [0]
        
        profile_data.append(row_data)
    
    # Create DataFrame for Great Tables
    profile_df = pl.DataFrame(profile_data)
    
    # Build Great Tables object
    gt_table = (
        GT(profile_df)
        .tab_header(
            title="📊 Enhanced Data Profile",
            subtitle=f"Dataset: {df.height:,} rows × {df.width} columns"
        )
        .tab_spanner(
            label="Basic Info",
            columns=["Column", "Type", "Count", "Missing", "Missing %", "Unique"]
        )
    )
    
    # Add numeric statistics spanner if we have numeric columns
    numeric_cols_exist = any(dtype.is_numeric() for dtype in df.dtypes)
    if numeric_cols_exist:
        gt_table = gt_table.tab_spanner(
            label="Statistics",
            columns=["Mean", "Std", "Min", "Q25", "Median", "Q75", "Max", "Skewness"]
        )
    
    # Add visualization spanners
    gt_table = gt_table.tab_spanner(
        label="Distribution",
        columns=["Histogram"]
    )
    
    if corr_target:
        gt_table = gt_table.tab_spanner(
            label=f"Correlation with '{corr_target}'",
            columns=["Correlation", "Corr_Bar"]
        )
    
    # Format the table
    gt_table = (
        gt_table
        .fmt_number(columns=["Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"], decimals=3)
        .fmt_number(columns=["Skewness"], decimals=3)
        .fmt_percent(columns=["Missing %"], decimals=1, scale_values=False)
        .fmt_integer(columns=["Count", "Missing", "Unique"])
    )
    
    # Add correlation formatting if target specified
    if corr_target:
        gt_table = gt_table.fmt_number(columns=["Correlation"], decimals=3)
    
    # Add nanoplots for histograms
    if 'Histogram' in profile_df.columns:
        gt_table = gt_table.fmt_nanoplot(
            columns="Histogram",
            plot_type="line",
            options=nanoplot_options(
                data_point_fill_color="steelblue",
                data_point_stroke_color="steelblue", 
                data_line_stroke_color="steelblue"
            )
        )
    
    # Add correlation bar plots if target specified
    if corr_target and 'Corr_Bar' in profile_df.columns:
        gt_table = gt_table.fmt_nanoplot(
            columns="Corr_Bar", 
            plot_type="bar",
            options=nanoplot_options(
                data_bar_stroke_color="darkred",
                data_bar_fill_color="red"
            )
        )
    
    # Style the table
    gt_table = (
        gt_table
        .tab_options(
            table_font_size="12px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef"
        )
    )
    
    return gt_table


