
from __future__ import annotations
from typing import Iterable, Sequence, Tuple, List, Optional
import polars as pl
import polars.selectors as cs

# ---------- helpers ----------

def _numeric_columns(df: pl.DataFrame) -> list[str]:
    try:
        from polars import selectors as cs  # type: ignore
        return list(df.select(cs.numeric()).columns)
    except Exception:
        return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]

def _ensure_columns(df: pl.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        cols = _numeric_columns(df)
    else:
        cols = [c for c in columns if c in df.columns]
        dtypes = dict(zip(df.columns, df.dtypes))
        cols = [c for c in cols if dtypes[c].is_numeric()]
    return cols

def _corr_pair(df: pl.DataFrame, c1: str, c2: str) -> float:
    return df.select(pl.corr(pl.col(c1), pl.col(c2), method="pearson")).item()

def _corr_matrix(df: pl.DataFrame, cols: Sequence[str]) -> List[List[float]]:
    n = len(cols)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            r = 1.0 if i == j else float(_corr_pair(df, cols[i], cols[j]))
            mat[i][j] = r
            mat[j][i] = r
    return mat

def _px_to_inches(w: Optional[int], h: Optional[int], default_size: float) -> Tuple[float,float]:
    if w is None or h is None:
        size = min(12, max(5, default_size))
        return size, size
    return max(w/100, 1.0), max(h/100, 1.0)

# ---------- Seaborn backends ----------

def _corr_heatmap_seaborn(cols: Sequence[str], mat: List[List[float]], annotate: bool, width, height):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert correlation matrix to numpy array for seaborn (no pandas needed!)
    corr_array = np.array(mat)
    
    n = len(cols)
    fig_w, fig_h = _px_to_inches(width, height, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Use seaborn heatmap with better styling (klib-like)
    sns.heatmap(
        corr_array,
        xticklabels=cols,
        yticklabels=cols,
        annot=annotate, 
        cmap='RdBu_r', 
        center=0, 
        vmin=-1, 
        vmax=1,
        square=True,
        ax=ax,
        fmt='.2f',
        cbar_kws={'label': 'Correlation coefficient'}
    )
    
    ax.set_title("Correlation Heatmap (Pearson)", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    
    return fig

def _dist_plot_seaborn(s: pl.Series, column: str, bins: int, width, height):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Use numpy array directly (no pandas needed!)
    data = s.to_numpy()
    
    fig_w, fig_h = _px_to_inches(width, height, 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Use seaborn histplot with KDE overlay (like klib)
    sns.histplot(data, bins=bins, kde=True, stat='count', alpha=0.7, ax=ax)
    
    ax.set_title(f"Distribution of {column}", fontsize=14, pad=20)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def _missingval_plot_seaborn(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig_w, fig_h = _px_to_inches(width, height, max(3.0, len(cols) * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Use seaborn barplot for better styling
    sns.barplot(x=ratios, y=cols, ax=ax, palette='viridis')
    
    # Add text annotations
    for y, v, a in zip(range(len(cols)), ratios, counts):
        if normalize:
            ax.text(v, y, f" {v*100:.1f}%", va="center", fontsize=10)
        else:
            ax.text(v, y, f" {a}", va="center", fontsize=10)
    
    ax.set_xlabel("Share of missing values", fontsize=12)
    ax.set_ylabel("Columns", fontsize=12)
    ax.set_title("Missing Values per Column", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    return fig

def _cat_plot_seaborn(cat_data: dict, width, height):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    n_cols = len(cat_data)
    fig_w, fig_h = _px_to_inches(width, height, max(10.0, n_cols * 3))
    
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = [axes]
    
    for i, (col_name, data) in enumerate(cat_data.items()):
        ax = axes[i]
        
        # Combine top and bottom values
        all_values = data['top_values'] + data['bottom_values']
        all_counts = data['top_counts'] + data['bottom_counts']
        
        if all_values:
            # Create color palette (different colors for top vs bottom)
            colors = ['skyblue'] * len(data['top_values']) + ['lightcoral'] * len(data['bottom_values'])
            
            bars = ax.bar(range(len(all_values)), all_counts, color=colors)
            ax.set_xticks(range(len(all_values)))
            ax.set_xticklabels(all_values, rotation=45, ha='right')
            ax.set_title(f'{col_name}\n(Top {len(data["top_values"])}, Bottom {len(data["bottom_values"])})', 
                        fontsize=12, pad=10)
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for bar, count in zip(bars, all_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom', fontsize=10)
    
    fig.suptitle('Categorical Value Frequencies', fontsize=14, y=0.98)
    fig.tight_layout()
    return fig

# ---------- Plotly backends ----------

def _corr_heatmap_plotly(cols: Sequence[str], mat: List[List[float]], annotate: bool, width, height):
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(z=mat, x=list(cols), y=list(cols), zmin=-1, zmax=1, colorbar=dict(title="r")))
    fig.update_layout(title="Correlation heatmap (Pearson)", width=width, height=height)
    if annotate:
        annotations = []
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                annotations.append(dict(showarrow=False, text=f"{val:.2f}", x=cols[j], y=cols[i]))
        fig.update_layout(annotations=annotations)
    return fig

def _distplot_plotly(s: pl.Series, column: str, bins: int, width, height):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Histogram(x=s.to_numpy(), nbinsx=bins)])
    fig.update_layout(title=f"Distribution: {column}", xaxis_title=column, yaxis_title="Count",
                      width=width, height=height)
    return fig

def _missingval_plot_plotly(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import plotly.graph_objects as go
    if normalize:
        text_values = [f"{r*100:.1f}%" for r in ratios]
    else:
        text_values = counts
    fig = go.Figure(data=[go.Bar(y=cols, x=ratios, orientation="h", text=text_values, textposition="outside")])
    fig.update_layout(title="Missing values per column", xaxis_title="Share of missing",
                      yaxis_title="Columns", width=width, height=height)
    return fig

def _cat_plot_plotly(cat_data: dict, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    n_cols = len(cat_data)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=list(cat_data.keys()),
        horizontal_spacing=0.1
    )
    
    for i, (col_name, data) in enumerate(cat_data.items(), 1):
        # Combine top and bottom values
        all_values = data['top_values'] + data['bottom_values']
        all_counts = data['top_counts'] + data['bottom_counts']
        
        if all_values:
            # Create colors (blue for top, red for bottom)
            colors = ['steelblue'] * len(data['top_values']) + ['indianred'] * len(data['bottom_values'])
            
            fig.add_trace(
                go.Bar(
                    x=all_values,
                    y=all_counts,
                    marker_color=colors,
                    text=all_counts,
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ),
                row=1, col=i
            )
            
            # Update x-axis for this subplot
            fig.update_xaxes(tickangle=45, row=1, col=i)
            fig.update_yaxes(title_text="Count" if i == 1 else "", row=1, col=i)
    
    fig.update_layout(
        title="Categorical Value Frequencies",
        width=width or max(800, n_cols * 300),
        height=height or 500,
        showlegend=False
    )
    
    return fig

# ---------- Altair backends ----------

def _corr_heatmap_altair(cols: Sequence[str], mat: List[List[float]], annotate: bool, width, height):
    import altair as alt
    data = [{"row": r, "col": c, "r": mat[i][j]} for i, r in enumerate(cols) for j, c in enumerate(cols)]
    df_data = pl.DataFrame(data)
    base = alt.Chart(df_data).mark_rect().encode(
        x=alt.X("col:N", title=None),
        y=alt.Y("row:N", title=None),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1]))
    ).properties(title="Correlation heatmap (Pearson)")
    if width:  base = base.properties(width=width)
    if height: base = base.properties(height=height)
    if annotate:
        text = alt.Chart(df_data).mark_text().encode(x="col:N", y="row:N", text=alt.Text("r:Q", format=".2f"))
        if width:  text = text.properties(width=width)
        if height: text = text.properties(height=height)
        return base + text
    return base

def _distplot_altair(s: pl.Series, column: str, bins: int, width, height):
    import altair as alt
    data = [{column: v} for v in s.to_list()]
    df_data = pl.DataFrame(data)
    chart = alt.Chart(df_data).mark_bar().encode(
        x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=bins)),
        y="count()",
        tooltip=[column]
    ).properties(title=f"Distribution: {column}")
    if width:  chart = chart.properties(width=width)
    if height: chart = chart.properties(height=height)
    return chart

def _missingval_plot_altair(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import altair as alt
    if normalize:
        text_values = [f"{r*100:.1f}%" for r in ratios]
    else:
        text_values = counts
    data = [{"column": c, "ratio": r, "count": n, "text_value": t} for c, r, n, t in zip(cols, ratios, counts, text_values)]
    df_data = pl.DataFrame(data)
    base = alt.Chart(df_data).mark_bar().encode(
        y=alt.Y("column:N", sort=None, title="Columns"),
        x=alt.X("ratio:Q", title="Share of missing"),
        tooltip=["column", "count", "ratio"]
    ).properties(title="Missing values per column")
    if width:  base = base.properties(width=width)
    if height: base = base.properties(height=height)
    text = alt.Chart(df_data).mark_text(align="left", baseline="middle", dx=3).encode(
        y="column:N", x="ratio:Q", text="text_value:N"
    )
    if width:  text = text.properties(width=width)
    if height: text = text.properties(height=height)
    return base + text

def _cat_plot_altair(cat_data: dict, width, height):
    import altair as alt
    
    # Prepare data for all columns
    all_data = []
    for col_name, data in cat_data.items():
        # Add top values
        for val, count in zip(data['top_values'], data['top_counts']):
            all_data.append({
                'column': col_name,
                'value': str(val),
                'count': count,
                'type': 'top'
            })
        # Add bottom values
        for val, count in zip(data['bottom_values'], data['bottom_counts']):
            all_data.append({
                'column': col_name,
                'value': str(val),
                'count': count,
                'type': 'bottom'
            })
    
    if not all_data:
        # Return empty chart
        return alt.Chart(pl.DataFrame({'x': [0], 'y': [0]})).mark_point()
    
    df_data = pl.DataFrame(all_data)
    
    chart = alt.Chart(df_data).mark_bar().encode(
        x=alt.X('value:N', title='Categories'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('type:N', 
                       scale=alt.Scale(domain=['top', 'bottom'], 
                                     range=['steelblue', 'indianred']),
                       legend=alt.Legend(title="Category Type")),
        tooltip=['column', 'value', 'count', 'type'],
        facet=alt.Facet('column:N', title='Categorical Variables')
    ).properties(
        title="Categorical Value Frequencies"
    ).resolve_scale(
        x='independent'
    )
    
    if width: chart = chart.properties(width=width // len(cat_data))
    if height: chart = chart.properties(height=height)
    
    return chart

# ---------- Public APIs ----------

def corr_heatmap(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    annotate: bool = True,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create a correlation heatmap for numeric columns in a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing numeric columns to correlate.
    columns : Sequence[str] | None, optional
        Specific columns to include in correlation. If None, uses all numeric columns.
    annotate : bool, default True
        Whether to display correlation values as text on each cell.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "matplotlib"
        Plotting backend to use. Options: "seaborn", "plotly", "altair".

    Returns
    -------
    Figure object
        The correlation heatmap figure (type depends on backend).

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    >>> plc.corr_heatmap(df, backend="plotly", annotate=False)
    """
    cols = _ensure_columns(df, columns)
    if len(cols) == 0:
        if backend == "seaborn":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_axis_off()
            fig.suptitle("No numeric columns to correlate")
            return fig
        elif backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(title="No numeric columns to correlate",
                                                 width=width, height=height); return fig
        elif backend == "altair":
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_rect()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
        else:
            raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")

    mat = _corr_matrix(df, cols)

    if backend == "seaborn":
        return _corr_heatmap_seaborn(cols, mat, annotate, width, height)
    elif backend == "plotly":
        return _corr_heatmap_plotly(cols, mat, annotate, width, height)
    elif backend == "altair":
        return _corr_heatmap_altair(cols, mat, annotate, width, height)
    else:
        raise ValueError("backend must be 'seaborn', 'plotly', or 'altair'")

def dist_plot(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    bins: int = 30,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create a distribution plot (histogram) for a numeric column.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing the column to plot.
    column : str | None, optional
        Name of the numeric column to plot. If None, uses the first numeric column.
    bins : int, default 30
        Number of histogram bins to use.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "matplotlib"
        Plotting backend to use. Options: "seaborn", "plotly", "altair".

    Returns
    -------
    Figure object
        The distribution plot figure (type depends on backend).

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({'age': [25, 30, 35, 40, 45, 50]})
    >>> plc.distplot(df, column="age", bins=10, backend="plotly")
    """
    cols = _numeric_columns(df)
    if column is None:
        if not cols:
            if backend == "seaborn":
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.set_axis_off()
                fig.suptitle("No numeric column found")
                return fig
            elif backend == "plotly":
                import plotly.graph_objects as go
                fig = go.Figure(); fig.update_layout(title="No numeric column found",
                                                     width=width, height=height); return fig
            elif backend == "altair":
                import altair as alt
                chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
                if width:  chart = chart.properties(width=width)
                if height: chart = chart.properties(height=height)
                return chart
            else:
                raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")
        column = cols[0]
    elif column not in df.columns:
        if backend == "seaborn":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_axis_off()
            fig.suptitle(f"Column '{column}' not found")
            return fig
        elif backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(title=f"Column '{column}' not found",
                                                 width=width, height=height); return fig
        elif backend == "altair":
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
        else:
            raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")
    else:
        if not df[column].dtype.is_numeric():
            if backend == "seaborn":
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.set_axis_off()
                fig.suptitle(f"Column '{column}' is not numeric")
                return fig
            elif backend == "plotly":
                import plotly.graph_objects as go
                fig = go.Figure(); fig.update_layout(title=f"Column '{column}' is not numeric",
                                                     width=width, height=height); return fig
            elif backend == "altair":
                import altair as alt
                chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
                if width:  chart = chart.properties(width=width)
                if height: chart = chart.properties(height=height)
                return chart
            else:
                raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")

    s = df.select(pl.col(column).drop_nulls()).to_series()
    if s.dtype not in (pl.Float32, pl.Float64):
        s = s.cast(pl.Float64)

    if backend == "seaborn":
        return _dist_plot_seaborn(s, column, bins, width, height)
    elif backend == "plotly":
        return _distplot_plotly(s, column, bins, width, height)
    elif backend == "altair":
        return _distplot_altair(s, column, bins, width, height)
    else:
        raise ValueError("backend must be 'seaborn', 'plotly', or 'altair'")


# Backward compatibility alias
distplot = dist_plot

def missingval_plot(
    df: pl.DataFrame,
    *,
    sort: str = "desc",
    normalize: bool = False,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create a horizontal bar plot showing missing values per column.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to analyze for missing values.
    sort : str, default "desc"
        How to sort columns by missing value ratio. Options: "desc", "asc", "none".
    normalize : bool, default False
        If True, display percentages instead of absolute counts on bars.
        Similar to normalize parameter in pandas value_counts().
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "matplotlib"
        Plotting backend to use. Options: "seaborn", "plotly", "altair".

    Returns
    -------
    Figure object
        The missing values plot figure (type depends on backend).

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({'a': [1, None, 3], 'b': [4, 5, None], 'c': [7, 8, 9]})
    >>> plc.missingval_plot(df, normalize=True, backend="plotly")
    """
    cols = list(df.columns)
    if not cols:
        if backend == "seaborn":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_axis_off()
            fig.suptitle("No columns")
            return fig
        elif backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(title="No columns", width=width, height=height); return fig
        elif backend == "altair":
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart

    null_counts_row = df.select([pl.col(c).is_null().sum().alias(c) for c in cols])
    total = df.height
    counts = [int(null_counts_row.select(pl.col(c)).item()) for c in cols]
    ratios = [cnt / max(total, 1) for cnt in counts]

    order = list(range(len(cols)))
    if sort == "desc":
        order.sort(key=lambda i: ratios[i], reverse=True)
    elif sort == "asc":
        order.sort(key=lambda i: ratios[i])

    cols_o  = [cols[i] for i in order]
    ratios_o = [ratios[i] for i in order]
    counts_o = [counts[i] for i in order]

    if backend == "seaborn":
        return _missingval_plot_seaborn(cols_o, ratios_o, counts_o, width, height, normalize)
    elif backend == "plotly":
        return _missingval_plot_plotly(cols_o, ratios_o, counts_o, width, height, normalize)
    elif backend == "altair":
        return _missingval_plot_altair(cols_o, ratios_o, counts_o, width, height, normalize)
    else:
        raise ValueError("backend must be 'seaborn', 'plotly', or 'altair'")


def cat_plot(
    df: pl.DataFrame,
    *,
    top: int = 10,
    bottom: int = 10,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create categorical value frequency plots showing top and bottom categories.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing categorical columns.
    top : int, default 10
        Number of most frequent categories to show per column.
    bottom : int, default 10  
        Number of least frequent categories to show per column.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "plotly"
        Plotting backend to use. Options: "seaborn", "plotly", "altair".

    Returns
    -------
    Figure object
        The categorical plot figure (type depends on backend).

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'A']})
    >>> plc.cat_plot(df, top=3, bottom=2)
    """
    # Get categorical columns (string/categorical types)
    cat_cols = [c for c, dt in zip(df.columns, df.dtypes) 
                if dt in (pl.String, pl.Utf8, pl.Categorical)]
    
    if not cat_cols:
        # No categorical columns found
        if backend == "seaborn":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_axis_off()
            fig.suptitle("No categorical columns found")
            return fig
        elif backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(title="No categorical columns found", width=width, height=height)
            return fig
        elif backend == "altair":
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
        else:
            raise ValueError("backend must be 'seaborn', 'plotly', or 'altair'")
    
    # Calculate value counts for each categorical column
    cat_data = {}
    for col in cat_cols:
        value_counts = (df.select(pl.col(col).value_counts(sort=True))
                       .unnest(col)
                       .head(top + bottom))  # Get top + bottom values
        
        # Split into top and bottom
        counts = value_counts.get_column('count').to_list()
        values = value_counts.get_column(col).to_list()
        
        if len(counts) > top:
            top_values = values[:top]
            top_counts = counts[:top]
            bottom_values = values[-bottom:] if bottom > 0 else []
            bottom_counts = counts[-bottom:] if bottom > 0 else []
        else:
            top_values = values
            top_counts = counts
            bottom_values = []
            bottom_counts = []
            
        cat_data[col] = {
            'top_values': top_values,
            'top_counts': top_counts,
            'bottom_values': bottom_values,
            'bottom_counts': bottom_counts
        }
    
    if backend == "seaborn":
        return _cat_plot_seaborn(cat_data, width, height)
    elif backend == "plotly":
        return _cat_plot_plotly(cat_data, width, height)
    elif backend == "altair":
        return _cat_plot_altair(cat_data, width, height)
    else:
        raise ValueError("backend must be 'seaborn', 'plotly', or 'altair'")
