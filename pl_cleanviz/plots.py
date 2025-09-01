
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

# ---------- Matplotlib backends ----------

def _corr_heatmap_matplotlib(cols: Sequence[str], mat: List[List[float]], annotate: bool, width, height):
    import matplotlib.pyplot as plt
    n = len(cols)
    fig_w, fig_h = _px_to_inches(width, height, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    ax.set_title("Correlation heatmap (Pearson)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel("r", rotation=0, labelpad=10)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    plt.close(fig)  # Prevent automatic display
    return fig

def _distplot_matplotlib(s: pl.Series, column: str, bins: int, width, height):
    import matplotlib.pyplot as plt
    fig_w, fig_h = _px_to_inches(width, height, 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.hist(s.to_numpy(), bins=bins)
    ax.set_title(f"Distribution: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    plt.close(fig)  # Prevent automatic display
    return fig

def _missingval_plot_matplotlib(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import matplotlib.pyplot as plt
    fig_w, fig_h = _px_to_inches(width, height, max(3.0, len(cols) * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.barh(cols, ratios)
    for y, v, a in zip(range(len(cols)), ratios, counts):
        if normalize:
            ax.text(v, y, f" {v*100:.1f}%", va="center")
        else:
            ax.text(v, y, f" {a}", va="center")
    ax.set_xlabel("Share of missing")
    ax.set_ylabel("Columns")
    ax.set_title("Missing values per column")
    ax.invert_yaxis()
    fig.tight_layout()
    plt.close(fig)  # Prevent automatic display
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

# ---------- Public APIs ----------

def corr_heatmap(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    annotate: bool = True,
    width: int | None = None,
    height: int | None = None,
    backend: str = "matplotlib",
):
    cols = _ensure_columns(df, columns)
    if len(cols) == 0:
        if backend == "matplotlib":
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

    if backend == "matplotlib":
        return _corr_heatmap_matplotlib(cols, mat, annotate, width, height)
    elif backend == "plotly":
        return _corr_heatmap_plotly(cols, mat, annotate, width, height)
    elif backend == "altair":
        return _corr_heatmap_altair(cols, mat, annotate, width, height)
    else:
        raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")

def distplot(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    bins: int = 30,
    width: int | None = None,
    height: int | None = None,
    backend: str = "matplotlib",
):
    cols = _numeric_columns(df)
    if column is None:
        if not cols:
            if backend == "matplotlib":
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
        if backend == "matplotlib":
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
            if backend == "matplotlib":
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

    if backend == "matplotlib":
        return _distplot_matplotlib(s, column, bins, width, height)
    elif backend == "plotly":
        return _distplot_plotly(s, column, bins, width, height)
    elif backend == "altair":
        return _distplot_altair(s, column, bins, width, height)
    else:
        raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")

def missingval_plot(
    df: pl.DataFrame,
    *,
    sort: str = "desc",
    normalize: bool = False,
    width: int | None = None,
    height: int | None = None,
    backend: str = "matplotlib",
):
    """
    Plot missing-values per column.
    sort: 'desc'|'asc'|'none' to control column order by missing ratio.
    normalize: if True, display percentages instead of absolute counts on bars.
    """
    cols = list(df.columns)
    if not cols:
        if backend == "matplotlib":
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

    if backend == "matplotlib":
        return _missingval_plot_matplotlib(cols_o, ratios_o, counts_o, width, height, normalize)
    elif backend == "plotly":
        return _missingval_plot_plotly(cols_o, ratios_o, counts_o, width, height, normalize)
    elif backend == "altair":
        return _missingval_plot_altair(cols_o, ratios_o, counts_o, width, height, normalize)
    else:
        raise ValueError("backend must be 'matplotlib', 'plotly', or 'altair'")
