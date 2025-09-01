
from __future__ import annotations
from typing import Dict, Any
import polars as pl
from .plots import corr_heatmap, distplot, missingval_plot

def profile_quick(
    df: pl.DataFrame,
    *,
    backend: str = "matplotlib",
    width: int | None = None,
    height: int | None = None,
    bins: int = 30,
) -> Dict[str, Any]:
    """
    Generate a quick profiling summary with basic statistics and plots.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to profile.
    backend : str, default "matplotlib"
        Plotting backend to use. Options: "matplotlib", "plotly", "altair".
    width : int | None, optional
        Width of the plots in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plots in pixels. If None, uses backend default.
    bins : int, default 30
        Number of histogram bins for distribution plots.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'summary': Basic DataFrame statistics (n_rows, n_cols)
        - 'dtypes': Column data types
        - 'missing': Missing value information per column
        - 'plots': Dictionary with 'missing', 'corr', and 'dist' plots

    Examples
    --------
    >>> import polars as pl
    >>> import pl_cleanviz as plc
    >>> df = pl.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
    >>> result = plc.profile_quick(df, backend="plotly")
    >>> print(result['summary'])
    """
    n_rows, n_cols = df.height, df.width
    dtypes = {c: str(dt) for c, dt in zip(df.columns, df.dtypes)}

    # missing stats
    cols = list(df.columns)
    null_counts_row = df.select([pl.col(c).is_null().sum().alias(c) for c in cols])
    counts = [int(null_counts_row.select(pl.col(c)).item()) for c in cols]
    ratios = [cnt / max(n_rows, 1) for cnt in counts]
    missing_list = [{"column": c, "missing": n, "ratio": r} for c, n, r in zip(cols, counts, ratios)]

    # choose a numeric column for the distplot (first numeric)
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    dist_col = num_cols[0] if num_cols else None

    plots = {
        "missing": missingval_plot(df, backend=backend, width=width, height=height),
        "corr": corr_heatmap(df, backend=backend, width=width, height=height),
        "dist": distplot(df, column=dist_col, bins=bins, backend=backend, width=width, height=height),
    }

    return {
        "summary": {"n_rows": n_rows, "n_cols": n_cols},
        "dtypes": dtypes,
        "missing": {"per_column": missing_list},
        "plots": plots,
    }

def save_profile(report: Dict[str, Any], directory: str):
    import os, json
    from .utils import save_fig
    os.makedirs(directory, exist_ok=True)

    plots = report.get("plots", {})
    for key, obj in plots.items():
        name_html = os.path.join(directory, f"{key}.html")
        name_png  = os.path.join(directory, f"{key}.png")
        t = str(type(obj))
        if "plotly" in t or "altair" in t:
            save_fig(obj, name_html)
        else:
            save_fig(obj, name_png)

    meta = {
        "summary": report.get("summary", {}),
        "dtypes": report.get("dtypes", {}),
        "missing": report.get("missing", {}),
        "files": sorted(os.listdir(directory)),
    }
    with open(os.path.join(directory, "profile.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
