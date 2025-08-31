# pl-cleanviz

Polars-only utilities with multiple plotting backends (no pandas dependency).

Functions:
- `clean_column_names(df)`
- `corr_heatmap(df, columns=None, annotate=True, backend="matplotlib", width=None, height=None)`
- `distplot(df, column=None, bins=30, backend="matplotlib", width=None, height=None)`
- `missingval_plot(df, sort="desc", backend="matplotlib", width=None, height=None)`
- `save_fig(obj, path, scale=1.0)` — save Matplotlib/Plotly/Altair outputs.
- `profile_quick(df, backend="matplotlib", width=None, height=None, bins=30)` — minimal profiler.
- `save_profile(report, directory)` — save profiler outputs to files.

Supported backends: `"matplotlib"` (default), `"plotly"`, `"altair"`.
