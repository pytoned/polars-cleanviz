from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl

from .plots import corr_heatmap, distplot, missingval_plot
from .utils import fig_to_base64_png


@dataclass
class ProfileConfig:
    title: str = "Polars Profile Report"
    backend: str = "matplotlib"  # use matplotlib for embedded PNGs
    bins: int = 30
    max_numeric_dists: int = 8
    max_categorical_bars: int = 8
    sample_rows: int = 10
    infer_categorical_threshold: int = 50  # treat small-cardinality strings as categorical


class ProfileReport:
    """
    Polars-only profiling report inspired by ydata-profiling, simplified.

    Provides basic table/variable stats, correlations, missingness, and
    embeds plots as base64 PNG in a single self-contained HTML.
    """

    def __init__(self, df: pl.DataFrame, *, config: Optional[ProfileConfig] = None):
        self.df = df
        self.config = config or ProfileConfig()
        self._model: Dict[str, Any] = {}

    # --------- public API ---------
    def to_dict(self) -> Dict[str, Any]:
        if not self._model:
            self._model = self._build_model()
        return self._model

    def to_html(self) -> str:
        data = self.to_dict()
        return _render_html(data)

    def to_file(self, path: str) -> str:
        html = self.to_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    # --------- model builders ---------
    def _build_model(self) -> Dict[str, Any]:
        df = self.df
        cfg = self.config

        n_rows, n_cols = df.height, df.width

        # table-level stats
        total_cells = n_rows * max(n_cols, 1)
        null_per_col = df.select([pl.col(c).is_null().sum().alias(c) for c in df.columns]) if df.width else None
        missing_per_col = {c: int(null_per_col.select(pl.col(c)).item()) for c in df.columns} if null_per_col is not None else {}
        total_missing = sum(missing_per_col.values())

        # duplicates (full-row duplicates)
        n_dupes = 0
        try:
            n_dupes = int(df.height - df.unique().height)
        except Exception:
            n_dupes = 0

        # memory estimate (if available)
        est_mem = None
        try:
            est_mem = int(df.estimated_size())
        except Exception:
            pass

        # variable-level stats
        variables: List[Dict[str, Any]] = []
        num_cols: List[str] = []
        cat_cols: List[str] = []
        bool_cols: List[str] = []
        dt_cols: List[str] = []

        dtypes = dict(zip(df.columns, df.dtypes))
        for c in df.columns:
            dt = dtypes[c]
            miss = missing_per_col.get(c, 0)
            distinct = int(df.select(pl.col(c).n_unique()).item())

            # classify
            kind = "other"
            if pl.datatypes.is_numeric(dt):
                kind = "numeric"; num_cols.append(c)
            elif dt == pl.Boolean:
                kind = "boolean"; bool_cols.append(c)
            elif pl.datatypes.is_utf8(dt):
                # heuristic: small-cardinality strings -> categorical
                kind = "categorical" if distinct <= cfg.infer_categorical_threshold else "text"
                if kind == "categorical":
                    cat_cols.append(c)
            elif pl.datatypes.is_datetime(dt) or pl.datatypes.is_date(dt) or pl.datatypes.is_time(dt):
                kind = "datetime"; dt_cols.append(c)

            vstats: Dict[str, Any] = {
                "name": c,
                "dtype": str(dt),
                "type": kind,
                "n_missing": miss,
                "p_missing": (miss / max(n_rows, 1)) if n_rows else 0.0,
                "n_distinct": distinct,
                "is_unique": distinct == n_rows and miss == 0,
            }

            if kind == "numeric":
                s = df.select(pl.col(c)).drop_nulls().get_column(c)
                if s.len() > 0:
                    q = df.select([
                        pl.col(c).min().alias("min"), pl.col(c).max().alias("max"),
                        pl.col(c).mean().alias("mean"), pl.col(c).std().alias("std"),
                        pl.col(c).quantile(0.05).alias("p05"), pl.col(c).quantile(0.25).alias("p25"),
                        pl.col(c).median().alias("p50"), pl.col(c).quantile(0.75).alias("p75"), pl.col(c).quantile(0.95).alias("p95"),
                    ]).to_dicts()[0]
                    vstats.update(q)
                # zero count
                try:
                    zeros = int(df.select(pl.col(c).eq(0).sum()).item())
                    vstats["n_zeros"] = zeros
                except Exception:
                    vstats["n_zeros"] = None
            elif kind in ("categorical", "text", "boolean"):
                # top frequency
                try:
                    top = (
                        df.select(pl.col(c))
                        .drop_nulls()
                        .group_by(c)
                        .len()
                        .sort("len", descending=True)
                        .head(1)
                        .to_dicts()
                    )
                    if top:
                        vstats["mode"] = {"value": top[0][c], "count": int(top[0]["len"]) }
                except Exception:
                    pass
                if pl.datatypes.is_utf8(dt):
                    # string lengths
                    try:
                        lens = df.select(pl.col(c).str.len_chars()).drop_nulls()
                        if lens.height:
                            vstats["str_len_min"] = int(lens.select(pl.min("len_chars")).item())
                            vstats["str_len_max"] = int(lens.select(pl.max("len_chars")).item())
                            vstats["str_len_mean"] = float(lens.select(pl.mean("len_chars")).item())
                    except Exception:
                        pass

            variables.append(vstats)

        # plots (matplotlib -> base64)
        plots: Dict[str, str] = {}
        try:
            # missing per column
            fig_miss = missingval_plot(df, backend="matplotlib")
            plots["missing"] = fig_to_base64_png(fig_miss)
        except Exception:
            pass
        try:
            # correlation heatmap on numeric columns
            fig_corr = corr_heatmap(df, backend="matplotlib")
            plots["correlation"] = fig_to_base64_png(fig_corr)
        except Exception:
            pass
        # distributions for a subset of numeric columns
        for c in num_cols[: self.config.max_numeric_dists]:
            try:
                fig = distplot(df, column=c, bins=cfg.bins, backend="matplotlib")
                plots[f"dist__{c}"] = fig_to_base64_png(fig)
            except Exception:
                continue

        # samples
        head_rows = df.head(cfg.sample_rows).to_dicts()
        tail_rows = df.tail(cfg.sample_rows).to_dicts()

        return {
            "config": {
                "title": cfg.title,
            },
            "table": {
                "n_rows": n_rows,
                "n_columns": n_cols,
                "n_cells": total_cells,
                "n_missing": total_missing,
                "p_missing": (total_missing / max(total_cells, 1)) if total_cells else 0.0,
                "n_duplicates": n_dupes,
                "estimated_size_bytes": est_mem,
            },
            "variables": variables,
            "plots": plots,
            "samples": {
                "head": head_rows,
                "tail": tail_rows,
            },
        }


def _render_html(model: Dict[str, Any]) -> str:
    cfg = model.get("config", {})
    table = model.get("table", {})
    variables: List[Dict[str, Any]] = model.get("variables", [])
    plots: Dict[str, str] = model.get("plots", {})
    samples = model.get("samples", {})

    # simple CSS for readability
    css = """
    body {font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px;}
    h1 {margin-bottom: 8px;}
    h2 {margin-top: 28px; border-bottom: 1px solid #eee; padding-bottom: 4px;}
    .kpi {display:flex; gap:16px; flex-wrap:wrap; margin: 10px 0 16px;}
    .kpi .box {background:#f7f7f7; border:1px solid #eee; border-radius:6px; padding:10px 12px;}
    table {border-collapse:collapse; width: 100%;}
    th, td {border:1px solid #e6e6e6; padding:6px 8px; text-align:left; font-size: 13px;}
    th {background:#fafafa;}
    .grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px;}
    .card {border:1px solid #eee; border-radius:6px; padding:10px;}
    img.plot {max-width: 100%; height:auto; border:1px solid #eee;}
    code {background:#f3f3f3; padding:2px 4px; border-radius:3px;}
    """

    # variables table (limited columns but informative)
    var_rows = "".join(
        f"<tr><td>{v.get('name')}</td><td><code>{v.get('dtype')}</code></td><td>{v.get('type')}</td>"
        f"<td>{v.get('n_missing')}</td><td>{v.get('n_distinct')}</td>"
        f"<td>{_fmt(v.get('mean'))}</td><td>{_fmt(v.get('std'))}</td><td>{_fmt(v.get('min'))}</td><td>{_fmt(v.get('max'))}</td></tr>"
        for v in variables
    )

    # distribution plots
    dist_imgs = [
        (k.split("__", 1)[1], src) for k, src in plots.items() if k.startswith("dist__")
    ]
    dist_html = "".join(
        f"<div class='card'><div style='font-weight:600;margin-bottom:6px'>{col}</div><img class='plot' src='{src}'/></div>"
        for col, src in dist_imgs
    )

    # build HTML
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{_esc(cfg.get('title','Polars Profile Report'))}</title>
  <style>{css}</style>
 </head>
<body>
  <h1>{_esc(cfg.get('title','Polars Profile Report'))}</h1>
  <div class='kpi'>
    <div class='box'><div style='font-weight:600'>Rows</div><div>{table.get('n_rows')}</div></div>
    <div class='box'><div style='font-weight:600'>Columns</div><div>{table.get('n_columns')}</div></div>
    <div class='box'><div style='font-weight:600'>Missing</div><div>{table.get('n_missing')} ({_pct(table.get('p_missing'))})</div></div>
    <div class='box'><div style='font-weight:600'>Duplicates</div><div>{table.get('n_duplicates')}</div></div>
    <div class='box'><div style='font-weight:600'>Size (est.)</div><div>{_bytes(table.get('estimated_size_bytes'))}</div></div>
  </div>

  <h2>Missingness</h2>
  {f"<img class='plot' src='{plots.get('missing','')}'/>" if plots.get('missing') else '<div>No plot</div>'}

  <h2>Correlations</h2>
  {f"<img class='plot' src='{plots.get('correlation','')}'/>" if plots.get('correlation') else '<div>No plot</div>'}

  <h2>Variables</h2>
  <div style='overflow:auto;'>
    <table>
      <thead><tr><th>Name</th><th>Dtype</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>
        {var_rows}
      </tbody>
    </table>
  </div>

  <h2>Distributions</h2>
  <div class='grid'>
    {dist_html if dist_html else '<div>No numeric distributions plotted</div>'}
  </div>

  <h2>Samples</h2>
  <div class='grid'>
    <div class='card'>
      <div style='font-weight:600;margin-bottom:6px'>Head</div>
      { _simple_table(samples.get('head', [])) }
    </div>
    <div class='card'>
      <div style='font-weight:600;margin-bottom:6px'>Tail</div>
      { _simple_table(samples.get('tail', [])) }
    </div>
  </div>

  <div style='color:#888; margin-top:24px'>Generated by polars-cleanviz</div>
</body>
</html>
"""
    return html


def _esc(x: Any) -> str:
    import html
    return html.escape(str(x))


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x)
    except Exception:
        return str(x)


def _pct(x: Any) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return ""


def _bytes(n: Any) -> str:
    try:
        n = int(n)
    except Exception:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.2f} {units[i]}"


def _simple_table(rows: List[dict]) -> str:
    if not rows:
        return "<div style='color:#888'>No rows</div>"
    cols = list(rows[0].keys())
    thead = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    tb = "".join("<tr>" + "".join(f"<td>{_esc(r.get(c,''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<div style='overflow:auto'><table><thead><tr>{thead}</tr></thead><tbody>{tb}</tbody></table></div>"

