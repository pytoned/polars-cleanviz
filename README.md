# polars-cleanviz

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polars.svg)](https://pypi.org/project/polars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**polars-cleanviz** is a lightweight Polars-only data cleaning & visualization helper.  
It’s inspired by [`klib`](https://github.com/akanz1/klib) and [`ydata-profiling`](https://github.com/ydataai/ydata-profiling), but kept **fast, small, and focused**.

---

## ✨ Features

- 🔤 `clean_column_names(df)` → normalize & deduplicate column names  
- 📊 `corr_heatmap(df, backend=...)` → correlation heatmap (`matplotlib` / `plotly` / `altair`)  
- 📈 `distplot(df, backend=...)` → numeric column distribution plot  
- ❓ `missingval_plot(df, backend=...)` → missing values per column  
- 💾 `save_fig(obj, path)` → save any figure (`.png`, `.html`, etc.)  
- 📋 `profile_quick(df)` → **minimal profiler**: summary, dtypes, missing, correlation, distplot  
- 📂 `save_profile(report, dir)` → save profiler outputs to files

New (Polars-only, ydata-style):

- 🧭 `ProfileReport(df)` → build a single-file HTML profiling report (table stats, variables, missingness, correlations, distributions, samples)

---

## 📦 Installation

### From GitHub
```bash
pip install git+https://github.com/pytoned/polars-cleanviz.git
