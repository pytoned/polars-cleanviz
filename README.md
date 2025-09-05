# 🔬 Polarscope

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polars.svg)](https://pypi.org/project/polars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Simple data inspection tools for Polars** 🐻‍❄️

Polarscope is a basic data analysis library for Polars DataFrames. It provides an `xray()` function for data inspection and some plotting utilities. Still early in development with more features planned.

---

## ✨ Current Features

### 🔬 **Data Inspection**
- **`xray(df)`** → Basic data summary with statistics and data quality info
- Shows column types, missing values, basic stats (mean, std, percentiles)
- Optional expanded mode with additional metrics
- Great Tables formatting for nice output

### 📊 **Built-in Datasets**
- **`from polarscope.datasets import titanic, diabetes`** → Small datasets for testing
- Useful for trying out the library functions

### 🧹 **Basic Utilities** *(Limited functionality)*
- Some data cleaning functions (still being developed)
- Plotting functions (basic implementation)

---

## 🚀 Quick Start

### Installation

```bash
pip install polarscope
```

That's it! The main dependencies (Polars, Great Tables) will be installed automatically.

### Basic Usage

```python
import polars as pl
import polarscope as ps
from polarscope.datasets import titanic

# Use a built-in dataset or load your own
df = titanic()
# df = pl.read_csv("your_data.csv")

# Get basic data summary
ps.xray(df)

# More detailed analysis
ps.xray(df, expanded=True)

# Custom title and correlation analysis
ps.xray(df, title="My Data Analysis", corr_target="Survived")
```

### Available Options

```python
# Some useful parameters for xray()
ps.xray(
    df,
    expanded=True,              # Show more statistics
    title="Custom Title",       # Custom title for output
    corr_target="column_name",  # Show correlations with this column
    decimals=2,                 # Number formatting
    great_tables=False          # Return DataFrame instead of formatted table
)
```

---

## 🚧 Current Status

**Early development** - expect bugs and missing features!

- Main focus is on the `xray()` function
- Uses Polars for data processing (no Pandas dependency)
- Great Tables for nice-looking output
- Other functions are basic implementations or placeholders

## 📋 What Works

- ✅ Basic data inspection with `xray()`
- ✅ Built-in datasets (titanic, diabetes)
- ✅ Great Tables formatting
- ✅ Custom titles and correlation analysis
- ⚠️ Some plotting functions (limited)
- ⚠️ Some cleaning utilities (basic)

---

## 🤝 Contributing

This is a small project, but contributions are welcome! Feel free to report bugs or suggest improvements.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by [`klib`](https://github.com/akanz1/klib) and built with [`great_tables`](https://github.com/posit-dev/great-tables).

---

**🔬 A simple tool for basic Polars data inspection.**