# 🔬 Polarscope

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polars.svg)](https://pypi.org/project/polars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Professional data inspection and visualization toolkit for Polars** 🐻‍❄️

Polarscope is a modern, fast, and comprehensive data analysis library built exclusively for Polars. It provides advanced X-ray analysis, beautiful visualizations, and comprehensive data quality assessment - all with native Polars performance.

---

## ✨ Key Features

### 🔬 **X-ray Analysis**
- **`xray(df)`** → Comprehensive data quality assessment with beautiful Great Tables output
- Advanced statistics: skewness, kurtosis, outliers, normality tests, data quality flags
- Performance metrics: execution timing and memory usage tracking
- Customizable output: minimal or expanded views with professional formatting

### 📊 **Multi-Backend Visualization**
- **`corr_heatmap(df)`** → Correlation analysis with advanced filtering and target analysis
- **`dist_plot(df)`** → Distribution plotting with statistical overlays
- **`missingval_plot(df)`** → Missing value pattern analysis
- **`cat_plot(df)`** → Categorical data frequency analysis
- **`corr_plot(df)`** → Interactive correlation exploration
- **3 backends supported**: Plotly (default), Seaborn, Altair

### 🧹 **Data Processing**
- **`clean_column_names(df)`** → Normalize and deduplicate column names
- **`data_cleaning(df)`** → Comprehensive automated data cleaning pipeline
- **`convert_datatypes(df)`** → Intelligent dtype optimization
- **`drop_missing(df)`** → Advanced missing value handling

### 🛠️ **Utilities**
- **`save_fig()`** → Universal figure saving (PNG, HTML, SVG, etc.)
- Native Polars performance - no Pandas dependency
- Type-safe with comprehensive docstrings

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install polarscope

# With all optional dependencies
pip install polarscope[all]

# Individual extras
pip install polarscope[plotly,seaborn,great_tables]
```

### Basic Usage

```python
import polars as pl
import polarscope as plc

# Load your data
df = pl.read_csv("data.csv")

# 🔬 Get comprehensive X-ray analysis
plc.xray(df)

# 📊 Create beautiful visualizations
plc.corr_heatmap(df, backend="plotly")
plc.dist_plot(df, column="price", backend="seaborn")
plc.missingval_plot(df, backend="altair")

# 🧹 Clean and optimize your data
df_clean = plc.data_cleaning(df)
df_optimized = plc.convert_datatypes(df_clean)
```

### Advanced X-ray Analysis

```python
# Expanded analysis with custom settings
plc.xray(
    df,
    expanded=True,                    # Show all statistics
    corr_target="target_column",      # Correlation to specific column
    outlier_method="iqr",            # Outlier detection method
    decimals=3,                      # Formatting precision
    compact=True                     # Compact number formatting
)

# Custom percentiles and quality thresholds
plc.xray(
    df,
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    missing_threshold=0.3,           # Flag high missingness
    outlier_threshold=0.05,          # Flag outlier-heavy columns
    normality_test=True,             # Include normality tests
    uniformity_test=True             # Include uniformity tests
)
```

---

## 📈 Performance

Polarscope is built for speed and efficiency:

- **Lightning fast**: Native Polars operations throughout
- **Memory efficient**: Optimized for large datasets
- **Scalable**: Handles millions of rows with ease
- **Professional**: Comprehensive test suite with 100% pass rate

---

## 🎯 Why Polarscope?

### **Polars-Native**
- Built exclusively for Polars - no Pandas dependencies
- Native performance and memory efficiency
- Type-safe operations with Polars' query engine

### **Professional Quality**
- Production-ready with comprehensive testing
- Beautiful, customizable output with Great Tables
- Scientific precision in statistical analysis

### **Multi-Backend Flexibility**
- Choose your preferred visualization backend
- Consistent API across Plotly, Seaborn, and Altair
- Easy switching between backends

### **Comprehensive Analysis**
- Goes beyond basic `.describe()` functionality
- Advanced statistics and quality assessment
- Data quality flags and recommendations

---

## 📚 Documentation

### Core Functions

| Function | Description | Key Features |
|----------|-------------|--------------|
| `xray()` | Comprehensive data analysis | Statistics, quality flags, performance metrics |
| `corr_heatmap()` | Correlation visualization | Multiple backends, filtering, target analysis |
| `dist_plot()` | Distribution analysis | Statistical overlays, multiple backends |
| `missingval_plot()` | Missing value patterns | Percentage/absolute counts, pattern analysis |
| `data_cleaning()` | Automated cleaning | Duplicates, missing values, optimization |

### Backends

- **Plotly** (default): Interactive, web-ready visualizations
- **Seaborn**: Statistical plotting with beautiful aesthetics  
- **Altair**: Grammar of graphics approach

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by [`klib`](https://github.com/akanz1/klib) and [`ydata-profiling`](https://github.com/ydataai/ydata-profiling), but built from the ground up for Polars performance and modern data science workflows.

---

**🔬 Ready to X-ray your data? Install Polarscope today!**