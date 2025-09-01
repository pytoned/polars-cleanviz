from .clean import clean_column_names
from .plots import corr_heatmap, dist_plot, missingval_plot, cat_plot, corr_plot, convert_datatypes, drop_missing, data_cleaning
from .utils import save_fig
from .xray import xray
from . import datasets

__all__ = [
    "cat_plot",        # New klib function
    "clean_column_names",
    "convert_datatypes",  # New klib function
    "corr_heatmap",
    "corr_plot",       # New enhanced klib function
    "data_cleaning",   # New klib function
    "datasets",        # Dataset loading utilities
    "dist_plot",       # Distribution plotting
    "drop_missing",    # New klib function
    "missingval_plot",
    "save_fig",
    "xray",            # Comprehensive data analysis and quality assessment
]
__version__ = "1.0.0"

# Package metadata
__title__ = "polarscope"
__description__ = "🔬 Professional data inspection and visualization toolkit for Polars"
__author__ = "Anders & Co."
