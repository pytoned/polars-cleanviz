from .clean import clean_column_names
from .plots import corr_heatmap, dist_plot, distplot, missingval_plot, cat_plot, corr_plot, convert_datatypes, drop_missing, data_cleaning
from .utils import save_fig
from .profiler import profile_quick, save_profile
from .profiling import profile_report, profile_config, ProfileReport, ProfileConfig

__all__ = [
    "cat_plot",        # New klib function
    "clean_column_names",
    "convert_datatypes",  # New klib function
    "corr_heatmap",
    "corr_plot",       # New enhanced klib function
    "data_cleaning",   # New klib function  
    "dist_plot",       # New klib-compatible name
    "distplot",        # Backward compatibility
    "drop_missing",    # New klib function
    "missingval_plot",
    "save_fig",
    "profile_quick",
    "save_profile",
    "profile_report",  # New lowercase name
    "profile_config",  # New lowercase name
    "ProfileReport",   # Backward compatibility
    "ProfileConfig",   # Backward compatibility
]
__version__ = "0.4.1"
