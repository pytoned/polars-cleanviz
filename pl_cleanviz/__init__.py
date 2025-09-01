from .clean import clean_column_names
from .plots import corr_heatmap, distplot, missingval_plot
from .utils import save_fig
from .profiler import profile_quick, save_profile
from .profiling import profile_report, profile_config, ProfileReport, ProfileConfig

__all__ = [
    "clean_column_names",
    "corr_heatmap",
    "distplot",
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
