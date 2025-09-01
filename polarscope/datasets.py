"""
Dataset loading utilities for Polarscope.

This module provides easy access to built-in datasets for testing and demonstration
of Polarscope functionality.
"""

import polars as pl
from pathlib import Path
from typing import Optional
import importlib.resources


def _get_data_path(filename: str) -> Path:
    """Get the path to a dataset file, handling both development and installed package scenarios."""
    try:
        # Try to use importlib.resources for installed packages
        import importlib.resources as pkg_resources
        try:
            # Python 3.9+ style
            with pkg_resources.files("polarscope") / "data" / filename as data_path:
                return Path(data_path)
        except AttributeError:
            # Python 3.8 fallback
            with pkg_resources.path("polarscope.data", filename) as data_path:
                return Path(data_path)
    except (ImportError, FileNotFoundError):
        # Development mode or fallback - look for datasets folder
        current_file = Path(__file__).parent
        project_root = current_file.parent
        datasets_path = project_root / "datasets" / filename
        
        if datasets_path.exists():
            return datasets_path
        
        # Last resort - check if we're in the polarscope directory
        alt_path = current_file.parent / "datasets" / filename
        if alt_path.exists():
            return alt_path
            
        raise FileNotFoundError(f"Could not find dataset file: {filename}")


def load_titanic(return_polars: bool = True) -> pl.DataFrame:
    """
    Load the Titanic dataset.
    
    The Titanic dataset contains passenger information from the RMS Titanic,
    including survival status, passenger class, demographics, and fare information.
    Perfect for demonstrating data analysis, missing value handling, and 
    classification tasks.
    
    Parameters
    ----------
    return_polars : bool, default True
        If True, returns a Polars DataFrame. If False, returns the file path
        as a string for custom loading.
    
    Returns
    -------
    pl.DataFrame or str
        Either a Polars DataFrame containing the Titanic data, or the path
        to the CSV file.
        
    Examples
    --------
    Load the Titanic dataset:
    
    >>> from polarscope.datasets import load_titanic
    >>> df = load_titanic()
    >>> print(df.shape)
    (156, 12)
    
    Get the file path instead:
    
    >>> file_path = load_titanic(return_polars=False)
    >>> df = pl.read_csv(file_path)
    
    Dataset Information
    -------------------
    - **Rows**: 156 passengers (subset of original dataset)
    - **Columns**: 12 features
    - **Missing values**: Yes (Age, Cabin, Embarked)
    - **Data types**: Mixed (numeric, string, categorical)
    
    Columns:
    - PassengerId: Unique passenger identifier
    - Survived: Survival status (0 = No, 1 = Yes)
    - Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - Name: Passenger name
    - Sex: Gender (male/female)
    - Age: Age in years
    - SibSp: Number of siblings/spouses aboard
    - Parch: Number of parents/children aboard
    - Ticket: Ticket number
    - Fare: Passenger fare
    - Cabin: Cabin number
    - Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    """
    data_path = _get_data_path("titanic.csv")
    
    if not return_polars:
        return str(data_path)
    
    return pl.read_csv(data_path)


def load_diabetes(return_polars: bool = True) -> pl.DataFrame:
    """
    Load the Diabetes dataset.
    
    The Pima Indians Diabetes dataset contains medical diagnostic information
    for predicting diabetes onset. All features are numeric, making it excellent
    for statistical analysis and demonstrating numeric data processing.
    
    Parameters
    ----------
    return_polars : bool, default True
        If True, returns a Polars DataFrame. If False, returns the file path
        as a string for custom loading.
    
    Returns
    -------
    pl.DataFrame or str
        Either a Polars DataFrame containing the diabetes data, or the path
        to the CSV file.
        
    Examples
    --------
    Load the diabetes dataset:
    
    >>> from polarscope.datasets import load_diabetes
    >>> df = load_diabetes()
    >>> print(df.shape)
    (768, 9)
    
    Get the file path instead:
    
    >>> file_path = load_diabetes(return_polars=False)
    >>> df = pl.read_csv(file_path)
    
    Dataset Information
    -------------------
    - **Rows**: 768 patients
    - **Columns**: 9 features (8 predictors + 1 target)
    - **Missing values**: None (but contains zeros that may represent missing)
    - **Data types**: All numeric
    
    Columns:
    - Pregnancies: Number of times pregnant
    - Glucose: Plasma glucose concentration
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index (weight in kg/(height in m)^2)
    - DiabetesPedigreeFunction: Diabetes pedigree function
    - Age: Age in years
    - Outcome: Class variable (0 or 1) - diabetes diagnosis
    """
    data_path = _get_data_path("diabetes.csv")
    
    if not return_polars:
        return str(data_path)
    
    return pl.read_csv(data_path)


# Convenience aliases for easier access
titanic = load_titanic
diabetes = load_diabetes


def list_datasets() -> list[str]:
    """
    List all available datasets.
    
    Returns
    -------
    list[str]
        List of available dataset names.
        
    Examples
    --------
    >>> from polarscope.datasets import list_datasets
    >>> datasets = list_datasets()
    >>> print(datasets)
    ['titanic', 'diabetes']
    """
    return ['titanic', 'diabetes']


def dataset_info(name: str) -> str:
    """
    Get information about a specific dataset.
    
    Parameters
    ----------
    name : str
        Name of the dataset ('titanic' or 'diabetes').
        
    Returns
    -------
    str
        Detailed information about the dataset.
        
    Examples
    --------
    >>> from polarscope.datasets import dataset_info
    >>> info = dataset_info('titanic')
    >>> print(info)
    """
    if name.lower() == 'titanic':
        return """
Titanic Dataset
===============
Famous passenger manifest from RMS Titanic with survival outcomes.

• Rows: 156 passengers
• Columns: 12 features  
• Missing values: Yes (Age, Cabin, Embarked)
• Use case: Classification, missing value analysis, categorical data
• Perfect for: Demonstrating xray(), missing value plots, correlation analysis
        """.strip()
    
    elif name.lower() == 'diabetes':
        return """
Diabetes Dataset  
================
Pima Indians Diabetes medical diagnostic data.

• Rows: 768 patients
• Columns: 9 features
• Missing values: None (but zeros may represent missing)
• Use case: Medical prediction, statistical analysis
• Perfect for: Demonstrating statistical tests, distribution analysis, correlation
        """.strip()
    
    else:
        available = ', '.join(list_datasets())
        return f"Unknown dataset '{name}'. Available datasets: {available}"


# For backward compatibility and convenience
__all__ = [
    'load_titanic', 'load_diabetes', 'titanic', 'diabetes',
    'list_datasets', 'dataset_info'
]
