"""Inspection Utilities for RA-MAP Dataset

Module provides simple overview functions for DataFrames loaded by src.load_data.
Designed for initial exploration.
Output is intended for Jupyter notebooks

Functions:
    show_sheet_overview: List all keys in a dict of DataFrames
    show_dataset_structure: Print shape of each DataFrame in the dict
    inspect_dataset: Generic inspection fo DataFrame
    inspect_omics_dataset: Specific inspection of omics Dataframe. Borrows inspect_dataset
    
Example Usage:
    from src.load_data import load_all_sheets
    from src.check_data import show_sheet_overview, inspect_dataset

    data = load_all_sheets()
    show_sheet_overview(data)
    inspect_dataset(data['df_clinical'], name="clinical")
    inspect_omics_dataset(data['df_expMatrix'], name="Expression Matrix")
"""


from IPython.core.display_functions import display
import pandas as pd


def show_sheet_overview(data: dict) -> None:
    """Print the keys (sheet names) of our dictionary of DataFrames produced by load_data.py

    Args:
        data: Dictionary mapping sheet names to DataFrames
    """
    print("All dataset keys:")
    print(data.keys())


def show_dataset_structure(data: dict) -> None:
    """Prints shape overview of all loaded DataFrames.
    
    Args:
        data: Dictionary mapping sheet names to Dataframes
    """
    print("\n===== DATASET STRUCTURE OVERVIEW =====\n")

    for name, df in data.items():
        print(f"{name}: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n===== END =====")


def inspect_dataset(
        df: pd.DataFrame,
        name: str="dataset",
        n_rows: int=5,
        missing_top: int=10
        ) -> None:
    """Generic function to inspect a DataFrame by samples, shape, dtypes, and missing values.

    Args:
        df: Dataframe to inspect
        name: Display name to set header
        n_rows: Number of shown rows
        missing_top: Number of columns to show for the missing-values ranking in descending order
    """
    print(f"\n===== {name.upper()} =====")

    print("\nSAMPLE:")
    display(df.sample(n_rows))

    print("\nSHAPE:")
    print(df.shape)

    print("\nDTYPES:")
    print(df.dtypes)

    print(f"\nMISSING VALUES (Top {missing_top}):")
    print(df.isnull().sum().sort_values(ascending=False).head(missing_top))


def inspect_omics_dataset(
    df: pd.DataFrame,
    name: str = "omics dataset",
    dtype_sample: int = 10,
    describe_sample: int = 10,
) -> None:
    """Inspect omics-style datasets (gene expression, protein expression, etc.).
    
    Extends inspect_dataset with random samples of dtypes and numeric
    summary statistics, which are typical for omics data.
    
    Args:
        df: DataFrame with omics data. Typical inputs:
            - data['df_Samples']: Protogen samples
            - data['df_Samples_Annotation']: Sample annotations  
            - data['df_expMatrix']: SOMAscan expression matrix
            - data['df_sampMatrix']: SOMAscan sample matrix
        name: Display name for the header. Defaults to "omics dataset".
        dtype_sample: Number of dtype samples to show. Defaults to 10.
        describe_sample: Number of features in numeric summary. Defaults to 10.
    """
    inspect_dataset(df, name)
    print(f"\nDTYPES ({dtype_sample} random samples):")
    print(df.dtypes.sample(dtype_sample))
    print(f"\nSHORT NUMERIC SUMMARY (random {describe_sample} samples):")
    display(df.describe().T.sample(describe_sample))