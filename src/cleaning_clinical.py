from typing import Tuple, List
import pandas as pd
import numpy as np

def clean_invalid_numeric_formats(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """ Cleans malformed numeric inputs before dtype conversion."""
    df = df.copy()
    numeric_cols = []
    numeric_cols += contract.get("numeric_columns", [])
    numeric_cols += contract.get("static_numeric", [])
    numeric_cols += contract.get("longitudinal_numeric", [])
    for col in numeric_cols:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .replace([
                    "NA", "N/A", "ND", "Missing",
                    "missing", "null", "None", "",
                    " ", "-99", "-99.0"
                ], np.nan)
            )
            # European decimal commas
            cleaned = cleaned.str.replace(",", ".", regex=False)
            # Remove inequality signs
            cleaned = cleaned.str.replace(">", "", regex=False)
            cleaned = cleaned.str.replace("<", "", regex=False)
            # Remove impossible slash values like 120/80
            cleaned = cleaned.replace(
                to_replace=r"^\d+/\d+$",
                value=np.nan,
                regex=True
            )
            df[col] = cleaned
    return df

def fix_dtypes(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """ Standardizes missing values and converts columns to appropriate dtypes based on the data contract."""
    df = df.copy()
    # 1. Standardize missing values
    missing_values = ["NA", "N/A", "ND","Missing", "missing","null", "None","", " "]
    df = df.replace(missing_values, np.nan)
    # 2. Convert numeric columns
    numeric_cols = []
    numeric_cols += contract.get("numeric_columns", [])
    numeric_cols += contract.get("static_numeric", [])
    numeric_cols += contract.get("longitudinal_numeric", [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 3. Convert datetime columns
    datetime_cols = contract.get("datetime_columns", [])
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def standardize_binary(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Standardizes binary columns into 0/1 values based on the data contract."""
    df = df.copy()
    binary_map = {"y": 1,"n": 0,"yes": 1,"no": 0,"true": 1,"false": 0,"1": 1,"0": 0}
    binary_cols = contract.get("binary_columns", [])
    for col in binary_cols:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            cleaned = cleaned.replace(binary_map)
            # restore proper NaN values
            cleaned = cleaned.replace("nan", np.nan)
            df[col] = cleaned
    return df

def drop_useless_columns(df: pd.DataFrame, contract: dict, missing_threshold: float = 0.4) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Drops:- fully empty columns - high missing columns - constant columns while protecting important contract-defined columns.
    Returns: - cleaned DataFrame - list of dropped high-missing columns - list of dropped constant columns
    """
    df = df.copy()
    # Protected columns
    protected_cols = set()
    protected_cols.update(contract.get("id_columns", []))
    protected_cols.update(contract.get("key_columns", {}).keys())
    protected_cols.update(contract.get("cohort_columns", []))
    protected_cols.update(contract.get("labels", []))
    # 1. Fully missing columns
    fully_missing = [
        col for col in df.columns
        if df[col].isna().all() and col not in protected_cols
    ]
    df = df.drop(columns=fully_missing)
    # 2. High missing columns
    missing_ratio = df.isna().mean()
    high_missing_cols = [
        col for col in missing_ratio[missing_ratio > missing_threshold].index
        if col not in protected_cols
    ]
    df = df.drop(columns=high_missing_cols)
    # 3. Constant columns
    nunique = df.nunique(dropna=True)
    constant_cols = [
        col for col in nunique[nunique <= 1].index
        if col not in protected_cols
    ]
    df = df.drop(columns=constant_cols)
    return df, high_missing_cols, constant_cols

def encode_categories(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Encodes categorical columns using factorization based on the data contract. IDs and datetime columns are protected automatically.
    """
    df = df.copy()
    # Collect categorical columns
    categorical_cols = []
    categorical_cols += contract.get("categorical_columns", [])
    categorical_cols += contract.get("cohort_columns", [])
    categorical_cols += contract.get("coded_categorical_columns", [])
    # Encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col], _ = pd.factorize(df[col])
    return df

import pandas as pd


def impute_missing(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Imputes missing values based on column roles defined in the data contract. Numeric columns -> median Categorical columns -> mode ID columns are never imputed.
    """
    df = df.copy()
    # Numeric columns
    numeric_cols = []
    numeric_cols += contract.get("numeric_columns", [])
    numeric_cols += contract.get("static_numeric", [])
    numeric_cols += contract.get("longitudinal_numeric", [])

    for col in numeric_cols:
        if col in df.columns:
            if df[col].isna().sum() > 0:
                median = df[col].median()
                df[col] = df[col].fillna(median)

    # Categorical columns
    categorical_cols = []
    categorical_cols += contract.get("categorical_columns", [])
    categorical_cols += contract.get("cohort_columns", [])
    categorical_cols += contract.get("coded_categorical_columns", [])
    categorical_cols += contract.get("binary_columns", [])

    for col in categorical_cols:
        if col in df.columns:
            if df[col].isna().sum() > 0:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
    return df

import pandas as pd
import numpy as np

def clean_raw_artifacts(df, contract):
    df = df.copy()
    # 1. basic missing encodings
    df = df.replace([
        "NA", "N/A", "", " ", "null", "None", "ND",
        "-99", "-99.0"
    ], np.nan)
    # 2. datetime cleanup
    for col in contract.get("datetime_columns", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].replace(pd.Timestamp("1900-01-01"), pd.NaT)
    return df