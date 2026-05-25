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
    missing_values = ["NA", "N/A", "ND","Missing", "missing","null", "None","N","", " "]
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

#Remove non-RA patients (VACCINE cohort) and rows without RA clinical measurements
def filter_ra_cohort(df: pd.DataFrame, study_col: str = "Study", ra_value: str = "TACERA", verbose: bool = True) -> pd.DataFrame:
    """ Filters dataset to include only RA patients (TACERA cohort).
    This removes: - VACCINE cohort (non-RA) - rows without RA clinical measurements
    Parameters:
    -----------
    df : pd.DataFrame - Input clinical dataset
    study_col : str - Column indicating study/cohort membership
    ra_value : str - Value representing RA cohort (default: TACERA)
    verbose : bool - If True, prints filtering summary
    Returns:
    --------
    pd.DataFrame - Filtered RA-only dataset
    """
    if study_col not in df.columns:
        raise ValueError(f"Column '{study_col}' not found in DataFrame")
    before = len(df)
    df_filtered = df[df[study_col] == ra_value].copy()
    after = len(df_filtered)
    if verbose:
        print("\n===== RA COHORT FILTER =====")
        print(f"Before filtering: {before} rows")
        print(f"After filtering:  {after} rows")
        print(f"Removed (non-RA / other cohorts): {before - after}")
        print("===========================\n")
    return df_filtered

# Patient-level clinical cleaning
def clean_patient_level_data(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Full contract-driven clinical preprocessing pipeline."""
    df = df.copy()
    # 0. Normalize raw inconsistencies
    df = clean_invalid_numeric_formats(df, contract)
    # 1. Fix data types
    df = fix_dtypes(df, contract)
    # 2. Standardize binary variables
    df = standardize_binary(df, contract)
    # 3. Drop useless columns (safe at patient level)
    df, _, _ = drop_useless_columns(df, contract)
    return df

def clean_event_level_data(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Light preprocessing for event-level datasets (steroids, meds) Goal: make raw event data aggregation-safe WITHOUT changing meaning.
    Does NOT: - impute missing values - encode categories - drop columns
    """
    df = df.copy()
    # 1. STANDARDISE MISSING VALUES
    df = df.replace(["NA", "N/A", "", " ", "null", "None", "ND"], np.nan)

    # 2. PARSE DATE COLUMNS
    for col in contract.get("datetime_columns", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # 3. CLEAN NUMERIC FIELDS (safe coercion only, no imputation)
    for col in contract.get("numeric_columns", []):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)   # decimal fix
                .str.replace(">", "", regex=False)    # censoring removal
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 4. STANDARDISE BINARY FIELDS (light only)
    binary_map = {"y": 1, "n": 0,"yes": 1, "no": 0,"true": 1, "false": 0,"0": 0, "1": 1}
    for col in contract.get("binary_columns", []):
        if col in df.columns:
            cleaned = df[col].astype(str).str.strip().str.lower()
            df[col] = cleaned.replace(binary_map)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 5. LIGHT STRING NORMALISATION (no encoding)
    for col in contract.get("categorical_columns", []):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def aggregate_steroids(df: pd.DataFrame) -> pd.DataFrame:
    """ Aggregate steroid event data to patient level. Output: one row per Digest (patient).
    """
    df = df.copy()
    # Ensure dates are sorted correctly for time-based features
    if "Date Given" in df.columns:
        df = df.sort_values(["Digest", "Date Given"])
    agg = df.groupby("Digest").agg(
        steroid_injection_count=("Steroid", "count"),
        total_dose=("Dose", "sum"),
        mean_dose=("Dose", "mean"),
        max_dose=("Dose", "max"),
        # how many intra vs intra-articular / intramuscular events
        intraarticular_count=("Route", lambda x: (x.str.lower().str.contains("intraarticular")).sum()),
        intramuscular_count=("Route", lambda x: (x.str.lower().str.contains("intramuscular")).sum()),
        first_injection=("Date Given", "min"),
        last_injection=("Date Given", "max")
    ).reset_index()
    return agg

def aggregate_meds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate medication event data to patient level. Output: one row per Digest (patient)."""
    df = df.copy()
    if "Date Started" in df.columns:
        df = df.sort_values(["Digest", "Date Started"])
    agg = df.groupby("Digest").agg(
        med_event_count=("RA Medication", "count"),
        unique_medications=("RA Medication", "nunique"),

        total_dose=("Dose", "sum"),
        mean_dose=("Dose", "mean"),

        first_med_date=("Date Started", "min"),
        last_med_date=("Date Started", "max")
    ).reset_index()
    return agg

def merge_patient_datasets(df_clinical, df_steroids, df_meds):
    """Merge clinical + aggregated event data into one patient-level dataset."""
    df = df_clinical.copy()
    # 1. Merge steroids
    df = df.merge(df_steroids, on="Digest", how="left")
    # 2. Merge meds
    df = df.merge(df_meds, on="Digest", how="left")
    return df

def final_preprocessing_full_dataset(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Final preprocessing step after merging all patient-level datasets. Produces analysis-ready dataset (NOT feature-selected ML dataset)."""
    df = df.copy()
    # 1. REMOVE CONSTANT FEATURES
    nunique = df.nunique(dropna=True)
    df = df.loc[:, nunique > 1]
    # 2. IMPUTE MISSING VALUES
    df = impute_missing(df, contract)
    # 3. ENCODE CATEGORICAL VARIABLES
    df = encode_categories(df, contract)
    return df


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