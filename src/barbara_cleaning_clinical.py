from typing import Tuple, List
import pandas as pd
import numpy as np
from typing import Union, List
from typing import Dict, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

def drop_cohort_columns(df: pd.DataFrame, contract: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Removes cohort / administrative columns defined in the contract.
    """

    df = df.copy()

    cohort_cols = [
        col
        for col in contract.get("cohort_columns", [])
        if col in df.columns
    ]

    df = df.drop(columns=cohort_cols)

    if verbose:
        print("\n===== COHORT COLUMN REMOVAL =====")
        print(f"Removed {len(cohort_cols)} columns:")
        print(cohort_cols)
        print(f"Remaining shape: {df.shape}")
        print("=================================\n")

    return df


def clean_binary_columns(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """
    Standardises binary columns:
    - Yes/No, Y/N, True/False → 1/0
    - Female/Male → 0/1
    - ND / missing → NaN
    """

    import numpy as np

    df = df.copy()

    binary_map = {
        "y": 1, "yes": 1, "true": 1, "1": 1,
        "n": 0, "no": 0, "false": 0, "0": 0,
        "female": 0, "male": 1
    }

    cols = contract.get("binary_columns", [])

    for col in cols:
        if col in df.columns:

            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            cleaned = cleaned.replace(binary_map)

            # everything unknown → NaN
            cleaned = cleaned.replace(
                ["nd", "n/a", "na", "missing", "none", ""],
                np.nan
            )

            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df

def drop_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drops one or more columns from a DataFrame safely.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    columns : str or list of str
        Column name(s) to drop
    verbose : bool
        Print what was removed

    Returns:
    --------
    pd.DataFrame
        DataFrame with selected columns removed
    """

    df = df.copy()

    # allow single column or list
    if isinstance(columns, str):
        columns = [columns]

    # only drop columns that actually exist
    existing_cols = [col for col in columns if col in df.columns]

    df = df.drop(columns=existing_cols)

    if verbose:
        print("\n===== COLUMN DROP =====")
        print(f"Dropped columns: {existing_cols}")
        print(f"Remaining shape: {df.shape}")
        print("=======================\n")

    return df

def clean_remission_month(df: pd.DataFrame, col: str = "Remission month") -> pd.DataFrame:
    """
    Splits remission month into:
    - remission_event (0/1)
    - remission_time (numeric, NaN if no remission)
    """

    df = df.copy()

    cleaned = df[col].astype(str).str.strip().str.upper()

    # event indicator
    df["remission_event"] = cleaned.apply(lambda x: 0 if x == "N" else 1)

    # time variable
    df["remission_time"] = pd.to_numeric(cleaned, errors="coerce")

    # optional: drop original column
    df = df.drop(columns=[col])

    return df


def clean_clinical_numeric(df: pd.DataFrame, contract: dict, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1. build numeric list
    numeric_cols = (
        contract.get("longitudinal_numeric", []) +
        contract.get("static_numeric", [])
    )

    # 2. KEEP ONLY existing columns (CRITICAL FIX)
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    missing_values = {
        "ND", "N/A", "NA", "missing", "Missing",
        "null", "None", "", " ", "-99", "-99.0"
    }

    for col in numeric_cols:
        cleaned = df[col].astype(str).str.strip()

        # unify missing
        cleaned = cleaned.replace(list(missing_values), np.nan)

        # numeric normalization
        cleaned = cleaned.str.replace(",", ".", regex=False)
        cleaned = cleaned.str.replace(">", "", regex=False)
        cleaned = cleaned.str.replace("<", "", regex=False)

        # remove ratio-like values
        cleaned = cleaned.replace(r"^\d+/\d+$", np.nan, regex=True)

        df[col] = pd.to_numeric(cleaned, errors="coerce")

    if verbose:
        print("\nNumeric preprocessing completed")
        print("Shape:", df.shape)

        if len(numeric_cols) > 0:
            print("\nTop missingness:")
            print(df[numeric_cols].isna().mean().sort_values(ascending=False).head(10))

    return df


def postprocess_missing_values(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """
    Standardises missing values AFTER type-specific preprocessing.
    Keeps:
    - NaNs in features
    - NaNs in remission_time (meaning: no remission)
    """

    df = df.copy()

    # -------------------------
    # 1. Binary columns
    # -------------------------
    binary_cols = contract.get("binary_columns", [])

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                ["ND", "N/A", "NA", "missing", "Missing", "", " "],
                np.nan
            )

    # -------------------------
    # 2. Numeric columns
    # -------------------------
    numeric_cols = (
        contract.get("longitudinal_numeric", []) +
        contract.get("static_numeric", [])
    )

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 3. IMPORTANT: outcome handling
    # -------------------------
    # remission_time: NaN = no remission (keep as is!)
    # DO NOT impute

    return df

#***************** Normalisation Clinical ************************#
# Normalise binary_columns (only imputation possible, no need to impute - missing values are also signals)
# Normalise outcome_binary and outcome_continuous (no imputation needed - rows will be dropped)
# Normalise static_numeric and longitudinal_numeric (impute missing values with median)

def impute_numeric_values(
    df: pd.DataFrame,
    contract: Dict,
    strategy: str = "median",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Imputes ONLY numeric clinical features.

    - longitudinal_numeric → median imputation
    - static_numeric → median imputation
    - does NOT touch binary or outcome columns

    Assumes numeric columns are already cleaned (pd.to_numeric applied).
    """

    df = df.copy()

    # -------------------------
    # Collect numeric columns
    # -------------------------
    numeric_cols = (
        contract.get("longitudinal_numeric", []) +
        contract.get("static_numeric", [])
    )

    # keep only existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if len(numeric_cols) == 0:
        if verbose:
            print("[Numeric Imputation] No numeric columns found.")
        return df

    # -------------------------
    # Imputation
    # -------------------------
    imputer = SimpleImputer(strategy=strategy)

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    if verbose:
        print("\n===== NUMERIC IMPUTATION =====")
        print(f"Strategy: {strategy}")
        print(f"Imputed columns: {len(numeric_cols)}")
        print(f"Shape: {df.shape}")
        print("==============================\n")

    return df

# Normalising and scaling the entire merged set: clinical, meds, steroids #

def build_global_transform_groups(clinical, steroids, meds):
    """
    Build unified feature groups across all datasets
    """

    binary_cols = clinical.get("binary_columns", [])

    numeric_cols = (
        clinical.get("longitudinal_numeric", []) +
        clinical.get("static_numeric", []) +
        steroids.get("numeric_columns", []) +
        meds.get("numeric_columns", [])
    )

    datetime_cols = (
        steroids.get("datetime_columns", []) +
        meds.get("datetime_columns", [])
    )

    return binary_cols, numeric_cols, datetime_cols

def define_log_columns():
    """
    Define skewed variables across ALL feature groups
    (clinical + steroids + meds)
    """

    log_cols = [

        # -------------------------
        # inflammation markers
        # -------------------------
        "CRP.0M", "CRP.6M", "CRP.9M",
        "ESR.0M",

        # -------------------------
        # blood counts (clinical)
        # -------------------------
        "PLT.0M", "PLT.6M",
        "WBC.0M", "WBC.6M",

        "NEUTROPHILS.0M", "NEUTROPHILS.6M",
        "LYMPHOCYTES.0M", "LYMPHOCYTES.6M",
        "MONOCYTES.0M", "MONOCYTES.6M",
        "EOSINOPHILS.0M", "EOSINOPHILS.6M",
        "BASOPHILS.0M", "BASOPHILS.6M",

        # -------------------------
        # symptoms
        # -------------------------
        "PAIN.0M", "PAIN.6M",
        "TOTAL.SWOLLEN.0M", "TOTAL.SWOLLEN.6M", "TOTAL.SWOLLEN.9M",
        "TOTAL.TENDER.0M", "TOTAL.TENDER.6M", "TOTAL.TENDER.9M",

        # -------------------------
        # steroid features
        # -------------------------
        "steroid_injection_count",
        "total_dose_x", "mean_dose_x", "max_dose",
        "intraarticular_count", "intramuscular_count",

        # -------------------------
        # medication features
        # -------------------------
        "med_event_count",
        "unique_medications",
        "total_dose_y", "mean_dose_y"
    ]

    return log_cols

def apply_log_transform(df, cols):
    df = df.copy()

    cols = [c for c in cols if c in df.columns]

    for col in cols:
        df[col] = np.log1p(df[col])

    return df

def apply_scaling(df, cols):
    df = df.copy()

    cols = [c for c in cols if c in df.columns]

    scaler = StandardScaler()

    df[cols] = scaler.fit_transform(df[cols])

    return df, scaler

def preprocess_numeric_pipeline(df, clinical, steroids, meds):
    """
    Full numeric preprocessing pipeline:
    - log transform skewed variables
    - scale all numeric variables
    - leave binary variables untouched
    """

    df = df.copy()

    # 1. build global groups
    binary_cols, numeric_cols, datetime_cols = build_global_transform_groups(
        clinical, steroids, meds
    )

    # 2. define log columns (global)
    log_cols = define_log_columns()

    # 3. log transform
    df = apply_log_transform(df, log_cols)

    # 4. scale numeric
    df, scaler = apply_scaling(df, numeric_cols)

    return df, scaler