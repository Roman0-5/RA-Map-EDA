from typing import Tuple, List
import pandas as pd
import numpy as np
from typing import Union, List
from typing import Dict, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_all_sheets():
    """Loads all sheets as a dataframe\n
        Returns:
                dict: Dictionary with the following DataFrames:
                    Clinical:
                    - df_clinical: Clinical patient data
                    - df_steroids: Intramuscular steroids data
                    - df_meds: RA medications
                    - df_glossary: Glossary
                    Protogen:
                    - df_Samples: Protogen samples
                    - df_Samples_Annotation: Sample annotations
                    Somascan:
                    - df_expMatrix: SOMAscan expression matrix
                    - df_sampMatrix: SOMAscan sample matrix
    """
    print('Loading clinical...')
    clinical_sheet = pd.read_excel('../../datasets_final/Clinical.xlsx', sheet_name=None)
    print('Loading protogen...')
    protogen_sheet = pd.read_excel('../../datasets_final/Protein.xlsx', sheet_name=None)

    print("Done loading")
    return {
        # clinical
        'df_clinical': clinical_sheet['OpenPseudonymised_RA_MAP_Clinic'],
        'df_steroids': clinical_sheet['intramuscular steroids'],
        'df_meds': clinical_sheet['RA Meds'],
        'df_glossary': clinical_sheet['Glossary'],
        # protogen
        'df_Samples': protogen_sheet['LIS_PG665-P01 RA MAP Samples Ex'],
        'df_Samples_Annotation': protogen_sheet['Sample annotation']
    }


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

def build_global_transform_groups(clinical, steroids, meds):
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
    Return candidate log columns (NOT guaranteed to exist)
    """
    return [
        "CRP.0M", "CRP.6M", "CRP.9M",
        "ESR.0M",

        "PLT.0M", "PLT.6M",
        "WBC.0M", "WBC.6M",

        "NEUTROPHILS.0M", "NEUTROPHILS.6M",
        "LYMPHOCYTES.0M", "LYMPHOCYTES.6M",
        "MONOCYTES.0M", "MONOCYTES.6M",
        "EOSINOPHILS.0M", "EOSINOPHILS.6M",
        "BASOPHILS.0M", "BASOPHILS.6M",

        "PAIN.0M", "PAIN.6M",
        "TOTAL.SWOLLEN.0M", "TOTAL.SWOLLEN.6M", "TOTAL.SWOLLEN.9M",
        "TOTAL.TENDER.0M", "TOTAL.TENDER.6M", "TOTAL.TENDER.9M",

        "steroid_injection_count",
        "total_dose_x", "mean_dose_x", "max_dose",
        "intraarticular_count", "intramuscular_count",

        "med_event_count",
        "unique_medications",
        "total_dose_y", "mean_dose_y"
    ]

def apply_log_transform(df, cols, debug=True):
    df = df.copy()

    cols = [c for c in cols if c in df.columns]

    if debug:
        print(f"[LOG] applying log1p to {len(cols)} columns")

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.log1p(df[col])

    return df

def apply_scaling(df, cols, debug=True):
    df = df.copy()

    cols = [c for c in cols if c in df.columns]

    if debug:
        print(f"[ZSCORE] scaling {len(cols)} columns")

    scaler = StandardScaler()

    df[cols] = scaler.fit_transform(df[cols])

    return df, scaler

def preprocess_log_pipeline(df, clinical, steroids, meds):
    """
    LOG ONLY (feature selection stage)
    """

    df = df.copy()

    _, numeric_cols, _ = build_global_transform_groups(
        clinical, steroids, meds
    )

    log_cols = define_log_columns()

    # IMPORTANT FIX: resolve to actual df columns
    log_cols = [c for c in log_cols if c in df.columns]

    df = apply_log_transform(df, log_cols)

    return df, numeric_cols

def preprocess_zscore_pipeline(df, clinical, steroids, meds):
    """
    Z-score AFTER log transform (modeling stage)
    """

    df = df.copy()

    _, numeric_cols, _ = build_global_transform_groups(
        clinical, steroids, meds
    )

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    df, scaler = apply_scaling(df, numeric_cols)

    return df, scaler

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

