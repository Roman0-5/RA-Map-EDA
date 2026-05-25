import pandas as pd
import numpy as np
from src.cleaning_clinical import (clean_invalid_numeric_formats, fix_dtypes, standardize_binary, drop_useless_columns, encode_categories, impute_missing)

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
    """Light preprocessing for event-level datasets (steroids, meds).
    Goal: make raw event data aggregation-safe WITHOUT changing meaning.
    Does NOT: impute missing values, encode categories, or drop meaningful data.
    """
    df = df.copy()
    # 1. STANDARDISE MISSING VALUES
    df = df.replace(["NA", "N/A", "", " ", "null", "None", "ND"], np.nan)
    # 2. PARSE DATE COLUMNS
    for col in contract.get("datetime_columns", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # 3. CLEAN NUMERIC FIELDS (safe coercion only)
    for col in contract.get("numeric_columns", []):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 4. STANDARDISE BINARY FIELDS
    binary_map = {
        "y": 1, "n": 0,
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "0": 0, "1": 1
    }
    for col in contract.get("binary_columns", []):
        if col in df.columns:
            cleaned = df[col].astype(str).str.strip().str.lower()
            df[col] = cleaned.replace(binary_map)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 5. LIGHT STRING NORMALISATION
    for col in contract.get("categorical_columns", []):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # 6. DROP TRUE DUPLICATE EVENTS (IMPORTANT FIX)
    dedup_cols = [
        "Digest",
        "Date Given",
        "Steroid",
        "Dose",
        "Route",
        "Joint Injected"
    ]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)
    return df

"""def aggregate_steroids(df: pd.DataFrame) -> pd.DataFrame:
    """""" Aggregate steroid event data to patient level. Output: one row per Digest (patient).
    """"""
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
    return agg"""

def aggregate_steroids(df: pd.DataFrame) -> pd.DataFrame:
    """ Aggregate steroid event data to patient level. Output: One row per patient (Digest)."""
    df = df.copy()
    # Sort chronologically
    if "Date Given" in df.columns:
        df = df.sort_values(["Digest", "Date Given"])
    agg = df.groupby("Digest").agg(
        # Event counts
        steroid_injection_count=("Steroid", "count"),
        # Dose summaries
        steroid_total_dose=("Dose", "sum"),
        steroid_mean_dose=("Dose", "mean"),
        steroid_max_dose=("Dose", "max"),
        # Route counts
        intraarticular_count=(
            "Route",
            lambda x: (
                x.astype(str)
                 .str.lower()
                 .str.contains("intraarticular", na=False)
            ).sum()
        ),
        intramuscular_count=(
            "Route",
            lambda x: (
                x.astype(str)
                 .str.lower()
                 .str.contains("intramuscular", na=False)
            ).sum()
        ),
        # Temporal features
        first_injection=("Date Given", "min"),
        last_injection=("Date Given", "max")
    ).reset_index()
    return agg

"""def aggregate_meds(df: pd.DataFrame) -> pd.DataFrame:
    """"""Aggregate medication event data to patient level. Output: one row per Digest (patient).""""""
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
    return agg """

def aggregate_meds(df: pd.DataFrame) -> pd.DataFrame:
    """ Aggregate medication event data to patient level. Output: One row per patient (Digest). """
    df = df.copy()
    # Sort chronologically
    if "Date Started" in df.columns:
        df = df.sort_values(["Digest", "Date Started"])
    agg = df.groupby("Digest").agg(
        # Medication exposure counts
        med_event_count=("RA Medication", "count"),
        unique_medications=("RA Medication", "nunique"),
        # Dose summaries
        med_total_dose=("Dose", "sum"),
        med_mean_dose=("Dose", "mean"),
        # Temporal features
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