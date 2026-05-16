import pandas as pd
import numpy as np
from IPython.core.display_functions import display

from src.extraction import extract_dtypes


def data_quality_report(df):
    """Function to make a Dataqualityreport with df as Input"""
    #initializing objects from other function
    df, only_num, only_str, only_obj = extract_dtypes(df)
    print("Dataquality report")
    print()
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Total of {len(df) * len(df.columns)} data")

    
    print("")
    # 1. Missing values per Row
    print("Missing Values")
    missing = df.isna().sum() # total sum of missing values
    missing_pct = (missing / len(df) * 100).round(2) 
    missing_report = pd.DataFrame({ # new dataframe for missing values
        'Missing': missing,
        'Percentage': missing_pct
    })
    # Only show rows with missing values
    problems = missing_report[missing_report['Missing'] > 0]
    if len(problems) > 0:
        print(problems.sort_values('Missing', ascending=False))
    else:
        print("No missing values were found!")
    print()
    
    # 2. duplicates
    print("How many duplicates?")
    print(f"Duplicates: {df.duplicated().sum()}")
    print()
    
    # 3. dtype overview 
    for name, subset in [('numeric', only_num), ('string', only_str), ('object', only_obj)]:
        cols = subset.columns.tolist()
        print(f"{name}: {len(cols)} columns")
        print(f"{cols[:5]}{'...' if len(cols) > 5 else ''}")
        print()
    
    # 4. Numerical Values special cases and distribution
    print("Numerical Values")
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe()
    print(stats)
    print()

    for col in num_df.columns:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]

        lower = mean - (3 * std)
        upper = mean + (3 * std)
        outliers = ((num_df[col] < lower) | (num_df[col] > upper)).sum()

        print(f"{col}")
        print(f" Min: {stats.loc['min', col]:.2f} |"
              f" Max: {stats.loc['max', col]:.2f} |"
              f" Mean: {mean:.2f} |"
              f" Std: {std:.2f} |"
              f" Outliers: {outliers}")
    
    # 5. Consistency of string rows
    print("Strings")
    str_df = pd.concat([only_str, only_obj], axis=1)

    # all stats at once
    summary = str_df.agg([
        'nunique',
        'count',
        lambda x: x.dropna().str.startswith(' ').any() or x.dropna().str.endswith(' ').any(),
        lambda x: x.nunique() != x.dropna().str.lower().nunique()
    ])
    summary.index = ['Unique', 'Count', 'Has_Whitespace', 'Mixed_Case']
    print(summary.T)  # transposed for readability

def clinical_data_audit(df, name="dataset", missing_threshold=0.4):
    """
    Pre-cleaning audit for RA clinical-related tables.

    PURPOSE:
    - Understand data quality BEFORE any cleaning or feature engineering
    - Works on individual tables (clinical_scores, steroids, meds)
    - Does NOT assume ML-ready structure
    """

    print(f"\n{'='*70}")
    print(f"CLINICAL AUDIT: {name.upper()}")
    print(f"{'='*70}\n")

    # 1. STRUCTURE
    print("1. STRUCTURE")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. DUPLICATES
    print("\n2. DUPLICATES")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # 3. MISSING VALUES
    print("\n3. MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)).sort_values(ascending=False)
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    }).sort_values("missing_pct", ascending=False)
    high_missing = missing_report[missing_report["missing_pct"] > missing_threshold]
    mid_missing = missing_report[
        (missing_report["missing_pct"] > 0) &
        (missing_report["missing_pct"] <= missing_threshold)
    ]
    print(f"\nColumns > {missing_threshold*100:.0f}% missing (likely drop):")
    display(high_missing)
    print("\nModerate missing values (likely impute):")
    display(mid_missing.head(15))

    # 4. NUMERIC FEATURES
    print("\n4. NUMERIC FEATURES")
    num_df = df.select_dtypes(include=[np.number])
    print(f"Numeric columns: {num_df.shape[1]}")
    if num_df.shape[1] > 0:
        desc = num_df.describe().T
        outlier_counts = []
        for col in num_df.columns:
            q1 = num_df[col].quantile(0.25)
            q3 = num_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_counts.append(((num_df[col] < lower) | (num_df[col] > upper)).sum())
        desc["outliers_iqr"] = outlier_counts
        print("\nTop features by outliers:")
        display(desc.sort_values("outliers_iqr", ascending=False).head(10))

    # 5. CATEGORICAL CHECKS
    print("\n5. CATEGORICAL CONSISTENCY")
    cat_df = df.select_dtypes(include=["object"])
    inconsistent = []
    for col in cat_df.columns:
        vals = cat_df[col].dropna().astype(str)
        if len(vals) > 0 and any(vals.str.lower() != vals):
            inconsistent.append(col)
    print(f"Columns with inconsistent casing: {inconsistent}")

    # 6. SUMMARY
    print("\n6. CLEANING SUMMARY")
    print(f"- High missing (> {missing_threshold*100:.0f}%): {len(high_missing)}")
    print(f"- Moderate missing: {len(mid_missing)}")
    print(f"- Numeric features: {num_df.shape[1]}")
    print("\n" + "="*70 + "\n")
    return {
        "high_missing": high_missing,
        "mid_missing": mid_missing
    }

def low_variance_report(df, name="dataset", threshold=0.01, return_drop_list=True, verbose=True):
    """
    Detects low-variance features in a dataset.
    This is a feature screening step BEFORE preprocessing.
    Parameters:
    df : pd.DataFrame
        Input dataset
    name : str
        Dataset name for reporting
    threshold : float
        Variance threshold below which features are considered low-variance
    return_drop_list : bool
        If True, returns list of columns to drop
    verbose : bool
        If True, prints report
    Returns:
    dict containing:
        - low_variance_features
        - variance_series
        - suggested_drop_columns
    """

    if verbose:
        print(f"\n===== LOW VARIANCE REPORT: {name.upper()} =====\n")
    # keep only numeric features (important for clustering)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        if verbose:
            print("No numeric features found.")
        return {
            "low_variance_features": [],
            "variance": pd.Series(dtype=float),
            "drop_columns": []
        }
    # compute variance
    variance = num_df.var().sort_values()
    # identify low variance features
    low_var_features = variance[variance < threshold]
    if verbose:
        print(f"Total numeric features: {num_df.shape[1]}")
        print(f"Low variance threshold: {threshold}")
        print(f"Low variance features found: {len(low_var_features)}\n")
        if len(low_var_features) > 0:
            print("Top low-variance features:")
            print(low_var_features.head(20))
        else:
            print("No low-variance features detected.")
    # suggest full drop list
    drop_cols = low_var_features.index.tolist()
    if return_drop_list:
        return {
            "low_variance_features": low_var_features,
            "variance": variance,
            "drop_columns": drop_cols
        }
    return {
        "low_variance_features": low_var_features,
        "variance": variance
    }