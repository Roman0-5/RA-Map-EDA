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
    print(high_missing)
    print("\nModerate missing values (likely impute):")
    print(mid_missing.head(15))

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
        print(desc.sort_values("outliers_iqr", ascending=False).head(10))

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
if __name__ == "__main__":
    clinical_data_audit()