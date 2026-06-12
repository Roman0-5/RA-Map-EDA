def ensure_patient_id(df):
    df = df.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    return df

def ensure_patient_timepoint(df):
    df = df.copy()
    df["Patient_Timepoint"] = df["Patient_Timepoint"].astype(str)
    return df

import pandas as pd

def enforce_flat_dataframe(df):
    """
    Ensures:
    - no MultiIndex
    - no hidden index data
    - clean flat table structure
    """

    df = df.copy()

    # 1. flatten index if needed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # 2. bring index into columns if meaningful
    df = df.reset_index(drop=False)

    # 3. remove duplicate index column if created
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    return df

def standardize_patient_column(df):
    """
    Ensures dataset always has:
    Patient_ID column (clean, consistent)
    """

    df = df.copy()
    df = enforce_flat_dataframe(df)

    # detect possible ID columns
    candidate_cols = ["Patient_ID", "Patient_Timepoint"]

    found = None
    for col in candidate_cols:
        if col in df.columns:
            found = col
            break

    if found is None:
        raise KeyError(f"No patient ID column found. Columns: {df.columns.tolist()[:10]}")

    # standardize to Patient_ID
    df["Patient_ID"] = (
        df[found]
        .astype(str)
        .str.replace("_BL", "", regex=False)
    )

    return df

def validate_ml_ready(df, name="dataset"):
    """
    Hard check before ML
    """

    assert "Patient_ID" in df.columns, f"{name}: Missing Patient_ID"

    assert df.index.name is None or df.index.name == "", f"{name}: index not clean"

    # check duplicates
    dup = df["Patient_ID"].duplicated().sum()
    print(f"{name}: duplicates in Patient_ID = {dup}")

    print(f"{name}: shape = {df.shape}")
    print(f"{name}: OK")

def select_features_from_zscore(df_zscore: pd.DataFrame, feature_list: list, id_col="Patient_ID"):
    df = df_zscore.copy()

    cols = []
    if id_col in df.columns:
        cols.append(id_col)

    valid_features = [f for f in feature_list if f in df.columns]

    missing = set(feature_list) - set(valid_features)
    if missing:
        print(f"[WARN] {len(missing)} features not found in z-score data")

    cols += valid_features

    return df[cols], valid_features

def standardize_protein_id(df):
    df = df.copy()

    if "Patient_Timepoint" in df.columns:
        df["Patient_ID"] = (
            df["Patient_Timepoint"]
            .astype(str)
            .str.replace("_BL", "", regex=False)
        )
        df = df.drop(columns=["Patient_Timepoint"])

    if "Digest" in df.columns:
        df = df.drop(columns=["Digest"])

    df["Patient_ID"] = df["Patient_ID"].astype(str)

    # move Patient_ID to front
    cols = ["Patient_ID"] + [c for c in df.columns if c != "Patient_ID"]
    df = df[cols]

    return df