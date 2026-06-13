import pandas as pd

# SANITY CHECKS POST DATA LOADING / CLEANING, ETC:

def sanity_structure(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissingness (top 10):\n", df.isna().mean().sort_values(ascending=False).head(10))

def sanity_contract_check(df: pd.DataFrame, contract: dict):
    expected_cols = (
        contract.get("datetime_columns", []) +
        contract.get("numeric_columns", []) +
        contract.get("categorical_columns", []) +
        contract.get("binary_columns", []) +
        list(contract.get("key_columns", {}).keys())
    )
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    print("Missing expected columns:", missing)
    print("Extra columns:", extra)


def sanity_numeric(df: pd.DataFrame, contract: dict):
    cols = contract.get("numeric_columns", [])
    for col in cols:
        if col in df.columns:
            print(f"\n{col}")
            print("dtype:", df[col].dtype)
            print("non-numeric (NaN count):", df[col].isna().sum())
            print("min/max:", df[col].min(), df[col].max())


def sanity_binary(df: pd.DataFrame, contract: dict):
    cols = contract.get("binary_columns", [])
    for col in cols:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].value_counts(dropna=False))


def sanity_categorical(df: pd.DataFrame, contract: dict):
    cols = contract.get("categorical_columns", [])
    for col in cols:
        if col in df.columns:
            print(f"\n{col} (top values):")
            print(df[col].value_counts(dropna=False).head(10))


def sanity_dates(df: pd.DataFrame, contract: dict):
    cols = contract.get("datetime_columns", [])
    for col in cols:
        if col in df.columns:
            print(f"\n{col}")
            print("NaT count:", df[col].isna().sum())
            print("min:", df[col].min())
            print("max:", df[col].max())

def sanity_duplicates(df: pd.DataFrame, subset=None):
    print("Duplicate rows:", df.duplicated(subset=subset).sum())

def check_unique_patients(df):
    print("Rows:", len(df))
    print("Unique Digest:", df["Digest"].nunique())

def sanity_check_clinical(df: pd.DataFrame, contract: dict):
    """Sanity check for patient-level clinical dataset after preprocessing."""
    print("\n===== SHAPE =====")
    print(df.shape)
    print("\n===== MISSINGNESS (TOP 10) =====")
    print(df.isna().mean().sort_values(ascending=False).head(10))
    print("\n===== DTYPE SUMMARY =====")
    print(df.dtypes.value_counts())

    # NUMERIC CHECK
    print("\n===== NUMERIC CHECK =====")
    for col in contract.get("longitudinal_numeric", []) + contract.get("static_numeric", []):
        if col in df.columns:
            print(f"{col}: dtype={df[col].dtype}, NaNs={df[col].isna().sum()}")
    # BINARY CHECK
    print("\n===== BINARY CHECK =====")
    for col in contract.get("binary_columns", []):
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].value_counts(dropna=False))
    # ID CHECK (must be clean)
    print("\n===== ID CHECK =====")
    for col in contract.get("id_columns", []):
        if col in df.columns:
            print(f"{col}: unique={df[col].nunique()}, nulls={df[col].isna().sum()}")