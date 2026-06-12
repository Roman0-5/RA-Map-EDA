import pandas as pd

def apply_clinical_selection_contract(df: pd.DataFrame,
                                      contract: dict,
                                      drop_missing_cols: bool = False):
    """
    Selects clinical features based on a contract while preserving ID columns.

    Parameters:
    - df: full clinical dataframe
    - contract: CLINICAL_CONTRACT_SELECTION
    - drop_missing_cols: if True, silently drops missing columns;
                         if False, raises warning

    Returns:
    - filtered dataframe containing:
        * id columns
        * selected binary + longitudinal + static features
    """

    df = df.copy()

    # -------------------------
    # 1. ID columns (always keep)
    # -------------------------
    id_cols = contract.get("id_columns", [])

    # -------------------------
    # 2. Feature groups
    # -------------------------
    binary_cols = contract.get("binary_columns", [])
    long_cols = contract.get("longitudinal_numeric", [])
    static_cols = contract.get("static_numeric", [])

    selected_cols = id_cols + binary_cols + long_cols + static_cols

    # -------------------------
    # 3. Handle missing columns
    # -------------------------
    missing = [c for c in selected_cols if c not in df.columns]

    if missing:
        if drop_missing_cols:
            selected_cols = [c for c in selected_cols if c in df.columns]
        else:
            print("⚠️ Missing columns in dataframe:")
            print(missing)

    # -------------------------
    # 4. Subset dataframe
    # -------------------------
    df_selected = df[selected_cols].copy()

    print("\n===== CLINICAL CONTRACT SELECTION =====")
    print(f"Total selected columns: {len(selected_cols)}")
    print(f"Missing columns: {len(missing)}")
    print(f"Final shape: {df_selected.shape}")
    print("======================================\n")

    return df_selected

def select_most_variable_features(df, contract, top_n=30, exclude_outcomes=True):
    """
    Selects top-N most variable CLINICAL features.
    """

    df = df.copy()

    # -------------------------
    # 1. Define feature pool
    # -------------------------
    feature_cols = (
        contract.get("binary_columns", []) +
        contract.get("longitudinal_numeric", []) +
        contract.get("static_numeric", [])
    )

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]

    # -------------------------
    # 2. Compute variance
    # -------------------------
    variances = X.var(numeric_only=True)

    top_features = variances.sort_values(ascending=False).head(top_n).index.tolist()

    # -------------------------
    # 3. Build reduced dataframe
    # -------------------------
    df_selected = df[top_features].copy()

    print("\n===== VARIANCE SELECTION =====")
    print(f"Input features: {len(feature_cols)}")
    print(f"Selected features: {len(top_features)}")
    print("==============================\n")

    return df_selected, variances.sort_values(ascending=False)