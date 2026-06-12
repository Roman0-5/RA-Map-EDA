from scipy.stats import pointbiserialr
import pandas as pd
from collections import defaultdict

def select_top_variance_features(df, top_n=50):
    """
    Select top N most variable protein features.
    Returns:
        selected_df
        variance_rank
    """

    df_numeric = df.select_dtypes(include=["number"])

    variance_rank = (
        df_numeric.var(axis=0)
        .sort_values(ascending=False)
    )

    selected_features = variance_rank.head(top_n).index

    selected_df = df_numeric[selected_features].copy()

    print("\n===== VARIANCE SELECTION =====")
    print(f"Input features: {df_numeric.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print("==============================\n")

    return selected_df, variance_rank

def add_patient_id(df):

    df = df.copy()

    df["Patient_ID"] = (
        df.index.astype(str)
        .str.replace("_BL", "", regex=False)
    )

    return df

def rank_proteins_by_remission(df, y_col="remission_event"):
    df = df.copy()

    y = df[y_col]

    # drop ID + outcome
    X = df.drop(columns=["Patient_ID", y_col])

    # keep only numeric (important safeguard)
    X = X.select_dtypes(include=["number"])

    results = {}

    for col in X.columns:
        corr, _ = pointbiserialr(X[col], y)
        results[col] = abs(corr)

    return pd.Series(results).sort_values(ascending=False)

def rank_proteins_with_direction(df, y_col="remission_event"):
    df = df.copy()

    y = df[y_col]
    X = df.drop(columns=["Patient_ID", y_col])
    X = X.select_dtypes(include=["number"])

    results = {}

    for col in X.columns:
        corr, _ = pointbiserialr(X[col], y)
        results[col] = corr   # keep SIGN

    return pd.Series(results).sort_values(key=lambda x: x.abs(), ascending=False)

def select_ra_literature_proteins(df, ra_gene_list, id_col="Patient_ID"):
    """
    Select RA literature proteins from a protein dataframe
    where columns are formatted as: <id>_<gene>

    Returns:
        filtered dataframe with matching proteins only
    """

    # 1. keep only protein columns
    protein_cols = [c for c in df.columns if "_" in c]

    # 2. map gene -> full column names
    gene_to_cols = defaultdict(list)

    for col in protein_cols:
        gene = col.split("_")[-1]
        gene_to_cols[gene].append(col)

    # 3. collect matching columns
    selected_cols = []

    for gene in ra_gene_list:
        if gene in gene_to_cols:
            selected_cols.extend(gene_to_cols[gene])

    # 4. build output dataframe
    cols_to_keep = [id_col] + selected_cols
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    return df[cols_to_keep]