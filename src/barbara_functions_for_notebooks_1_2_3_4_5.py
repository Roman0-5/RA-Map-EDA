import pandas as pd
import os
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import numpy as np

#---------------------------#
#LOAD DataFrame
#---------------------------#

def load_autoimmune_data(data_dir, file_name, sheet_name=0):
    file_path = Path(data_dir) / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)

    elif file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

    else:
        raise ValueError("Unsupported file type")

    # Restore patient index if present
    if "Patient_Timepoint" in df.columns:
        df = df.set_index("Patient_Timepoint")

    print(f"Loaded: {file_path}")
    print(f"Shape: {df.shape}")
    print(f"Index name: {df.index.name}")

    return df

#---------------------------#
#INSPECT data
#---------------------------#

def get_duplicate_patient_ids(df):
    """
    Count how often each patient ID appears.
    Assumes column names follow patterns like:
    TAC1241_BL, TAC1241_M6, TAC1241_M12
    """

    patient_ids = [
        col.rsplit("_", 1)[0]
        for col in df.columns
        if "_" in col
    ]

    counts = Counter(patient_ids)

    duplicate_df = (
        pd.DataFrame(
            counts.items(),
            columns=["Patient_ID", "Count"]
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    return duplicate_df

def missing_value_summary(df):
    missing = df.isna().sum()

    return (
        missing[missing > 0]
        .sort_values(ascending=False)
        .to_frame("Missing_Count")
    )

def get_duplicate_values(df, column):
    """
    Returns all duplicated values in a column with counts.
    """

    if column not in df.columns:
        return f"Column {column} not found"

    return (
        df[column]
        .value_counts()
        .loc[lambda x: x > 1]
        .sort_values(ascending=False)
    )

def check_duplicate_protein_ids(df):
    """
    Check duplicate ProteinIDs and return summary + full duplicate table.
    """

    import pandas as pd

    if "ProteinID" not in df.columns:
        return "ProteinID column not found"

    duplicate_count = df["ProteinID"].duplicated().sum()

    duplicate_rows = df[df["ProteinID"].duplicated(keep=False)] \
        .sort_values("ProteinID")

    summary = pd.DataFrame({
        "Duplicate_ProteinID_Count": [duplicate_count],
        "Unique_Duplicated_ProteinIDs": [duplicate_rows["ProteinID"].nunique()]
    })

    return summary, duplicate_rows

#---------------------------#
#HARMONISE data
#---------------------------#

def filter_ra_only(df):
    """
    Remove VAC (non-RA / control) samples from dataset.

    Parameters:
    - df: pandas DataFrame with patient columns

    Returns:
    - RA-only DataFrame
    """

    vac_cols = [col for col in df.columns if "VAC" in str(col)]

    df_ra = df.drop(columns=vac_cols)

    print(f"Removed {len(vac_cols)} VAC columns")
    print(f"Remaining dataset shape: {df_ra.shape}")

    return df_ra


def export_to_csv(df, output_path, index=True):
    """
    Export DataFrame safely to CSV (creates folders if needed).
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=index)

    print(f"File saved to: {output_path}")


def transpose_autoimmune_data(df):
    """
    Correct transpose for autoimmune dataset.
    """

    meta_cols = ["GeneID", "ProteinID", "Gene Symbol", "Gene Name"]

    # protein identifiers
    protein_ids = (
        df["ProteinID"].astype(str) + "_" +
        df["Gene Symbol"].astype(str)
    )

    # remove ALL metadata properly
    numeric = df.drop(columns=meta_cols)

    # transpose
    df_t = numeric.T

    # assign protein names
    df_t.columns = protein_ids.values

    # FIX: ensure index becomes patients ONLY
    df_t.index.name = "Patient_Timepoint"

    print("Transposition complete")
    print("Shape:", df_t.shape)
    print("Index preview:", df_t.index[:5])

    return df_t

def split_bl_m6(df):
    """
    Split transposed autoimmune dataset into BL and M6 subsets.

    Parameters:
    - df: patient × protein DataFrame (index contains TACxxx_BL / TACxxx_M6)

    Returns:
    - df_bl
    - df_m6
    """

    df_bl = df[df.index.str.contains("_BL")].copy()
    df_m6 = df[df.index.str.contains("_M6")].copy()

    print("BL shape:", df_bl.shape)
    print("M6 shape:", df_m6.shape)

    return df_bl, df_m6

def harmonisation_validator(df_bl, df_m6):
    """
    Harmonisation check for BL and M6 omics datasets.

    Checks:
    - patient overlap
    - missing patients
    - duplicate indices
    - feature alignment
    - column order consistency
    - basic structural sanity

    Returns:
    - dict of diagnostic results
    """

    report = {}

    # -----------------------------
    # 1. Patient overlap
    # -----------------------------
    bl_ids = set(df_bl.index.str.replace("_BL", ""))
    m6_ids = set(df_m6.index.str.replace("_M6", ""))

    report["bl_only"] = bl_ids - m6_ids
    report["m6_only"] = m6_ids - bl_ids
    report["shared_patients"] = bl_ids & m6_ids

    # -----------------------------
    # 2. Duplicates check
    # -----------------------------
    report["bl_duplicate_patients"] = df_bl.index.duplicated().sum()
    report["m6_duplicate_patients"] = df_m6.index.duplicated().sum()

    # -----------------------------
    # 3. Feature alignment
    # -----------------------------
    bl_features = set(df_bl.columns)
    m6_features = set(df_m6.columns)

    report["features_only_in_bl"] = bl_features - m6_features
    report["features_only_in_m6"] = m6_features - bl_features
    report["shared_features"] = bl_features & m6_features

    # -----------------------------
    # 4. Column order consistency
    # -----------------------------
    report["feature_order_match"] = list(df_bl.columns) == list(df_m6.columns)

    # -----------------------------
    # 5. Shape summary
    # -----------------------------
    report["shape_bl"] = df_bl.shape
    report["shape_m6"] = df_m6.shape

    # -----------------------------
    # 6. Basic sanity check
    # -----------------------------
    report["bl_non_numeric_columns"] = df_bl.select_dtypes(include=["object"]).columns.tolist()
    report["m6_non_numeric_columns"] = df_m6.select_dtypes(include=["object"]).columns.tolist()

    # -----------------------------
    # PRINT SUMMARY
    # -----------------------------
    print("\n===== HARMONISATION REPORT =====")

    print(f"BL shape: {df_bl.shape}")
    print(f"M6 shape: {df_m6.shape}")

    print(f"\nShared patients: {len(report['shared_patients'])}")
    print(f"BL-only patients: {len(report['bl_only'])}")
    print(f"M6-only patients: {len(report['m6_only'])}")

    print(f"\nFeature mismatch BL-only: {len(report['features_only_in_bl'])}")
    print(f"Feature mismatch M6-only: {len(report['features_only_in_m6'])}")

    print(f"\nFeature order identical: {report['feature_order_match']}")

    print("================================\n")

    return report


#---------------------------#
#VISUALISE data - prior to normalisation - at post-harmonisation
#---------------------------#

def run_pca_visualisation(df, n_components=2, title="PCA Plot"):
    """
    Runs PCA on a DataFrame and plots first 2 components.

    Parameters:
    - df: BL dataset (patients × proteins)
    - n_components: number of PCA components
    - title: plot title

    Returns:
    - PCA transformed array
    """

    X = df.values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    return X_pca

def run_umap_visualisation(df, n_neighbors=15, min_dist=0.1, title="UMAP Plot"):
    """
    Runs UMAP projection and visualises structure.

    Parameters:
    - df: BL dataset
    - n_neighbors: local structure parameter
    - min_dist: cluster tightness
    - title: plot title

    Returns:
    - UMAP embedding
    """

    X = df.values
    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )

    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_umap[:, 0], X_umap[:, 1])

    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.show()

    return X_umap

def plot_heatmap(df, n_samples=50, n_features=50, title="Heatmap (subset)"):
    """
    Plots a heatmap of a subset of the dataset.

    Parameters:
    - df: BL dataset
    - n_samples: number of patients to show
    - n_features: number of proteins to show
    """

    X = df.iloc[:n_samples, :n_features].values
    X_scaled = StandardScaler().fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.heatmap(X_scaled, cmap="viridis")

    plt.title(title)
    plt.xlabel("Proteins")
    plt.ylabel("Patients")
    plt.show()

#---------------------------#
#VERIFY NORMALISATION in data
#---------------------------#

def plot_global_distribution(df, title="Global distribution of values", bins=100):
    """
    Global distribution assessment of protein expression values.
    Includes:
    - histogram
    - skewness (right-skew detection)
    - range (min/max)
    - quantiles (long-tail detection)
    """

    data = df.stack()
    data = pd.to_numeric(data, errors="coerce").dropna()

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.hist(data.values, bins=bins)
    plt.title(title)
    plt.xlabel("Protein expression value")
    plt.ylabel("Frequency")
    plt.show()

    # --- Statistics ---
    skewness = data.skew()
    min_val = data.min()
    max_val = data.max()
    quantiles = data.quantile([0.01, 0.05, 0.5, 0.95, 0.99])

    print("\n===== GLOBAL DISTRIBUTION SUMMARY =====")
    print(f"Skewness: {skewness:.3f}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print("\nQuantiles:")
    print(quantiles)

    return {
        "skewness": skewness,
        "min": min_val,
        "max": max_val,
        "quantiles": quantiles
    }


def compute_global_skewness(df):
    """
    Returns skewness of all values in dataframe.
    """
    return df.stack().skew()


def compute_global_range(df):
    """
    Returns min and max values in dataframe.
    """
    return df.min().min(), df.max().max()


def compute_quantiles(df, quantiles=None):
    """
    Returns global quantiles of all values.
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]

    return df.stack().quantile(quantiles)

def log_transform_assessment(df, bins=100):
    """
    Compares raw vs log-transformed protein distributions.

    Outputs:
    - histograms (raw vs log1p)
    - skewness comparison
    - range compression
    """

    data_raw = df.stack()
    data_raw = pd.to_numeric(data_raw, errors="coerce").dropna()

    data_log = np.log1p(data_raw)

    # --- PLOTS ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(data_raw, bins=bins)
    plt.title("Raw distribution")
    plt.xlabel("Value")

    plt.subplot(1, 2, 2)
    plt.hist(data_log, bins=bins)
    plt.title("Log-transformed distribution")
    plt.xlabel("log(1 + x)")

    plt.tight_layout()
    plt.show()

    # --- METRICS ---
    results = {
        "raw_skew": data_raw.skew(),
        "log_skew": data_log.skew(),
        "raw_min": data_raw.min(),
        "raw_max": data_raw.max(),
        "log_min": data_log.min(),
        "log_max": data_log.max()
    }

    print("\n===== LOG TRANSFORM ASSESSMENT =====")
    print(f"Raw skewness: {results['raw_skew']:.3f}")
    print(f"Log skewness: {results['log_skew']:.3f}")
    print(f"Raw range: {results['raw_min']} → {results['raw_max']}")
    print(f"Log range: {results['log_min']:.3f} → {results['log_max']:.3f}")

    return results

import matplotlib.pyplot as plt


def plot_protein_distributions(df, proteins=None, n=6):
    """
    Plot distributions of individual proteins to assess skewness per feature.
    """

    if proteins is None:
        proteins = df.columns[:n]

    plt.figure(figsize=(12, 8))

    for i, protein in enumerate(proteins):
        plt.subplot(2, 3, i + 1)
        plt.hist(df[protein].dropna(), bins=50)
        plt.title(str(protein))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def protein_skewness(df):
    """
    Returns skewness per protein.
    """

    return df.skew().sort_values(ascending=False)

def protein_outlier_summary(df):
    """
    Counts extreme values per protein (99th percentile cutoff).
    """

    threshold = df.quantile(0.99)

    outliers = (df > threshold).sum().sort_values(ascending=False)

    return outliers

#---------------------------#
#NORMALIZE data
#---------------------------#

def log_normalise(df, method="log1p"):
    """
    Normalise proteomics data using log transformation.

    Parameters:
    - df: patient × protein DataFrame
    - method: log1p (default) or log

    Returns:
    - log-transformed DataFrame
    """

    df_numeric = df.copy()

    if method == "log1p":
        df_log = np.log1p(df_numeric)
    elif method == "log":
        df_log = np.log(df_numeric)
    else:
        raise ValueError("method must be 'log1p' or 'log'")

    print("Log normalisation complete")
    print("Shape:", df_log.shape)


#---------------------------#
#STANDARDISE data
#---------------------------#

def standardise_data(df):
    """
    Standardise protein expression data (z-score per protein).

    Parameters:
    - df: patient × protein DataFrame (after log transform)

    Returns:
    - standardised DataFrame with same shape
    """

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(
        scaled_array,
        index=df.index,
        columns=df.columns
    )

    print("Standardisation complete")
    print("Shape:", df_scaled.shape)

    return df_scaled