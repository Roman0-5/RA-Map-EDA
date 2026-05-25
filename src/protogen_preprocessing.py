import pandas as pd


PROTOGEN_FILE = "../datasets/Protogen_RA_MAP_16_05_21.xlsx"
MEASUREMENT_SHEET = "LIS_PG665-P01 RA MAP Samples Ex"
ANNOTATION_SHEET = "Sample annotation"

#---------------------------------------------------------------------------------------------

def load_protogen_data(file_path: str = PROTOGEN_FILE) -> dict:
    """
    Load the two relevant Protogen sheets:
    - measurement sheet
    - sample annotation sheet
    """
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    df_measurement = all_sheets[MEASUREMENT_SHEET].copy()
    df_annotation = all_sheets[ANNOTATION_SHEET].copy()

    print("Protogen sheets loaded.")
    print(f"Measurement shape: {df_measurement.shape}")
    print(f"Annotation shape: {df_annotation.shape}")

    return {
        "df_measurement": df_measurement,
        "df_annotation": df_annotation,
    }

#---------------------------------------------------------------------------------------------

def filter_annotation_ra_bl(df_annotation: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
    - Study == TACERA
    - Timepoint == BL

    Also, keeping only the columns relevant for later merging / metadata.
    """
    relevant_cols = ["SampleId", "Timepoint", "Patient_ID", "Digest", "Study"]

    missing_cols = [col for col in relevant_cols if col not in df_annotation.columns]
    if missing_cols:
        raise KeyError(f"Missing required annotation columns: {missing_cols}")

    df_filtered = df_annotation.loc[
        (df_annotation["Study"] == "TACERA") &
        (df_annotation["Timepoint"] == "BL"),
        relevant_cols
    ].copy()

    print("Filtered annotation to TACERA + BL.")
    print(f"Filtered annotation shape: {df_filtered.shape}")
    print(f"Unique SampleIds: {df_filtered['SampleId'].nunique()}")
    print(f"Unique Patient_ID: {df_filtered['Patient_ID'].nunique()}")
    print(f"Unique Digest: {df_filtered['Digest'].nunique()}")

    return df_filtered

#---------------------------------------------------------------------------------------------

def select_relevant_measurement_columns(
    df_measurement: pd.DataFrame,
    df_annotation_ra_bl: pd.DataFrame
) -> pd.DataFrame:
    """
    Keep only the measurement metadata columns plus the relevant RA+BL sample columns.
    """
    base_cols = ["ProteinID", "GeneID", "Gene Symbol", "Gene Name"]

    sample_ids = df_annotation_ra_bl["SampleId"].dropna().astype(str).tolist()

    cols_to_keep = [col for col in base_cols + sample_ids if col in df_measurement.columns]

    df_selected = df_measurement[cols_to_keep].copy()

    print("Selected relevant measurement columns.")
    print(f"Measurement shape before: {df_measurement.shape}")
    print(f"Measurement shape after: {df_selected.shape}")
    print(f"Number of selected sample columns: {len(cols_to_keep) - len(base_cols)}")

    return df_selected

#---------------------------------------------------------------------------------------------

def add_marker_name(df_measurement_ra_bl: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unique marker name using Gene Symbol + ProteinID.
    """
    df = df_measurement_ra_bl.copy()

    df["MarkerName"] = (
        df["Gene Symbol"].astype(str).str.strip()
        + "_"
        + df["ProteinID"].astype(str).str.strip()
    )

    print("MarkerName column added.")
    print(f"Unique MarkerNames: {df['MarkerName'].nunique()} / {len(df)}")

    return df

#---------------------------------------------------------------------------------------------

def transpose_measurement(df_measurement_ra_bl: pd.DataFrame) -> pd.DataFrame:
    """
    Transpose the reduced measurement dataframe so that:
    - rows = samples
    - columns = markers

    Uses MarkerName as the future column names.
    """
    df = df_measurement_ra_bl.copy()

    if "MarkerName" not in df.columns:
        raise KeyError("MarkerName column not found. Run add_marker_name() first.")

    # Keep only MarkerName + sample columns
    metadata_cols = ["ProteinID", "GeneID", "Gene Symbol", "Gene Name", "MarkerName"]
    sample_cols = [col for col in df.columns if col not in metadata_cols]

    # Use MarkerName as index, then transpose sample columns
    df_t = df.set_index("MarkerName")[sample_cols].T

    # Make SampleId an explicit column
    df_t.index.name = "SampleId"
    df_t = df_t.reset_index()

    print("Measurement dataframe transposed.")
    print(f"Shape before transpose: {df_measurement_ra_bl.shape}")
    print(f"Shape after transpose: {df_t.shape}")

    return df_t

#---------------------------------------------------------------------------------------------

def merge_with_annotation(
    df_measurement_t: pd.DataFrame,
    df_annotation_ra_bl: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the transposed measurement dataframe with filtered annotation metadata
    using SampleId.
    """
    df_merged = df_annotation_ra_bl.merge(
        df_measurement_t,
        on="SampleId",
        how="inner"
    )

    print("Merged transposed measurement with annotation.")
    print(f"Transposed measurement shape: {df_measurement_t.shape}")
    print(f"Filtered annotation shape: {df_annotation_ra_bl.shape}")
    print(f"Merged shape: {df_merged.shape}")
    print(f"Unique SampleIds after merge: {df_merged['SampleId'].nunique()}")
    print(f"Unique Patient_ID after merge: {df_merged['Patient_ID'].nunique()}")
    print(f"Unique Digest after merge: {df_merged['Digest'].nunique()}")

    return df_merged

#---------------------------------------------------------------------------------------------

def split_metadata_features(df_gene_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged dataframe into:
    - metadata dataframe
    - feature dataframe
    """
    metadata_cols = ["SampleId", "Timepoint", "Patient_ID", "Digest", "Study"]

    missing_cols = [col for col in metadata_cols if col not in df_gene_merged.columns]
    if missing_cols:
        raise KeyError(f"Missing metadata columns: {missing_cols}")

    df_meta = df_gene_merged[metadata_cols].copy()
    df_features = df_gene_merged.drop(columns=metadata_cols).copy()

    print("Split merged dataframe into metadata and features.")
    print(f"Metadata shape: {df_meta.shape}")
    print(f"Feature shape: {df_features.shape}")

    return df_meta, df_features

#---------------------------------------------------------------------------------------------

def inspect_missing_values(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of missing values per feature.
    """
    missing_summary = pd.DataFrame({
        "feature": df_features.columns,
        "missing_count": df_features.isna().sum().values,
        "missing_pct": (df_features.isna().mean() * 100).values,
    }).sort_values("missing_pct", ascending=False)

    print("Missing values inspected.")
    print(f"Feature shape: {df_features.shape}")
    print(f"Total missing values: {df_features.isna().sum().sum()}")
    print(f"Features with any missing values: {(missing_summary['missing_count'] > 0).sum()}")

    return missing_summary

#---------------------------------------------------------------------------------------------

