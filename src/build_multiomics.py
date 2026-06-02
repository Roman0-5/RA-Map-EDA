"""Build a unified multi-omics Parquet file.

Merges three data sources on Patient_ID (inner join → 100 patients):
    - clinical_merged        : 68 clinical features
    - expression_matrix_bl   : 1310 SomaScan proteins (baseline, SeqID cols)
    - protogen_merged_bl     : 163 Protogen autoantigen features

Column prefixes make the source unambiguous:
    - clinical columns : no prefix  (Patient_ID, DAS28.0M, …)
    - expression cols  : ``expr_``  (expr_2597-8_3, …)
    - protogen cols    : ``pg_``    (pg_XRCC6_104740305, …)
"""

import pandas as pd


# ============================================================================
# Defaults — override via function arguments if paths differ
# ============================================================================

_CLINICAL_PATH  = "../../mid_processing_datasets/clinical_merged.parquet"
_EXPR_BL_PATH   = "../../mid_processing_datasets/expression_matrix_baseline.parquet"
_PROTOGEN_PATH  = "../../mid_processing_datasets/protogen_merged_bl.parquet"
_OUTPUT_PATH    = "../../mid_processing_datasets/multiomics_bl.parquet"

_PROTOGEN_META  = ["SampleId", "Timepoint", "Digest", "Study"]


# ============================================================================
# Main builder
# ============================================================================

def build_multiomics(
    clinical_path: str  = _CLINICAL_PATH,
    expr_bl_path: str   = _EXPR_BL_PATH,
    protogen_path: str  = _PROTOGEN_PATH,
    output_path: str    = _OUTPUT_PATH,
) -> pd.DataFrame:
    """Merge clinical, expression and protogen data into one Parquet file.

    Performs an inner join on ``Patient_ID`` across all three sources,
    keeping only the 100 patients present in every dataset.

    Column naming:
        - Clinical columns  : unchanged  (``Patient_ID``, ``DAS28.0M``, …)
        - Expression columns: prefixed with ``expr_``
        - Protogen columns  : prefixed with ``pg_``

    Args:
        clinical_path: Path to ``clinical_merged.parquet``.
        expr_bl_path:  Path to ``expression_matrix_baseline.parquet``.
        protogen_path: Path to ``protogen_merged_bl.parquet``.
        output_path:   Where to write the merged Parquet file.

    Returns:
        Merged DataFrame (100 patients × ~1541 columns).
    """
    print(f"\n{'='*70}")
    print("BUILDING MULTI-OMICS DATASET")
    print(f"{'='*70}")

    # ------------------------------------------------------------------ load
    print("\nLoading files …")
    clinical  = pd.read_parquet(clinical_path)
    expr_bl   = pd.read_parquet(expr_bl_path)
    protogen  = pd.read_parquet(protogen_path)

    print(f"  clinical  : {clinical.shape}")
    print(f"  expr_bl   : {expr_bl.shape}")
    print(f"  protogen  : {protogen.shape}")

    # normalise Patient_ID dtype across all sources
    for df in (clinical, expr_bl, protogen):
        df['Patient_ID'] = df['Patient_ID'].astype(str)

    # --------------------------------------------------------- common IDs
    ids_clinical = set(clinical['Patient_ID'])
    ids_expr     = set(expr_bl['Patient_ID'])
    ids_protogen = set(protogen['Patient_ID'])
    common_ids   = ids_clinical & ids_expr & ids_protogen

    print(f"\nPatient overlap:")
    print(f"  clinical  : {len(ids_clinical)}")
    print(f"  expr_bl   : {len(ids_expr)}")
    print(f"  protogen  : {len(ids_protogen)}")
    print(f"  inner join: {len(common_ids)}")

    # ------------------------------------------- filter to common patients
    clinical  = clinical[clinical['Patient_ID'].isin(common_ids)].copy()
    expr_bl   = expr_bl[expr_bl['Patient_ID'].isin(common_ids)].copy()
    protogen  = protogen[protogen['Patient_ID'].isin(common_ids)].copy()

    # ------------------------------------------- prefix expression columns
    expr_feature_cols = [c for c in expr_bl.columns if c != 'Patient_ID']
    expr_bl = expr_bl.rename(
        columns={c: f"expr_{c}" for c in expr_feature_cols}
    )

    # ----------------------------------------------- prefix protogen cols
    pg_drop_cols    = [c for c in _PROTOGEN_META if c in protogen.columns]
    protogen        = protogen.drop(columns=pg_drop_cols)
    pg_feature_cols = [c for c in protogen.columns if c != 'Patient_ID']
    protogen        = protogen.rename(
        columns={c: f"pg_{c}" for c in pg_feature_cols}
    )

    # ---------------------------------------------------------------- merge
    print("\nMerging …")
    merged = (
        clinical
        .merge(expr_bl,  on='Patient_ID', how='inner')
        .merge(protogen, on='Patient_ID', how='inner')
    )

    # ------------------------------------------------------------ summary
    n_clinical = len(clinical.columns) - 1          # excl. Patient_ID
    n_expr     = len(expr_feature_cols)
    n_pg       = len(pg_feature_cols)

    print(f"\nMerged shape: {merged.shape}")
    print(f"  Patient_ID + clinical features : {n_clinical}")
    print(f"  Expression features (expr_*)   : {n_expr}")
    print(f"  Protogen features   (pg_*)     : {n_pg}")
    print(f"  Total columns                  : {merged.shape[1]}")

    total_cells   = merged.shape[0] * merged.shape[1]
    missing_cells = merged.isnull().sum().sum()
    print(f"\nMissing values: {missing_cells} / {total_cells} "
          f"({missing_cells / total_cells * 100:.2f}%)")

    cols_with_na = (merged.isnull().sum() > 0).sum()
    print(f"Columns with any NaN: {cols_with_na}")

    # ---------------------------------------------------------------- save
    merged.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return merged


# ============================================================================
# Quick run
# ============================================================================

if __name__ == "__main__":
    df = build_multiomics()
    print(df.head(2))