"""Build a longitudinal multi-omics dataset (BL + M6).

Merges all cleaned sources on Patient_ID (inner join → 97 patients):

    clinical_merged.parquet      — clinical features (no prefix)
    exp_bl.parquet               — expression baseline  (prefix: expr_*_BL)
    exp_m6.parquet               — expression 6-month   (prefix: expr_*_M6)
    protogen_features.parquet    — protogen features BL (prefix: pg_*_BL)
                                   protogen features M6  (prefix: pg_*_M6)

Run from proj/src/:
    python build_multiomics_longitudinal.py
"""

import os
import pandas as pd
import numpy as np


# ============================================================================
# Paths
# ============================================================================

SRC_DIR     = "../cleaned_datasets"
OUT_DIR     = "../cleaned_datasets"

CLINICAL_PATH   = f"{SRC_DIR}/clinical_merged.parquet"
EXP_BL_PATH     = f"{SRC_DIR}/exp_bl.parquet"
EXP_M6_PATH     = f"{SRC_DIR}/exp_m6.parquet"
PG_FEAT_PATH    = f"{SRC_DIR}/protogen_features.parquet"
PG_META_PATH    = f"{SRC_DIR}/protogen_metadata.parquet"
OUTPUT_PATH     = f"{OUT_DIR}/multiomics_longitudinal.parquet"


# ============================================================================
# Main
# ============================================================================

def build_multiomics(
    clinical_path: str = CLINICAL_PATH,
    exp_bl_path:   str = EXP_BL_PATH,
    exp_m6_path:   str = EXP_M6_PATH,
    pg_feat_path:  str = PG_FEAT_PATH,
    pg_meta_path:  str = PG_META_PATH,
    output_path:   str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Merge clinical + expression BL/M6 + protogen BL/M6 longitudinally.

    Args:
        clinical_path: Path to clinical_merged.parquet.
        exp_bl_path:   Path to exp_bl.parquet (SampleId index: TAC1000_BL).
        exp_m6_path:   Path to exp_m6.parquet (SampleId index: TAC1000_M6).
        pg_feat_path:  Path to protogen_features.parquet (numeric index).
        pg_meta_path:  Path to protogen_metadata.parquet (numeric index).
        output_path:   Where to save the merged file.

    Returns:
        Merged DataFrame (97 patients × ~2500+ columns).
    """
    print(f"\n{'='*60}")
    print("BUILDING LONGITUDINAL MULTI-OMICS DATASET")
    print(f"{'='*60}")

    # ---------------------------------------------------------------- load
    print("\nLoading files ...")
    clinical = pd.read_parquet(clinical_path)
    exp_bl   = pd.read_parquet(exp_bl_path)
    exp_m6   = pd.read_parquet(exp_m6_path)
    pg_feat  = pd.read_parquet(pg_feat_path)
    pg_meta  = pd.read_parquet(pg_meta_path)

    clinical['Patient_ID'] = clinical['Patient_ID'].astype(str)

    # ----------------------------------------- expression: extract Patient_ID
    # SampleId index is e.g. "TAC1000_BL" or "TAC1000_M6"
    exp_bl = exp_bl.copy()
    exp_bl['Patient_ID'] = exp_bl.index.str.replace('_BL', '', regex=False)
    exp_bl = exp_bl.reset_index(drop=True)

    exp_m6 = exp_m6.copy()
    exp_m6['Patient_ID'] = exp_m6.index.str.replace('_M6', '', regex=False)
    exp_m6 = exp_m6.reset_index(drop=True)

    # Prefix expression columns
    expr_cols = [c for c in exp_bl.columns if c != 'Patient_ID']
    exp_bl = exp_bl.rename(columns={c: f"expr_{c}_BL" for c in expr_cols})
    exp_m6 = exp_m6.rename(columns={c: f"expr_{c}_M6" for c in expr_cols})

    # ----------------------------------------- protogen: attach Patient_ID
    # positional join — metadata and features share the same numeric index
    pg_feat = pg_feat.copy().reset_index(drop=True)
    pg_meta = pg_meta.copy().reset_index(drop=True)
    pg_meta['Patient_ID'] = pg_meta['Patient_ID'].astype(str)

    pg_combined = pd.concat([pg_meta[['Patient_ID', 'Timepoint']], pg_feat],
                            axis=1)

    pg_bl = pg_combined[pg_combined['Timepoint'] == 'BL'].drop(
        columns='Timepoint'
    ).reset_index(drop=True)

    pg_m6 = pg_combined[pg_combined['Timepoint'] == 'M6'].drop(
        columns='Timepoint'
    ).reset_index(drop=True)

    pg_cols = [c for c in pg_bl.columns if c != 'Patient_ID']
    pg_bl = pg_bl.rename(columns={c: f"pg_{c}_BL" for c in pg_cols})
    pg_m6 = pg_m6.rename(columns={c: f"pg_{c}_M6" for c in pg_cols})

    # ---------------------------------------------------- inner join
    print("\nMerging ...")
    merged = (
        clinical
        .merge(exp_bl,  on='Patient_ID', how='inner')
        .merge(exp_m6,  on='Patient_ID', how='inner')
        .merge(pg_bl,   on='Patient_ID', how='inner')
        .merge(pg_m6,   on='Patient_ID', how='inner')
    )

    # --------------------------------------------------------- summary
    n_clin  = len(clinical.columns) - 1
    n_expr  = len(expr_cols)
    n_pg    = len(pg_cols)

    print(f"\n  Patients            : {len(merged)}")
    print(f"  Clinical cols       : {n_clin}")
    print(f"  Expression BL cols  : {n_expr}  (expr_*_BL)")
    print(f"  Expression M6 cols  : {n_expr}  (expr_*_M6)")
    print(f"  Protogen BL cols    : {n_pg}   (pg_*_BL)")
    print(f"  Protogen M6 cols    : {n_pg}   (pg_*_M6)")
    print(f"  Total columns       : {merged.shape[1]}")

    missing = merged.isnull().sum().sum()
    print(f"  Missing values      : {missing} "
          f"({missing / merged.size * 100:.2f}%)")

    # ---------------------------------------------------------------- save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    return merged