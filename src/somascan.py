"""
All functions already have default fallbacks provided.
The output path is relative to where the function will and is intented to be called


Usage:
    1. data = load()
    2. splits = reshape_and_split(data['df_expMatrix'], data['df_sampMatrix'])
    3. save_splits(splits)
    4. create_gene_lookup(data['df_expMatrix'])
    
Note: helper_extract_timestamps is called internally by reshape_and_split 
Only call it directly if you have a pre-transposed expression matrix

Function calls:
    - load():
        Load only somascan data

    - reshape_and_split(df_exp, df_samp, id_col):
        Reshape and split df_exp by an id column lookup from df_samp

    - helper_extract_timestamps(df_exp, df_samp, patient_id_col, sample_id_col, timepoint_col):
        Extracts timepoints with the same method as reshape_and_split,
        is only called internally

    - save_splits(splits, output_dir):
        Save new files in new folder,
        created during runtime

    - create_gene_lookup(df_exp, output_dir):
        Creates pseudo glossary for protein lookup
"""

import pandas as pd
import os


def load() -> dict:
    """Load SOMAscan Excel file
    
    Returns:
        Dict mapping sheet names to DataFrames
    """
    print('Loading somascan...')
    df = pd.read_excel('../../datasets/SOMASCAN_RA-Map_figshare_17_11_20.xlsx', 
        sheet_name=None)
    print("Done loading!")
    return {
        'df_expMatrix': df['expression matrix'],
        'df_sampMatrix': df['sample matrix']
        }


def reshape_and_split(
    df_exp: pd.DataFrame,
    df_samp: pd.DataFrame,
    id_col: str = 'SeqId'
    ) -> dict:
    """Reshape expression matrix and split by timepoint.
    
    Transposes the raw expression matrix from (proteins x samples) to
    (samples x proteins) and groups the result by timepoint using the
    sample matrix as a lookup.
    
    Args:
        df_exp: expression matrix data
        df_samp: sample matrix
        id_col: Column to use as protein identifier. Defaults to 'SeqId'
    
    Returns:
        Dict mapping each timepoint to its samples x proteins DataFrame.
    """
    # check which columns are overlapping with SampleId, otherwise they are metadata
    sample_columns = df_samp['SampleId'].tolist()
    patient_cols_in_exp = [col for col in df_exp.columns if col in sample_columns]
    
    # transposing and indexing
    expr = df_exp.set_index(id_col)[patient_cols_in_exp].T
    expr.index.name = 'SampleId'
    
    # save splits by timepoint
    splits = helper_extract_timestamps(expr, df_samp)
    
    print(f"Timepoints found: {list(splits.keys())}")
    for tp, df in splits.items():
        print(f"  {tp}: {df.shape}")
    
    return splits


def helper_extract_timestamps(
    df_exp: pd.DataFrame,
    df_samp: pd.DataFrame,
    patient_id_col: str = 'Patient_ID',
    sample_id_col: str = 'SampleId',
    timepoint_col: str = 'TimePoint',
    ) -> dict:
    """Group expression matrix by timepoint using sample matrix as a column lookup.
    
    For each unique timepoint in the sample matrix select the corresponding
    rows from the expression matrix and prepend a Patient_ID column.
    Renames sample suffixes from '_Baseline'/'_6month' to '_BL'/'_M6'.
    
    Args:
        df_exp: Expression matrix already indexed by SampleId and transposed.
        df_samp: Sample Matrix.
        patient_id_col: Column name for patient identifier.
        sample_id_col: Column name for sample identifier.
        timepoint_col: Column name to group by.
    
    Returns:
        Dict mapping each timepoint to a DataFrame of its samples,
        with Patient_ID as the first column.
    """
    suffix_map = {'Baseline': 'BL', '6month': 'M6'}
    result = {}
    
    for timepoint, group in df_samp.groupby(timepoint_col):
        sample_ids = group[sample_id_col].tolist()
        subset = df_exp.loc[sample_ids].copy()
        
        id_to_patient = dict(zip(group[sample_id_col], group[patient_id_col]))
        subset.insert(0, 'Patient_ID', subset.index.map(id_to_patient))
        
        if timepoint in suffix_map:
            old_suffix = f'_{timepoint}'
            new_suffix = f'_{suffix_map[timepoint]}'
            subset.index = subset.index.str.replace(old_suffix, new_suffix, regex=False)
            subset.index.name = 'SampleId'
        
        result[timepoint] = subset
    
    return result


def save_splits(
    splits: dict,
    output_dir: str = '../../mid_processing_datasets'
    ) -> None:
    """Save each timepoint dataframe as a parquet file
    
    Creates the output directory if it does not exist and writes to one parquet
    per timepoint, using a short suffix in the filename.
    
    Args:
        splits: Dict mapping timepoint names to DataFrames
        output_dir: Target directory for parquet files. Defaults to
                    '../../mid_processing_datasets'.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix_map = {'bl': 'bl', 'm6': 'm6'}
    
    for timepoint, df in splits.items():
        suffix = suffix_map.get(timepoint, timepoint.lower())
        path = f'{output_dir}/expression_matrix_{suffix}.parquet'
        df.to_parquet(path)
        print(f"Saved: {path}")


def create_gene_lookup(
    df_exp: pd.DataFrame,
    output_dir: str = '../../mid_processing_datasets'
    ) -> pd.DataFrame:
    """Save complete gene lookup table by SeqId,
    acts as a pseduo glosarry of proteins
    
    Extracts annotation columns from the raw expression matrix, sets SeqId
    as the index, and writes the result to a parquet file
    
    Args:
        df_exp: Raw expression matrix containing annotation columns
        output_dir: Target directory for the parquet file, defaults to
                    '../../mid_processing_datasets'.
    
    Returns:
        Gene lookup DataFrame indexed by SeqId, with annotation columns
        (EntrezGeneSymbol, EntrezGeneID, Target, TargetFullName, UniProt)
    """
    lookup_cols = ['SeqId', 'EntrezGeneSymbol', 'EntrezGeneID', 
                   'Target', 'TargetFullName', 'UniProt']
    
    gene_lookup = df_exp[lookup_cols].set_index('SeqId').copy()
    # force a string type
    gene_lookup = gene_lookup.astype(str)
    os.makedirs(output_dir, exist_ok=True)
    path = f'{output_dir}/gene_lookup.parquet'
    gene_lookup.to_parquet(path)
    print(f"Saved: {path}")
    return gene_lookup