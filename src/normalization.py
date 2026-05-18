import pandas as pd
import numpy as np
from src.extraction import extract_dtypes
import src.check_data

#predefined exclusion from known values
exclude_columns = ['Study', 'Patient_ID', 'Digest']

def normalize_num_dtype(df):
    """
    chooses the optimal dtype by checking the bitsize with np methods
    """
    df = df.copy()
    df, only_num, _, _ = extract_dtypes(df)
    
    for col in only_num.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        
        # float or int?
        # if the series contains at least 1 float the whole row is float
        #dropna drops all NaN values
        #apply() applies the function
        #float_integer returns true if int or false if float by checking decimal point value
        #.all() only returns true if all values (float.is_integer) are true 
        if not series.dropna().apply(float.is_integer).all():
            max_abs = series.abs().max()
            # if the absolute value is under Float32 Bytesize cast Float32 on it else Float64
            if max_abs < np.finfo(np.float32).max:
                df[col] = series.astype('Float32')
            else:
                df[col] = series.astype('Float64')
            continue
        
        # integer logic for casting
        min_val = series.min()
        max_val = series.max()
        
        # only positive values get unsigned integers, negative values get normal int values
        if min_val >= 0:
            if max_val <= 255:
                df[col] = series.astype('UInt8')
            elif max_val <= 65535:
                df[col] = series.astype('UInt16')
            elif max_val <= 4294967295:
                df[col] = series.astype('UInt32')
            else:
                df[col] = series.astype('UInt64')
        else:
            if min_val >= -128 and max_val <= 127:
                df[col] = series.astype('Int8')
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = series.astype('Int16')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                df[col] = series.astype('Int32')
            else:
                df[col] = series.astype('Int64')
    
    return df

def normalize_bool_dtype(df):
    """
    converts obj and str dtypes to bool
    """
    df = df.copy()
    df, _,only_str, only_obj = extract_dtypes(df)
    yes_no_values = {'yes', 'no', 'y', 'n'}
    # cast booleans on words
    boolean_schema = {
        'yes': True, 'YES': True, 'Y': True, 'y': True, 'Yes': True,
        'no': False, 'NO': False, 'n': False, 'N': False, 'No': False
    }
    for col in list(only_str.columns) + list(only_obj.columns):
        unique_vals = set(df[col].dropna().str.lower().str.strip().unique())
        if unique_vals.issubset(yes_no_values):
            df[col] = df[col].map(boolean_schema).astype('boolean')
    return df

def normalize_str_dtype(df, exclude_columns=None):
    """
    Cleanup String rows
    pass exclude_columns to ignore specifc rows
    """
    df = df.copy()
    df, _, only_str, _ = extract_dtypes(df)
    
    if exclude_columns is None:
        exclude_columns = []
    
    for col in only_str.columns:
        if col in exclude_columns:
            continue
        # Strip leading/trailing whitespace
        df[col] = df[col].str.strip()
        # Collapse multiple internal spaces/tabs/newlines to single space
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        # Convert empty strings to actual NaN
        df[col] = df[col].replace('', pd.NA)
    return df

def normalize_obj_dtype(df, missing_variants=None):
    """
    Prepares obj rows for other functions, similar to normalize_str_dtypes
    
    Input missing_variants as needed
    - replaces missing_variants with NaN
    - strips whitespaces
    - empty strings get converted to NaN
    """
    df = df.copy()
    
    if missing_variants is None:
        missing_variants = ['ND', 'Missing', 'unknown', 'nd', 'missing', 
                            'Unknown', 'NA', 'N/A', '-', '']
    
    df, _, _, only_obj = extract_dtypes(df)
    
    for col in only_obj.columns:
        # strip strings
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass  # if not a string skip it
        
        # replace missing variants
        df[col] = df[col].replace(missing_variants, pd.NA)
    
    return df

def normalize_all(df, exclude_columns=None):
    """Run all normalization steps in correct order.
    
    Order matters:
    1. obj: prepare object columns (whitespace, missings)
    2. bool: detect and convert Yes/No columns
    3. num: optimize numeric dtypes
    4. str: clean string columns
    
    Args:
        df: DataFrame to normalize.
        exclude_columns: List of columns to skip in str normalization.
    
    Returns:
        Normalized DataFrame. Also runs check_data() for a quality report.
    """
    
    df = normalize_obj_dtype(df)
    df = normalize_bool_dtype(df)
    df = normalize_num_dtype(df)
    df = normalize_str_dtype(df, exclude_columns=exclude_columns)
    return df