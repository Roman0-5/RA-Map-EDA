import pandas as pd
import numpy as np
def low_variance_report(df, name="dataset", threshold=0.2, return_drop_list=True, verbose=True):
    """
    Detects low-variance features in a dataset.
    This is a feature screening step BEFORE preprocessing.
    Parameters:
    df : pd.DataFrame
        Input dataset
    name : str
        Dataset name for reporting
    threshold : float
        Variance threshold below which features are considered low-variance
    return_drop_list : bool
        If True, returns list of columns to drop
    verbose : bool
        If True, prints report
    Returns:
    dict containing:
        - low_variance_features
        - variance_series
        - suggested_drop_columns
    """

    if verbose:
        print(f"\n===== LOW VARIANCE REPORT: {name.upper()} =====\n")
    # keep only numeric features (important for clustering)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        if verbose:
            print("No numeric features found.")
        return {
            "low_variance_features": [],
            "variance": pd.Series(dtype=float),
            "drop_columns": []
        }
    # compute variance
    variance = num_df.var().sort_values()
    # identify low variance features
    low_var_features = variance[variance < threshold]
    if verbose:
        print(f"Total numeric features: {num_df.shape[1]}")
        print(f"Low variance threshold: {threshold}")
        print(f"Low variance features found: {len(low_var_features)}\n")
        if len(low_var_features) > 0:
            print("Top low-variance features:")
            print(low_var_features.head(20))
        else:
            print("No low-variance features detected.")
    # suggest full drop list
    drop_cols = low_var_features.index.tolist()
    if return_drop_list:
        return {
            "low_variance_features": low_var_features,
            "variance": variance,
            "drop_columns": drop_cols
        }
    return {
        "low_variance_features": low_var_features,
        "variance": variance
    }