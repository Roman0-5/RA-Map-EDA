"""Correlation-based feature selection.

Identifies redundant features (|r| >= threshold) and keeps the one with
higher variance from each correlated pair.  Results are saved as both
.txt (one feature per line) and .json (with metadata).
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------

def remove_high_correlation_features(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Identify and remove redundant features by pairwise Pearson correlation.

    For each correlated pair (|r| >= threshold) the feature with **lower
    variance** is marked for removal.  The feature with higher variance is
    always kept, even if it is correlated with several others.

    Args:
        df:        Input DataFrame.  Non-numeric columns are ignored.
        threshold: Absolute correlation threshold in [0, 1).
                   Features with |r| >= threshold are considered redundant.

    Returns:
        Tuple ``(df_reduced, to_keep, to_drop)`` — the filtered DataFrame
        and both sorted lists of column names.

    Raises:
        ValueError: If threshold is not in [0, 1).
    """
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        print("No numeric columns found — nothing to select.")
        return df.copy(), [], []

    print(f"Computing correlation matrix for {X.shape[1]} features "
          f"({X.shape[0]} samples) …")

    variances = X.var()
    corr  = X.corr().abs()
    #isolate upper triangle of matrix 
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()

    # iterate over upper triangle 
    for col in upper.columns:
        if col in to_drop:
            continue
        # all values that are above the threshold respective to their column 
        partners = upper.index[upper[col] >= threshold].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            if variances[col] >= variances[partner]:
                to_drop.add(partner)   # col wins -> drop partner
            else:
                to_drop.add(col)       # partner wins -> drop col
                break #if column is dropped, break

    all_features   = list(X.columns)
    to_drop_sorted = sorted(to_drop)
    to_keep_sorted = sorted(f for f in all_features if f not in to_drop)
    df_reduced     = df[to_keep_sorted].copy()

    return df_reduced, to_keep_sorted, to_drop_sorted


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_feature_lists(
    to_keep: list[str],
    to_drop: list[str],
    output_dir: str,
    name: str,
    threshold: float,
) -> None:
    """Save to_keep and to_drop lists as .txt and .json files.

    File naming convention::

        {output_dir}/{name}_features_to_keep.txt
        {output_dir}/{name}_features_to_drop.txt
        {output_dir}/{name}_feature_selection.json   <- both lists + metadata

    Args:
        to_keep:    Features that should be retained.
        to_drop:    Features marked as redundant.
        output_dir: Directory to write files to (created if missing).
        name:       Dataset name used as filename prefix.
        threshold:  Threshold used during selection (stored in metadata).
    """
    os.makedirs(output_dir, exist_ok=True)

    for label, features in [('to_keep', to_keep), ('to_drop', to_drop)]:
        path = os.path.join(output_dir, f"{name}_features_{label}.txt")
        with open(path, 'w') as fh:
            fh.write('\n'.join(features))
        print(f"  Saved: {path}  ({len(features)} features)")

    payload = {
        'metadata': {
            'dataset':    name,
            'threshold':  threshold,
            'n_kept':     len(to_keep),
            'n_dropped':  len(to_drop),
            'n_total':    len(to_keep) + len(to_drop),
            'created_at': datetime.now().isoformat(timespec='seconds'),
        },
        'to_keep': to_keep,
        'to_drop': to_drop,
    }
    json_path = os.path.join(output_dir, f"{name}_feature_selection.json")
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Saved: {json_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_correlation_selection(
    df: pd.DataFrame,
    name: str,
    threshold: float,
    output_dir: str = "../../reports/feature_selection",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run full correlation-based feature selection and save results.

    Combines :func:`remove_high_correlation_features` and
    :func:`save_feature_lists` into a single call.

    Args:
        df:         Input DataFrame (non-numeric columns are ignored).
        name:       Dataset name — used for print output and filenames.
        threshold:  Absolute Pearson |r| threshold in [0, 1).
        output_dir: Where to save the output files.

    Returns:
        Tuple ``(df_reduced, to_keep, to_drop)`` — the filtered DataFrame
        and both sorted lists of column names.

    Example::

        from src.feature_selection import run_correlation_selection

        df_reduced, to_keep, to_drop = run_correlation_selection(
            df        = expression_bl,
            name      = "expression_bl",
            threshold = 0.95,
            output_dir= "reports/feature_selection",
        )

        # Use in downstream pipeline
        X_train = train_df[to_keep]
        X_test  = test_df[to_keep]
    """
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION: {name.upper()}  (threshold = {threshold})")
    print(f"{'='*70}")

    df_reduced, to_keep, to_drop = remove_high_correlation_features(df, threshold)

    total = len(to_keep) + len(to_drop)
    print(f"\nResults:")
    print(f"  Total numeric features : {total}")
    print(f"  Kept                   : {len(to_keep)} "
          f"({len(to_keep)/total*100:.1f}%)")
    print(f"  Dropped (redundant)    : {len(to_drop)} "
          f"({len(to_drop)/total*100:.1f}%)")

    print("\nSaving feature lists ...")
    save_feature_lists(to_keep, to_drop, output_dir, name, threshold)

    return df_reduced, to_keep, to_drop