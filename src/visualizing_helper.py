"""Shared plotting helpers for exploratory data analysis.

All functions take pre-processed numeric data and produce a single plot.
Used by eda_raw.py and eda_log.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_distribution(X: pd.DataFrame, name: str, output_dir: str,
                      xlabel: str = 'Value') -> None:
    """Histogram of all numeric values in the dataset.
    
    Args:
        X: DataFrame with only numeric columns.
        name: Used for title and filename.
        output_dir: Where to save the plot.
        xlabel: Label for x-axis (e.g. 'RFU' or 'log2(RFU + 1)').
    """
    all_values = X.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_values, bins=100, color='steelblue', alpha=0.7)
    ax.set_title(f'{name}: Value Distribution')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = f'{output_dir}/{name}_distribution.svg'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_pca_scatter(X_scaled: np.ndarray, pca: PCA, name: str,
                     output_dir: str,
                     labels: pd.Series | None = None,
                     label_name: str = 'Group') -> pd.DataFrame | None:
    """PC1 vs PC2 scatter plot, optionally coloured by a clinical label.

    When labels are provided the function also saves a .txt file containing
    one row per sample with columns: PC1, PC2, Label.

    Args:
        X_scaled:   Standardized data (samples × features).
        pca:        Fitted PCA object.
        name:       For title and filename.
        output_dir: Save location.
        labels:     Optional Series/array (same length as X_scaled rows) with
                    group strings per sample.  When None the plot is monochrome.
        label_name: Legend title shown in the plot.

    Returns:
        DataFrame with columns PC1, PC2, Label when labels are provided,
        otherwise None.
    """
    pcs = pca.transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 8))

    label_df = None

    if labels is None:
        ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6, s=50,
                   edgecolor='white', linewidths=0.5)
    else:
        labels = pd.Series(labels).reset_index(drop=True)
        groups = labels.unique()

        palette = {
            'Remission':        '#2196F3',
            'Non-Remission':    '#F44336',
            'Good Responder':   '#4CAF50',
            'Moderate':         '#FF9800',
            'Non-Responder':    '#F44336',
            'Unknown':          '#BDBDBD',
        }
        fallback = sns.color_palette('tab10', n_colors=len(groups))
        for i, g in enumerate(groups):
            if g not in palette:
                palette[g] = fallback[i]

        for group in groups:
            mask = labels == group
            ax.scatter(
                pcs[mask, 0], pcs[mask, 1],
                label=f'{group} (n={mask.sum()})',
                color=palette.get(group, 'gray'),
                alpha=0.7, s=55,
                edgecolor='white', linewidths=0.5,
            )

        ax.legend(title=label_name, framealpha=0.9, loc='best', fontsize=9)

        # Build label DataFrame and save as .txt
        label_df = pd.DataFrame({
            'PC1':   pcs[:, 0],
            'PC2':   pcs[:, 1],
            'Label': labels,
        })
        txt_path = f'{output_dir}/{name}_pca_labels.txt'
        label_df.to_csv(txt_path, sep='\t', index=False)
        print(f"Saved: {txt_path}")

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'{name}: PCA (PC1 vs PC2)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{output_dir}/{name}_pca_scatter.svg'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return label_df


def plot_scree(pca: PCA, name: str, output_dir: str, 
               n_show: int = 30) -> None:
    """Scree plot: individual + cumulative explained variance.
    
    Args:
        pca: Fitted PCA object.
        name: For title and filename.
        output_dir: Save location.
        n_show: How many PCs to display.
    """
    n_show = min(n_show, len(pca.explained_variance_ratio_))
    explained = pca.explained_variance_ratio_[:n_show]
    cumulative = np.cumsum(pca.explained_variance_ratio_)[:n_show]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(1, n_show + 1)
    
    ax1.bar(x, explained * 100, alpha=0.6, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual variance (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, 'ro-', linewidth=2, markersize=5)
    ax2.set_ylabel('Cumulative variance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    for threshold in [50, 80, 90]:
        ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.4)
    
    ax1.set_title(f'{name}: Scree Plot (first {n_show} PCs)')
    ax1.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = f'{output_dir}/{name}_scree.svg'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_boxplot(X: pd.DataFrame, name: str, output_dir: str,
                 top_n: int = 20, ylabel: str = 'Value') -> None:
    """Boxplot with stripplot overlay for top-N high-variance features.
    
    Args:
        X: DataFrame with numeric features.
        name: For title and filename.
        output_dir: Save location.
        top_n: Number of features to show (sorted by variance).
        ylabel: Y-axis label.
    """
    top_features = X.var().sort_values(ascending=False).head(top_n).index
    data = X[top_features]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.boxplot(data=data, ax=ax, whis=1.5, color='lightgray',
                showfliers=False)
    sns.stripplot(data=data, ax=ax, color='steelblue', alpha=0.5,
                  size=3, jitter=0.25)
    
    ax.set_title(f'{name}: Top {top_n} high-variance features')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = f'{output_dir}/{name}_boxplot_top{top_n}.svg'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def build_remission_labels(df: pd.DataFrame,
                           clinical_df: pd.DataFrame) -> pd.Series:
    """Derive per-sample Remission / Non-Remission / Unknown labels.

    Args:
        df: Feature DataFrame whose row order must be preserved.
        clinical_df: Clinical data containing ``Patient_ID`` and
                     ``Remission month``.

    Returns:
        Series of label strings, same length and order as ``df``.
    """
    def _get_ids(frame: pd.DataFrame) -> pd.Series:
        if 'Patient_ID' in frame.columns:
            return frame['Patient_ID'].astype(str)
        return frame.index.astype(str).rename('Patient_ID')

    ids = _get_ids(df).reset_index(drop=True)

    clin = clinical_df.copy()
    if 'Patient_ID' not in clin.columns:
        clin = clin.reset_index()
    clin['Patient_ID'] = clin['Patient_ID'].astype(str)

    lookup = clin.set_index('Patient_ID')['Remission month']

    def _map(pid: str) -> str:
        if pid not in lookup.index:
            return 'Unknown'
        return 'Remission' if pd.notna(lookup[pid]) else 'Non-Remission'

    return ids.map(_map)


def build_eular_labels(df: pd.DataFrame,
                       clinical_df: pd.DataFrame) -> pd.Series:
    """Derive per-sample EULAR response labels (3 classes).

    Args:
        df: Feature DataFrame whose row order must be preserved.
        clinical_df: Clinical data with ``Patient_ID``, ``DAS28.0M``,
                     ``DAS28.6M``.

    Returns:
        Series of label strings, same length and order as ``df``.
    """
    def _get_ids(frame: pd.DataFrame) -> pd.Series:
        if 'Patient_ID' in frame.columns:
            return frame['Patient_ID'].astype(str)
        return frame.index.astype(str).rename('Patient_ID')

    ids = _get_ids(df).reset_index(drop=True)

    clin = clinical_df.copy()
    if 'Patient_ID' not in clin.columns:
        clin = clin.reset_index()
    clin['Patient_ID'] = clin['Patient_ID'].astype(str)
    clin = clin.set_index('Patient_ID')[['DAS28.0M', 'DAS28.6M']]

    def _eular(pid: str) -> str:
        if pid not in clin.index:
            return 'Unknown'
        row = clin.loc[pid]
        bl, m6 = row['DAS28.0M'], row['DAS28.6M']
        if pd.isna(bl) or pd.isna(m6):
            return 'Unknown'
        delta = bl - m6
        if delta > 1.2 and m6 <= 3.2:
            return 'Good Responder'
        if (delta > 1.2 and m6 > 3.2) or (delta > 0.6 and m6 <= 5.1):
            return 'Moderate'
        return 'Non-Responder'

    return ids.map(_eular)


def prepare_pca(X: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Standardize data and fit PCA (all components).
    
    Args:
        X: DataFrame with numeric features.
    
    Returns:
        Tuple of (X_scaled, fitted_pca).
    """
    X_imputed = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    return X_scaled, pca