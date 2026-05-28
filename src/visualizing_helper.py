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
                     output_dir: str) -> None:
    """PC1 vs PC2 scatter plot.
    
    Args:
        X_scaled: Standardized data (samples × features).
        pca: Fitted PCA object.
        name: For title and filename.
        output_dir: Save location.
    """
    pcs = pca.transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6, s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'{name}: PCA (PC1 vs PC2)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = f'{output_dir}/{name}_pca_scatter.svg'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


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


def prepare_pca(X: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Standardize data and fit PCA (all components).
    
    Args:
        X: DataFrame with numeric features.
    
    Returns:
        Tuple of (X_scaled, fitted_pca).
    """
    # Median-imputation für NaN
    X_imputed = X.fillna(X.median())
    
    # Standardisieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # PCA mit allen Komponenten
    pca = PCA()
    pca.fit(X_scaled)
    
    return X_scaled, pca