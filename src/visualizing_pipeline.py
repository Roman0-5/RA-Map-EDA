"""EDA visualizations on raw (untransformed) data.

Produces distribution, PCA, scree, and boxplot for a single dataset.
"""

import os
import numpy as np
import pandas as pd
from src.visualizing_helper import (
    plot_distribution,
    plot_pca_scatter,
    plot_scree,
    plot_boxplot,
    prepare_pca,
)


def visualize_raw(df: pd.DataFrame, name: str = "dataset",
                  output_dir: str = "../../reports/raw",
                  top_n_boxplot: int = 20,
                  n_show_scree: int = 30) -> None:
    """Run all EDA visualizations on raw data.
    
    Generates:
        - Distribution histogram (raw values)
        - PCA scatter (PC1 vs PC2)
        - Scree plot
        - Boxplot of top-N high-variance features
    
    Args:
        df: Input DataFrame (with or without ID columns).
        name: Display name and filename prefix.
        output_dir: Where to save all plots.
        top_n_boxplot: Features to show in boxplot.
        n_show_scree: PCs to show in scree plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RAW EDA: {name.upper()}")
    print(f"{'='*70}")
    
    # Nur numerische Features
    X = df.select_dtypes(include=[np.number])
    
    if X.shape[1] == 0:
        print(f"No numeric columns in {name}. Skipping.")
        return
    
    print(f"Shape: {X.shape}")
    print(f"Value range: [{X.min().min():.1f}, {X.max().max():.1f}]")
    
    # 1. Verteilung
    plot_distribution(X, name, output_dir, xlabel='Raw value')
    
    # 2. PCA + Scree (gemeinsame Vorbereitung)
    X_scaled, pca = prepare_pca(X)
    plot_pca_scatter(X_scaled, pca, name, output_dir)
    plot_scree(pca, name, output_dir, n_show=n_show_scree)
    
    # 3. Boxplot
    plot_boxplot(X, name, output_dir, top_n=top_n_boxplot, 
                 ylabel='Raw value')
    
    print(f"\nDone with {name}\n")