"""EDA visualizations on log2-transformed data.

For RFU and similar lognormal-distributed data. Applies log2(X + 1)
before plotting.
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


def visualize_log(df: pd.DataFrame, name: str = "dataset",
                  output_dir: str = "../../reports/log2",
                  top_n_boxplot: int = 20,
                  n_show_scree: int = 30) -> None:
    """Run all EDA visualizations on log2-transformed data.
    
    Applies log2(X + 1) transformation, then generates:
        - Distribution histogram (log-scale)
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
    print(f"LOG2 EDA: {name.upper()}")
    print(f"{'='*70}")
    
    # Nur numerische Features
    X = df.select_dtypes(include=[np.number])
    
    if X.shape[1] == 0:
        print(f"No numeric columns in {name}. Skipping.")
        return
    
    # Log2-Transformation
    X_log = np.log2(X + 1)
    
    print(f"Shape: {X_log.shape}")
    print(f"Log2 value range: [{X_log.min().min():.2f}, {X_log.max().max():.2f}]")
    
    # 1. Verteilung
    plot_distribution(X_log, f"{name}_log2", output_dir, 
                     xlabel='log2(value + 1)')
    
    # 2. PCA + Scree
    X_scaled, pca = prepare_pca(X_log)
    plot_pca_scatter(X_scaled, pca, f"{name}_log2", output_dir)
    plot_scree(pca, f"{name}_log2", output_dir, n_show=n_show_scree)
    
    # 3. Boxplot
    plot_boxplot(X_log, f"{name}_log2", output_dir, top_n=top_n_boxplot,
                 ylabel='log2(value + 1)')
    
    print(f"\nDone with {name}\n")