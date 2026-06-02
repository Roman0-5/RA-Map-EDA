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
    build_remission_labels,
    build_eular_labels,
)

# Maps label_type string to (builder_function, legend_title)
_LABEL_BUILDERS = {
    'remission': (build_remission_labels, 'Remission'),
    'eular':     (build_eular_labels,     'EULAR Response'),
}


def visualize_raw(df: pd.DataFrame, name: str = "dataset",
                  output_dir: str = "../../reports/raw",
                  top_n_boxplot: int = 20,
                  n_show_scree: int = 30,
                  clinical_df: pd.DataFrame | None = None,
                  label_type: str | list[str] = 'remission') -> None:
    """Run all EDA visualizations on raw data.

    Generates:
        - Distribution histogram (raw values)
        - PCA scatter (PC1 vs PC2), one per label_type, coloured accordingly
        - Scree plot
        - Boxplot of top-N high-variance features

    When labels are provided each PCA scatter also saves a .txt file with
    columns PC1, PC2, Label — one row per sample.

    Args:
        df:            Input DataFrame (with or without ID columns).
        name:          Display name and filename prefix.
        output_dir:    Where to save all plots.
        top_n_boxplot: Features to show in boxplot.
        n_show_scree:  PCs to show in scree plot.
        clinical_df:   Optional clinical DataFrame containing ``Patient_ID``
                       and the columns required for the chosen label type(s).
        label_type:    One label type or a list of label types to run.
                       Supported values: ``'remission'``, ``'eular'``.
                       Each produces a separate coloured PCA scatter + .txt.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"RAW EDA: {name.upper()}")
    print(f"{'='*70}")

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        print(f"No numeric columns in {name}. Skipping.")
        return

    print(f"Shape: {X.shape}")
    print(f"Value range: [{X.min().min():.1f}, {X.max().max():.1f}]")

    # 1. Distribution
    plot_distribution(X, name, output_dir, xlabel='Raw value')

    # 2. PCA — computed once, plotted once per label type
    X_scaled, pca = prepare_pca(X)

    # Extract Patient_ID if available — used to annotate the .txt output
    if 'Patient_ID' in df.columns:
        patient_ids = df['Patient_ID'].reset_index(drop=True)
    else:
        patient_ids = None

    label_types = [label_type] if isinstance(label_type, str) else label_type

    for lt in label_types:
        if lt not in _LABEL_BUILDERS:
            print(f"Unknown label_type '{lt}' — skipping. "
                  f"Valid options: {list(_LABEL_BUILDERS)}")
            continue

        labels, legend_title = None, 'Group'

        if clinical_df is not None:
            builder, legend_title = _LABEL_BUILDERS[lt]
            labels = builder(df, clinical_df)
            print(f"{legend_title} labels: {labels.value_counts().to_dict()}")

        # Filename includes label type so files don't overwrite each other
        scatter_name = f"{name}_{lt}"
        plot_pca_scatter(X_scaled, pca, scatter_name, output_dir,
                         labels=labels, label_name=legend_title,
                         patient_ids=patient_ids)

    plot_scree(pca, name, output_dir, n_show=n_show_scree)

    # 3. Boxplot
    plot_boxplot(X, name, output_dir, top_n=top_n_boxplot,
                 ylabel='Raw value')

    print(f"\nDone with {name}\n")