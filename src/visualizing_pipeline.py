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
    build_total_dose_y_therapy_labels,
    build_total_dose_x_therapy_labels,
    build_inflammation_labels
)

_LABEL_BUILDERS = {
    'remission':    (build_remission_labels,           'Remission'),
    'eular':        (build_eular_labels,               'EULAR Response'),
    'inflammation': (build_inflammation_labels,        'Inflammation'),
    'therapy_x':    (build_total_dose_x_therapy_labels,'Total Dose X'),
    'therapy_y':    (build_total_dose_y_therapy_labels,'Total Dose Y'),
}


def visualize_raw(df: pd.DataFrame, name: str = "dataset",
                  modality: str = "expr",
                  base_dir: str = "../../reports/raw",
                  top_n_boxplot: int = 20,
                  n_show_scree: int = 30,
                  clinical_df: pd.DataFrame | None = None,
                  label_type: str | list[str] = 'remission') -> None:
    """Run all EDA visualizations on raw data.

    Saves plots into structured subdirectories:
        {base_dir}/pca/{modality}/
        {base_dir}/scree/{modality}/
        {base_dir}/distribution/{modality}/
        {base_dir}/boxplot/{modality}/

    Args:
        df:            Input DataFrame (with or without ID columns).
        name:          Display name and filename prefix.
        modality:      Data modality: 'expr', 'pg', 'clinical', 'multiomics'.
        base_dir:      Root reports directory (default: ../../reports/raw).
        top_n_boxplot: Features to show in boxplot.
        n_show_scree:  PCs to show in scree plot.
        clinical_df:   Optional clinical DataFrame for label colouring.
        label_type:    One label type or list of label types.
    """
    def _dir(plot_type: str) -> str:
        path = os.path.join(base_dir, plot_type, modality)
        os.makedirs(path, exist_ok=True)
        return path

    print(f"\n{'='*70}")
    print(f"RAW EDA: {name.upper()}  [{modality}]")
    print(f"{'='*70}")

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        print(f"No numeric columns in {name}. Skipping.")
        return

    print(f"Shape: {X.shape}")
    print(f"Value range: [{X.min().min():.1f}, {X.max().max():.1f}]")

    # 1. Distribution
    plot_distribution(X, name, _dir('distribution'), xlabel='Raw value')

    # 2. PCA — computed once, plotted once per label type
    X_scaled, pca = prepare_pca(X)

    # Extract Patient_ID for .txt output
    patient_ids = df['Patient_ID'].reset_index(drop=True) \
        if 'Patient_ID' in df.columns else None

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

        pca_dir = os.path.join(_dir('pca'), lt)
        os.makedirs(pca_dir, exist_ok=True)
        plot_pca_scatter(X_scaled, pca, name, pca_dir,
                         labels=labels, label_name=legend_title,
                         patient_ids=patient_ids)

    plot_scree(pca, name, _dir('scree'), n_show=n_show_scree)

    # 3. Boxplot
    plot_boxplot(X, name, _dir('boxplot'), top_n=top_n_boxplot,
                 ylabel='Raw value')

    print(f"\nDone with {name}\n")