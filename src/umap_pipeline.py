"""UMAP dimensionality reduction pipeline for multi-omics data.

Pipeline:
    Raw features
        → StandardScaler
        → PCA(n_pca_components)       noise reduction
        → UMAP(n_components=2)        2D embedding for visualisation
        → Scatter plot coloured by clinical label
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP


# ============================================================================
# Colour palette (shared with visualizing_helper)
# ============================================================================

PALETTE = {
    'Remission':        '#2196F3',
    'Non-Remission':    '#F44336',
    'Good Responder':   '#4CAF50',
    'Moderate':         '#FF9800',
    'Non-Responder':    '#F44336',
    'Unknown':          '#BDBDBD',
}


# ============================================================================
# Label builders  (same logic as visualizing_helper)
# ============================================================================

def _get_patient_ids(df: pd.DataFrame) -> pd.Series:
    if 'Patient_ID' in df.columns:
        return df['Patient_ID'].astype(str).reset_index(drop=True)
    return pd.Series(df.index.astype(str), name='Patient_ID').reset_index(drop=True)


def build_remission_labels(df: pd.DataFrame,
                           clinical_df: pd.DataFrame) -> pd.Series:
    ids  = _get_patient_ids(df)
    clin = clinical_df.copy()
    if 'Patient_ID' not in clin.columns:
        clin = clin.reset_index()
    clin['Patient_ID'] = clin['Patient_ID'].astype(str)
    lookup = clin.set_index('Patient_ID')['Remission month']

    def _map(pid):
        if pid not in lookup.index:
            return 'Unknown'
        return 'Remission' if pd.notna(lookup[pid]) else 'Non-Remission'

    return ids.map(_map)


def build_eular_labels(df: pd.DataFrame,
                       clinical_df: pd.DataFrame) -> pd.Series:
    ids  = _get_patient_ids(df)
    clin = clinical_df.copy()
    if 'Patient_ID' not in clin.columns:
        clin = clin.reset_index()
    clin['Patient_ID'] = clin['Patient_ID'].astype(str)
    clin = clin.set_index('Patient_ID')[['DAS28.0M', 'DAS28.6M']]

    def _eular(pid):
        if pid not in clin.index:
            return 'Unknown'
        bl, m6 = clin.loc[pid, 'DAS28.0M'], clin.loc[pid, 'DAS28.6M']
        if pd.isna(bl) or pd.isna(m6):
            return 'Unknown'
        delta = bl - m6
        if delta > 1.2 and m6 <= 3.2:
            return 'Good Responder'
        if (delta > 1.2 and m6 > 3.2) or (delta > 0.6 and m6 <= 5.1):
            return 'Moderate'
        return 'Non-Responder'

    return ids.map(_eular)


_LABEL_BUILDERS = {
    'remission': (build_remission_labels, 'Remission'),
    'eular':     (build_eular_labels,     'EULAR Response'),
}


# ============================================================================
# Core pipeline
# ============================================================================

def fit_umap_embedding(
    df: pd.DataFrame,
    n_pca_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Preprocess and compute 2D UMAP embedding.

    Steps:
        1. Select numeric columns, median-impute NaNs
        2. StandardScaler
        3. PCA(n_pca_components) for noise reduction
        4. UMAP(n_components=2)

    Args:
        df:               Input DataFrame (non-numeric cols ignored).
        n_pca_components: Number of PCA components before UMAP.
                          Set to None to skip PCA.
        n_neighbors:      UMAP neighbourhood size.  Smaller = more local
                          structure, larger = more global structure.
        min_dist:         Minimum distance between points in 2D embedding.
        metric:           Distance metric ('cosine' recommended for proteomics).
        random_state:     Reproducibility seed.

    Returns:
        Tuple (embedding, info) where embedding is (n_samples, 2) array
        and info is a dict with variance explained by PCA etc.
    """
    X = df.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    print(f"Input shape       : {X.shape}")

    # 1. Scale
    X_scaled = StandardScaler().fit_transform(X)

    # 2. PCA
    info = {}
    if n_pca_components is not None:
        n_pca_components = min(n_pca_components, X_scaled.shape[1],
                               X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_.sum()
        info['pca_components']    = n_pca_components
        info['pca_var_explained'] = var_explained
        print(f"PCA components    : {n_pca_components}  "
              f"({var_explained*100:.1f}% variance)")
        X_input = X_pca
    else:
        print("PCA skipped")
        X_input = X_scaled

    # 3. UMAP
    print(f"Running UMAP "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric='{metric}') …")
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_input)
    info['umap_params'] = dict(n_neighbors=n_neighbors, min_dist=min_dist,
                               metric=metric)
    print(f"Embedding shape   : {embedding.shape}")

    return embedding, info


# ============================================================================
# Plotting
# ============================================================================

def plot_umap_scatter(
    embedding: np.ndarray,
    name: str,
    output_dir: str,
    labels: pd.Series | None = None,
    label_name: str = 'Group',
    patient_ids: pd.Series | None = None,
) -> pd.DataFrame | None:
    """Scatter plot of 2D UMAP embedding, optionally coloured by label.

    Also saves a .txt file (Patient_ID, UMAP1, UMAP2, Label) when labels
    are provided.

    Args:
        embedding:   (n_samples, 2) UMAP coordinates.
        name:        Filename prefix.
        output_dir:  Save location.
        labels:      Optional group label per sample.
        label_name:  Legend title.
        patient_ids: Optional Patient_ID per sample for .txt output.

    Returns:
        DataFrame with UMAP coordinates and labels, or None.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    label_df = None

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   alpha=0.6, s=50, edgecolor='white', linewidths=0.5)
    else:
        labels  = pd.Series(labels).reset_index(drop=True)
        groups  = labels.unique()
        fallback = sns.color_palette('tab10', n_colors=len(groups))

        for i, group in enumerate(groups):
            mask  = labels == group
            color = PALETTE.get(group, fallback[i])
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                label=f'{group} (n={mask.sum()})',
                color=color, alpha=0.7, s=55,
                edgecolor='white', linewidths=0.5,
            )

        ax.legend(title=label_name, framealpha=0.9, loc='best', fontsize=9)

        label_df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Label': labels,
        })
        if patient_ids is not None:
            label_df.insert(0, 'Patient_ID',
                            pd.Series(patient_ids).reset_index(drop=True))

        txt_path = os.path.join(output_dir, f'{name}_umap_labels.txt')
        label_df.to_csv(txt_path, sep='\t', index=False)
        print(f"Saved: {txt_path}")

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'{name}: UMAP')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    svg_path = os.path.join(output_dir, f'{name}_umap_scatter.svg')
    plt.savefig(svg_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {svg_path}")

    return label_df


# ============================================================================
# Convenience wrapper
# ============================================================================

def run_umap(
    df: pd.DataFrame,
    name: str,
    output_dir: str,
    clinical_df: pd.DataFrame | None = None,
    label_type: str | list[str] = 'remission',
    n_pca_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Full UMAP pipeline: preprocess → embed → plot.

    Args:
        df:               Input DataFrame.
        name:             Dataset name for filenames and titles.
        output_dir:       Where to save plots and .txt files.
        clinical_df:      Clinical data for label colouring.
        label_type:       ``'remission'``, ``'eular'``, or a list of both.
        n_pca_components: PCA components before UMAP (None to skip).
        n_neighbors:      UMAP n_neighbors parameter.
        min_dist:         UMAP min_dist parameter.
        metric:           Distance metric for UMAP.
        random_state:     Random seed for reproducibility.

    Returns:
        Tuple (embedding, label_df) — 2D coordinates and label DataFrame.

    Example::

        from src.umap_pipeline import run_umap

        embedding, label_df = run_umap(
            df          = expr_bl,
            name        = "expression_bl",
            output_dir  = "reports/umap",
            clinical_df = clinical,
            label_type  = ['remission', 'eular'],
        )
    """
    print(f"\n{'='*70}")
    print(f"UMAP: {name.upper()}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # Compute embedding once
    embedding, info = fit_umap_embedding(
        df, n_pca_components=n_pca_components,
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=random_state,
    )

    patient_ids = _get_patient_ids(df)
    label_types = [label_type] if isinstance(label_type, str) else label_type

    last_label_df = None
    for lt in label_types:
        if lt not in _LABEL_BUILDERS:
            print(f"Unknown label_type '{lt}' — skipping.")
            continue

        labels, legend_title = None, 'Group'
        if clinical_df is not None:
            builder, legend_title = _LABEL_BUILDERS[lt]
            labels = builder(df, clinical_df)
            print(f"{legend_title} labels: {labels.value_counts().to_dict()}")

        scatter_name  = f"{name}_{lt}"
        last_label_df = plot_umap_scatter(
            embedding, scatter_name, output_dir,
            labels=labels, label_name=legend_title,
            patient_ids=patient_ids,
        )

    print(f"\nDone with {name}\n")
    return embedding, last_label_df