"""K-Means clustering pipeline for multi-omics data.

Pipeline per dataset:
    Raw features
        → StandardScaler
        → PCA(n_pca_components)          noise reduction
        → UMAP(n_umap_components)        non-linear feature reduction
        → Silhouette-based k selection   k = 2..k_max
        → K-Means(best_k)
        → Cluster labels saved as .txt and .parquet
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from umap import UMAP


# ============================================================================
# Feature preparation
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
    n_pca_components: int = 50,
    n_umap_components: int = 10,
    metric: str = 'cosine',
    random_state: int = 42,
) -> tuple[np.ndarray, pd.Series]:
    """Preprocess DataFrame into a UMAP-reduced feature matrix.

    Steps:
        1. Select numeric columns, median-impute NaNs
        2. StandardScaler
        3. PCA(n_pca_components) for noise reduction
        4. UMAP(n_umap_components) for non-linear feature reduction

    Args:
        df:                Input DataFrame (non-numeric cols ignored).
        n_pca_components:  PCA components before UMAP.
        n_umap_components: Final feature dimensions fed to clustering.
                           Use 2 for visualisation only, 10+ for clustering.
        metric:            Distance metric for UMAP ('cosine' recommended
                           for proteomics/expression data).
        random_state:      Reproducibility seed.

    Returns:
        Tuple (X_umap, patient_ids) where X_umap is (n_samples,
        n_umap_components) and patient_ids is a Series of ID strings.
    """
    # Patient IDs
    if 'Patient_ID' in df.columns:
        patient_ids = df['Patient_ID'].astype(str).reset_index(drop=True)
    else:
        patient_ids = pd.Series(
            df.index.astype(str), name='Patient_ID'
        ).reset_index(drop=True)

    X = df.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    print(f"  Input shape       : {X.shape}")

    # Scale
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    n_pca = min(n_pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA components    : {n_pca}  ({var*100:.1f}% variance)")

    # UMAP
    n_umap = min(n_umap_components, n_pca)
    print(f"  UMAP components   : {n_umap}  (metric='{metric}')")
    reducer = UMAP(
        n_components=n_umap,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"  Feature matrix    : {X_umap.shape}")

    return X_umap, patient_ids


# ============================================================================
# K selection via Silhouette Score
# ============================================================================

def find_optimal_k(
    X: np.ndarray,
    k_range: range | list[int],
    output_dir: str,
    name: str,
    random_state: int = 42,
) -> tuple[int, dict[int, dict[str, float]]]:
    """Evaluate K-Means for each k in k_range using multiple clustering metrics.

    Higher silhouette score = better-separated clusters.
    Score range: [-1, 1].  Values above 0.5 indicate good structure.

    Args:
        X:            Feature matrix (n_samples, n_features).
        k_range:      Iterable of k values to test.
        output_dir:   Where to save the silhouette score plot.
        name:         Dataset name for plot title and filename.
        random_state: Reproducibility seed.

    Returns:
        Tuple (best_k, scores) where scores is {k: silhouette_score}.
    """
    scores: dict[int, dict[str, float]] = {}

    for k in k_range:
        if k >= X.shape[0]:
            print(f"  k={k} skipped (>= n_samples)")
            continue

        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        if len(set(labels)) < 2:
            print(f"  k={k} skipped (only 1 cluster found)")
            continue

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        inertia = km.inertia_

        scores[k] = {
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "inertia": inertia,
        }

        print(
            f"  k={k}  "
            f"silhouette={sil:.4f}  "
            f"calinski_harabasz={ch:.4f}  "
            f"davies_bouldin={db:.4f}  "
            f"inertia={inertia:.4f}"
        )

    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    print(
        f"  Best k: {best_k}  "
        f"(silhouette={scores[best_k]['silhouette']:.4f})"
    )

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = sorted(scores)
    vals = [scores[k]["silhouette"] for k in ks]
    ax.plot(ks, vals, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.axvline(best_k, color='tomato', linestyle='--', alpha=0.7,
               label=f'Best k={best_k}')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'{name}: Silhouette Score per k')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{name}_silhouette.svg')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return best_k, scores


# ============================================================================
# Cluster label output
# ============================================================================

def save_cluster_labels(
    patient_ids: pd.Series,
    labels: np.ndarray,
    output_dir: str,
    name: str,
    metadata: dict,
) -> None:
    """Save cluster assignments as .txt (TSV) and .parquet.

    Files created::

        {output_dir}/{name}_cluster_labels.txt
        {output_dir}/{name}_cluster_labels.parquet

    Args:
        patient_ids: Series of Patient_ID strings.
        labels:      Cluster label per sample (0-indexed).
        output_dir:  Save directory (created if missing).
        name:        Dataset name used as filename prefix.
        metadata:    Dict of run parameters stored in the parquet attrs.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_out = pd.DataFrame({
        'Patient_ID': patient_ids.reset_index(drop=True),
        'Cluster':    labels + 1,   # 1-indexed for readability
    })

    # .txt — tab-separated, human-readable
    txt_path = os.path.join(output_dir, f'{name}_cluster_labels.txt')
    df_out.to_csv(txt_path, sep='\t', index=False)
    print(f"  Saved: {txt_path}")

    # .parquet — includes metadata as custom attributes
    pq_path = os.path.join(output_dir, f'{name}_cluster_labels.parquet')
    df_out.to_parquet(pq_path, index=False)
    print(f"  Saved: {pq_path}")

    # Metadata sidecar as JSON
    meta_path = os.path.join(output_dir, f'{name}_clustering_meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  Saved: {meta_path}")

    # Cluster summary
    counts = df_out['Cluster'].value_counts().sort_index()
    print(f"\n  Cluster distribution:")
    for cluster, count in counts.items():
        print(f"    Cluster {cluster}: {count} patients "
              f"({count/len(df_out)*100:.1f}%)")


# ============================================================================
# Main wrapper
# ============================================================================

def run_kmeans_clustering(
    df: pd.DataFrame,
    name: str,
    output_dir: str,
    n_pca_components: int = 50,
    n_umap_components: int = 10,
    k_range: range | list[int] = range(2, 7),
    metric: str = 'cosine',
    random_state: int = 42,
) -> pd.DataFrame:
    """Full K-Means clustering pipeline: preprocess → k selection → cluster.

    Args:
        df:                Input DataFrame (non-numeric cols ignored).
        name:              Dataset name for filenames and print output.
        output_dir:        Where to save all outputs.
        n_pca_components:  PCA components before UMAP (noise reduction).
        n_umap_components: UMAP output dimensions for clustering.
        k_range:           K values to evaluate (default: 2 to 6).
        metric:            UMAP distance metric.
        random_state:      Seed for reproducibility.

    Returns:
        DataFrame with columns Patient_ID and Cluster (1-indexed).

    Example::

        from src.clustering_pipeline import run_kmeans_clustering

        labels = run_kmeans_clustering(
            df         = expression_bl,
            name       = "expression_bl",
            output_dir = "reports/clustering",
        )

        labels = run_kmeans_clustering(
            df         = multiomics_bl,
            name       = "multiomics_bl",
            output_dir = "reports/clustering",
        )
    """
    print(f"\n{'='*70}")
    print(f"CLUSTERING: {name.upper()}")
    print(f"{'='*70}")

    # 1. Feature preparation
    print("\n[1] Preparing features ...")
    X_umap, patient_ids = prepare_features(
        df, n_pca_components, n_umap_components, metric, random_state
    )

    # 2. Find optimal k
    print("\n[2] Selecting optimal k ...")
    best_k, k_scores = find_optimal_k(
        X_umap, k_range, output_dir, name, random_state
    )

    # 3. Final K-Means with best k
    print(f"\n[3] Running K-Means with k={best_k} ...")
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    final_labels = km_final.fit_predict(X_umap)
    final_sil = silhouette_score(X_umap, final_labels)
    print(f"  Final silhouette score: {final_sil:.4f}")

    final_ch = calinski_harabasz_score(X_umap, final_labels)
    final_db = davies_bouldin_score(X_umap, final_labels)
    final_inertia = km_final.inertia_

    print(f"  Final Calinski-Harabasz score: {final_ch:.4f}")
    print(f"  Final Davies-Bouldin score: {final_db:.4f}")
    print(f"  Final inertia: {final_inertia:.4f}")

    # 4. Save outputs
    print("\n[4] Saving outputs ...")
    metadata = {
        'dataset':            name,
        'n_samples':          len(patient_ids),
        'n_pca_components':   n_pca_components,
        'n_umap_components':  n_umap_components,
        'umap_metric':        metric,
        'k_range':            list(k_range),
        'best_k':             best_k,
        #'silhouette_scores':  {str(k): round(v, 4)
        #                      for k, v in silhouette_scores.items()},
        'k_scores': {
            str(k): {
                'silhouette': round(v['silhouette'], 4),
                'calinski_harabasz': round(v['calinski_harabasz'], 4),
                'davies_bouldin': round(v['davies_bouldin'], 4),
                'inertia': round(v['inertia'], 4),
            }
            for k, v in k_scores.items()
        },
        'final_silhouette':   round(final_sil, 4),
        'final_calinski_harabasz': round(final_ch, 4),
        'final_davies_bouldin': round(final_db, 4),
        'final_inertia': round(final_inertia, 4),
        'random_state':       random_state,
    }
    save_cluster_labels(patient_ids, final_labels, output_dir, name, metadata)

    result = pd.DataFrame({
        'Patient_ID': patient_ids.reset_index(drop=True),
        'Cluster':    final_labels + 1,
    })

    print(f"\nDone: {name}\n")
    return result


# ============================================================================
# Run directly:  python clustering_pipeline.py
# ============================================================================

if __name__ == "__main__":
    OUTPUT = "reports/clustering"

    DATASETS = [
        ("ml_variance", "datasets_final/ml_ready/ml_variance.csv"),
        ("ml_correlation", "datasets_final/ml_ready/ml_correlation.csv"),
        ("ml_literature", "datasets_final/ml_ready/ml_literature.csv"),
    ]

    N_PCA   = 50
    N_UMAP  = 10
    K_RANGE = range(2, 7)
    METRIC  = "cosine"
    SEED    = 42

    for name, path in DATASETS:
        if not os.path.exists(path):
            print(f"[SKIP] {name} — file not found: {path}")
            continue
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        run_kmeans_clustering(
            df                = df,
            name              = name,
            output_dir        = OUTPUT,
            n_pca_components  = N_PCA,
            n_umap_components = N_UMAP,
            k_range           = K_RANGE,
            metric            = METRIC,
            random_state      = SEED,
        )
