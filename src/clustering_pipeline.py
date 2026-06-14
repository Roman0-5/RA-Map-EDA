"""K-Means clustering pipeline for multi-omics data.

Pipeline per dataset:
    Raw features
        → StandardScaler
        → PCA(n_pca_components)          noise reduction
        → UMAP(n_umap_components)        non-linear feature reduction
        → Silhouette-based k selection   k = 2..k_max
        → K-Means(best_k)
        → Cluster labels saved as .csv
"""

import json
import os
from datetime import datetime
from itertools import product

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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
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
    print(f"  UMAP components   : {n_umap}  (metric='{metric}', "
          f"n_neighbors={n_neighbors}, min_dist={min_dist})")
    reducer = UMAP(
        n_components=n_umap,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
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
    file_prefix: str | None = None,
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

    # Plot: 2x2 panel of all four metrics across k.
    # Each panel marks its own optimal k. Note the directions:
    #   silhouette / calinski_harabasz -> HIGHER is better
    #   davies_bouldin / inertia       -> LOWER  is better (inertia = elbow)
    os.makedirs(output_dir, exist_ok=True)
    ks = sorted(scores)

    panels = [
        ("silhouette",        "Silhouette Score",  "higher = better", max),
        ("calinski_harabasz", "Calinski-Harabasz", "higher = better", max),
        ("davies_bouldin",    "Davies-Bouldin",    "lower = better",  min),
        ("inertia",           "Inertia (Elbow)",   "elbow: read the bend", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, label, direction, opt) in zip(axes.ravel(), panels):
        vals = [scores[k][key] for k in ks]
        ax.plot(ks, vals, 'o-', color='steelblue', linewidth=2, markersize=6)

        if opt is None:
            # Inertia always falls with k -> its minimum is meaningless for
            # picking k. Mark the silhouette-selected k as reference instead.
            ax.axvline(best_k, color='tomato', linestyle='--', alpha=0.7,
                       label=f'selected k={best_k}')
        else:
            # Mark the optimum of THIS metric (not necessarily best_k).
            opt_k = opt(ks, key=lambda k: scores[k][key])
            ax.axvline(opt_k, color='tomato', linestyle='--', alpha=0.7,
                       label=f'optimal k={opt_k}')

        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel(label)
        ax.set_title(f'{label}  ({direction})', fontsize=11)
        ax.set_xticks(ks)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'{name}: cluster metrics per k   '
                 f'(silhouette-selected best k = {best_k})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(output_dir, f'{file_prefix or name}_k_diagnostics.svg')
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
    """Save cluster assignments as .csv, plus a metadata sidecar as .json.

    Files created::

        {output_dir}/{name}_cluster_labels.csv
        {output_dir}/{name}_clustering_meta.json

    Args:
        patient_ids: Series of Patient_ID strings.
        labels:      Cluster label per sample (0-indexed).
        output_dir:  Save directory (created if missing).
        name:        Dataset name used as filename prefix.
        metadata:    Dict of run parameters stored in the JSON sidecar.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_out = pd.DataFrame({
        'Patient_ID': patient_ids.reset_index(drop=True),
        'Cluster':    labels + 1,   # 1-indexed for readability
    })

    # .csv — comma-separated labels
    csv_path = os.path.join(output_dir, f'{name}_cluster_labels.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

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
# PCA cluster visualisation
# ============================================================================

def plot_clusters_pca_projection(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_dir: str,
    name: str,
    random_state: int = 42,
) -> None:
    """
    Create a 2D PCA projection plot of the clustered patients.

    Note:
    - This is for visualisation only.
    - Your clustering can still be based on PCA + UMAP.
    - The plot shows the patients projected into 2 PCA dimensions.
    """
    # keep only numeric feature columns
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.median())

    # scale again for PCA projection plot
    X_scaled = StandardScaler().fit_transform(X)

    # 2D PCA for plotting
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    explained = pca_2d.explained_variance_ratio_ * 100

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, cluster in enumerate(unique_labels):
        mask = labels == cluster
        ax.scatter(
            X_pca_2d[mask, 0],
            X_pca_2d[mask, 1],
            s=45,
            alpha=0.8,
            color=cmap(i),
            label=f"{cluster + 1}"
        )

    ax.set_title(f"Patient clustering ({name}, PCA projection)")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.legend(title="Cluster")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(output_dir, f"{name}_cluster_pca_projection.svg")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {path}")

# ============================================================================
# Run naming  (parameters encoded into filenames; datetime is in the folder)
# ============================================================================

def build_run_name(
    dataset: str,
    n_pca: int,
    n_umap: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    best_k: int | None = None,
) -> str:
    """Build a self-describing run name from the parameters.

    Example::

        ml_correlation_pca50_umap10_nn30_md0.0_cos_k3

    The name encodes every parameter that changes the result. The datetime
    is NOT part of the name — it lives in the parent run folder created by
    ``run_parameter_sweep`` (e.g. ``ml_correlation_20260614-143207/``).

    Args:
        dataset:     Base dataset name (e.g. "ml_correlation").
        n_pca:       PCA components.
        n_umap:      UMAP output dimensions.
        n_neighbors: UMAP n_neighbors.
        min_dist:    UMAP min_dist.
        metric:      UMAP distance metric.
        best_k:      Chosen k (appended only once known).

    Returns:
        A filesystem-safe run name string.
    """
    metric_short = {
        "cosine": "cos",
        "euclidean": "euc",
        "manhattan": "man",
        "correlation": "cor",
    }.get(metric, metric[:3])

    parts = [
        dataset,
        f"pca{n_pca}",
        f"umap{n_umap}",
        f"nn{n_neighbors}",
        f"md{min_dist}",
        metric_short,
    ]
    if best_k is not None:
        parts.append(f"k{best_k}")
    return "_".join(parts)


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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Full K-Means clustering pipeline: preprocess → k selection → cluster.

    All outputs are saved under a self-describing run name that encodes
    every parameter plus a datetime stamp (see ``build_run_name``), so
    repeated runs with different parameters never overwrite each other.

    Args:
        df:                Input DataFrame (non-numeric cols ignored).
        name:              Dataset name; base of the run name / filenames.
        output_dir:        Where to save all outputs.
        n_pca_components:  PCA components before UMAP (noise reduction).
        n_umap_components: UMAP output dimensions for clustering.
        k_range:           K values to evaluate (default: 2 to 6).
        metric:            UMAP distance metric.
        n_neighbors:       UMAP n_neighbors (local vs. global structure).
        min_dist:          UMAP min_dist (cluster compactness).
        random_state:      Seed for reproducibility.
        return_meta:       If True, return (DataFrame, metadata dict).

    Returns:
        DataFrame with columns Patient_ID and Cluster (1-indexed), or
        a (DataFrame, metadata) tuple when ``return_meta`` is True.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # provenance only (meta.json)

    print(f"\n{'='*70}")
    print(f"CLUSTERING: {name.upper()}  "
          f"[pca{n_pca_components} umap{n_umap_components} "
          f"nn{n_neighbors} md{min_dist} {metric}]")
    print(f"{'='*70}")

    # Run name WITHOUT k (k is only known after selection) — used for the
    # k-diagnostics plot, which spans all k values.
    pre_tag = build_run_name(
        name, n_pca_components, n_umap_components,
        n_neighbors, min_dist, metric, best_k=None,
    )

    # 1. Feature preparation
    print("\n[1] Preparing features ...")
    X_umap, patient_ids = prepare_features(
        df, n_pca_components, n_umap_components,
        metric, n_neighbors, min_dist, random_state,
    )

    # 2. Find optimal k
    print("\n[2] Selecting optimal k ...")
    best_k, k_scores = find_optimal_k(
        X_umap, k_range, output_dir, name,
        file_prefix=pre_tag, random_state=random_state,
    )

    # Full run name now that best_k is known.
    run_name = build_run_name(
        name, n_pca_components, n_umap_components,
        n_neighbors, min_dist, metric, best_k=best_k,
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
        'run_name':           run_name,
        'timestamp':          timestamp,
        'dataset':            name,
        'n_samples':          len(patient_ids),
        'n_pca_components':   n_pca_components,
        'n_umap_components':  n_umap_components,
        'umap_metric':        metric,
        'umap_n_neighbors':   n_neighbors,
        'umap_min_dist':      min_dist,
        'k_range':            list(k_range),
        'best_k':             best_k,
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
    save_cluster_labels(patient_ids, final_labels, output_dir, run_name, metadata)

    print("\n[5] Saving PCA projection plot ...")
    plot_clusters_pca_projection(
        df=df,
        labels=final_labels,
        output_dir=output_dir,
        name=name,
        random_state=random_state,
    )

    result = pd.DataFrame({
        'Patient_ID': patient_ids.reset_index(drop=True),
        'Cluster':    final_labels + 1,
    })

    print(f"\nDone: {run_name}\n")
    if return_meta:
        return result, metadata
    return result

# ============================================================================
# Parameter sweep  (many runs, one coloured summary)
# ============================================================================

def run_parameter_sweep(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    param_grid: dict[str, list],
    k_range: range | list[int] = range(2, 7),
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the pipeline over every combination in a parameter grid.

    Each combination is saved with its own parameter+datetime run name. In
    addition, one summary is written twice: a plain ``.csv`` (machine-readable)

    Args:
        df:           Input DataFrame (same df reused for every combo).
        dataset_name: Base dataset name (e.g. "ml_correlation").
        output_dir:   Where to save all outputs.
        param_grid:   Dict mapping parameter name -> list of values to try.
                      Recognised keys: n_pca_components, n_umap_components,
                      metric, n_neighbors, min_dist, random_state.
                      Keys left out fall back to sensible defaults.
        k_range:      K values evaluated per run.
        random_state: Default seed (overridden if 'random_state' is in grid).

    Returns:
        Summary DataFrame (one row per run), also saved as .csv
    """
    defaults = {
        "n_pca_components":  50,
        "n_umap_components": 10,
        "metric":            "cosine",
        "n_neighbors":       15,
        "min_dist":          0.1,
        "random_state":      random_state,
    }

    combos = [dict(zip(param_grid, vals)) for vals in product(*param_grid.values())]
    sweep_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Dedicated folder per dataset+datetime; all runs of this sweep go inside.
    #   reports/clustering/ml_variance_20260614-143207/<all run files>
    run_dir = os.path.join(output_dir, f"{dataset_name}_{sweep_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# SWEEP: {dataset_name}  ({len(combos)} runs)")
    print(f"# Folder: {run_dir}")
    print(f"{'#'*70}")

    summary_rows: list[dict] = []
    for i, combo in enumerate(combos, 1):
        params = {**defaults, **combo}
        print(f"\n>>> Run {i}/{len(combos)}: {combo}")

        _, meta = run_kmeans_clustering(
            df                = df,
            name              = dataset_name,
            output_dir        = run_dir,
            n_pca_components  = params["n_pca_components"],
            n_umap_components = params["n_umap_components"],
            k_range           = k_range,
            metric            = params["metric"],
            n_neighbors       = params["n_neighbors"],
            min_dist          = params["min_dist"],
            random_state      = params["random_state"],
            return_meta       = True,
        )

        summary_rows.append({
            "run_name":          meta["run_name"],
            "dataset":           meta["dataset"],
            "n_pca":             meta["n_pca_components"],
            "n_umap":            meta["n_umap_components"],
            "metric":            meta["umap_metric"],
            "n_neighbors":       meta["umap_n_neighbors"],
            "min_dist":          meta["umap_min_dist"],
            "random_state":      meta["random_state"],
            "best_k":            meta["best_k"],
            "silhouette":        meta["final_silhouette"],
            "calinski_harabasz": meta["final_calinski_harabasz"],
            "davies_bouldin":    meta["final_davies_bouldin"],
            "inertia":           meta["final_inertia"],
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "silhouette", ascending=False
    ).reset_index(drop=True)

    # Summary lives in the dated folder, so its name needs no datetime.
    base = os.path.join(run_dir, f"{dataset_name}_sweep_summary")
    summary.to_csv(f"{base}.csv", index=False)
    print(f"\n  Saved: {base}.csv")

    print(f"\n{'#'*70}")
    print(f"# SWEEP DONE — folder: {run_dir}")
    print(f"{'#'*70}")
    print(summary.to_string(index=False))

    return summary


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

    K_RANGE = range(2, 7)
    SEED    = 42

    # Parameter grid. Each combination becomes its own run (with a unique
    # parameter+datetime filename and a 4-metric diagnostics plot); the runs
    # Keys left out fall back to defaults. Tip: vary ONE parameter at a time.
    PARAM_GRID = {
        "n_pca_components": [50],
        "n_neighbors":      [5, 15, 30, 50],
        "min_dist":         [0.0, 0.1, 0.25],
        "metric":           ["cosine"],
        # "n_umap_components": [5, 10, 15, 20],  # tune separately, afterwards
        # "random_state":      [0, 1, 2],        # to test cluster stability
    }

    for name, path in DATASETS:
        if not os.path.exists(path):
            print(f"[SKIP] {name} — file not found: {path}")
            continue
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)

        run_parameter_sweep(
            df           = df,
            dataset_name = name,
            output_dir   = OUTPUT,
            param_grid   = PARAM_GRID,
            k_range      = K_RANGE,
            random_state = SEED,
        )