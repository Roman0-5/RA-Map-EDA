import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_value_distribution(
    df: pd.DataFrame,
    name : str = 'dataset',
    output_dir : str = '../reports',
    log_transform: bool = True
) -> None:
    
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] == 0:
        print(f"No numeric columns in {name}")
        return
    
    all_values = num_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if log_transform:
            fig, axes = plt.subplots(1, 2, figsize=(14,5))
            
            axes[0].hist(all_values, bins=100, color='steelblue', alpha=0.7)
            axes[0].set_title('Raw Values')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_yscale('log')
            axes[0].grid(alpha=0.3)
            
            axes[1].hist(np.log2(np.abs(all_values) + 1), bins=100, color='coral', alpha=0.7)
            axes[1].set_title('Log2 transformed values')
            axes[1].set_xlabel('log2(|value| + 1)')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_values, bins=100, color='steelblue', alpha=0.7)
        ax.set_title(f'Value distribution: {name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'{name.upper()}', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}_value_distribution.svg"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_missing_heatmap(df: pd.DataFrame, name: str = "dataset",
                        output_dir: str = "../reports") -> None:
    """Plot missing value heatmap.
    
    Each row = sample, each column = feature.
    White = present, dark = missing.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Boolean Matrix: True wenn missing
    missing = df.isna()
    
    ax.imshow(missing.values, aspect='auto', cmap='Greys', interpolation='none')
    ax.set_xlabel('Features (columns)')
    ax.set_ylabel('Samples (rows)')
    ax.set_title(f'Missing values pattern: {name}\n(dark = missing)')
    
    # Optional: gesamtes Missingness
    total_missing = missing.sum().sum()
    total_cells = df.size
    pct = total_missing / total_cells * 100
    ax.text(0.02, 0.98, f'Total missing: {total_missing} / {total_cells} ({pct:.2f}%)',
           transform=ax.transAxes, va='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}_missing_heatmap.svg"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_variance_distribution(df: pd.DataFrame, name: str = "dataset",
                              output_dir: str = "../reports") -> None:
    """Plot variance distribution across features.
    
    Helps identify low-variance features.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return
    
    variances = num_df.var().sort_values()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram der Varianzen
    axes[0].hist(variances, bins=50, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Variance')
    axes[0].set_ylabel('Number of features')
    axes[0].set_title('Variance distribution')
    axes[0].set_xscale('log')
    axes[0].grid(alpha=0.3)
    
    # Sortierte Varianzen
    axes[1].plot(range(len(variances)), variances.values, color='coral')
    axes[1].set_xlabel('Feature rank (sorted)')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Variance per feature (sorted)')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'{name.upper()} - Variance Analysis', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}_variance.svg"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_outlier_boxplot(df: pd.DataFrame, name: str = "dataset",
                         output_dir: str = "../reports",
                         top_n: int = 20) -> None:
    """Boxplot of top-N features with most outliers.
    
    Visualizes which features have extreme values.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return
    
    # Top-N Features mit höchster Varianz
    top_features = num_df.var().sort_values(ascending=False).head(top_n).index
    
    fig, ax = plt.subplots(figsize=(14, 6))
    num_df[top_features].boxplot(ax=ax)
    ax.set_title(f'{name}: Top {top_n} highest-variance features')
    ax.set_ylabel('Value')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}_boxplot_top{top_n}.svg"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    
def plot_strip_boxplot(df, name="dataset", output_dir="../reports", 
                       top_n=20, log_transform=True):
    """Boxplot with stripplot overlay showing all individual datapoints.
    
    Uses log2 transformation for skewed RFU data so that the IQR-based
    whiskers are statistically meaningful.
    
    Args:
        df: DataFrame with numeric features in columns.
        name: For title and filename.
        output_dir: Where to save the plot.
        top_n: Number of highest-variance features to show.
        log_transform: Apply log2(x+1) before plotting.
    """
    # Nur numerische Features
    num_df = df.select_dtypes(include=[np.number])
    
    if num_df.shape[1] == 0:
        print(f"No numeric columns in {name}")
        return
    
    # Top-N nach Varianz
    top_features = num_df.var().sort_values(ascending=False).head(top_n).index
    data = num_df[top_features].copy()
    
    # Log2-Transformation
    if log_transform:
        data = np.log2(data + 1)
        ylabel = 'log2(RFU + 1)'
    else:
        ylabel = 'RFU'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Boxplot in hellem Grau (Hintergrund-Struktur)
    sns.boxplot(data=data, ax=ax, whis=1.5, 
                color='lightgray', showfliers=False)
    
    # 2. Strip Plot drüber (alle einzelnen Patienten)
    sns.stripplot(data=data, ax=ax, 
                  color='steelblue', alpha=0.5, size=4, 
                  jitter=0.25)
    
    ax.set_title(f'{name}: Top {top_n} high-variance proteins')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Protein (SeqId)')
    
    # X-Labels rotieren damit sie lesbar sind
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{name}_strip_boxplot.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")