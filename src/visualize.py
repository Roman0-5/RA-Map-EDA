"""Visualizing baseline vs 6-month protein profiles via PCA."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_timepoints():
    """PCA visualization of TAC patients at Baseline vs 6 months.
    
    Creates three plots:
    1. PC1 vs PC2 scatter colored by timepoint
    2. PC1 vs PC3 scatter colored by timepoint
    3. Trajectory plot showing patient movement from BL to 6M
    """
    # 1. Beide Zeitpunkte laden
    df_bl = pd.read_parquet('../mid_processing_datasets/expression_matrix_bl.parquet')
    df_6m = pd.read_parquet('../mid_processing_datasets/expression_matrix_6m.parquet')
    
    # 2. Nur TAC-Patienten (VAC ausfiltern)
    df_bl = df_bl[df_bl['Patient_ID'].str.startswith('TAC')].copy()
    df_6m = df_6m[df_6m['Patient_ID'].str.startswith('TAC')].copy()
    
    # 3. Timepoint-Label hinzufügen
    df_bl['TimePoint'] = 'Baseline'
    df_6m['TimePoint'] = '6month'
    
    # 4. Kombinieren
    df_combined = pd.concat([df_bl, df_6m], axis=0)
    print(f"Combined shape: {df_combined.shape}")
    print(f"  Baseline: {(df_combined['TimePoint'] == 'Baseline').sum()}")
    print(f"  6month:   {(df_combined['TimePoint'] == '6month').sum()}")
    
    # 5. Index für späteres Tracking zurücksetzen (wichtig!)
    df_combined = df_combined.reset_index(drop=True)
    
    # 6. Metadaten und Numerik trennen
    meta = df_combined[['Patient_ID', 'TimePoint']].copy()
    X = df_combined.drop(columns=['Patient_ID', 'TimePoint'])
    
    # 7. Preprocessing
    X_log = np.log2(X + 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # 8. PCA
    pca = PCA(n_components=5)
    pcs = pca.fit_transform(X_scaled)
    
    print(f"\nExplained variance:")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ev:.1%}")
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Cumulative (PC1-5): {cum[-1]:.1%}")
    
    # 9. Plot 1+2: Scatter PC1vsPC2 und PC1vsPC3
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Baseline': '#1f77b4', '6month': '#ff7f0e'}
    timepoints = meta['TimePoint'].values
    
    # PC1 vs PC2
    for tp in ['Baseline', '6month']:
        mask = (timepoints == tp)
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1], 
                       c=colors[tp], label=tp, alpha=0.6, s=50)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PC1 vs PC2 (TAC patients only)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PC1 vs PC3
    for tp in ['Baseline', '6month']:
        mask = (timepoints == tp)
        axes[1].scatter(pcs[mask, 0], pcs[mask, 2], 
                       c=colors[tp], label=tp, alpha=0.6, s=50)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    axes[1].set_title('PC1 vs PC3 (TAC patients only)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/pca_timepoints_scatter.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # 10. Plot 3: Trajectory Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Patienten finden die BEIDE Zeitpunkte haben
    bl_patients = set(meta[meta['TimePoint'] == 'Baseline']['Patient_ID'])
    sm_patients = set(meta[meta['TimePoint'] == '6month']['Patient_ID'])
    both = bl_patients & sm_patients
    print(f"\nPatients with both timepoints: {len(both)}")
    
    # Für jeden Patienten: Linie von BL zu 6M
    for patient_id in both:
        # Indizes in meta finden
        bl_pos = meta[(meta['Patient_ID'] == patient_id) & 
                     (meta['TimePoint'] == 'Baseline')].index[0]
        sm_pos = meta[(meta['Patient_ID'] == patient_id) & 
                     (meta['TimePoint'] == '6month')].index[0]
        
        # Linie zwischen den zwei Punkten zeichnen
        ax.plot([pcs[bl_pos, 0], pcs[sm_pos, 0]], 
                [pcs[bl_pos, 1], pcs[sm_pos, 1]],
                'gray', alpha=0.4, linewidth=0.5, zorder=1)
    
    # Punkte farbig drüber zeichnen
    for tp in ['Baseline', '6month']:
        mask = (timepoints == tp)
        ax.scatter(pcs[mask, 0], pcs[mask, 1], 
                  c=colors[tp], label=tp, alpha=0.7, s=60, zorder=2,
                  edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'Protein Profile Trajectory: Baseline → 6 months\n'
                 f'({len(both)} patients with both timepoints)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/pca_trajectory.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return pcs, meta, pca