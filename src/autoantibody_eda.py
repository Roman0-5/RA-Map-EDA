"""
autoantibody_eda_plots.py  (angepasst an die bioplotr / GitHub-Originale)
-------------------------------------------------------------------------
Aenderungen ggü. der Ursprungsfassung (jeweils mit # >>> markiert):
  - Palette: Baseline = #00BFC4 (ggplot 2-Farben-Default) statt #619CFF
  - Mean-Variance: Achsen mu / log2(sigma), LOWESS frac 0.3 -> 0.5
  - Similarity: nur oberes Dendrogramm + Farbstreifen, Colorbar rechts
  - PCA: zusaetzlich PC1 gespiegelt (Orientierung wie R)
  - Drivers: PCA auf NA-bereinigtem Subset NEU gerechnet (wie R na.omit),
             Tests: Spearman (numerisch) / Kruskal-Wallis (kategorial),
             Colormap Reds statt YlOrRd
  - Legenden ausserhalb rechts, theme_bw-artiges Grid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.stats import pearsonr, f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

def autoantibody():
    PROTOGEN_FILE = "../datasets/protogen_log_cpm.csv"
    CLINICAL_FILE = "../datasets/protogen_clin.csv"
    OUTPUT_DIR    = "../reports/replicating/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "0.3",
        "axes.linewidth":   0.8,
        "axes.grid":        True,
        "grid.color":       "0.90",
        "grid.linewidth":   0.8,
        "font.size":        11,
    })

    df_protogen = pd.read_csv(PROTOGEN_FILE, index_col=0)
    df_clinical = pd.read_csv(CLINICAL_FILE)

    if "s" in df_clinical.columns:
        df_clinical = df_clinical.set_index("s")

    common      = [s for s in df_clinical.index if s in df_protogen.columns]
    df_protogen = df_protogen[common]
    df_clinical = df_clinical.loc[common]

    print(f"Matrix:     {df_protogen.shape}  (Antigene x Samples)")
    print(f"Klinik:     {df_clinical.shape}")
    print(f"Timepoints: {df_clinical['TIME'].value_counts().to_dict()}")
    # >>> kurze Kontrolle des ALCOHOL-Typs (siehe Caveat im Chat)
    print(f"ALCOHOL dtype: {df_clinical['ALCOHOL'].dtype} | "
        f"Beispiele: {df_clinical['ALCOHOL'].dropna().unique()[:6]}")

    # >>> Farben EXAKT wie ggplot2-2-Farben-Default (bioplotr) ----------------------
    #     vorher faelschlich {"Baseline": "#619CFF"} (das ist das 3-Farben-Blau)
    palette = {"Baseline": "#00BFC4", "6-month": "#F8766D"}
    markers = {"Baseline": "^", "6-month": "o"}

    # Legenden-Helfer: ausserhalb rechts platzieren (ggplot-Stil)
    def _legend_right(ax, title, **kw):
        ax.legend(title=title, loc="center left",
                bbox_to_anchor=(1.02, 0.5), frameon=False, **kw)

    # == 1. MEAN-VARIANCE PLOT =====================================================
    print("\n[1/5] Mean-Variance Plot ...")

    antigen_means = df_protogen.mean(axis=1)
    antigen_sds   = df_protogen.std(axis=1)
    log2_sd       = np.log2(antigen_sds)

    # >>> frac 0.3 -> 0.5: glattere, monotone Kurve wie im R-Original
    lw = lowess(log2_sd, antigen_means, frac=0.5, return_sorted=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(antigen_means, log2_sd, s=9, color="black", edgecolors="none")
    ax.plot(lw[:, 0], lw[:, 1], color="#3366FF", linewidth=1.5, label="LOWESS")
    ax.set_xlabel(r"$\mu$", fontsize=12)                  # >>> war "u"
    ax.set_ylabel(r"$\log_2(\sigma)$", fontsize=12)       # >>> war "log2(o)"
    ax.set_title("Mean-Variance Plot", fontsize=13)
    _legend_right(ax, "Curve")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}01_mean_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 01_mean_variance.png gespeichert")

    # == 2. DENSITY PLOT ===========================================================
    print("[2/5] Density Plot ...")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sample in df_protogen.columns:
        time  = df_clinical.loc[sample, "TIME"]
        color = palette.get(time, "#888888")
        vals  = df_protogen[sample].dropna()
        sns.kdeplot(vals, ax=ax, color=color, alpha=0.6, linewidth=0.7,
                    fill=False, bw_adjust=0.8)

    ax.set_xlim(4, 19)            # >>> 18 -> 19 (R laeuft bis ~19)
    ax.set_ylim(bottom=0)
    patches = [mpatches.Patch(color=palette[t], label=t)
            for t in ["6-month", "Baseline"]]
    ax.legend(handles=patches, title="Time", loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Density Plot", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}02_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 02_density.png gespeichert")

    # == 3. SUBJECT SIMILARITY MATRIX ==============================================
    print("[3/5] Subject Similarity Matrix ...")

    dist_array  = pdist(df_protogen.T, metric="euclidean")
    dist_df     = pd.DataFrame(
        squareform(dist_array),
        index=df_protogen.columns,
        columns=df_protogen.columns
    )
    row_linkage = linkage(dist_array, method="average")
    time_colors = df_clinical["TIME"].map(palette).rename("Time")

    g = sns.clustermap(
        dist_df,
        row_linkage=row_linkage,
        col_linkage=row_linkage,
        col_colors=time_colors,          # >>> nur oben (wie R) - keine row_colors
        cmap="RdBu",
        figsize=(10, 10),
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        vmin=0,
        vmax=60,
        dendrogram_ratio=(0.02, 0.15),   # >>> linkes Dendrogramm fast auf 0
        cbar_pos=(0.95, 0.35, 0.02, 0.25),  # >>> Colorbar rechts statt links
        cbar_kws={"label": "Euclidean Distance"},
    )
    g.ax_row_dendrogram.set_visible(False)   # >>> linkes Dendrogramm ausblenden
    g.fig.suptitle("Sample Similarity Matrix", fontsize=14,
                fontweight="bold", y=1.00)
    legend_patches = [mpatches.Patch(color=palette[t], label=t)
                    for t in ["6-month", "Baseline"]]
    g.fig.legend(handles=legend_patches, title="Time",
                loc="upper right", bbox_to_anchor=(0.99, 0.92), frameon=True)
    plt.savefig(f"{OUTPUT_DIR}03_similarity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 03_similarity_matrix.png gespeichert")

    # == 4. PCA ====================================================================
    print("[4/5] PCA ...")

    X       = df_protogen.T.values
    pca     = PCA(n_components=10)
    scores  = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    # >>> Orientierung an R angleichen: PC1 UND PC2 spiegeln
    scores[:, 0] *= -1     # >>> NEU: PC1 war gegenueber R seitenverkehrt
    scores[:, 1] *= -1     #     PC2 wie zuvor

    fig, ax = plt.subplots(figsize=(7, 6))
    for time in ["6-month", "Baseline"]:
        idx = [i for i, s in enumerate(common)
            if df_clinical.loc[s, "TIME"] == time]
        ax.scatter(scores[idx, 0], scores[idx, 1],
                color=palette[time], label=time,
                marker=markers[time], alpha=0.8, s=30, edgecolors="none")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.2f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.2f}%)", fontsize=12)
    ax.set_title("PCA", fontsize=13)
    _legend_right(ax, "Time")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}04_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 04_pca.png gespeichert")

    # == 5. DRIVERS PLOT ===========================================================
    print("[5/5] Drivers Plot ...")

    driver_cols = ["TIME", "SEX", "DAS28", "BMI", "ALCOHOL", "AGE"]

    # >>> 1:1 wie R: erst gemeinsam NA-bereinigen, DANN PCA neu rechnen
    clin_drv = df_clinical[driver_cols].copy()
    # ALCOHOL ist kategorial ("Yes"/"No") -> bleibt Text und wird via assoc_p()
    # als Faktor (Kruskal-Wallis) behandelt, genau wie in bioplotr.
    clin_drv = clin_drv.dropna()
    samples_drv = clin_drv.index.tolist()
    print(f"   Drivers-Subset nach na.omit: {len(samples_drv)} Samples")

    X_drv       = df_protogen[samples_drv].T.values
    pca_drv     = PCA(n_components=10).fit(X_drv)
    scores_drv  = pca_drv.transform(X_drv)
    var_exp_drv = pca_drv.explained_variance_ratio_ * 100

    n_pcs     = min(5, scores_drv.shape[1])
    pc_labels = [f"PC{i+1}\n({var_exp_drv[i]:.2f}%)" for i in range(n_pcs)]


    def assoc_p(pc_vec, feat):
        """Pearson fuer numerische, einfaktorielle ANOVA (F-Test) fuer kategoriale
        Features - reproduziert die parametrischen Tests aus bioplotr::plot_drivers
        (lm(PC ~ feature))."""
        if pd.api.types.is_numeric_dtype(feat):
            _, p = pearsonr(pc_vec, feat.values)
            return p
        groups = [pc_vec[feat.values == lvl] for lvl in pd.unique(feat)]
        groups = [grp for grp in groups if len(grp) > 1]
        if len(groups) < 2:
            return 1.0
        return f_oneway(*groups).pvalue


    pvals = np.ones((len(driver_cols), n_pcs))
    for j in range(n_pcs):
        for i, col in enumerate(driver_cols):
            pvals[i, j] = assoc_p(scores_drv[:, j], clin_drv[col])

    _, flat_qvals, _, _ = multipletests(pvals.flatten(), method="fdr_bh")
    qvals    = flat_qvals.reshape(pvals.shape)
    log_q    = -np.log10(np.clip(qvals, 1e-10, 1))
    log_q_df = pd.DataFrame(log_q, index=driver_cols, columns=pc_labels)
    sig_mask = qvals < 0.05

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        log_q_df, ax=ax,
        cmap="Reds",                 # >>> war YlOrRd
        annot=False,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "-log(q)"},
        vmin=0,
    )
    ax.grid(False)
    for i in range(len(driver_cols)):
        for j in range(n_pcs):
            if sig_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="black", linewidth=2.5
                ))

    ax.set_yticklabels(driver_cols, rotation=0)
    ax.set_title("Variation By Feature", fontsize=13)
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}05_drivers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 05_drivers.png gespeichert")

    print(f"\nAlle Plots gespeichert in: {OUTPUT_DIR}")