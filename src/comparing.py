"""
compare_rds_excel.py
--------------------
Vergleicht protogen_mat.rds und Protogen_RA_MAP_16_05_21.xlsx
auf Shape-Übereinstimmung und Werte-Identität.

Da die Sample-Namen unterschiedlich sind (Anonymisierung),
werden die Werte spaltenunabhängig verglichen:
- Sortierte Werteverteilungen
- Statistische Kennzahlen
- Antigen-weise Verteilungen (über Zeilenindex)
"""

import numpy as np
import pandas as pd
import pyreadr

def compare():
    RDS_FILE   = "../datasets/protogen_mat.rds"
    EXCEL_FILE = "../datasets/Protogen_RA_MAP_16_05_21.xlsx"

    META_COLS  = ["ProteinID", "GeneID", "Gene Symbol", "Gene Name"]

    # ── Laden ────────────────────────────────────────────────────────────────────
    print("Lade Dateien ...")
    rds   = pyreadr.read_r(RDS_FILE)[None]
    excel = pd.read_excel(EXCEL_FILE, sheet_name=0)

    excel_vals = excel[[c for c in excel.columns if c not in META_COLS]]

    print(f"  RDS   geladen: {rds.shape}")
    print(f"  Excel geladen: {excel_vals.shape}")
    print()

    # ── 1. Shape ─────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("1. SHAPE VERGLEICH")
    print("=" * 50)
    shape_match = rds.shape == excel_vals.shape
    print(f"  RDS Shape:   {rds.shape}  (Antigene × Samples)")
    print(f"  Excel Shape: {excel_vals.shape}  (Antigene × Samples)")
    print(f"  Übereinstimmung: {'✓ JA' if shape_match else '✗ NEIN'}")
    print()

    # ── 2. Globale Statistiken ───────────────────────────────────────────────────
    print("=" * 50)
    print("2. GLOBALE WERTE-STATISTIK")
    print("=" * 50)

    rds_flat   = rds.values.flatten().astype(float)
    excel_flat = excel_vals.values.flatten().astype(float)

    stats = {
        "Min":      (np.nanmin,    rds_flat, excel_flat),
        "Max":      (np.nanmax,    rds_flat, excel_flat),
        "Median":   (np.nanmedian, rds_flat, excel_flat),
        "Mean":     (np.nanmean,   rds_flat, excel_flat),
        "Std":      (np.nanstd,    rds_flat, excel_flat),
    }

    print(f"  {'Kennzahl':<12} {'RDS':>12} {'Excel':>12} {'Gleich':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for name, (fn, r, e) in stats.items():
        rv, ev = fn(r), fn(e)
        match  = "✓" if np.isclose(rv, ev, rtol=1e-3) else "✗"
        print(f"  {name:<12} {rv:>12.2f} {ev:>12.2f} {match:>8}")
    print()

    # ── 3. Quantile-Vergleich ────────────────────────────────────────────────────
    print("=" * 50)
    print("3. QUANTILE-VERGLEICH (sortierte Verteilung)")
    print("=" * 50)
    quantiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    rds_q   = np.nanpercentile(rds_flat, quantiles)
    excel_q = np.nanpercentile(excel_flat, quantiles)

    print(f"  {'Perzentil':>10} {'RDS':>12} {'Excel':>12} {'Gleich':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    all_match = True
    for q, rv, ev in zip(quantiles, rds_q, excel_q):
        match = np.isclose(rv, ev, rtol=1e-3)
        if not match:
            all_match = False
        print(f"  {q:>9}% {rv:>12.1f} {ev:>12.1f} {'✓' if match else '✗':>8}")
    print(f"\n  Alle Quantile identisch: {'✓ JA' if all_match else '✗ NEIN'}")
    print()

    # ── 4. Antigen-weise Zeilenstatistik ────────────────────────────────────────
    print("=" * 50)
    print("4. ANTIGEN-WEISE VERGLEICH (Zeile für Zeile)")
    print("=" * 50)
    print("  (Annahme: Antigene stehen in derselben Reihenfolge)")

    rds_row_med   = np.nanmedian(rds.values.astype(float),   axis=1)
    excel_row_med = np.nanmedian(excel_vals.values.astype(float), axis=1)

    diffs    = np.abs(rds_row_med - excel_row_med)
    matching = np.isclose(rds_row_med, excel_row_med, rtol=1e-3)

    print(f"  Antigene mit identischem Zeilen-Median: {matching.sum()} / {len(matching)}")
    print(f"  Max. Abweichung: {diffs.max():.4f}")
    print(f"  Mittlere Abweichung: {diffs.mean():.4f}")

    if not matching.all():
        print("\n  Abweichende Antigene (Zeilenindex):")
        for i in np.where(~matching)[0]:
            print(f"    Zeile {i}: RDS={rds_row_med[i]:.2f}  Excel={excel_row_med[i]:.2f}")
    print()

    # ── 5. Fehlende Werte ────────────────────────────────────────────────────────
    print("=" * 50)
    print("5. FEHLENDE WERTE")
    print("=" * 50)
    print(f"  RDS   NaN: {np.isnan(rds_flat).sum()}")
    print(f"  Excel NaN: {np.isnan(excel_flat).sum()}")
    print()

    # ── Gesamturteil ─────────────────────────────────────────────────────────────
    print("=" * 50)
    print("GESAMTURTEIL")
    print("=" * 50)
    if shape_match and all_match and matching.all():
        print("  ✓ RDS und Excel sind inhaltlich IDENTISCH.")
        print("  ✓ Ihr könnt sicher mit beiden arbeiten.")
    else:
        print("  ✗ Es gibt Unterschiede — Details oben prüfen.")
        
if __name__ == '__main__':
    compare()