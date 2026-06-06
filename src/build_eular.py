"""
build_eular_labels.py
---------------------
Berechnet EULAR Response Labels aus DAS28-Werten (Baseline + 6 Monate)
und exportiert das Ergebnis als CSV und Parquet.

EULAR Response Kriterien (ACR/EULAR 2010):
  - Good:     DAS28 @ 6M <= 3.2  UND  ΔDAS28 > 1.2
  - Moderate: (DAS28 @ 6M <= 3.2 UND ΔDAS28 0.6–1.2)
              ODER (DAS28 @ 6M > 3.2 UND <= 5.1  UND  ΔDAS28 > 0.6)
  - None:     alle anderen (inkl. Verschlechterung)

Quelle: https://das-score.nl/en/das-and-das28/das28-why/eular-response-criteria
"""

import pandas as pd
from pathlib import Path

INPUT_FILE  = Path("../cleaned_datasets/clinical_merged.parquet")
OUT_CSV     = Path("../cleaned_datasets/eular_labels.csv")
OUT_PARQUET = Path("../cleaned_datasets/eular_labels.parquet")


def compute_eular_response(das_bl: pd.Series, das_m6: pd.Series) -> pd.Series:
    """
    Gibt eine kategorische Serie zurück: 'Good', 'Moderate', 'No response'.
    Zeilen mit fehlendem DAS28-Wert werden als NaN belassen.
    """
    delta = das_bl - das_m6  # positiv = Verbesserung

    good = (das_m6 <= 3.2) & (delta > 1.2)

    moderate = (
        ((das_m6 <= 3.2) & (delta >= 0.6) & (delta <= 1.2))
        | ((das_m6 > 3.2) & (das_m6 <= 5.1) & (delta > 0.6))
    )

    labels = pd.Series("No response", index=das_bl.index, dtype="object")
    labels[moderate] = "Moderate"
    labels[good]     = "Good"

    # Zeilen mit fehlendem DAS28 auf NaN setzen
    missing = das_bl.isna() | das_m6.isna()
    labels[missing] = pd.NA

    return labels.astype("category")


def main():
    print("Lade klinische Daten ...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df)} Patienten, {df.shape[1]} Spalten geladen")

    # EULAR Label berechnen
    df["EULAR_Response"] = compute_eular_response(df["DAS28.0M"], df["DAS28.6M"])
    df["DAS28_Delta"]    = (df["DAS28.0M"] - df["DAS28.6M"]).round(3)

    # Nur relevante Spalten behalten
    out = df[[
        "Patient_ID",
        "DAS28.0M",
        "DAS28.6M",
        "DAS28_Delta",
        "EULAR_Response",
    ]].copy()

    # Zeilen ohne Label entfernen
    n_before = len(out)
    out = out.dropna(subset=["EULAR_Response"])
    n_dropped = n_before - len(out)
    print(f"\n  {n_dropped} Patienten ohne DAS28-Wert entfernt (fehlende BL oder M6)")
    print(f"  {len(out)} Patienten mit gültigem Label\n")

    # Verteilung ausgeben
    print("EULAR Response Verteilung:")
    counts = out["EULAR_Response"].value_counts()
    for label, n in counts.items():
        pct = 100 * n / len(out)
        print(f"  {label:<15} {n:>3}  ({pct:.1f}%)")

    # DAS28-Statistik
    print(f"\nDAS28 @ Baseline:  mean={out['DAS28.0M'].mean():.2f}  "
          f"std={out['DAS28.0M'].std():.2f}")
    print(f"DAS28 @ 6 Monate:  mean={out['DAS28.6M'].mean():.2f}  "
          f"std={out['DAS28.6M'].std():.2f}")
    print(f"Delta DAS28:       mean={out['DAS28_Delta'].mean():.2f}  "
          f"std={out['DAS28_Delta'].std():.2f}")

    # Export
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_CSV, index=False)
    print(f"\nCSV gespeichert:     {OUT_CSV}")

    out_parquet = out.copy()
    out_parquet["EULAR_Response"] = out_parquet["EULAR_Response"].astype(str)
    out_parquet.to_parquet(OUT_PARQUET, index=False)
    print(f"Parquet gespeichert: {OUT_PARQUET}")

    print("\nFertig.")


if __name__ == "__main__":
    main()