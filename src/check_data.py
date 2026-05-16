from IPython.core.display_functions import display
import pandas as pd

def check_data(df):
    return df


def inspect_dataset(df, name="dataset", n_rows=5, dtype_head=20, missing_top=10):
    print(f"\n===== {name.upper()} =====")

    print("\nHEAD:")
    display(df.head(n_rows))

    print("\nSHAPE:")
    print(df.shape)

    print("\nDTYPES:")
    print(df.dtypes)

    print(f"\nMISSING VALUES (top {missing_top}):")
    print(df.isnull().sum().sort_values(ascending=False).head(missing_top))


def inspect_gene_dataset(df, name="gene dataset"):
    inspect_dataset(df, name)

    print("\nDTYPES (sample 20):")
    print(df.dtypes.head(20))

    print("\nQUICK NUMERIC SUMMARY:")
    display(df.describe().T.head(10))


def inspect_protein_dataset(df, name="protein dataset"):
    inspect_dataset(df, name)

    print("\nDTYPES (sample 20):")
    print(df.dtypes.head(20))

    print("\nSUMMARY STATISTICS (first 10 proteins):")
    display(df.describe().T.head(10))

def show_sheet_overview(data):
    print("All dataset keys:")
    print(data.keys())

def show_dataset_structure(data):
    """
    Prints shape overview of all loaded DataFrames.
    """
    print("\n===== DATASET STRUCTURE OVERVIEW =====\n")

    for name, df in data.items():
        print(f"{name}: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n===== END =====")