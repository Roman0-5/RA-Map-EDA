import pandas as pd
import numpy as np
from src.extraction import extract_dtypes
from src.normalization import normalize_all

def data_quality_report(df):
    """Function to make a Dataqualityreport with df as Input"""
    #initializing objects from other function
    df = normalize_all(df)
    df, only_num, only_str, only_obj = extract_dtypes(df)
    print("Dataquality report")
    print()
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Total of {len(df) * len(df.columns)} data")

    
    print("")
    # 1. Missing values per Row
    print("Missing Values")
    missing = df.isna().sum() # total sum of missing values
    missing_pct = (missing / len(df) * 100).round(2) 
    missing_report = pd.DataFrame({ # new dataframe for missing values
        'Missing': missing,
        'Percentage': missing_pct
    })
    # Only show rows with missing values
    problems = missing_report[missing_report['Missing'] > 0]
    if len(problems) > 0:
        print(problems.sort_values('Missing', ascending=False))
    else:
        print("No missing values were found!")
    print()
    
    # 2. duplicates
    print("How many duplicates?")
    print(f"Duplicates: {df.duplicated().sum()}")
    print()
    
    # 3. dtype overview 
    for name, subset in [('numeric', only_num), ('string', only_str), ('object', only_obj)]:
        cols = subset.columns.tolist()
        print(f"{name}: {len(cols)} columns")
        print(f"{cols[:5]}{'...' if len(cols) > 5 else ''}")
        print()
    
    # 4. Numerical Values special cases and distribution
    print("Numerical Values")
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe()
    print(stats)
    print()

    for col in num_df.columns:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]

        lower = mean - (3 * std)
        upper = mean + (3 * std)
        outliers = ((num_df[col] < lower) | (num_df[col] > upper)).sum()

        print(f"{col}")
        print(f" Min: {stats.loc['min', col]:.2f} |"
              f" Max: {stats.loc['max', col]:.2f} |"
              f" Mean: {mean:.2f} |"
              f" Std: {std:.2f} |"
              f" Outliers: {outliers}")
    
    # 5. Consistency of string rows
    print("Strings")
    str_df = pd.concat([only_str, only_obj], axis=1)

    # all stats at once
    summary = str_df.agg([
        'nunique',
        'count',
        lambda x: x.dropna().str.startswith(' ').any() or x.dropna().str.endswith(' ').any(),
        lambda x: x.nunique() != x.dropna().str.lower().nunique()
    ])
    summary.index = ['Unique', 'Count', 'Has_Whitespace', 'Mixed_Case']
    print(summary.T)  # transposed for readability