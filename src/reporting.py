def data_quality_report(df):
    """Function to make a Dataqualityreport with df as Input"""
    
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
    
    # 2. Duplikate
    print("How many duplicates?")
    print(f"Duplicates: {df.duplicated().sum()}")
    print()
    
    # 3. Datentypen-Überblick
    for dtype in df.dtypes.unique():
        if str(dtype) == 'object':
            rows = df.select_dtypes(include=['object', 'str']).columns.tolist()
        else:
            rows = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"{dtype}: {len(rows)} columns")
        print(f"{rows[:5]}{'...' if len(rows) > 5 else ''}")
    print()
    
    # 4. Numerical Values special cases and distribution
    print("Numerical Values")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        special = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"{col}:")
        print(f"  Min: {df[col].min()} | Max: {df[col].max()} | "
              f"Mean: {df[col].mean():.2f} | Special values: {special}")
    print()
    
    # 5. String-Spalten: Konsistenz
    print("Strings")
    str_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in str_cols:
        unique = df[col].nunique()
        print(f"{col}:")
        print(f"Unique Strings: {unique}")
        
        # Leerzeichen-Probleme
        if df[col].dropna().str.startswith(' ').any() or df[col].dropna().str.endswith(' ').any():
            print(f"Enthält führende/nachfolgende Leerzeichen!")
        
        # Groß-/Kleinschreibung inkonsistent?
        if unique != df[col].str.lower().nunique():
            print(f"Inkonsistente Groß-/Kleinschreibung!")
        
        # Bei wenigen einzigartigen Werten: zeig sie
        if unique <= 15:
            print(f"  Werte: {df[col].value_counts().to_dict()}")
    print()

os.makedirs('../reports', exist_ok=True)
with open('../reports/data_quality_report.txt', 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        data_quality_report(df_clinical)
print("saved to /reports/quality_report.txt")