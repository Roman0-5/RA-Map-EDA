def select_features(df, contract):
    cols = []

    # IDs
    cols += [
        contract["primary_key"],
        *contract["secondary_keys"]
    ]

    # Cohort (only what you want)
    cols += ["RACE", "GENDER"]

    # Binary features
    cols += contract["binary_clinical"]
    cols += contract["binary_behavioral"]

    # Static numeric (exclude height/weight)
    static = [
        c for c in contract["static_numeric"]
        if c not in ["HEIGHT", "WEIGHT"]
    ]
    cols += static

    # Baseline features
    cols += contract["baseline_numeric"]

    # Engineered features (keep for now)
    cols += contract["engineered_features"]["steroids"]
    cols += contract["engineered_features"]["meds"]

    # explicitly remove date columns (important)
    date_cols = [
        "first_injection",
        "last_injection",
        "first_med_date",
        "last_med_date"
    ]
    cols = [c for c in cols if c not in date_cols]

    # keep only existing columns
    cols = [c for c in cols if c in df.columns]

    return df[cols]