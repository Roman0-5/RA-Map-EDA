"""
Updated RA Data Contract
Fully aligned with original clinical + event datasets.
Exhaustive feature coverage, structured by source + timepoint.
"""

# =========================
# CLINICAL DATA CONTRACT
# =========================

CLINICAL_CONTRACT = {
    # ---------------------
    # IDENTIFIERS
    # ---------------------
    "primary_key": "Digest",
    "secondary_keys": ["Patient_ID"],

    # ---------------------
    # COHORT / META FEATURES
    # ---------------------
    "cohort_columns": [
        "Study",
        "Region",
        "Hub",
        "REGION_HUB",
        "RACE",
        "GENDER"
    ],

    # ---------------------
    # BINARY FEATURES
    # ---------------------
    "binary_clinical": [
        "ACPA.POSITIVE",
        "RHUEMATOID.FACTOR",
        "Erosive",
        "IM.STEROIDS.3MONTHS",
        "ORAL.STEROIDS.3M"
    ],

    "binary_behavioral": [
        "ALCOHOL_Y_N",
        "CURENT SMOKER"
    ],

    # ---------------------
    # STATIC NUMERIC
    # ---------------------
    "static_numeric": [
        "AGE",
        "HEIGHT",
        "WEIGHT",
        "Symp_Duration",
        "InitialxRAYScore",
        "FinalxRAYScore"
    ],

    # ---------------------
    # BASELINE (0M)
    # ---------------------
    "baseline_numeric": [
        "DAS28.0M",
        "HAQ.0M",
        "SDAI.0M",
        "CRP.0M",
        "ESR.0M",
        "BASOPHILS.0M",
        "EOSINOPHILS.0M",
        "HB.0M",
        "LYMPHOCYTES.0M",
        "MONOCYTES.0M",
        "NEUTROPHILS.0M",
        "PLT.0M",
        "WBC.0M",
        "FATIQUE.0M",
        "PAIN.0M",
        "TOTAL.SWOLLEN.0M",
        "TOTAL.TENDER.0M"
    ],

    # ---------------------
    # 3 MONTHS
    # ---------------------
    "month_3_numeric": [
        "DAS28.3M"
    ],

    # ---------------------
    # 6 MONTHS
    # ---------------------
    "month_6_numeric": [
        "DAS28.6M",
        "HAQ.6M",
        "SDAI.6M",
        "CRP.6M",
        "FATIQUE.6M",
        "PAIN.6M",
        "TOTAL.SWOLLEN.6M",
        "TOTAL.TENDER.6M",
        "BASOPHILS.6M",
        "EOSINOPHILS.6M",
        "HB.6M",
        "LYMPHOCYTES.6M",
        "MONOCYTES.6M",
        "NEUTROPHILS.6M",
        "PLT.6M",
        "WBC.6M"
    ],

    # ---------------------
    # 9 MONTHS
    # ---------------------
    "month_9_numeric": [
        "DAS28.9M",
        "CRP.9M",
        "TOTAL.SWOLLEN.9M",
        "TOTAL.TENDER.9M"
    ],

    # ---------------------
    # 12 MONTHS
    # ---------------------
    "month_12_numeric": [
        "DAS28.12M",
        "SDAI.12M"
    ],

    # ---------------------
    # 18 MONTHS
    # ---------------------
    "month_18_numeric": [
        "DAS28.18M"
    ],

    # ---------------------
    # LABELS / OUTCOMES
    # ---------------------
    "labels": [
        "Remission month",
        "Remission(<2.6DAS)",
        "HighDisease(>4DAS)",
    ],

    # ---------------------
    # ENGINEERED FEATURES
    # ---------------------
    "engineered_features": {
        "steroids": [
            "steroid_injection_count",
            "total_dose_x",
            "mean_dose_x",
            "max_dose",
            "intraarticular_count",
            "intramuscular_count",
            "first_injection",
            "last_injection"
        ],
        "meds": [
            "med_event_count",
            "unique_medications",
            "total_dose_y",
            "mean_dose_y",
            "first_med_date",
            "last_med_date"
        ]
    }
}


# =========================
# STEROID EVENT CONTRACT
# =========================

STEROIDS_CONTRACT = {
    "primary_key": "Digest",

    "datetime_columns": [
        "Date of Assessment",
        "Date Given"
    ],

    "categorical_columns": [
        "Steroid",
        "Route",
        "Joint Injected",
        "Unit"
    ],

    "numeric_columns": [
        "Dose",
        "Assessment"
    ],

    "binary_columns": [
        "4. Has the patient received a steroid injection?"
    ]
}


# =========================
# MEDICATION EVENT CONTRACT
# =========================

MEDS_CONTRACT = {
    "primary_key": "Digest",

    "datetime_columns": [
        "Date of Assessment",
        "Date Started"
    ],

    "categorical_columns": [
        "RA Medication",
        "Frequency",
        "Route"
    ],

    "coded_categorical_columns": [
        "Assessment",
        "Unit"
    ],

    "numeric_columns": [
        "Dose"
    ]
}


# =========================
# CONTRACT REGISTRY
# =========================

ALL_CONTRACTS = {
    "clinical": CLINICAL_CONTRACT,
    "steroids": STEROIDS_CONTRACT,
    "meds": MEDS_CONTRACT
}