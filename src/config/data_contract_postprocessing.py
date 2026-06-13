"""
Data Contract for RA dataset preprocessing pipeline.
This defines column roles across: - Clinical baseline data - Steroid administration events - Medication administration events
"""

CLINICAL_CONTRACT_POST = {

    "id_columns": ["Patient_ID", "Digest"],

    # -------------------------
    # PURE FEATURES (X)
    # -------------------------
    "binary_columns": [
        "ACPA.POSITIVE",
        "RHUEMATOID.FACTOR",
        "ALCOHOL_Y_N",
        "CURENT SMOKER",
        "Erosive",
        "GENDER",
    ],

    # -------------------------
    # OUTCOMES (y)
    # -------------------------
    "outcome_binary": [
        "remission_event",
        "Remission(<2.6DAS)",
        "HighDisease(>4DAS)"
    ],

    "outcome_continuous": [
        "remission_time"
    ],

    # -------------------------
    # LONGITUDINAL FEATURES
    # -------------------------
    "longitudinal_numeric": [
        "DAS28.0M", "DAS28.3M", "DAS28.6M",
        "DAS28.9M", "DAS28.12M", "DAS28.18M",

        "HAQ.0M", "HAQ.6M",
        "SDAI.0M", "SDAI.6M", "SDAI.12M",

        "CRP.0M", "CRP.6M", "CRP.9M",
        "ESR.0M",

        "BASOPHILS.0M", "EOSINOPHILS.0M", "HB.0M",
        "LYMPHOCYTES.0M", "MONOCYTES.0M",
        "NEUTROPHILS.0M", "PLT.0M", "WBC.0M",

        "BASOPHILS.6M", "EOSINOPHILS.6M", "HB.6M",
        "LYMPHOCYTES.6M", "MONOCYTES.6M",
        "NEUTROPHILS.6M", "PLT.6M", "WBC.6M",

        "FATIQUE.0M", "FATIQUE.6M",
        "PAIN.0M", "PAIN.6M",

        "TOTAL.SWOLLEN.0M", "TOTAL.TENDER.0M",
        "TOTAL.SWOLLEN.6M", "TOTAL.TENDER.6M",
        "TOTAL.SWOLLEN.9M", "TOTAL.TENDER.9M"
    ],

    # -------------------------
    # STATIC FEATURES
    # -------------------------
    "static_numeric": [
        "AGE",
        "HEIGHT",
        "WEIGHT",
        "Symp_Duration",
        "InitialxRAYScore",
        "FinalxRAYScore"
    ]
}

STEROID_FEATURES_CONTRACT_POST = {
    "id_columns": ["Patient_ID", "Digest"],

    # -------------------------
    # AGGREGATED STEROID FEATURES
    # -------------------------
    "numeric_columns": [
        "steroid_injection_count",
        "total_dose_x",
        "mean_dose_x",
        "max_dose",
        "intraarticular_count",
        "intramuscular_count"
    ],

    # -------------------------
    # TEMPORAL FEATURES (DATES → should NOT be normalized)
    # -------------------------
    "datetime_columns": [
        "first_injection",
        "last_injection"
    ]
}

MEDS_FEATURES_CONTRACT_POST = {
    "id_columns": ["Patient_ID", "Digest"],

    # -------------------------
    # AGGREGATED MEDICATION FEATURES
    # -------------------------
    "numeric_columns": [
        "med_event_count",
        "unique_medications",
        "total_dose_y",
        "mean_dose_y"
    ],

    # -------------------------
    # TEMPORAL FEATURES
    # -------------------------
    "datetime_columns": [
        "first_med_date",
        "last_med_date"
    ]
}

# OPTIONAL: helpers
ALL_CONTRACTS = {
    "clinical": CLINICAL_CONTRACT_POST,
    "steroids": STEROID_FEATURES_CONTRACT_POST,
    "meds": MEDS_FEATURES_CONTRACT_POST
}