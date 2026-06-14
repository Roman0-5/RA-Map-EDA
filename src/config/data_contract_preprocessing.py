"""
Data Contract for RA dataset preprocessing pipeline.
This defines column roles across: - Clinical baseline data - Steroid administration events - Medication administration events
"""
CLINICAL_CONTRACT = {
    "id_columns": ["Patient_ID", "Digest"],

    "cohort_columns": [
        "Study",
        "Region",
        "Hub",
        "REGION_HUB",
        "RACE",
        "vaccine centre"
    ],

    "binary_columns": [
        "ACPA.POSITIVE",
        "RHUEMATOID.FACTOR",
        "IM.STEROIDS.3MONTHS",
        "ALCOHOL_Y_N",
        "CURENT SMOKER",
        "Erosive",
        "ORAL.STEROIDS.3M",
        "Remission(<2.6DAS)",
        "HighDisease(>4DAS)",
        "GENDER"
    ],

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

        "TOTAL.SWOLLEN.9M", "TOTAL.TENDER.9M",

        "Hep B serology wk 9 (IU/mL)"
    ],

    "static_numeric": [
        "AGE",
        "HEIGHT",
        "WEIGHT",
        "Symp_Duration",
        "InitialxRAYScore",
        "FinalxRAYScore"
    ],

    "labels": [
        "Remission month",
    ]
}

# STEROID EVENT CONTRACT
STEROIDS_CONTRACT = {
    "key_columns": {
        "Digest": "patient_id"
    },

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

# MEDICATION EVENT CONTRACT

MEDS_CONTRACT = {
    "key_columns": {
        "Digest": "patient_id"
    },

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

# OPTIONAL: helpers
ALL_CONTRACTS = {
    "clinical": CLINICAL_CONTRACT,
    "steroids": STEROIDS_CONTRACT,
    "meds": MEDS_CONTRACT
}