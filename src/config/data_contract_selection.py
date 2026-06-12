"""
Data Contract for RA dataset preprocessing pipeline.
This defines column roles across: - Clinical baseline data - Steroid administration events - Medication administration events
"""

CLINICAL_CONTRACT_SELECTION = {

    "id_columns": ["Patient_ID", "Digest"],

    # -------------------------
    # PURE FEATURES (X)
    # -------------------------
    "binary_columns": [
        "ACPA.POSITIVE", #anti-CCP antibody status (RA marker)
        "RHUEMATOID.FACTOR", #rheumatoid factor positivity (autoantibody)
        "Erosive", #presence of joint erosions
    ],

    # -------------------------
    # LONGITUDINAL FEATURES
    # -------------------------
    "longitudinal_numeric": [
        "DAS28.0M", #disease activity composite score
        "HAQ.0M", #functional disability index score
        "SDAI.0M", #simplified disease activity index

        "CRP.0M", #C-reactive protein inflammation level
        "ESR.0M", #erythrocyte sedimentation rate

        #Bloodmarkers (routine clinical relevance)
        "HB.0M", #haemoglobin concentration level
        "LYMPHOCYTES.0M", #lymphocyte blood cell count
        "NEUTROPHILS.0M", #neutrophil blood cell count
        "PLT.0M", #platelet blood count
        "WBC.0M", #total white blood cell count

        "PAIN.0M", #patient-reported pain score
        "TOTAL.SWOLLEN.0M", "TOTAL.TENDER.0M", #joint symptom
    ],

    # -------------------------
    # STATIC FEATURES
    # -------------------------
    "static_numeric": [
        "AGE", #patient age at baseline
        "Symp_Duration", #symptom duration before diagnosis
        "InitialxRAYScore" #baseline radiographic damage score
    ]
}

