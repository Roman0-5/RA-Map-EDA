import pandas as pd

def load_all_sheets():
    """Loads all sheets as a dataframe\n
        Returns:
                dict: Dictionary with the following DataFrames:
                    - df_clinical: Clinical patient data
                    - df_steroids: Intramuscular steroids data
                    - df_meds: RA medications
                    - df_glossary: Glossary
                    - df_Samples: Protogen samples
                    - df_Samples_Annotation: Sample annotations
                    - df_expMatrix: SOMAscan expression matrix
                    - df_sampMatrix: SOMAscan sample matrix
    """
    print('Loading clinical...')
    clinical_sheet = pd.read_excel('../datasets/RA_MAP_Clinical_Figshare_17_5_21.xlsx', sheet_name=None)
    print('Loading protogen...')
    protogen_sheet = pd.read_excel('../datasets/Protogen_RA_MAP_16_05_21.xlsx', sheet_name=None)
    print('Loading somascan...')
    somascan_sheet = pd.read_excel('../datasets/SOMASCAN_RA-Map_figshare_17_11_20.xlsx', sheet_name=None)
    
    print("Done loading")
    return {
        #clinical
        'df_clinical': clinical_sheet['OpenPseudonymised_RA_MAP_Clinic'],
        'df_steroids': clinical_sheet['intramuscular steroids'],
        'df_meds': clinical_sheet['RA Meds'],
        'df_glossary': clinical_sheet['Glossary'],
        #protogen
        'df_Samples': protogen_sheet['LIS_PG665-P01 RA MAP Samples Ex'],
        'df_Samples_Annotation': protogen_sheet['Sample annotation'],
        #somascan
        'df_expMatrix': somascan_sheet['expression matrix'],
        'df_sampMatrix': somascan_sheet['sample matrix']
    }