# Setup, Structure, Data Collection, Cleaning, Prepping
This repo includes python scripts in jupyter notebook cells

Its purpose is to share my documentation of the data- analysis, cleaning/prepping, and creation. To use it for ML/DL models

All datasets were provided by the study [RA-Map, a multi-omic immune cell landscape in early RA](https://figshare.com/collections/_/5491611) under the [Creative Commons license](https://creativecommons.org/licenses/by/4.0/)
## Setup
Creating Jupyter Notebook environment and getting all data
These are bash commands
> All commands have to be run at the project root
### Jupyter Notebook
```bash
python -m venv venv #create virtual environment
pip install -r requirements.txt #install packages
nbstriprout --install #git filter for notebook runtime outputs
```
### Data from figshare
Get all Data and extract them from figshare.com 

#### Manually
follow the [link](https://figshare.com/collections/RA-Map_a_multi-omic_immune_cell_landscape_in_early_RA/5491611/1) and do it manually
or

#### Script
run this script
first it creates a new folder 'datasets' and then it downloads the 1st, 2nd, and 7th dataset, putting them in the newly created [folder](datasets/)

```bash
mkdir -p datasets; curl \
-o datasets/RA_MAP_Clinical_Figshare_17_5_21.xlsx https://figshare.com/ndownloader/files/28864710 \
-o datasets/Protogen_RA_MAP_16_05_21.xlsx https://figshare.com/ndownloader/files/28033515 \
-o datasets/SOMASCAN_RA-Map_figshare_17_11_20.xlsx https://figshare.com/ndownloader/files/25666688
``` 
## Notebooks
Now to the actual notebooks

[Clinical Data](notebooks/clinical.ipynb)

Patient_ID, Digest, Age, DAS28 Score,...

[Protein Analysis, Collection](notebooks/protogen.ipynb)

Antibody cells response to treatment

[Response to Treatment](notebooks/somascan.ipynb)

Labels such as Responder/Non-Responder, Samples with timestamps

[EDA with a sample patient](notebooks/exploring_patient.ipynb)

Exploratory Data Analysis using a Responder that spans across all Sheets 

[Data Cleanup/Preparation](notebooks/cleaning.ipynb)

Normalizing, Cleaning Datatypes, Adding new Data, All saved in [cleaned_datasets](cleaned_datasets/)