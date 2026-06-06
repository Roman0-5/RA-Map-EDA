import pyreadr
import pandas as pd

#this function requires a lot of RAM and CPU Usage (17626x2196 raw data matrix) so run it on the provided server
def convert():
    result = pyreadr.read_r('../datasets/microarray_dat.rds')
    df = result[None]
    print(f'shape: {df.shape}')
    df.to_csv('../datasets/microarray_dat.csv')
    df.to_parquet('../datasets/microarray_dat.parquet')
    print("saved.")
    #sanitycheck
    df_new = pd.read_parquet('../datasets/microarray_dat.parquet')
    print(f'shape: {df_new.shape}')
    df_new = pd.read_parquet('../datasets/microarray_data.csv')
    print(f'shape: {df_new.shape}')
if __name__ == '__main__':
    convert()
