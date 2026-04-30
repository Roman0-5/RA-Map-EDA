import numpy as np
def extract_dtypes(df):
    df = df.copy()
    print('Schema of total values \n(column, row)')
    print(f'Absolute Values: {df.shape}')
    print(list(df))
    only_num = df.select_dtypes(include=np.number)
    print(f'Numerical Values: {only_num.shape}')
    print(list(only_num.columns))
    #selecting only str dtypes
    only_str = df.select_dtypes(include=['str'])
    print(f'String values: {only_str.shape}')
    print(list(only_str.columns))
    #selecting only obj types
    only_obj = df.select_dtypes(include='object', exclude='str')
    print(f'Object values: {only_obj.shape}')
    print(list(only_obj.columns))
