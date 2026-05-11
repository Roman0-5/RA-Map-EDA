import numpy as np
def extract_dtypes(df, verbose=False):
    """
    Extract all dtypes of an input df\n
    Can be verbose if needed
    """
    df = df.copy()
    only_num = df.select_dtypes(include=np.number)
    #selecting only str dtypes
    only_str = df.select_dtypes(include=['str'])
    #selecting only obj types
    only_obj = df.select_dtypes(include='object', exclude='str')
    if verbose:
        print('Schema of total values \n(column, row)')
        print(f'Absolute Values: {df.shape}')
        print(list(df))
        print(f'Numerical Values: {only_num.shape}')
        print(list(only_num.columns))
        print(f'String values: {only_str.shape}')
        print(list(only_str.columns))
        print(f'Object values: {only_obj.shape}')
        print(list(only_obj.columns))

    return df, only_num, only_str, only_obj
