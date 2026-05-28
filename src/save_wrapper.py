import os
from contextlib import redirect_stdout

def save_report(report_func, df, name, output_dir='../../reports', **kwargs):
    """Generic save-to-file wrapper for any report function.
    
    Args:
        report_func: Function that prints a report (e.g. data_audit, data_quality_report)
        df: DataFrame to analyze
        name: Identifier for the output file
        output_dir: Where to save
        **kwargs: Extra arguments passed to report_func
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = f'{output_dir}/{name}_{report_func.__name__}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            report_func(df, name=name, **kwargs)
    
    print(f"Saved: {report_path}")