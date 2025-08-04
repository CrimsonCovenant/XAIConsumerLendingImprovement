import sys
import os
import yaml
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader import load_hmda_data_from_csv

def analyze_and_optimize_dtypes(df):
    print("Analyzing and Optimizing Data Types")
    optimized_dtypes = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if "int" in str(dtype):
            # Find the smallest integer type that can hold the min/max values
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                optimized_dtypes[col] = 'int8'
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                optimized_dtypes[col] = 'int16'
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                optimized_dtypes[col] = 'int32'
            else:
                optimized_dtypes[col] = 'int64'
        
        elif "float" in str(dtype):
            # Find the smallest float type
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                optimized_dtypes[col] = 'float16'
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                optimized_dtypes[col] = 'float32'
            else:
                optimized_dtypes[col] = 'float64'
        
        elif "object" in str(dtype):
            # If the number of unique values is less than 50% of the total,
            # it's a good candidate for 'category' type.
            if df[col].nunique() / df[col].notna().sum() < 0.5:
                optimized_dtypes[col] = 'category'
            else:
                optimized_dtypes[col] = 'object'
    
    print("Analysis complete")
    return optimized_dtypes

def main():
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    year_to_load = config['hmda_data']['initial_year_to_load']

    # Load the data
    df = load_hmda_data_from_csv(year=year_to_load)

    if df is not None:
        print("Initial Data Exploration")
        
        # 1. Display initial info
        print("Initial DataFrame Info:")
        df.info(verbose=False)
        
        # 2. Analyze and get optimized dtypes
        optimized_dtypes = analyze_and_optimize_dtypes(df)
        
        print("Optimized Dtypes")
        print("Recommended dtypes to add to your config file:")
        print(yaml.dump({'optimized_dtypes': optimized_dtypes}, indent=4))
        
        # 3. Show potential memory savings
        print("Potential Memory Savings")
        initial_mem = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Temporarily apply optimized dtypes to calculate new memory usage
        df_optimized = df.astype(optimized_dtypes)
        optimized_mem = df_optimized.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"Initial Memory Usage: {initial_mem:.2f} MB")
        print(f"Optimized Memory Usage: {optimized_mem:.2f} MB")
        print(f"Savings: {initial_mem - optimized_mem:.2f} MB ({((initial_mem - optimized_mem) / initial_mem) * 100:.2f}%)")

if __name__ == '__main__':
    main()