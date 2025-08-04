import pandas as pd
import os
import yaml
import time

def load_hmda_data_from_csv(year, config_path='config/config.yml'):
    print(f"Starting data load for year: {year}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None

    # Construct file path
    raw_data_path = config['data']['raw']
    file_name = f"year_{year}.csv"
    full_path = os.path.join(raw_data_path, file_name)

    if not os.path.exists(full_path):
        print(f"Error: Data file not found at {full_path}")
        return None

    # Get the optimized dtypes from the config file
    try:
        dtypes = config['optimized_dtypes']
        print("Found optimized dtypes in config")
    except KeyError:
        dtypes = None
        print("'optimized_dtypes' not found in config. Loading with default types.")


    # Load the data and time the process
    start_time = time.time()
    try:
        # Use the specified dtypes to load the data efficiently
        df = pd.read_csv(full_path, dtype=dtypes)
        end_time = time.time()
        
        print(f"Successfully loaded {file_name}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return df
        
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None