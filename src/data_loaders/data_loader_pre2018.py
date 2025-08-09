import pandas as pd
import os
import yaml
import time

def load_pre2018_data(year, config):
    """
    Efficiently loads a single year of HMDA data from the pre-2018 schema.
    """
    print(f"Loading PRE-2018 data for year: {year}")
    
    raw_data_path = config['data']['raw']
    file_name = f"year_{year}.csv"
    full_path = os.path.join(raw_data_path, file_name)

    all_dtypes = config.get('optimized_dtypes', {})
    
    try:
        header_cols = pd.read_csv(full_path, nrows=0).columns.tolist()
        dtypes_to_apply = {k: v for k, v in all_dtypes.items() if k in header_cols}
        
        start_time = time.time()
        # Use pyarrow for speed, as this schema is well-defined
        df = pd.read_csv(full_path, dtype=dtypes_to_apply, engine='pyarrow')
        end_time = time.time()
        
        print(f"Successfully loaded {file_name} in {end_time - start_time:.2f} seconds.")
        return df

    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")
        return None