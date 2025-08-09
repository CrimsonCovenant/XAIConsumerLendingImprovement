import pandas as pd
import os
import time

def load_post2018_data(year, config):
    """
    Robustly loads a single year of HMDA data from the post-2018 schema in chunks.
    """
    print(f"Loading POST-2018 data for year: {year}")
    
    raw_data_path = config['data']['raw']
    file_name = f"year_{year}.csv"
    full_path = os.path.join(raw_data_path, file_name)

    chunk_list = []
    chunk_size = 100000 

    try:
        start_time = time.time()
        # Use the robust c engine with chunking for this complex schema
        with pd.read_csv(full_path, chunksize=chunk_size, low_memory=False) as reader:
            for chunk in reader:
                chunk_list.append(chunk)
        
        df = pd.concat(chunk_list, ignore_index=True)
        del chunk_list
        
        end_time = time.time()
        print(f"Successfully loaded {file_name} in {end_time - start_time:.2f} seconds.")
        return df

    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")
        return None