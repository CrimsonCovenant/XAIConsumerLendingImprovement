import sys
import os
import yaml
import pandas as pd
from tqdm import tqdm

def main():
    """
    Processes and unifies all POST-2018 HMDA data files 2018-2024
    using a memory-efficient chunking pipeline.
    """
    print("Starting MEMORY-EFFICIENT Unification of POST-2018 Schemas")
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Define the mapping for the new schema
    schema_map = {
        'action_taken': 'action_taken',
        'loan_amount': 'loan_amount',
        'income': 'income',
        'loan_purpose': 'loan_purpose',
        'loan_type': 'loan_type',
        'lien_status': 'lien_status',
        'hoepa_status': 'hoepa_status',
        'occupancy_type': 'occupancy_type',
        'applicant_sex': 'applicant_sex',
        'derived_race': 'applicant_race',
        'derived_ethnicity': 'applicant_ethnicity'
    }
    
    # The final, unified columns we want to keep
    unified_schema = list(schema_map.values())

    years = range(2018, 2025) # 2018 to 2024
    
    for year in years:
        print(f"Processing Year: {year}")
        
        raw_data_path = config['data']['raw']
        file_name = f"year_{year}.csv"
        full_path = os.path.join(raw_data_path, file_name)

        if not os.path.exists(full_path):
            print(f"File not found for {year}, skipping.")
            continue

        processed_dir = config['data']['processed']
        output_path = os.path.join(processed_dir, f"unified_processed_{year}.csv")
        
        chunk_size = 100000 
        is_first_chunk = True

        try:
            with pd.read_csv(full_path, chunksize=chunk_size, low_memory=False) as reader:
                for chunk in tqdm(reader, desc=f"Processing {year} in chunks"):
                    # Scale loan_amount and income
                    if 'loan_amount' in chunk.columns:
                        chunk['loan_amount'] = chunk['loan_amount'] / 1000
                    if 'income' in chunk.columns:
                        chunk['income'] = chunk['income'] / 1000
                    
                    # Rename columns based on the map
                    chunk.rename(columns=schema_map, inplace=True)
                    
                    # Select only the columns that are in our unified schema
                    # This is crucial for handling files with varying extra columns
                    final_cols = [col for col in unified_schema if col in chunk.columns]
                    chunk_unified = chunk[final_cols]
                    
                    # Basic Cleaning on the chunk
                    chunk_filtered = chunk_unified[chunk_unified['action_taken'].isin([1, 3])].copy()
                    chunk_filtered['loan_status'] = (chunk_filtered['action_taken'] == 1).astype(int)
                    chunk_cleaned = chunk_filtered.drop(columns=['action_taken']).dropna()

                    # Save the processed chunk
                    if is_first_chunk:
                        # For the first chunk, write the header
                        chunk_cleaned.to_csv(output_path, index=False, mode='w', header=True)
                        is_first_chunk = False
                    else:
                        # For subsequent chunks, append without the header
                        chunk_cleaned.to_csv(output_path, index=False, mode='a', header=False)
            
            print(f"Successfully processed and saved unified data for {year}.")

        except Exception as e:
            print(f"An error occurred while processing {year}: {e}")

    print("POST-2018 Schema Unification Complete")

if __name__ == '__main__':
    main()