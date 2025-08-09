import sys
import os
import yaml
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# We will use a more robust, chunking-based loader for this specific file
from src.data_loaders.data_loader_post2018 import load_post2018_data as load_robust_data

def main():
    """
    Processes and unifies the 2017 HMDA data file, handling specific
    data type issues present in that year.
    """
    year = 2017
    print(f"Starting Unification FIX for year: {year}")
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Define the mapping for the old schema
    schema_map = {
        'action_taken': 'action_taken',
        'loan_amount_000s': 'loan_amount',
        'applicant_income_000s': 'income',
        'loan_purpose': 'loan_purpose',
        'loan_type': 'loan_type',
        'lien_status': 'lien_status',
        'hoepa_status': 'hoepa_status',
        'owner_occupancy': 'occupancy_type',
        'applicant_sex': 'applicant_sex',
        'applicant_race_1': 'applicant_race',
        'applicant_ethnicity': 'applicant_ethnicity'
    }
    
    # Use the robust loader for the problematic 2017 file
    df = load_robust_data(year, config)
    
    if df is not None:
        df_renamed = df.rename(columns=schema_map)
        
        # Filter for only the columns in our final unified schema
        final_cols = list(schema_map.values())
        df_unified = df_renamed[final_cols]
        
        # Basic Cleaning
        df_filtered = df_unified[df_unified['action_taken'].isin([1, 3])].copy()
        df_filtered['loan_status'] = (df_filtered['action_taken'] == 1).astype(int)
        df_cleaned = df_filtered.drop(columns=['action_taken']).dropna()

        # Save the unified file
        processed_dir = config['data']['processed']
        output_path = os.path.join(processed_dir, f"unified_processed_{year}.csv")
        print(f"Saving cleaned 2017 data to {output_path}")
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"2017 Schema Unification FIX Complete")
    else:
        print(f"Failed to load data for 2017. Aborting fix")

if __name__ == '__main__':
    main()