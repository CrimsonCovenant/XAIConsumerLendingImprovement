import sys
import os
import yaml
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loaders.data_loader_pre2018 import load_pre2018_data

def main():
    """
    Processes and unifies all PRE-2018 HMDA data files (2008-2017).
    """
    print("Starting Unification of PRE-2018 HMDA Schemas")
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
    
    years = range(2008, 2018) # 2008 to 2017
    
    for year in tqdm(years, desc="Unifying Pre-2018 Data"):
        df = load_pre2018_data(year, config)
        if df is None: continue

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
        df_cleaned.to_csv(output_path, index=False)
        
    print("PRE-2018 Schema Unification Complete")

if __name__ == '__main__':
    main()