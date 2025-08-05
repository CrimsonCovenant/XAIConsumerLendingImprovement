import sys
import os
import yaml
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader import load_hmda_data_from_csv

def preprocess_hmda_data(df, config):
    print("Starting Preprocessing")
    
    # 1. Select a subset of relevant columns for our initial model
    selected_features = [
        'loan_amount_000s', 'applicant_income_000s', 'loan_purpose_name',
        'lien_status_name', 'property_type_name', 'owner_occupancy_name',
        'applicant_sex_name', 'applicant_race_name_1', 'applicant_ethnicity_name', 'hoepa_status_name']

    target_variable = config['hmda_data']['target_variable']
    
    # Ensure all selected features and the target variable are in the DataFrame
    all_cols_to_keep = [col for col in selected_features + [target_variable] if col in df.columns]
    
    print(f"Subsetting to {len(all_cols_to_keep)} columns.")
    df_subset = df[all_cols_to_keep].copy()

    # 2. Define the binary target variable
    # We'll model 'Loan Originated' (action_taken == 1) vs. 'Application Denied' (action_taken == 3)
    print("Defining binary target variable: 1 for Originated, 0 for Denied.")
    df_filtered = df_subset[df_subset[target_variable].isin([1, 3])].copy()
    df_filtered['loan_status'] = (df_filtered[target_variable] == 1).astype(int)
    df_filtered = df_filtered.drop(columns=[target_variable]) # Drop original target column

    # 3. Handle Missing Values
    # For this initial pass, we will drop rows with any missing values for simplicity.
    initial_rows = len(df_filtered)
    df_cleaned = df_filtered.dropna()
    final_rows = len(df_cleaned)
    print(f"Handling missing values: Dropped {initial_rows - final_rows} rows.")
    
    print(f"Preprocessing complete. Final shape: {df_cleaned.shape}")
    return df_cleaned

def main():
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    year_to_process = config['hmda_data']['initial_year_to_load']

    # Load the data
    df_raw = load_hmda_data_from_csv(year=year_to_process, config_path='config/config.yml')

    if df_raw is not None:
        # Preprocess the data
        df_processed = preprocess_hmda_data(df_raw, config)

        # Save the processed data
        processed_dir = config['data']['processed']
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        output_path = os.path.join(processed_dir, f"processed_{year_to_process}.csv")
        print(f"Saving processed data to {output_path}")
        df_processed.to_csv(output_path, index=False)
        print("Save complete.")

if __name__ == '__main__':
    main()