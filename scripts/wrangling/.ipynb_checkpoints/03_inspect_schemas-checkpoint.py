import os
import yaml
import pandas as pd
from tqdm import tqdm
import sys

def main():
    """
    Inspects the column schemas of all raw HMDA data files to find
    common and differing columns across years, saving the output to a file.
    """
    print("Starting Schema Inspection")
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = config['data']['raw']
    reports_path = config['outputs']['reports']
    
    # Ensure the reports directory exists
    os.makedirs(reports_path, exist_ok=True)
    output_file_path = os.path.join(reports_path, 'schema_analysis_report.txt')

    years = range(2008, 2025) # 2008 to 2024
    
    columns_by_year = {}
    all_columns = set()

    for year in tqdm(years, desc="Inspecting Files"):
        file_name = f"year_{year}.csv"
        full_path = os.path.join(raw_data_path, file_name)

        if os.path.exists(full_path):
            try:
                # Read only the first row to get the header efficiently
                header_df = pd.read_csv(full_path, nrows=0)
                columns = set(header_df.columns)
                columns_by_year[year] = columns
                all_columns.update(columns)
            except Exception as e:
                print(f"Could not read {file_name}: {e}")
        else:
            print(f"Warning: Data file for year {year} not found.")

    if not columns_by_year:
        print("No data files found to inspect.")
        return
        
    # Analysis & Writing to File
    print(f"Saving Schema Analysis to {output_file_path}")
    with open(output_file_path, 'w') as f:
        # Find the common columns present in all files
        common_columns = set.intersection(*columns_by_year.values())
        
        f.write("SCHEMA ANALYSIS REPORT")

        f.write(f"Found {len(common_columns)} common columns across all {len(columns_by_year)} years found:\n")
        for col in sorted(list(common_columns)):
            f.write(f"{col}\n")
        
        # Find columns that are NOT in all files
        f.write(f"\n\nFound {len(all_columns - common_columns)} columns that vary across years:\n")
        for year, columns in sorted(columns_by_year.items()):
            unique_to_year = columns - common_columns
            if unique_to_year:
                f.write(f"\nYear {year} has {len(unique_to_year)} unique columns:\n")
                for col in sorted(list(unique_to_year)):
                    f.write(f"  - {col}\n")

    print("Inspection complete. Report saved")

if __name__ == '__main__':
    main()