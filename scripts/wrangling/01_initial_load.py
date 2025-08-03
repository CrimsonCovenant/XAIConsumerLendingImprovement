import sys
import os
import yaml

# Add the src directory to the Python path
# This allows us to import our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader import load_hmda_data_from_csv

def main():
    """
    Main function to execute the initial data loading and assessment.
    """
    # Load configuration to get the year to process
    try:
        with open('config/config.yml', 'r') as f:
            config = yaml.safe_load(f)
        year_to_load = config['hmda_data']['initial_year_to_load']
    except FileNotFoundError:
        print("Error: config.yml not found. Please ensure it is in the project root.")
        return
    except KeyError:
        print("Error: 'initial_year_to_load' not found in config.yml.")
        return

    # Load the data using our reusable function
    df = load_hmda_data_from_csv(year=year_to_load)

    # Perform a quick assessment if the data loaded successfully
    if df is not None:
        print("\n--- Data Assessment ---")
        print(f"Shape of the DataFrame: {df.shape}")
        
        # Calculate and display memory usage in a readable format
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"Memory Usage: {memory_usage_mb:.2f} MB")
        print("-----------------------\n")

if __name__ == '__main__':
    main()