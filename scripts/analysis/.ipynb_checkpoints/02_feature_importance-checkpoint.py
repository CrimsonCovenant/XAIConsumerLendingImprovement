import sys
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# We need to import the class to be able to load the pickled object
from src.models.logistic_regression import LogisticRegression

def main():
    """
    Main function to load a trained model, extract feature importances,
    and visualize them.
    """
    print("Loading Configuration and Data")
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data just to get the column names after one-hot encoding
    processed_path = config['data']['processed']
    year = config['hmda_data']['initial_year_to_load']
    data_path = os.path.join(processed_path, f"processed_{year}.csv")
    df = pd.read_csv(data_path)
    
    # Recreate the feature set to get the correct column order
    df_dummies = pd.get_dummies(df, drop_first=True)
    X = df_dummies.drop('loan_status', axis=1)
    feature_names = X.columns

    # Load the trained model
    model_path = os.path.join(config['outputs']['models'], f'logistic_regression_{year}.pkl')
    print(f"Loading Model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Run the training script '01_train_logistic_regression.py' first.")
        return

    print("Extracting and Visualizing Feature Importance")
    # In a linear model like logistic regression, the weights are the importances
    importances = model.weights
    
    # Create a DataFrame for easier visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Most Important Features Positive Impact:")
    print(feature_importance_df.head(10))
    
    print("\n10 Least Important Features Most Negative Impact:")
    print(feature_importance_df.tail(10))

    # --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 18))
    
    # Create a color palette: green for positive, red for negative
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in feature_importance_df['Importance']]
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=colors, hue='Feature', dodge=False, legend=False)
    
    plt.title(f'Feature Importance from Logistic Regression - HMDA {year}', fontsize=16)
    plt.xlabel('Coefficient (Weight)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    # Save the visualization
    viz_dir = config['outputs']['visualizations']
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    output_path = os.path.join(viz_dir, f'feature_importance_{year}.png')
    
    plt.savefig(output_path)
    print(f"\nSaved feature importance plot to {output_path}")
    plt.show()


if __name__ == '__main__':
    main()