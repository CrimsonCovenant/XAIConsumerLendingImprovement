import sys
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.logistic_regression import LogisticRegression

def main():
    """
    Main function to perform bootstrap analysis on logistic regression feature importances.
    """
    print("Loading Configuration and Data")
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    processed_path = config['data']['processed']
    year = config['hmda_data']['initial_year_to_load']
    data_path = os.path.join(processed_path, f"processed_{year}.csv")
    df = pd.read_csv(data_path)
    
    # Prepare data for modeling
    df_dummies = pd.get_dummies(df, drop_first=True)
    X = df_dummies.drop('loan_status', axis=1)
    y = df_dummies['loan_status']
    feature_names = X.columns

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Bootstrapping
    n_bootstraps = 100 # We will run 100 iterations
    bootstrap_coefs = []
    print(f"Starting Bootstrap Analysis with {n_bootstraps} Iterations")

    for i in tqdm(range(n_bootstraps), desc="Bootstrap Progress"):
        # Resample the data with replacement
        X_sample, y_sample = resample(X_scaled, y.values, random_state=i, stratify=y)
        
        # Train a new model on the resampled data
        # Note: verbose=False to keep the log clean
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
        model.fit(X_sample, y_sample)
        
        # Get the learned weights and store them
        weights, _ = model.get_params()
        bootstrap_coefs.append(weights)

    # Convert the list of coefficients into a DataFrame
    coefs_df = pd.DataFrame(bootstrap_coefs, columns=feature_names)

    # Analyze Bootstrap Results
    print("Analyzing Bootstrap Results")
    
    # Calculate statistics: mean, std dev, and 95% confidence intervals
    results = pd.DataFrame({
        'Mean': coefs_df.mean(),
        'Std Dev': coefs_df.std(),
        'CI Lower (2.5%)': coefs_df.quantile(0.025),
        'CI Upper (97.5%)': coefs_df.quantile(0.975)
    }).sort_values(by='Mean', ascending=False)

    print("Feature Importance with Confidence Intervals:")
    print(results)
    
    # Identify features whose confidence intervals do NOT cross zero
    significant_features = results[(results['CI Lower (2.5%)'] > 0) | (results['CI Upper (97.5%)'] < 0)]
    print("\nStatistically Significant Features (95% Confidence):")
    print(significant_features)

    # Visualization
    results_to_plot = results.sort_values(by='Mean', ascending=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 18))
    
    # Plot the mean coefficient as a point
    ax.scatter(x=results_to_plot['Mean'], y=results_to_plot.index, color='black', s=50, zorder=2)
    
    # Plot the confidence interval as a line (error bar)
    errors = [results_to_plot['Mean'] - results_to_plot['CI Lower (2.5%)'], 
              results_to_plot['CI Upper (97.5%)'] - results_to_plot['Mean']]
    ax.errorbar(results_to_plot['Mean'], results_to_plot.index, xerr=errors,
                fmt='none', capsize=5, color='gray', zorder=1)

    ax.axvline(x=0, linestyle='--', color='red', linewidth=1)
    ax.set_title(f'Feature Importance and 95% Confidence Intervals - HMDA {year}', fontsize=16)
    ax.set_xlabel('Coefficient (Weight)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()

    # Save the visualization
    viz_dir = config['outputs']['visualizations']
    output_path = os.path.join(viz_dir, f'feature_importance_bootstrap_{year}.png')
    
    plt.savefig(output_path)
    print(f"\nSaved bootstrap analysis plot to {output_path}")
    plt.show()

if __name__ == '__main__':
    main()