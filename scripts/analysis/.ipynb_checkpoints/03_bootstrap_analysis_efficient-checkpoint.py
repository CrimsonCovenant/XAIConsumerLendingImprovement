import sys
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm # A library to show progress bars

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.logistic_regression import LogisticRegression

def main():
    """
    Main function to perform an EFFICIENT bootstrap analysis by training
    on smaller subsamples of the data.
    """
    print("Loading Configuration and Data")
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    processed_path = config['data']['processed']
    year = config['hmda_data']['initial_year_to_load']
    data_path = os.path.join(processed_path, f"processed_{year}.csv")
    df = pd.read_csv(data_path)
    
    df_dummies = pd.get_dummies(df, drop_first=True)
    X = df_dummies.drop('loan_status', axis=1)
    y = df_dummies['loan_status']
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # EFFICIENT BOOTSTRAPPING
    n_bootstraps = 100
    subsample_size = 100000 
    bootstrap_coefs = []
    print(f"Starting Efficient Bootstrap Analysis")
    print(f"Iterations: {n_bootstraps}, Subsample Size: {subsample_size}")

    for i in tqdm(range(n_bootstraps), desc="Bootstrap Progress"):
        # Create a smaller, random subsample for this iteration
        X_subsample, y_subsample = resample(X_scaled, y.values, n_samples=subsample_size, random_state=i, stratify=y)
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
        model.fit(X_subsample, y_subsample)
        
        weights, _ = model.get_params()
        bootstrap_coefs.append(weights)

    coefs_df = pd.DataFrame(bootstrap_coefs, columns=feature_names)

    # Analyze and Visualize Results same as before
    print("Analyzing Bootstrap Results")
    results = pd.DataFrame({
        'Mean': coefs_df.mean(),
        'Std Dev': coefs_df.std(),
        'CI Lower (2.5%)': coefs_df.quantile(0.025),
        'CI Upper (97.5%)': coefs_df.quantile(0.975)
    }).sort_values(by='Mean', ascending=False)

    print("Feature Importance with Confidence Intervals:")
    print(results)
    
    # Visualization
    results_to_plot = results.sort_values(by='Mean', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.scatter(x=results_to_plot['Mean'], y=results_to_plot.index, color='black', s=50, zorder=2)
    errors = [results_to_plot['Mean'] - results_to_plot['CI Lower (2.5%)'], 
              results_to_plot['CI Upper (97.5%)'] - results_to_plot['Mean']]
    ax.errorbar(results_to_plot['Mean'], results_to_plot.index, xerr=errors,
                fmt='none', capsize=5, color='gray', zorder=1)

    ax.axvline(x=0, linestyle='--', color='red', linewidth=1)
    ax.set_title(f'Feature Importance (Efficient Bootstrap) - HMDA {year}', fontsize=16)
    ax.set_xlabel('Coefficient (Weight)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()

    viz_dir = config['outputs']['visualizations']
    output_path = os.path.join(viz_dir, f'feature_importance_bootstrap_efficient_{year}.png')
    plt.savefig(output_path)
    print(f"\nSaved efficient bootstrap analysis plot to {output_path}")
    plt.show()

if __name__ == '__main__':
    main()