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
from scipy.stats import t

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.logistic_regression import LogisticRegression

def welch_ttest_from_scratch(x1, x2, equal_var=False):
    n1, n2 = x1.size, x2.size
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    t_statistic = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    
    df_num = (v1 / n1 + v2 / n2)**2
    df_den = (v1**2) / ((n1**2) * (n1 - 1)) + (v2**2) / ((n2**2) * (n2 - 1))
    df = df_num / df_den
    
    p_value = t.sf(np.abs(t_statistic), df) * 2
    
    return t_statistic, p_value

def run_bootstrap_for_year(year, config):
    print(f"Running Bootstrap Analysis for Year: {year}")
    
    processed_path = config['data']['processed']
    data_path = os.path.join(processed_path, f"unified_processed_{year}.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Unified data file not found for {year}. Skipping.")
        return None, None

    df = pd.read_csv(data_path)
    
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    X = df_dummies.drop('loan_status', axis=1)
    y = df_dummies['loan_status']
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_bootstraps = 100
    subsample_size = 100000 
    bootstrap_coefs = []

    for i in tqdm(range(n_bootstraps), desc=f"Bootstrap {year}"):
        X_subsample, y_subsample = resample(X_scaled, y.values, 
                                            n_samples=min(subsample_size, len(y)), 
                                            random_state=i, stratify=y)
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
        model.fit(X_subsample, y_subsample)
        weights, _ = model.get_params()
        bootstrap_coefs.append(weights)

    return pd.DataFrame(bootstrap_coefs, columns=feature_names)

def main():
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    year1, year2 = 2023, 2024
    
    coefs_year1 = run_bootstrap_for_year(year1, config)
    coefs_year2 = run_bootstrap_for_year(year2, config)
    
    if coefs_year1 is None or coefs_year2 is None:
        print("\nAborting drift analysis due to missing data.")
        return

    print(f"Comparing Feature Importance Distributions ({year1} vs {year2})")
    
    common_features = list(set(coefs_year1.columns) & set(coefs_year2.columns))
    
    drift_results = []
    for feature in common_features:
        dist1 = coefs_year1[feature].values
        dist2 = coefs_year2[feature].values
        
        t_stat, p_value = welch_ttest_from_scratch(dist1, dist2)
        
        drift_results.append({
            'Feature': feature,
            f'Mean_{year1}': dist1.mean(),
            f'Mean_{year2}': dist2.mean(),
            'p_value': p_value,
            'Significant_Drift': 'Yes' if p_value < 0.05 else 'No'
        })

    drift_df = pd.DataFrame(drift_results).sort_values(by=f'Mean_{year2}', ascending=False)
    print("\nDrift Analysis Results (p-value < 0.05 indicates significant drift):")
    print(drift_df.to_string())
    
    # Visualization
    results1 = coefs_year1[common_features].mean().rename(str(year1))
    results2 = coefs_year2[common_features].mean().rename(str(year2))
    
    plot_df = pd.concat([results1, results2], axis=1).sort_values(by=str(year2))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(plot_df))
    
    ax.barh(y_pos - 0.2, plot_df[str(year1)], height=0.4, label=str(year1), color='skyblue', alpha=0.8)
    ax.barh(y_pos + 0.2, plot_df[str(year2)], height=0.4, label=str(year2), color='salmon', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index)
    ax.axvline(x=0, linestyle='--', color='black', linewidth=0.8)
    ax.set_title(f'Explanation Drift: Feature Importance Comparison ({year1} vs {year2})', fontsize=16)
    ax.set_xlabel('Coefficient (Weight)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.legend()
    plt.tight_layout()

    viz_dir = config['outputs']['visualizations']
    output_path = os.path.join(viz_dir, f'drift_analysis_{year1}_vs_{year2}.png')
    plt.savefig(output_path)
    print(f"\nSaved drift analysis plot to {output_path}")
    plt.show()

if __name__ == '__main__':
    main()