import sys
import os
import yaml
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.logistic_regression import LogisticRegression
from src.models.logistic_regression_optimized import LogisticRegressionOptimized

def main():
    """
    Main function to benchmark the original and optimized logistic regression models.
    """
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    processed_path = config['data']['processed']
    year = config['hmda_data']['initial_year_to_load']
    data_path = os.path.join(processed_path, f"processed_{year}.csv")
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    X = df_dummies.drop('loan_status', axis=1)
    y = df_dummies['loan_status'].values # Use .values for numpy array

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Benchmark Original Model
    print("Benchmarking Original Model (Batch Gradient Descent)")
    model_original = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=False)
    start_time_orig = time.time()
    model_original.fit(X_train, y_train)
    end_time_orig = time.time()
    time_original = end_time_orig - start_time_orig
    print(f"Original Model Training Time: {time_original:.2f} seconds")

    # Benchmark Optimized Model
    print("Benchmarking Optimized Model (Mini-Batch Gradient Descent)")
    # Note: We use fewer epochs because each epoch sees the whole dataset.
    # 100 epochs with mini-batch is a lot of updates.
    model_optimized = LogisticRegressionOptimized(learning_rate=0.01, n_epochs=100, batch_size=512, verbose=False)
    start_time_opt = time.time()
    model_optimized.fit(X_train, y_train)
    end_time_opt = time.time()
    time_optimized = end_time_opt - start_time_opt
    print(f"Optimized Model Training Time: {time_optimized:.2f} seconds")

    # Summary
    print("Benchmark Summary")
    print(f"Original (Batch GD) Time: {time_original:.2f} seconds")
    print(f"Optimized (Mini-Batch GD) Time: {time_optimized:.2f} seconds")
    if time_optimized > 0:
        speedup = time_original / time_optimized
        print(f"Speedup: {speedup:.2f}x")

if __name__ == '__main__':
    main()