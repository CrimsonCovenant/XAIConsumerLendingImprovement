import sys
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.logistic_regression import LogisticRegression

def main():
    """
    Main function to train and evaluate the Logistic Regression model.
    """
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    processed_path = config['data']['processed']
    year = config['hmda_data']['initial_year_to_load']
    data_path = os.path.join(processed_path, f"processed_{year}.csv")
    
    print(f"--- Loading data from {data_path} ---")
    df = pd.read_csv(data_path)

    # Prepare data for modeling
    # Convert categorical variables to dummy variables
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    X = df_dummies.drop('loan_status', axis=1)
    y = df_dummies['loan_status']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training Logistic Regression Model from Scratch")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, verbose=True)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating Model Performance")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_dir = config['outputs']['models']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, f'logistic_regression_{year}.pkl')
    print(f"Saving trained model to {model_path}")
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully.")

if __name__ == '__main__':
    main()