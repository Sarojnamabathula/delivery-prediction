import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configuration
DATASET_PATH = 'data/dataset.csv'
MODELS_DIR = 'models/'

def preprocess_data():
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # 1. Feature Engineering & Selection
    features = [
        'Origin_Location', 'Destination_Location', 'TRANSPORTATION_DISTANCE_IN_KM',
        'vehicleType', 'start_hour', 'weekday', 'start_month',
        'Weather_Condition', 'Traffic_Index', 'Driver_Experience', 'Package_Weight'
    ]
    target = 'delivery_duration_hours'
    
    X = df[features].copy()
    y = df[target].copy()
    
    # 2. One-Hot Encoding
    categorical_cols = ['Origin_Location', 'Destination_Location', 'vehicleType', 'Weather_Condition']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns are regular python standard types (bool to int)
    for col in X_encoded.select_dtypes(include=['bool']):
        X_encoded[col] = X_encoded[col].astype(int)
        
    feature_columns = X_encoded.columns.tolist()
    
    # 3. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    return X_scaled, y, scaler, feature_columns

def train_and_save_models(X, y, scaler, feature_columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n--- Training Models ---")
    
    # Linear Regression (Baseline)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    # Ensemble
    ensemble_model = VotingRegressor(estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ])
    print("Training Ensemble...")
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate and Save
    models = {
        'rf': rf_model,
        'xgb': xgb_model,
        'ensemble': ensemble_model,
        'lr': lr_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'R2': r2, 'MAE': mae}
        print(f"[{name.upper()}] R2: {r2:.4f} | MAE: {mae:.2f} hrs")
        
        # Save individual model
        if name != 'lr': # Don't really need LR for the dashboard unless requested
            joblib.dump(model, os.path.join(MODELS_DIR, f'delivery_model_{name}.pkl'))
            
    # Save Preprocessor and Column Info
    preprocessor_data = {
        'scaler': scaler,
        'columns': feature_columns,
        'results': results
    }
    joblib.dump(preprocessor_data, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    print(f"\n✅ All models and preprocessor saved to {MODELS_DIR}")

if __name__ == "__main__":
    X_scaled, y, scaler, feature_cols = preprocess_data()
    train_and_save_models(X_scaled, y, scaler, feature_cols)
    print("Model Training Pipeline Finished.")
