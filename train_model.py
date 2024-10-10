import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the datasets
load_data = pd.read_csv('data/load_forecasting_dataset.csv')
solar_data = pd.read_csv('data/solar_pv_generation_dataset.csv')

# Function to train the load forecasting model
def train_load_forecasting_model():
    X = load_data[['hour', 'day_of_week', 'month']]
    y = load_data['load_consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Load Forecasting Model MSE: {mse}')
    
    # Save the model
    joblib.dump(model, 'models/load_forecasting_model.pkl')
    print('Load forecasting model saved.')

# Function to train the solar PV generation model
def train_solar_pv_generation_model():
    X = solar_data[['hour', 'day_of_year', 'cloud_cover', 'temperature']]
    y = solar_data['solar_generation']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Solar PV Generation Model MSE: {mse}')
    
    # Save the model
    joblib.dump(model, 'models/solar_pv_generation_model.pkl')
    print('Solar PV generation model saved.')

# Train both models
train_load_forecasting_model()
train_solar_pv_generation_model()
