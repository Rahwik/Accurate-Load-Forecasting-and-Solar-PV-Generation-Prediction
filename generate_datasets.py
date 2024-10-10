import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset for load forecasting
def generate_load_forecasting_data(num_records=10000):
    hours = np.random.randint(0, 24, num_records)
    days_of_week = np.random.randint(0, 7, num_records)   # 0 = Sunday, 6 = Saturday
    months = np.random.randint(1, 13, num_records)
    load_consumption = hours * 5 + days_of_week * 3 + months * 4 + np.random.normal(0, 5, num_records)  # Arbitrary function

    load_data = pd.DataFrame({
        'hour': hours,
        'day_of_week': days_of_week,
        'month': months,
        'load_consumption': load_consumption
    })
    load_data.to_csv('data/load_forecasting_dataset.csv', index=False)
    print("Load forecasting dataset generated.")

# Generate synthetic dataset for solar PV generation
def generate_solar_pv_generation_data(num_records=10000):
    hours = np.random.randint(0, 24, num_records)
    days_of_year = np.random.randint(1, 366, num_records)
    cloud_cover = np.random.uniform(0, 100, num_records)  # Percentage
    temperature = np.random.uniform(15, 35, num_records)  # Celsius
    solar_generation = (hours * 2 - cloud_cover * 0.5 + temperature * 0.3 + np.random.normal(0, 3, num_records))  # Arbitrary function

    solar_data = pd.DataFrame({
        'hour': hours,
        'day_of_year': days_of_year,
        'cloud_cover': cloud_cover,
        'temperature': temperature,
        'solar_generation': solar_generation
    })
    solar_data.to_csv('data/solar_pv_generation_dataset.csv', index=False)
    print("Solar PV generation dataset generated.")

# Generate both datasets
generate_load_forecasting_data()
generate_solar_pv_generation_data()
