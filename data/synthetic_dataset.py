import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_advanced_data(num_samples=2500, save_path='dataset.csv'):
    np.random.seed(42)
    
    # 1. Temporal Features
    start_dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730), 
                                                 hours=np.random.randint(0, 24),
                                                 minutes=np.random.randint(0, 60)) 
                   for _ in range(num_samples)]
    
    hours = [d.hour for d in start_dates]
    days = [d.weekday() for d in start_dates] # 0-6
    months = [d.month for d in start_dates] # 1-12
    
    # 2. Location Features
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow']
    origins = np.random.choice(cities, size=num_samples)
    destinations = [np.random.choice([c for c in cities if c != origins[i]]) for i in range(num_samples)]
    
    # 3. Base Features
    distances = np.random.randint(10, 2500, size=num_samples) # 10 to 2500 KM
    vehicle_types = ['Bike', 'Van', 'Truck', 'Trailer']
    speeds = {'Bike': 30, 'Van': 60, 'Truck': 50, 'Trailer': 40}
    vehicles = np.random.choice(vehicle_types, size=num_samples)
    
    # 4. New Advanced Features
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Foggy', 'Stormy']
    weather = np.random.choice(weather_conditions, size=num_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    
    traffic_index = np.random.randint(1, 11, size=num_samples) # 1 to 10
    driver_experience = np.random.randint(1, 21, size=num_samples) # 1 to 20 years
    
    # Package weight depends on vehicle roughly
    weights = []
    for v in vehicles:
        if v == 'Bike': weights.append(np.random.uniform(0.5, 15.0))
        elif v == 'Van': weights.append(np.random.uniform(50, 1500.0))
        elif v == 'Truck': weights.append(np.random.uniform(1000, 10000.0))
        else: weights.append(np.random.uniform(5000, 25000.0))
        
    packet_weights = np.array(weights).round(1)
    
    # 5. Calculating Duration (Target Variable)
    durations = []
    for i in range(num_samples):
        base_dur = distances[i] / speeds[vehicles[i]]
        
        # Adjust for weather
        w_factor = {'Sunny': 1.0, 'Cloudy': 1.05, 'Rainy': 1.2, 'Foggy': 1.3, 'Stormy': 1.5}[weather[i]]
        # Adjust for traffic (index 1 to 10 -> 0% to 50% increase)
        t_factor = 1.0 + (traffic_index[i] * 0.05)
        # Adjust for experience (20 years -> 10% faster, 1 year -> 10% slower)
        e_factor = 1.1 - (driver_experience[i] * 0.01)
        
        # Add pure noise
        noise_factor = np.random.normal(1.0, 0.1)
        
        final_dur = base_dur * w_factor * t_factor * e_factor * noise_factor
        durations.append(max(0.5, final_dur)) # minimum 30 minutes
        
    end_dates = [start_dates[i] + timedelta(hours=durations[i]) for i in range(num_samples)]
    
    df = pd.DataFrame({
        'trip_start_date': start_dates,
        'trip_end_date': end_dates,
        'Origin_Location': origins,
        'Destination_Location': destinations,
        'TRANSPORTATION_DISTANCE_IN_KM': distances,
        'vehicleType': vehicles,
        'start_hour': hours,
        'weekday': days,
        'start_month': months,
        'Weather_Condition': weather,
        'Traffic_Index': traffic_index,
        'Driver_Experience': driver_experience,
        'Package_Weight': packet_weights,
        'delivery_duration_hours': durations
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {num_samples} samples and saved to {save_path}")

if __name__ == "__main__":
    generate_advanced_data(save_path='d:/ML/delivery_prediction/data/dataset.csv')
