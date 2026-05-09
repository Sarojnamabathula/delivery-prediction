import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Static data for maps
CITY_COORDS = {
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.7041, 77.1025),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Kolkata': (22.5726, 88.3639),
    'Ahmedabad': (23.0225, 72.5714),
    'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462)
}

def load_data(path='data/dataset.csv'):
    df = pd.read_csv(path)
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    df['trip_end_date'] = pd.to_datetime(df['trip_end_date'])
    df['route'] = df['Origin_Location'] + " ➔ " + df['Destination_Location']
    return df

def get_category(hours):
    if hours <= 6: return "Express 🟢"
    elif hours <= 24: return "Standard 🟡"
    else: return "Long Haul 🔴"
