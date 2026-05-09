from fastapi.testclient import TestClient
import sys
import os
import pytest

# Add the 'app' directory to the path so we can import the FastAPI app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.app_api import app, load_models

# Initialize models manually for tests if needed
load_models()

client = TestClient(app)

HEADERS = {
    "X-API-Key": "delivai-super-secret-key"
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()

def test_model_info_no_auth():
    response = client.get("/model-info")
    assert response.status_code == 401

def test_model_info_with_auth():
    response = client.get("/model-info", headers=HEADERS)
    assert response.status_code == 200
    assert "model_name" in response.json()

def test_predict_endpoint():
    payload = {
        "origin": "Mumbai",
        "destination": "Delhi",
        "distance": 1400.5,
        "vehicleType": "Truck",
        "hour": 8,
        "day": 1,
        "month": 6,
        "weather": "Sunny",
        "traffic": 3,
        "weight": 2000.0,
        "experience": 10
    }
    
    response = client.post("/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_hours" in data
    assert "delivery_category" in data
    assert data["predicted_hours"] > 0

def test_predict_invalid():
    payload = {
        "origin": "Mumbai",
        # distance missing
        "vehicleType": "Truck"
    }
    
    response = client.post("/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422 # Pydantic Validation Error
