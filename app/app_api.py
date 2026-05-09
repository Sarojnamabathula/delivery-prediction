from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import os
import time
import pandas as pd
from typing import List, Optional

# Setup
app = FastAPI(
    title="DelivAI Prediction Service",
    description="Real-time delivery duration prediction API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth Header
API_KEY = "delivai-super-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# Models
MODELS_DIR = "models/"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "delivery_model_ensemble.pkl")

preprocessor = None
model = None

@app.on_event("startup")
def load_models():
    global preprocessor, model
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load models: {e}")

# Pydantic Schemas
class PredictionInput(BaseModel):
    origin: str = Field(..., max_length=50)
    destination: str = Field(..., max_length=50)
    distance: float = Field(..., gt=0, le=5000)
    vehicleType: str = Field(...)
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    weather: str = Field(...)
    traffic: int = Field(..., ge=1, le=10)
    weight: float = Field(..., gt=0)
    experience: int = Field(..., ge=1, le=50)

class PredictionOutput(BaseModel):
    predicted_hours: float
    confidence_interval: float
    delivery_category: str
    model_used: str

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]

# Middleware Logging & Rate Limiting (Hint)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # TO DO: Add slowapi rate limiter here (e.g. @limiter.limit("5/minute"))
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {request.method} {request.url.path} - Time: {process_time:.4f}s")
    return response

# Helper mapping
def prepare_input(data: PredictionInput) -> pd.DataFrame:
    if not preprocessor:
        raise HTTPException(status_code=500, detail="Model pipeline not initialized")
    
    input_dict = {
        'Origin_Location': data.origin,
        'Destination_Location': data.destination,
        'TRANSPORTATION_DISTANCE_IN_KM': data.distance,
        'vehicleType': data.vehicleType,
        'start_hour': data.hour,
        'weekday': data.day,
        'start_month': data.month,
        'Weather_Condition': data.weather,
        'Traffic_Index': data.traffic,
        'Driver_Experience': data.experience,
        'Package_Weight': data.weight
    }
    
    df = pd.DataFrame([input_dict])
    
    # Needs to match training process
    X_encoded = pd.get_dummies(df)
    
    # Align columns
    model_columns = preprocessor['columns']
    for col in model_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[model_columns]
    
    # Scale
    scaler = preprocessor['scaler']
    X_scaled = scaler.transform(X_encoded)
    
    return X_scaled

def get_category(hours: float) -> str:
    if hours <= 6.0: return "Express"
    if hours <= 24.0: return "Standard"
    return "Long Haul"

# Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": (model is not None and preprocessor is not None)
    }

@app.get("/model-info", dependencies=[Security(get_api_key)])
def model_info():
    if not preprocessor:
        raise HTTPException(status_code=500, detail="Models missing")
    
    results = preprocessor.get('results', {}).get('ensemble', {})
    
    return {
        "model_name": "VotingRegressor (RF + XGB)",
        "r2_score": results.get("R2", "N/A"),
        "mae": results.get("MAE", "N/A"),
        "feature_count": len(preprocessor['columns'])
    }

@app.post("/predict", response_model=PredictionOutput, dependencies=[Security(get_api_key)])
def predict(data: PredictionInput):
    try:
        X = prepare_input(data)
        prediction = model.predict(X)[0]
        
        # Approximate std for CV (could be hardcoded or calc from ensemble variance)
        confidence = prediction * 0.15 # 15% margin
        
        return PredictionOutput(
            predicted_hours=round(prediction, 2),
            confidence_interval=round(confidence, 2),
            delivery_category=get_category(prediction),
            model_used="Ensemble (RF+XGB)"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch", response_model=List[PredictionOutput], dependencies=[Security(get_api_key)])
def predict_batch(data: BatchPredictionInput):
    outputs = []
    for item in data.inputs:
        try:
            X = prepare_input(item)
            pred = model.predict(X)[0]
            outputs.append(PredictionOutput(
                predicted_hours=round(pred, 2),
                confidence_interval=round(pred * 0.15, 2),
                delivery_category=get_category(pred),
                model_used="Ensemble (RF+XGB)"
            ))
        except Exception as e:
            # For batch, could append error or fail completely. We fail item gracefully.
            raise HTTPException(status_code=400, detail=f"Error processing item: {e}")
    return outputs
