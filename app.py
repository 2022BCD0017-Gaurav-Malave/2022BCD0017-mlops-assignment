from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import os

STUDENT_NAME = "Gaurav Malave"    
ROLL_NO      = "2022BCD0017" 


app = FastAPI(title=f"{ROLL_NO} MLOps API")

# Load model on startup
MODEL_PATH  = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

model  = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model loaded successfully.")
    else:
        print("WARNING: Model files not found. Train first.")


class PredictRequest(BaseModel):
    features: conlist(float, min_length=4, max_length=4)


@app.get("/")
@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "name":    STUDENT_NAME,
        "roll_no": ROLL_NO,
        "message": "MLOps Pipeline API is running"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    try:
        features = np.array(request.features, dtype=float).reshape(1, -1)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    return {
        "prediction":       int(prediction),
        "predicted_class":  labels.get(int(prediction), "Unknown"),
        "probabilities":    proba,
        "name":             STUDENT_NAME,
        "roll_no":          ROLL_NO
    }