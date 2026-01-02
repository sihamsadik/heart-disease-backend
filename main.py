from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Enable CORS so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
# We use try/except to handle paths depending on where the script is run
try:
    log_reg_model = joblib.load("log_reg_model.joblib")
    dt_model = joblib.load("decision_tree_model.joblib")
except Exception as e:
    print(f"Error loading models: {e}")

# Define Input Structure
class PatientData(BaseModel):
    age: int
    sex: int  # 1: Male, 0: Female
    cp: int   # Chest Pain Type (0-3)
    trestbps: int # Resting Blood Pressure
    chol: int # Cholesterol
    fbs: int  # Fasting Blood Sugar > 120 (1: True, 0: False)
    thalach: int # Max Heart Rate
    exang: int # Exercise Induced Angina (1: Yes, 0: No)

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Running"}

def predict_heart_disease(model, data: PatientData):
    # Convert input to DataFrame (names must match training columns)
    features = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] # Prob of class 1
    
    result = "Positive (Heart Disease)" if prediction == 1 else "Negative (Healthy)"
    return {"prediction": result, "probability": float(probability)}

@app.post("/predict/logistic")
def predict_logistic(data: PatientData):
    return predict_heart_disease(log_reg_model, data)

@app.post("/predict/tree")
def predict_tree(data: PatientData):
    return predict_heart_disease(dt_model, data)