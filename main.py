from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# 1. Enable CORS (Allows your frontend to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Models
# Ensure 'log_reg_model.joblib' and 'decision_tree_model.joblib' are in the same folder
try:
    log_reg_model = joblib.load("log_reg_model.joblib")
    dt_model = joblib.load("decision_tree_model.joblib")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")


# 3. Define Input Data Structure (Must match the 13 columns from training)
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Running"}

# Helper function to handle prediction logic
def get_prediction(model, data: PatientData):
    # Convert input data to a Pandas DataFrame
    features = pd.DataFrame([data.dict()])
    
    # Get the Class Prediction (0 or 1)
    prediction = model.predict(features)[0]
    
    # Get the Probability (Confidence)
    probs = model.predict_proba(features)[0]
    prob_sick = model.predict_proba(features)[0][1] # Probability of being class 1
    
    # Determine string result
    if prediction == 1:
        result = "Positive (Heart Disease)"
    else:
        result = "Negative (Healthy)"
# Determine string result based on probability threshold to reflect confidence of prediction in heatmap because model may predict class 0 with low confidence and it is critical to inform user with high probability to be positive with negative
    if prob_sick < 0.35:
        result = "Positive (Heart Disease)"
        final_prob = prob_sick
    else:
        result = "Negative (Healthy)"
        final_prob = prob_sick

    

    return {
        "prediction": result, 
        "probability": float(prob_sick)
    }

# Endpoint 1: Logistic Regression
@app.post("/predict/logistic")
def predict_logistic(data: PatientData):
    return get_prediction(log_reg_model, data)

# Endpoint 2: Decision Tree
@app.post("/predict/tree")
def predict_tree(data: PatientData):
    return get_prediction(dt_model, data)