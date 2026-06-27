from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import joblib
from pydantic import BaseModel
app=FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "artifacts" / "model.pkl")
preprocessor = joblib.load(BASE_DIR / "artifacts" / "scaler.pkl")

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: Transaction):

    df= pd.DataFrame([data.model_dump()])
    df[["Amount","Time"]] = preprocessor.transform(df[["Amount","Time"]])

    prediction=model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    result = "Fraud" if prediction == 1 else "Legitimate"

    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 6)
    }