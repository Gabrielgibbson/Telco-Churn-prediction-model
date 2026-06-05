from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title='Telco Churn Prediction API')
model = joblib.load("TelcoChurnModel.pkl")
original = joblib.load("label_encoder.pkl")
print("model loaded successfully")

class UserInput(BaseModel):
  gender: float
  SeniorCitizen: float
  Partner: float
  Dependents: float
  tenure: float
  PhoneService: float
  MultipleLines: float
  InternetService: float
  OnlineSecurity: float
  OnlineBackup: float
  DeviceProtection: float
  TechSupport: float
  StreamingTV: float
  StreamingMovies: float
  Contract: float
  PaperlessBilling: float
  PaymentMethod: float
  MonthlyCharges: float
  TotalCharges: float


@app.post("/predict")
def predict_churn(feature: UserInput):
  features = np.array([[
    feature.gender,
    feature.SeniorCitizen,
    feature.Partner,
    feature.Dependents,
    feature.tenure,
    feature.PhoneService,
    feature.MultipleLines,
    feature.InternetService,
    feature.OnlineSecurity,
    feature.OnlineBackup,
    feature.DeviceProtection,
    feature.TechSupport,
    feature.StreamingTV,
    feature.StreamingMovies,
    feature.Contract,
    feature.PaperlessBilling,
    feature.PaymentMethod,
    feature.MonthlyCharges,
    feature.TotalCharges
  ]])
  prediction = model.predict(features)[0]
  prediction_label = original.inverse_transform([prediction])[0]
  return {
        "input": feature,
        "Churn": prediction_label
    }
