import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

model = joblib.load('churn_predictor.pkl')
encoder = joblib.load('churn_encoder.pkl')

partner_encoder = encoder['Partner']
dependent_encoder = encoder['Dependents']
security_encoder = encoder['OnlineSecurity']
backup_encoder = encoder['OnlineBackup']
protection_encoder = encoder['DeviceProtection']
tech_encoder = encoder['TechSupport']
movie_encoder = encoder['StreamingMovies']
contract_encoder = encoder['Contract']
billing_encoder = encoder['PaperlessBilling']
payment_encoder = encoder['PaymentMethod']
churn_encoder = encoder['Churn']

print("model loaded successfully")

class UserInput(BaseModel):
  MonthlyCharges: float
  PaperlessBilling: str
  SeniorCitizen: int
  PaymentMethod: str
  StreamingMovies: str
  Partner: str
  Dependents: str
  DeviceProtection: str
  OnlineBackup: str
  TotalCharges: float
  TechSupport: str
  OnlineSecurity: str
  tenure: int
  Contract: str

@app.post("/predict")
def predict_customer_churn(data: UserInput):
  features = np.array([[
   data.MonthlyCharges,
   billing_encoder.transform([data.PaperlessBilling])[0],
   data.SeniorCitizen,
   payment_encoder.transform([data.PaymentMethod])[0],
   movie_encoder.transform([data.StreamingMovies])[0],
   partner_encoder.transform([data.Partner])[0],
   dependent_encoder.transform([data.Dependents])[0],
   protection_encoder.transform([data.DeviceProtection])[0],
   backup_encoder.transform([data.OnlineBackup])[0],
   data.TotalCharges,
   tech_encoder.transform([data.TechSupport])[0],
   security_encoder.transform([data.OnlineSecurity])[0],
   data.tenure,
   contract_encoder.transform([data.Contract])[0]
  ]])

  prediction = model.predict(features)[0]

  churn_text = churn_encoder.inverse_transform([int(prediction)])[0]
  return {
    'data': data,
    'churn_prediction': churn_text
  }
