from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Wine Acidity Prediction API")

model = joblib.load("predicting_wine_acidity.pkl")
print("model loaded successfully")

class AcidInput(BaseModel):
    citric_acid: float
    density: float


@app.post("/predict")
def predict_wine_acidity(fixed_acidity: AcidInput):
    features = np.array([[fixed_acidity.citric_acid, fixed_acidity.density]])
    prediction = model.predict(features)
    if prediction < 4.0 or prediction > 15:
        status_message ="Not advisable to consume"
    else:
        status_message ="Advisable to consume"
    return {
        "citric_acid": fixed_acidity.citric_acid,
        "density": fixed_acidity.density,
        "predicted_wine_acidity": float(prediction[0]),
        "status": status_message
    }