from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model_GPA = joblib.load("GPA_predictor.pkl")
model_burn = joblib.load("Burnout_risk_level.pkl")
encoders = joblib.load("encoders.pkl")

major_encoder = encoders["Major_Category"]
year_encoder = encoders["Year_of_Study"]
prompt_encoder = encoders["Prompt_Engineering_Skill"]
policy_encoder = encoders["Institutional_Policy"]
burnout_encoder = encoders["Burnout_Risk_Level"]
print("All model loaded successfully")

class UserInput(BaseModel):
  Major_Category: str
  Year_of_Study: str
  Pre_Semester_GPA: float
  Weekly_GenAI_Hours: float
  Prompt_Engineering_Skill: str
  Tool_Diversity: float
  Traditional_Study_Hours: float
  Perceived_AI_Dependency: int
  Institutional_Policy: str
  Anxiety_Level_During_Exams: int
  Skill_Retention_Score: float

@app.post("/predict")
def predict_Next_gpa_and_BurnOut_Risk_Level(data: UserInput):
  features = np.array([[
    major_encoder.transform([data.Major_Category])[0],
    year_encoder.transform([data.Year_of_Study])[0],
    data.Pre_Semester_GPA,
    data.Weekly_GenAI_Hours,
    prompt_encoder.transform([data.Prompt_Engineering_Skill])[0],
    data.Tool_Diversity,
    data.Traditional_Study_Hours,
    data.Perceived_AI_Dependency,
    policy_encoder.transform([data.Institutional_Policy])[0],
    data.Anxiety_Level_During_Exams,
    data.Skill_Retention_Score
  ]])

  prediction1 = model_GPA.predict(features)[0]
  prediction2 = model_burn.predict(features)[0]

  burnout_text = burnout_encoder.inverse_transform([int(prediction2)])[0]

  return {
    'data': data,
    'gpa_prediction': float(prediction1),
    'burnout_prediction': burnout_text
  }



