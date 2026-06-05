import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(data.head(15))

df = df.drop(["customerID"], axis=1)

# print("\n")
# print(df['SeniorCitizen'].dtype)

# print(df['SeniorCitizen'].corr(df['TotalCharges']))

new_columns = ['gender', 'Partner', 'Dependents', 'Contract', 'PaymentMethod', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Churn']
for column in new_columns:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])

# df = le.fit_transform(df[new_columns])


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
# print(df.info())
# print(df['SeniorCitizen'].corr(df['TotalCharges']))
# df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# print(df[df['TotalCharges'].isnull()])

# print(df.describe())
# print(df.info())

# print(df.corr(numeric_only=True))
df.to_csv("Clean_Telco_dataset.csv", index=False)


encoders = {}

for column in new_columns:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  encoders[column] = le   
model_filename = 'label_encoder.pkl'
joblib.dump(encoders, model_filename)
print(f"Encoders saved permanently {model_filename}")