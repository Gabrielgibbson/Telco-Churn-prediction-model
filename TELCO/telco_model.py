import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("New_clean_telco")
print(df.info())

texts = ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
encoders = {}
for column in texts:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  encoders[column] = le

corr = df.corr(numeric_only=True)
print(corr['Churn'].sort_values(ascending=False))

X = df.drop(['Churn', 'MultipleLines', 'PhoneService', 'gender', 'customerID', 'StreamingTV', 'InternetService'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = RandomForestClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
# print(accuracy)

joblib.dump(model, 'churn_predictor.pkl')
joblib.dump(encoders, 'churn_encoder.pkl')
print("Model saved successfully")